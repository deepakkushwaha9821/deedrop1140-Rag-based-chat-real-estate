import os
import shutil
from importlib import import_module
from typing import Any

import numpy as np
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from PIL import Image
from rapidocr_onnxruntime import RapidOCR
from sentence_transformers import CrossEncoder

try:
    from .config import Config
except ImportError:
    from config import Config

VECTOR_STORE_PATH = Config.VECTORSTORE_DIR
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
SUPPORTED_TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".json", ".log", ".rst"}


def get_ensemble_retriever_class():
    for module_name in ("langchain.retrievers", "langchain_classic.retrievers"):
        try:
            module = import_module(module_name)
            return getattr(module, "EnsembleRetriever")
        except (ImportError, AttributeError):
            continue
    raise ImportError("EnsembleRetriever not found in installed LangChain packages")

# ── Singleton embedding model (cached) ──────────────────────────────────────
_embeddings = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


# ── Singleton cross-encoder re-ranker ───────────────────────────────────────
_reranker = None
_ocr_engine = None

def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


def get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        _ocr_engine = RapidOCR()
    return _ocr_engine


# ── In-memory doc cache for BM25 (per chat_id) ──────────────────────────────
_docs_cache: dict = {}


def _extract_ocr_lines(ocr_result: Any) -> list[str]:
    lines: list[str] = []
    if not ocr_result:
        return lines

    for item in ocr_result:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            text = str(item[1]).strip()
            if text:
                lines.append(text)
    return lines


def _ocr_text_from_pil_image(image: Image.Image) -> str:
    rgb_image = image.convert("RGB")
    result, _ = get_ocr_engine()(np.array(rgb_image))
    lines = _extract_ocr_lines(result)
    return "\n".join(lines).strip()


def _load_image_documents(filepath: str) -> list[Document]:
    with Image.open(filepath) as image:
        text = _ocr_text_from_pil_image(image)
        description = f"Image file: {os.path.basename(filepath)} ({image.width}x{image.height})"

    if text:
        content = f"{description}\n\nExtracted text:\n{text}"
    else:
        content = f"{description}\n\nNo readable text detected in the image."

    return [Document(page_content=content, metadata={"source": filepath, "ocr": True, "type": "image"})]


def _load_pdf_documents(filepath: str) -> list[Document]:
    documents = PyPDFLoader(filepath).load()
    if any(doc.page_content.strip() for doc in documents):
        return documents

    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise RuntimeError("OCR for scanned PDFs requires pypdfium2") from exc

    pdf = pdfium.PdfDocument(filepath)
    ocr_docs: list[Document] = []

    for page_index in range(len(pdf)):
        page = pdf[page_index]
        bitmap = page.render(scale=2)
        image = bitmap.to_pil()
        text = _ocr_text_from_pil_image(image)
        if text:
            ocr_docs.append(
                Document(
                    page_content=text,
                    metadata={"source": filepath, "page": page_index + 1, "ocr": True, "type": "pdf"},
                )
            )

    if not ocr_docs:
        raise ValueError("No readable text found in PDF, including OCR fallback")

    return ocr_docs


def clear_chat_rag_data(chat_id: int):
    # Drop in-memory sparse retrieval cache for the deleted chat.
    _docs_cache.pop(chat_id, None)

    # Remove persisted Chroma storage for this chat.
    persist_path = os.path.join(VECTOR_STORE_PATH, str(chat_id))
    if os.path.isdir(persist_path):
        shutil.rmtree(persist_path, ignore_errors=True)


# ── CREATE VECTOR STORE ──────────────────────────────────────────────────────
def create_vectorstore(filepath: str, chat_id: int):
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".pdf":
        documents = _load_pdf_documents(filepath)
    elif ext in SUPPORTED_IMAGE_EXTENSIONS:
        documents = _load_image_documents(filepath)
    elif ext in SUPPORTED_TEXT_EXTENSIONS:
        loader = TextLoader(filepath, encoding="utf-8", autodetect_encoding=True)
        documents = loader.load()
    else:
        raise ValueError("Unsupported file type for RAG upload")

    if not documents or not any(d.page_content.strip() for d in documents):
        raise ValueError("Uploaded file has no readable text content")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=120,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    docs = splitter.split_documents(documents)

    for doc in docs:
        doc.metadata["chat_id"] = str(chat_id)

    # Cache docs for BM25
    _docs_cache[chat_id] = docs

    persist_path = os.path.join(VECTOR_STORE_PATH, str(chat_id))
    os.makedirs(persist_path, exist_ok=True)

    Chroma.from_documents(
        docs,
        get_embeddings(),
        persist_directory=persist_path,
        collection_name=f"chat_{chat_id}",
    )


# ── RE-RANK ──────────────────────────────────────────────────────────────────
def rerank_docs(query: str, docs: list, top_k: int = 3) -> list:
    if len(docs) <= top_k:
        return docs
    pairs = [(query, doc.page_content) for doc in docs]
    scores = get_reranker().predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_k]]


# ── GET RAG RESPONSE ─────────────────────────────────────────────────────────
def get_rag_response(chat_id: int, query: str, history: list = None) -> str:
    persist_path = os.path.join(VECTOR_STORE_PATH, str(chat_id))

    if not Config.GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not configured")

    embeddings = get_embeddings()

    # Dense retriever (ChromaDB)
    vectorstore = Chroma(
        persist_directory=persist_path,
        embedding_function=embeddings,
        collection_name=f"chat_{chat_id}",
    )
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    # Sparse retriever (BM25) — hybrid if docs are cached
    cached_docs = _docs_cache.get(chat_id)
    if cached_docs:
        bm25 = BM25Retriever.from_documents(cached_docs)
        bm25.k = 6
        EnsembleRetriever = get_ensemble_retriever_class()
        retriever = EnsembleRetriever(
            retrievers=[bm25, dense_retriever],
            weights=[0.4, 0.6],
        )
    else:
        retriever = dense_retriever

    # Retrieve → Re-rank → Format context
    raw_docs = retriever.invoke(query)
    top_docs = rerank_docs(query, raw_docs, top_k=3)

    context = "\n\n---\n\n".join(
        f"[Source {i+1}]\n{doc.page_content}" for i, doc in enumerate(top_docs)
    )

    # Build conversation history string (last 3 turns)
    history_text = ""
    if history:
        for msg in history[-6:]:
            role = "User" if msg.get("role") == "user" else "PropAI"
            history_text += f"{role}: {msg.get('content', '')}\n"

    llm = ChatGroq(
        groq_api_key=Config.GROQ_API_KEY,
        model="llama-3.1-8b-instant",
        temperature=0.1,
    )

    prompt = ChatPromptTemplate.from_template(
        """You are PropAI, an expert AI assistant specializing in real estate intelligence.
You help with property search, legal document analysis, market insights, lead qualification, and investment advice.
Use ONLY the context below to answer. If the answer is not in the context, say you need more document information.
Be specific, professional, and cite relevant details.

{history_section}

Context from documents:
{context}

User Question: {question}

Answer:"""
    )

    history_section = f"Previous conversation:\n{history_text}" if history_text else ""

    rag_chain = (
        {
            "context": lambda _: context,
            "question": RunnablePassthrough(),
            "history_section": lambda _: history_section,
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain.invoke(query)