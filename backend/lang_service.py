import os
import shutil
import math
import re
from functools import lru_cache
from importlib import import_module
from typing import Any
from collections import Counter

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
    from config import Config
except ImportError:
    from .config import Config

VECTOR_STORE_PATH = Config.VECTORSTORE_DIR
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
SUPPORTED_TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".json", ".log", ".rst"}


def _get_ensemble_cls():
    for module_name in ("langchain.retrievers", "langchain_classic.retrievers"):
        try:
            module = import_module(module_name)
            return getattr(module, "EnsembleRetriever")
        except (ImportError, AttributeError):
            continue
    raise ImportError("EnsembleRetriever not found in installed LangChain packages")


@lru_cache(maxsize=1)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )



@lru_cache(maxsize=1)
def get_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


@lru_cache(maxsize=1)
def _get_ocr():
    return RapidOCR()


# In-memory doc cache for BM25 (per chat_id)
_docs_cache: dict[int, list[Document]] = {}

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "he", "her", "his", "i", "in", "is", "it", "its", "me", "my", "of", "on",
    "or", "our", "she", "that", "the", "their", "them", "they", "this", "to", "was",
    "we", "were", "what", "when", "where", "which", "who", "why", "will", "with", "you",
}


def _ocr_lines(ocr_result: Any) -> list[str]:
    lines: list[str] = []
    if not ocr_result:
        return lines

    for item in ocr_result:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            text = str(item[1]).strip()
            if text:
                lines.append(text)
    return lines


def _tokenize(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2 and token not in _STOPWORDS]


def _safe_div(numerator: float, denominator: float) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _jaccard(tokens_a: set[str], tokens_b: set[str]) -> float:
    if not tokens_a or not tokens_b:
        return 0.0
    return round(len(tokens_a & tokens_b) / len(tokens_a | tokens_b), 4)


def _cosine_sim(text_a: str, text_b: str) -> float:
    tokens_a = _tokenize(text_a)
    tokens_b = _tokenize(text_b)
    if not tokens_a or not tokens_b:
        return 0.0

    counts_a = Counter(tokens_a)
    counts_b = Counter(tokens_b)
    common = set(counts_a) & set(counts_b)
    numerator = sum(counts_a[token] * counts_b[token] for token in common)
    norm_a = math.sqrt(sum(value * value for value in counts_a.values()))
    norm_b = math.sqrt(sum(value * value for value in counts_b.values()))
    if not norm_a or not norm_b:
        return 0.0
    return round(numerator / (norm_a * norm_b), 4)


def _doc_score(query_tokens: set[str], doc_text: str) -> float:
    doc_tokens = set(_tokenize(doc_text))
    if not query_tokens or not doc_tokens:
        return 0.0
    return len(query_tokens & doc_tokens) / len(query_tokens)


def _calc_metrics(query: str, answer: str, docs: list) -> dict[str, float]:
    context_text = "\n\n".join(doc.page_content for doc in docs)
    query_tokens = set(_tokenize(query))
    answer_tokens = _tokenize(answer)

    doc_scores = [_doc_score(query_tokens, doc.page_content) for doc in docs]
    relevance_threshold = 0.15 if query_tokens else 0.0
    labels = [1 if score >= relevance_threshold else 0 for score in doc_scores]

    relevant_docs = sum(labels)
    retrieved_docs = len(docs)
    possible_relevant_docs = max(1, sum(1 for score in doc_scores if score > 0))

    first_relevant_rank = next((index + 1 for index, label in enumerate(labels) if label), None)
    precision_at_k = _safe_div(relevant_docs, retrieved_docs)
    recall_at_k = _safe_div(relevant_docs, possible_relevant_docs)
    mrr = round(1 / first_relevant_rank, 4) if first_relevant_rank else 0.0

    precision_sum = 0.0
    hit_count = 0
    for index, label in enumerate(labels, start=1):
        if label:
            hit_count += 1
            precision_sum += hit_count / index
    map_score = _safe_div(precision_sum, possible_relevant_docs)

    dcg = sum(label / math.log2(index + 1) for index, label in enumerate(labels, start=1))
    ideal_labels = sorted(labels, reverse=True)
    idcg = sum(label / math.log2(index + 1) for index, label in enumerate(ideal_labels, start=1))
    ndcg = _safe_div(dcg, idcg)

    supported_tokens = sum(1 for token in answer_tokens if token in set(_tokenize(context_text)))
    faithfulness = _safe_div(supported_tokens, len(answer_tokens))
    answer_relevancy = _jaccard(set(answer_tokens), query_tokens)
    semantic_similarity = _cosine_sim(answer, context_text)
    correctness = round((faithfulness + answer_relevancy + semantic_similarity) / 3, 4)
    hallucination_rate = round(max(0.0, 1.0 - faithfulness), 4)

    return {
        "context_precision": precision_at_k,
        "context_recall": recall_at_k,
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
        "mrr": mrr,
        "map": map_score,
        "ndcg": ndcg,
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "correctness": correctness,
        "hallucination_rate": hallucination_rate,
        "semantic_similarity": semantic_similarity,
    }


def _ocr_from_image(image: Image.Image) -> str:
    rgb_image = image.convert("RGB")
    result, _ = _get_ocr()(np.array(rgb_image))
    lines = _ocr_lines(result)
    return "\n".join(lines).strip()


def _load_image_docs(filepath: str) -> list[Document]:
    with Image.open(filepath) as image:
        text = _ocr_from_image(image)
        description = f"Image file: {os.path.basename(filepath)} ({image.width}x{image.height})"

    if text:
        content = f"{description}\n\nExtracted text:\n{text}"
    else:
        content = f"{description}\n\nNo readable text detected in the image."

    return [Document(page_content=content, metadata={"source": filepath, "ocr": True, "type": "image"})]


def _load_pdf_docs(filepath: str) -> list[Document]:
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
        text = _ocr_from_image(image)
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


def clear_rag_data(chat_id: int):
    # Drop in-memory sparse retrieval cache for the deleted chat.
    _docs_cache.pop(chat_id, None)

    # Remove persisted Chroma storage for this chat.
    persist_path = os.path.join(VECTOR_STORE_PATH, str(chat_id))
    if os.path.isdir(persist_path):
        shutil.rmtree(persist_path, ignore_errors=True)



def create_vectorstore(filepath: str, chat_id: int):
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".pdf":
        documents = _load_pdf_docs(filepath)
    elif ext in SUPPORTED_IMAGE_EXTENSIONS:
        documents = _load_image_docs(filepath)
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



def rerank_docs(query: str, docs: list, top_k: int = 3) -> list:
    if len(docs) <= top_k:
        return docs
    pairs = [(query, doc.page_content) for doc in docs]
    scores = get_reranker().predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_k]]



def get_rag_response(chat_id: int, query: str, history: list = None) -> dict[str, Any]:
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
        EnsembleRetriever = _get_ensemble_cls()
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

    answer = rag_chain.invoke(query)
    metrics = _calc_metrics(query, answer, top_docs)

    return {
        "answer": answer,
        "metrics": metrics,
    }