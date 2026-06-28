import os
import json
from datetime import datetime, timedelta, timezone
from typing import Generator

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

try:
    from config import Config
    from lang_service import clear_rag_data, create_vectorstore, get_rag_response
    from langgraph_service import get_response
    from models import Chat, Message, SessionLocal, UploadedFile as ModelUploadedFile, User, init_db
except ImportError:
    from .config import Config
    from .lang_service import clear_rag_data, create_vectorstore, get_rag_response
    from .langgraph_service import get_response
    from .models import Chat, Message, SessionLocal, UploadedFile as ModelUploadedFile, User, init_db


app = FastAPI(
    title="Real Estate Intelligence API",
    description="AI-Powered Real Estate Intelligence Assistant — PropAI",
    version="1.0.0",
)
security = HTTPBearer()

app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.CORS_ORIGINS,
    allow_origin_regex=Config.CORS_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(Config.UPLOAD_DIR, exist_ok=True)
os.makedirs(Config.VECTORSTORE_DIR, exist_ok=True)
init_db()


class AuthInput(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    username: str


class ChatOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    user_id: int
    title: str
    mode: str
    is_pinned: bool
    is_archived: bool
    created_at: datetime


class MessageInput(BaseModel):
    message: str


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    chat_id: int
    role: str
    content: str
    metrics: dict[str, float] | None = None
    timestamp: datetime


class UploadedFileOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    chat_id: int
    filename: str
    filepath: str
    uploaded_at: datetime


class ChatDetailOut(BaseModel):
    chat: ChatOut
    messages: list[MessageOut]
    files: list[UploadedFileOut]


class AboutOut(BaseModel):
    app_name: str
    stack: list[str]


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_access_token(user_id: int) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=Config.ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": str(user_id), "exp": expire}
    return jwt.encode(payload, Config.JWT_SECRET_KEY, algorithm="HS256")


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
) -> User:
    token = credentials.credentials
    try:
        payload = jwt.decode(token, Config.JWT_SECRET_KEY, algorithms=["HS256"])
        user_id = int(payload.get("sub"))
    except (JWTError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token") from exc

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user


def message_to_out(message: Message) -> MessageOut:
    metrics = None
    if message.metrics_json:
        try:
            metrics = json.loads(message.metrics_json)
        except json.JSONDecodeError:
            pass

    out = MessageOut.model_validate(message)
    out.metrics = metrics
    return out


def get_user_chat_or_404(db: Session, chat_id: int, user_id: int) -> Chat:
    chat = db.query(Chat).filter(Chat.id == chat_id, Chat.user_id == user_id).first()
    if not chat:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found")
    return chat


@app.get("/")
def root():
    return {"message": "FastAPI backend is running"}


@app.post("/api/auth/register", response_model=UserOut)
def register(payload: AuthInput, db: Session = Depends(get_db)):
    exists = db.query(User).filter(User.username == payload.username).first()
    if exists:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username already exists")

    user = User(username=payload.username, password=generate_password_hash(payload.password))
    db.add(user)
    db.commit()
    db.refresh(user)

    return UserOut.model_validate(user)


@app.post("/api/auth/login", response_model=TokenResponse)
def login(payload: AuthInput, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == payload.username).first()
    if not user or not check_password_hash(user.password, payload.password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    token = create_access_token(user.id)
    return TokenResponse(access_token=token)


@app.get("/api/auth/me", response_model=UserOut)
def me(current_user: User = Depends(get_current_user)):
    return UserOut.model_validate(current_user)


@app.get("/api/chats", response_model=list[ChatOut])
def list_chats(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    chats = (
        db.query(Chat)
        .filter(Chat.user_id == current_user.id, Chat.is_archived.is_(False))
        .order_by(Chat.is_pinned.desc(), Chat.created_at.desc())
        .all()
    )
    return [ChatOut.model_validate(chat) for chat in chats]


@app.post("/api/chats", response_model=ChatOut)
def create_chat(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    chat = Chat(user_id=current_user.id, mode="normal")
    db.add(chat)
    db.commit()
    db.refresh(chat)
    return ChatOut.model_validate(chat)


@app.get("/api/chats/{chat_id}", response_model=ChatDetailOut)
def get_chat(chat_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    chat = get_user_chat_or_404(db, chat_id, current_user.id)

    chat_messages = db.query(Message).filter(Message.chat_id == chat_id).order_by(Message.timestamp).all()
    files = db.query(ModelUploadedFile).filter(ModelUploadedFile.chat_id == chat_id).all()

    return ChatDetailOut(
        chat=ChatOut.model_validate(chat),
        messages=[message_to_out(msg) for msg in chat_messages],
        files=[UploadedFileOut.model_validate(f) for f in files],
    )


@app.post("/api/chats/{chat_id}/messages", response_model=MessageOut)
def send_message(
    chat_id: int,
    payload: MessageInput,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    chat = get_user_chat_or_404(db, chat_id, current_user.id)

    user_input = payload.message.strip()
    if not user_input:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Message cannot be empty")

    if chat.title == "New Chat":
        chat.title = " ".join(user_input.split()[:6])

    MSG_CLASSES = {"user": HumanMessage, "ai": AIMessage}
    history_records = db.query(Message).filter(Message.chat_id == chat_id).order_by(Message.timestamp).all()

    if chat.mode == "rag":
        try:
            history_dicts = [{"role": m.role, "content": m.content} for m in history_records]
            rag_result = get_rag_response(chat_id, user_input, history=history_dicts)
            ai_text = rag_result["answer"]
            ai_metrics = rag_result["metrics"]
        except Exception:
            ai_text = "Error retrieving document context. Please check your uploaded file."
            ai_metrics = None
    else:
        history = [MSG_CLASSES.get(m.role, AIMessage)(content=m.content) for m in history_records]
        history.append(HumanMessage(content=user_input))
        try:
            ai_response = get_response(history)
            ai_text = ai_response.content
            ai_metrics = None
        except Exception:
            ai_text = "AI service is not configured. Please set GROQ_API_KEY on the server."
            ai_metrics = None

    db.add(Message(chat_id=chat_id, role="user", content=user_input))
    ai_message = Message(
        chat_id=chat_id,
        role="ai",
        content=ai_text,
        metrics_json=json.dumps(ai_metrics) if ai_metrics else None,
    )
    db.add(ai_message)
    db.commit()
    db.refresh(ai_message)

    return message_to_out(ai_message)


@app.post("/api/chats/{chat_id}/upload", response_model=UploadedFileOut)
async def upload_file(
    chat_id: int,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    chat = get_user_chat_or_404(db, chat_id, current_user.id)

    filename = secure_filename(file.filename or "document.txt")
    if not filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid filename")

    filepath = os.path.join(Config.UPLOAD_DIR, f"{chat_id}_{filename}")

    content = await file.read()
    with open(filepath, "wb") as out_file:
        out_file.write(content)

    try:
        create_vectorstore(filepath, chat_id)
    except Exception as exc:
        if os.path.exists(filepath):
            os.remove(filepath)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to process file. Upload a supported text file, image (png/jpg/webp), or PDF. Scanned PDFs/images use OCR.",
        ) from exc

    uploaded = ModelUploadedFile(chat_id=chat_id, filename=filename, filepath=filepath)
    chat.mode = "rag"

    db.add(uploaded)
    db.commit()
    db.refresh(uploaded)

    return UploadedFileOut.model_validate(uploaded)


@app.post("/api/chats/{chat_id}/pin", response_model=ChatOut)
def pin_chat(chat_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    chat = get_user_chat_or_404(db, chat_id, current_user.id)
    chat.is_pinned = not chat.is_pinned
    db.commit()
    db.refresh(chat)
    return ChatOut.model_validate(chat)


@app.post("/api/chats/{chat_id}/archive", response_model=ChatOut)
def archive_chat(chat_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    chat = get_user_chat_or_404(db, chat_id, current_user.id)
    chat.is_archived = True
    db.commit()
    db.refresh(chat)
    return ChatOut.model_validate(chat)


@app.delete("/api/chats/{chat_id}")
def delete_chat(chat_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    chat = get_user_chat_or_404(db, chat_id, current_user.id)

    uploaded_files = db.query(ModelUploadedFile).filter(ModelUploadedFile.chat_id == chat_id).all()
    for uploaded_file in uploaded_files:
        if uploaded_file.filepath and os.path.isfile(uploaded_file.filepath):
            try:
                os.remove(uploaded_file.filepath)
            except OSError:
                pass

    clear_rag_data(chat_id)

    db.query(Message).filter(Message.chat_id == chat_id).delete()
    db.query(ModelUploadedFile).filter(ModelUploadedFile.chat_id == chat_id).delete()
    db.delete(chat)
    db.commit()

    return {"deleted": True}


@app.get("/api/about", response_model=AboutOut)
def about():
    return AboutOut(
        app_name="PropAI — Real Estate Intelligence Assistant",
        stack=[
            "FastAPI", "React", "LangGraph", "ChromaDB",
            "Groq (LLaMA-3.1)", "HuggingFace Embeddings",
            "BM25 Hybrid Search", "Cross-Encoder Re-ranking",
        ],
    )
