import os
from datetime import datetime, timedelta, timezone
from typing import Generator

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel
from sqlalchemy.orm import Session
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

try:
    from .config import Config
    from .lang_service import clear_chat_rag_data, create_vectorstore, get_rag_response
    from .langgraph_service import get_response
    from .models import Chat, Message, SessionLocal, UploadedFile as ModelUploadedFile, User, init_db
except ImportError:
    from config import Config
    from lang_service import clear_chat_rag_data, create_vectorstore, get_rag_response
    from langgraph_service import get_response
    from models import Chat, Message, SessionLocal, UploadedFile as ModelUploadedFile, User, init_db


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

UPLOAD_FOLDER = Config.UPLOAD_DIR
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(Config.VECTORSTORE_DIR, exist_ok=True)
init_db()


class RegisterInput(BaseModel):
    username: str
    password: str


class LoginInput(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserOut(BaseModel):
    id: int
    username: str


class ChatOut(BaseModel):
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
    id: int
    chat_id: int
    role: str
    content: str
    timestamp: datetime


class UploadedFileOut(BaseModel):
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


def chat_to_out(chat: Chat) -> ChatOut:
    return ChatOut(
        id=chat.id,
        user_id=chat.user_id,
        title=chat.title,
        mode=chat.mode,
        is_pinned=chat.is_pinned,
        is_archived=chat.is_archived,
        created_at=chat.created_at,
    )


def message_to_out(message: Message) -> MessageOut:
    return MessageOut(
        id=message.id,
        chat_id=message.chat_id,
        role=message.role,
        content=message.content,
        timestamp=message.timestamp,
    )


def file_to_out(uploaded_file: ModelUploadedFile) -> UploadedFileOut:
    return UploadedFileOut(
        id=uploaded_file.id,
        chat_id=uploaded_file.chat_id,
        filename=uploaded_file.filename,
        filepath=uploaded_file.filepath,
        uploaded_at=uploaded_file.uploaded_at,
    )


def get_user_chat_or_404(db: Session, chat_id: int, user_id: int) -> Chat:
    chat = db.query(Chat).filter(Chat.id == chat_id, Chat.user_id == user_id).first()
    if not chat:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found")
    return chat


@app.get("/")
def root():
    return {"message": "FastAPI backend is running"}


@app.post("/api/auth/register", response_model=UserOut)
def register(payload: RegisterInput, db: Session = Depends(get_db)):
    exists = db.query(User).filter(User.username == payload.username).first()
    if exists:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username already exists")

    user = User(username=payload.username, password=generate_password_hash(payload.password))
    db.add(user)
    db.commit()
    db.refresh(user)

    return UserOut(id=user.id, username=user.username)


@app.post("/api/auth/login", response_model=TokenResponse)
def login(payload: LoginInput, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == payload.username).first()
    if not user or not check_password_hash(user.password, payload.password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    token = create_access_token(user.id)
    return TokenResponse(access_token=token)


@app.get("/api/auth/me", response_model=UserOut)
def me(current_user: User = Depends(get_current_user)):
    return UserOut(id=current_user.id, username=current_user.username)


@app.get("/api/chats", response_model=list[ChatOut])
def list_chats(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    chats = (
        db.query(Chat)
        .filter(Chat.user_id == current_user.id, Chat.is_archived.is_(False))
        .order_by(Chat.is_pinned.desc(), Chat.created_at.desc())
        .all()
    )
    return [chat_to_out(chat) for chat in chats]


@app.post("/api/chats", response_model=ChatOut)
def create_chat(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    chat = Chat(user_id=current_user.id, mode="normal")
    db.add(chat)
    db.commit()
    db.refresh(chat)
    return chat_to_out(chat)


@app.get("/api/chats/{chat_id}", response_model=ChatDetailOut)
def get_chat(chat_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    chat = get_user_chat_or_404(db, chat_id, current_user.id)

    messages = db.query(Message).filter(Message.chat_id == chat_id).order_by(Message.timestamp).all()
    files = db.query(ModelUploadedFile).filter(ModelUploadedFile.chat_id == chat_id).all()

    return ChatDetailOut(
        chat=chat_to_out(chat),
        messages=[message_to_out(message) for message in messages],
        files=[file_to_out(uploaded_file) for uploaded_file in files],
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

    messages = db.query(Message).filter(Message.chat_id == chat_id).order_by(Message.timestamp).all()

    if chat.mode == "rag":
        try:
            history_dicts = [
                {"role": msg.role, "content": msg.content}
                for msg in messages
            ]
            ai_text = get_rag_response(chat_id, user_input, history=history_dicts)
        except Exception:
            ai_text = "Error retrieving document context. Please check your uploaded file."
    else:
        history = []
        for msg in messages:
            if msg.role == "user":
                history.append(HumanMessage(content=msg.content))
            else:
                history.append(AIMessage(content=msg.content))

        history.append(HumanMessage(content=user_input))
        try:
            ai_response = get_response(history)
            ai_text = ai_response.content
        except Exception:
            ai_text = "AI service is not configured. Please set GROQ_API_KEY on the server."

    db.add(Message(chat_id=chat_id, role="user", content=user_input))
    ai_message = Message(chat_id=chat_id, role="ai", content=ai_text)
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

    filepath = os.path.join(UPLOAD_FOLDER, f"{chat_id}_{filename}")

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

    return file_to_out(uploaded)


@app.post("/api/chats/{chat_id}/pin", response_model=ChatOut)
def pin_chat(chat_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    chat = get_user_chat_or_404(db, chat_id, current_user.id)
    chat.is_pinned = not chat.is_pinned
    db.commit()
    db.refresh(chat)
    return chat_to_out(chat)


@app.post("/api/chats/{chat_id}/archive", response_model=ChatOut)
def archive_chat(chat_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    chat = get_user_chat_or_404(db, chat_id, current_user.id)
    chat.is_archived = True
    db.commit()
    db.refresh(chat)
    return chat_to_out(chat)


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

    clear_chat_rag_data(chat_id)

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
