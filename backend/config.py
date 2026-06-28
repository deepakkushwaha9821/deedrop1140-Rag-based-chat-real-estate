import os
from pathlib import Path

from dotenv import load_dotenv


BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent

load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(BACKEND_DIR / ".env")


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "change-me")
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", SECRET_KEY)
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL", f"sqlite:///{BACKEND_DIR / 'database.db'}")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    # Override via GROQ_MODEL env var (default: llama-3.3-70b-versatile)
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "10080"))
    CORS_ORIGINS = [
        origin.strip()
        for origin in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
        if origin.strip()
    ]
    CORS_ORIGIN_REGEX = os.getenv("CORS_ORIGIN_REGEX", r"^https://.*\.vercel\.app$").strip() or None
    UPLOAD_DIR = str(BACKEND_DIR / "uploads")
    VECTORSTORE_DIR = str(BACKEND_DIR / "vectorstores")