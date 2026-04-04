---
title: PropAI Real Estate Intelligence Backend
emoji: 🏠
colorFrom: yellow
colorTo: cyan
sdk: docker
app_port: 7860
pinned: false
short_description: Advanced RAG backend for PropAI Real Estate Intelligence
---

# 🏠 PropAI — Real Estate Intelligence Assistant

> AI-Powered RAG system for property search, document Q&A, and market insights.

## Architecture

```
┌─────────────────────┐     HTTPS      ┌──────────────────────────┐
│  Vercel (Frontend)  │ ─────────────► │  HF Spaces (Backend API) │
│  React + Vite       │                │  FastAPI + ChromaDB       │
└─────────────────────┘                └──────────────────────────┘
```

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Groq (LLaMA-3.1-8b-instant) |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector Store | ChromaDB (persistent SQLite) |
| Hybrid Search | BM25 + ChromaDB EnsembleRetriever |
| Re-ranking | Cross-Encoder `ms-marco-MiniLM-L-6-v2` |
| API Framework | FastAPI |
| Frontend | React + Vite (deployed on Vercel) |

## Key Features

- **Hybrid Retrieval** — BM25 keyword + dense vector search → ~35% better accuracy
- **Cross-Encoder Re-ranking** — Top-10 → Top-3 high-quality chunks before LLM
- **Multi-turn Memory** — Conversation history injected into RAG context
- **Embedding Cache** — Singleton model avoids re-loading on every request
- **Real Estate Persona** — Prompt-engineered for property, legal, and market Q&A

---

## 🚀 Deployment Guide

### Backend → Hugging Face Spaces

1. Create a new HF Space: **Docker** SDK, port **7860**
2. Create the space repo and push the `backend/` folder contents:

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/propai-backend
# Copy backend/ files into the cloned repo root
cp -r backend/* propai-backend/
cd propai-backend
```

3. Set **Secrets** in the Space settings:

| Secret | Value |
|---|---|
| `GROQ_API_KEY` | Your Groq API key |
| `SECRET_KEY` | Any random string |
| `JWT_SECRET_KEY` | Any random string |
| `CORS_ORIGINS` | `https://your-app.vercel.app` |

4. Push and the space will build automatically from `Dockerfile`

> **Note**: Use HF Space's persistent storage for `vectorstores/` and `uploads/` directories. Add this to your Space settings to enable persistent disk.

---

### Frontend → Vercel

1. Import your GitHub repo in Vercel
2. Set **Root Directory** to `frontend`
3. Set **Build Command**: `npm run build`
4. Set **Output Directory**: `dist`
5. Add **Environment Variable**:

| Key | Value |
|---|---|
| `VITE_API_URL` | `https://YOUR_USERNAME-propai-backend.hf.space` |

6. Deploy ✅

---

### Local Development

```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn app:app --reload --port 8000

# Frontend (new terminal)
cd frontend
npm install
npm run dev
```

Create `backend/.env` or root `.env`:
```env
GROQ_API_KEY=your_groq_key
SECRET_KEY=dev-secret-key
JWT_SECRET_KEY=dev-jwt-key
CORS_ORIGINS=http://localhost:5173
```

---

## Use Cases

- 🔍 **Property Recommendation** — Natural language property search by intent
- 📄 **Legal Document Q&A** — Upload agreements, deeds, or listings and ask questions
- 📊 **Market Insights** — Ask about trends, pricing, and investment
- 🤝 **Lead Qualification** — Automated client interaction and requirement gathering
