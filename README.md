# 💼 Job Application Assistant — AI Agentic Chatbot

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat-square&logo=fastapi)
![LangGraph](https://img.shields.io/badge/LangGraph-0.1-orange?style=flat-square)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5-purple?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.36-red?style=flat-square&logo=streamlit)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?style=flat-square&logo=docker)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-2088FF?style=flat-square&logo=githubactions)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

> An **agentic AI chatbot** that helps job seekers tailor their resume, identify skill gaps, generate cover letters, and prepare for interviews — all through a single conversational interface powered by LangGraph, RAG, and FastAPI.

---

## 📸 Demo

```
User: [uploads resume PDF]
Bot:  ✅ Resume indexed! Found 24 skills. Paste a job description to get started.

User: [pastes Senior ML Engineer JD from Google]
Bot:  🔍 Gap Analysis Complete
      Match Score: 71%
      ✓ Matched: Python, TensorFlow, Docker, REST APIs, SQL ...
      ✗ Missing: Kubernetes, JAX, TPU optimization, Go
      Summary: Strong candidate for the ML foundations. Focus on cloud-native ...

User: Generate a cover letter for this role
Bot:  📝 Your Tailored Cover Letter
      Dear Hiring Manager, ...

User: Give me interview prep questions
Bot:  🎯 Interview Prep Questions
      Q1. [Behavioral] Tell me about a time you optimized a model pipeline...
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      USER INTERFACE                         │
│                  Streamlit Chat App                         │
│              (file upload + chat history)                   │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP REST
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Backend                          │
│   POST /upload-resume   POST /chat   DELETE /session/{id}  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│               LangGraph Agent (State Graph)                 │
│                                                             │
│   ┌──────────────┐     conditional edges                    │
│   │   classify   │ ──────────────────────────────────────┐  │
│   │   intent     │                                       │  │
│   └──────────────┘                                       │  │
│         │                                                │  │
│    ┌────┴──────────────────────────────────┐             │  │
│    ▼           ▼              ▼            ▼             │  │
│ [gap_analysis] [cover_letter] [interview] [general]      │  │
│    │           │              │            │             │  │
│    └───────────┴──────────────┴────────────┘             │  │
│                        │                                 │  │
│                       END                                │  │
└────────────────────────┬─────────────────────────────────┘
                         │
          ┌──────────────┴──────────────┐
          ▼                             ▼
┌─────────────────┐           ┌──────────────────┐
│   ChromaDB      │           │   OpenAI / Groq  │
│  Vector Store   │           │   LLM API        │
│ (resume chunks) │           │  (gpt-4o-mini or │
│                 │           │   llama3-8b)     │
└─────────────────┘           └──────────────────┘
          ▲
          │ store/query embeddings
┌─────────────────┐
│ SentenceTransf. │
│ all-MiniLM-L6   │
│  (embeddings)   │
└─────────────────┘
```

### Agent Tool Nodes

| Tool | Trigger | What it does |
|------|---------|--------------|
| `gap_analysis` | User pastes JD | Extracts JD requirements → queries resume via vector similarity → scores match |
| `cover_letter` | "generate cover letter" | RAG over resume chunks + JD → LLM writes tailored letter |
| `interview_prep` | "interview questions" / "prepare" | RAG + LLM generates 8 questions with STAR-format hints |
| `general` | Everything else | Conversational LLM with chat history context |

---

## 🗂️ Project Structure

```
job-assistant-agent/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI entry point — 3 endpoints
│   ├── agent.py             # LangGraph state graph + 4 tool nodes
│   ├── vector_store.py      # ChromaDB wrapper (store / query / delete)
│   ├── models.py            # Pydantic request/response schemas
│   └── tools/
│       ├── __init__.py
│       ├── resume_tool.py   # PDF parser + section-aware chunker
│       ├── gap_tool.py      # JD requirement extractor + gap scorer
│       ├── coverletter_tool.py  # RAG-based cover letter generator
│       └── interview_tool.py    # Interview question + hint generator
├── ui/
│   └── streamlit_app.py     # Streamlit chat interface
├── tests/
│   ├── __init__.py
│   └── test_tools.py        # Unit tests (pytest)
├── .github/
│   └── workflows/
│       └── deploy.yml       # CI/CD: test → docker build → Render deploy
├── .env.example             # Environment variable template
├── Dockerfile               # Production Docker image
├── docker-compose.yml       # Local dev: API + UI together
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start (Local)

### Option A — Docker Compose (recommended)

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/job-assistant-agent.git
cd job-assistant-agent

# 2. Set up environment
cp .env.example .env
# Edit .env — add your OPENAI_API_KEY or GROQ_API_KEY

# 3. Run everything
docker compose up --build

# 4. Open in browser
# API:  http://localhost:8000/docs
# UI:   http://localhost:8501
```

### Option B — Without Docker

```bash
# 1. Clone & install
git clone https://github.com/YOUR_USERNAME/job-assistant-agent.git
cd job-assistant-agent
pip install -r requirements.txt

# 2. Environment
cp .env.example .env
# Edit .env

# 3. Start FastAPI backend
uvicorn app.main:app --reload --port 8000

# 4. Start Streamlit UI (new terminal)
cd ui
streamlit run streamlit_app.py
```

---

## 🔑 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | If using OpenAI | Your OpenAI API key |
| `GROQ_API_KEY` | If using Groq | Your Groq API key (free tier available) |
| `LLM_PROVIDER` | Yes | `openai` or `groq` |
| `LLM_MODEL` | Yes | e.g. `gpt-4o-mini` or `llama3-8b-8192` |
| `APP_HOST` | No | Default: `0.0.0.0` |
| `APP_PORT` | No | Default: `8000` |
| `BACKEND_URL` | UI only | Default: `http://localhost:8000` |

> 💡 **Free option**: Use Groq with `llama3-8b-8192` — it's free, fast, and works great for this project.
> Get a key at https://console.groq.com

---

## 🚀 Deployment

### Backend → Render.com (Free Tier)

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your GitHub repo
4. Settings:
   - **Build command**: `pip install -r requirements.txt`
   - **Start command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - **Environment variables**: Add your `OPENAI_API_KEY`, `LLM_PROVIDER`, `LLM_MODEL`
5. Deploy — you'll get a URL like `https://job-assistant-agent.onrender.com`

### UI → Streamlit Cloud (Free)

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Connect GitHub repo → select `ui/streamlit_app.py`
3. Add secret: `BACKEND_URL = https://job-assistant-agent.onrender.com`
4. Deploy — live URL shared instantly

### CI/CD (GitHub Actions)

Add these secrets to your GitHub repo (`Settings → Secrets`):

| Secret | Value |
|--------|-------|
| `OPENAI_API_KEY` | Your OpenAI key |
| `DOCKERHUB_USERNAME` | Your Docker Hub username |
| `DOCKERHUB_TOKEN` | Docker Hub access token |
| `RENDER_DEPLOY_HOOK_URL` | From Render dashboard → Settings → Deploy Hook |

Every push to `main` will:
1. Run `pytest` ✅
2. Build and push Docker image 🐳
3. Trigger Render re-deploy 🚀

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

```
tests/test_tools.py::test_section_aware_chunks_short_text       PASSED
tests/test_tools.py::test_section_aware_chunks_with_headings    PASSED
tests/test_tools.py::test_section_aware_chunks_long_section     PASSED
tests/test_tools.py::test_analyze_gap_no_requirements           PASSED
tests/test_tools.py::test_generate_cover_letter_no_resume       PASSED
tests/test_tools.py::test_generate_cover_letter_with_resume     PASSED
tests/test_tools.py::test_generate_interview_prep_no_resume     PASSED
tests/test_tools.py::test_route_intent_valid                    PASSED
```


## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Agent orchestration | LangGraph 0.1 (state graph) |
| LLM | OpenAI GPT-4o-mini / Groq Llama3 |
| LLM framework | LangChain 0.2 |
| Vector store | ChromaDB 0.5 (cosine similarity) |
| Embeddings | SentenceTransformers `all-MiniLM-L6-v2` |
| PDF parsing | PyPDF2 |
| Backend API | FastAPI + Uvicorn |
| Data validation | Pydantic v2 |
| Chat UI | Streamlit |
| Containerization | Docker + docker-compose |
| CI/CD | GitHub Actions |
| Hosting | Render.com + Streamlit Cloud |

---


## 📄 License

MIT — feel free to fork, extend, and use in your own projects.

---

## 🤝 Contributing

PRs welcome! Please open an issue first to discuss what you'd like to change.
