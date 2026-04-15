"""
FastAPI backend — exposes 3 endpoints:

POST /upload-resume   → parse PDF, store embeddings, return skills
POST /chat            → run LangGraph agent, return reply
DELETE /session/{id}  → clean up ChromaDB for session
"""

import uuid
import io
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.models import ChatRequest, ChatResponse, UploadResumeResponse
from app.agent import run_agent, LLM
from app.tools.resume_tool import parse_and_store_resume
from app.vector_store import delete_session

# ── In-memory session store (use Redis for production) ───────────────────────
# session_id → {jd_text, candidate_name}
_session_store: dict = {}

# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Job Application Assistant API",
    description="Agentic AI chatbot that helps tailor resumes, write cover letters, and prep for interviews.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health check ─────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


# ── Upload resume ─────────────────────────────────────────────────────────────
@app.post("/upload-resume", response_model=UploadResumeResponse)
async def upload_resume(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    candidate_name: Optional[str] = Form(""),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    session_id = session_id or str(uuid.uuid4())
    pdf_bytes = await file.read()

    if len(pdf_bytes) > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(status_code=413, detail="File too large. Max 10MB.")

    try:
        _, skills = parse_and_store_resume(session_id, pdf_bytes, LLM)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse resume: {str(e)}")

    _session_store[session_id] = {
        "candidate_name": candidate_name or "the candidate",
        "jd_text": None,
    }

    return UploadResumeResponse(
        session_id=session_id,
        message=f"Resume uploaded and indexed successfully. Found {len(skills)} skills.",
        extracted_skills=skills,
    )


# ── Chat ──────────────────────────────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if req.session_id not in _session_store:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Please upload your resume first.",
        )

    session_data = _session_store[req.session_id]
    history = [msg.model_dump() for msg in (req.history or [])]

    try:
        result = run_agent(
            session_id=req.session_id,
            user_message=req.message,
            history=history,
            jd_text=session_data.get("jd_text"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

    # Persist JD across turns if the agent used it
    # (The agent returns jd_text in result if it parsed one)
    if "jd_text" in result and result["jd_text"]:
        _session_store[req.session_id]["jd_text"] = result["jd_text"]

    return ChatResponse(
        session_id=req.session_id,
        reply=result["reply"],
        tool_used=result.get("tool_used"),
    )


# ── Session cleanup ───────────────────────────────────────────────────────────
@app.delete("/session/{session_id}")
def cleanup_session(session_id: str):
    if session_id in _session_store:
        del _session_store[session_id]
    delete_session(session_id)
    return {"message": f"Session {session_id} deleted."}


# ── Run directly (dev mode) ───────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    import os
    uvicorn.run(
        "app.main:app",
        host=os.getenv("APP_HOST", "0.0.0.0"),
        port=int(os.getenv("APP_PORT", 8000)),
        reload=True,
    )
