from pydantic import BaseModel
from typing import Optional, List


class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    session_id: str
    message: str
    history: Optional[List[ChatMessage]] = []


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    tool_used: Optional[str] = None


class UploadResumeResponse(BaseModel):
    session_id: str
    message: str
    extracted_skills: List[str]


class AgentState(BaseModel):
    session_id: str
    messages: List[dict]
    resume_text: Optional[str] = None
    jd_text: Optional[str] = None
    gap_result: Optional[str] = None
    cover_letter: Optional[str] = None
    interview_questions: Optional[str] = None
