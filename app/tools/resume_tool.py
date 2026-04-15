"""
Tool 1: Resume Analyzer
- Parses PDF resume → plain text
- Sections it into semantic chunks (by heading)
- Stores chunks in ChromaDB
- Returns extracted skills list via LLM
"""

import re
import io
from typing import List, Tuple

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

from app.vector_store import store_resume


def _extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract raw text from PDF bytes."""
    if PyPDF2 is None:
        raise ImportError("PyPDF2 is not installed")

    reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text.strip())
    return "\n\n".join(pages)


def _section_aware_chunks(text: str, max_chars: int = 600) -> List[str]:
    """
    Split resume text into section-aware chunks.
    Detects common resume headings and keeps sections together.
    Falls back to size-based splitting for long sections.
    """
    heading_pattern = re.compile(
        r"(?m)^(EXPERIENCE|EDUCATION|SKILLS|PROJECTS|SUMMARY|OBJECTIVE|"
        r"CERTIFICATIONS|AWARDS|PUBLICATIONS|WORK HISTORY|EMPLOYMENT)[\s:]*$",
        re.IGNORECASE,
    )

    # Split by detected headings
    parts = heading_pattern.split(text)
    chunks = []

    for part in parts:
        part = part.strip()
        if not part:
            continue
        # If section is short enough, keep as one chunk
        if len(part) <= max_chars:
            chunks.append(part)
        else:
            # Split long sections by paragraph
            paragraphs = [p.strip() for p in part.split("\n\n") if p.strip()]
            current = ""
            for para in paragraphs:
                if len(current) + len(para) <= max_chars:
                    current += "\n\n" + para if current else para
                else:
                    if current:
                        chunks.append(current)
                    current = para
            if current:
                chunks.append(current)

    return chunks if chunks else [text[:max_chars]]


def parse_and_store_resume(
    session_id: str,
    pdf_bytes: bytes,
    llm_client,  # langchain LLM instance
) -> Tuple[str, List[str]]:
    """
    Parse a PDF resume, store in vector DB, return (full_text, skills_list).
    """
    # Extract text
    raw_text = _extract_text_from_pdf(pdf_bytes)

    # Chunk and store
    chunks = _section_aware_chunks(raw_text)
    store_resume(session_id, chunks)

    # Extract skills via LLM
    prompt = f"""From the resume below, extract a clean list of technical skills, tools, and technologies.
Return ONLY a Python-style list of strings, e.g. ["Python", "FastAPI", "LangChain"].
No explanations. No markdown.

RESUME:
{raw_text[:3000]}
"""
    response = llm_client.invoke(prompt)
    skills_text = response.content if hasattr(response, "content") else str(response)

    # Parse skills from LLM output
    skills = []
    try:
        import ast
        skills = ast.literal_eval(skills_text.strip())
        if not isinstance(skills, list):
            skills = []
    except Exception:
        # Fallback: split by comma if LLM didn't return a list
        skills = [s.strip().strip('"').strip("'") for s in skills_text.split(",") if s.strip()]

    return raw_text, skills[:30]  # Cap at 30 skills
