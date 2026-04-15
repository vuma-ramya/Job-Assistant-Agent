"""
Basic tests — run with: pytest tests/ -v
"""

import pytest
from unittest.mock import MagicMock, patch


# ── Resume tool tests ─────────────────────────────────────────────────────────

def test_section_aware_chunks_short_text():
    from app.tools.resume_tool import _section_aware_chunks
    text = "John Doe\nPython developer with 5 years experience."
    chunks = _section_aware_chunks(text)
    assert isinstance(chunks, list)
    assert len(chunks) >= 1


def test_section_aware_chunks_with_headings():
    from app.tools.resume_tool import _section_aware_chunks
    text = """SKILLS
Python, FastAPI, LangChain

EXPERIENCE
Software Engineer at Acme Corp 2020-2024
Built scalable APIs."""
    chunks = _section_aware_chunks(text)
    assert len(chunks) >= 2


def test_section_aware_chunks_long_section():
    from app.tools.resume_tool import _section_aware_chunks
    long_text = "A" * 1500
    chunks = _section_aware_chunks(long_text, max_chars=600)
    assert len(chunks) >= 2


# ── Gap tool tests ────────────────────────────────────────────────────────────

def test_analyze_gap_no_requirements():
    from app.tools.gap_tool import analyze_gap

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="[]")

    with patch("app.tools.gap_tool.query_resume", return_value=[]):
        result = analyze_gap("test-session", "some JD text", mock_llm)

    assert "matched" in result
    assert "missing" in result
    assert "score" in result
    assert result["score"] == 0


# ── Cover letter tool tests ───────────────────────────────────────────────────

def test_generate_cover_letter_no_resume():
    from app.tools.coverletter_tool import generate_cover_letter

    mock_llm = MagicMock()

    with patch("app.tools.coverletter_tool.query_resume", return_value=[]):
        result = generate_cover_letter("test-session", "some JD", mock_llm)

    assert "No resume found" in result


def test_generate_cover_letter_with_resume():
    from app.tools.coverletter_tool import generate_cover_letter

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="Dear Hiring Manager, ...")

    with patch("app.tools.coverletter_tool.query_resume", return_value=["Python developer"]):
        result = generate_cover_letter("test-session", "some JD", mock_llm)

    assert "Dear Hiring Manager" in result


# ── Interview tool tests ──────────────────────────────────────────────────────

def test_generate_interview_prep_no_resume():
    from app.tools.interview_tool import generate_interview_prep

    mock_llm = MagicMock()

    with patch("app.tools.interview_tool.query_resume", return_value=[]):
        result = generate_interview_prep("test-session", "some JD", mock_llm)

    assert "No resume found" in result


# ── Agent intent classifier tests ─────────────────────────────────────────────

def test_route_intent_valid():
    """Ensure route_intent only returns valid intents."""
    from app.agent import route_intent

    valid_intents = {"gap_analysis", "cover_letter", "interview_prep", "general"}

    for intent in valid_intents:
        state = {
            "session_id": "x",
            "messages": [],
            "current_user_message": "test",
            "intent": intent,
            "jd_text": None,
            "tool_output": None,
            "tool_used": None,
        }
        result = route_intent(state)
        assert result in valid_intents
