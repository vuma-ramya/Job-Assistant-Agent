"""
Tool 4: Interview Prep Generator
- Generates role-specific behavioral + technical questions
- Provides STAR-format answer hints based on the candidate's own resume
- Tailored to the JD provided
"""

from app.vector_store import query_resume


INTERVIEW_PREP_PROMPT = """You are an expert interview coach preparing a candidate for a job interview.

Generate 8 interview questions for this role — a mix of:
- 3 behavioral questions (STAR format expected)
- 3 technical questions specific to the JD skills
- 2 situational / problem-solving questions

For each question, also provide a SHORT answer hint (2-3 sentences) using details from 
the candidate's resume highlights below.

FORMAT each question exactly like this:
---
Q1. [Question text]
TYPE: Behavioral / Technical / Situational
HINT: [2-3 sentence answer hint using their actual experience]
---

RESUME HIGHLIGHTS:
{resume_context}

JOB DESCRIPTION:
{jd_text}

Generate the 8 questions now:
"""


def generate_interview_prep(
    session_id: str,
    jd_text: str,
    llm_client,
) -> str:
    """
    Generate tailored interview questions + answer hints using RAG over resume.
    """
    # Pull the most relevant resume sections
    resume_chunks = query_resume(session_id, jd_text, top_k=6)

    if not resume_chunks:
        return (
            "No resume found for this session. "
            "Please upload your resume first."
        )

    resume_context = "\n\n".join(resume_chunks)

    prompt = INTERVIEW_PREP_PROMPT.format(
        resume_context=resume_context[:2500],
        jd_text=jd_text[:1500],
    )

    response = llm_client.invoke(prompt)
    result = response.content if hasattr(response, "content") else str(response)

    return result.strip()
