"""
Tool 3: Cover Letter Generator
- RAG-based: retrieves most relevant resume sections
- Combines with JD context
- Generates a tailored, professional cover letter
"""

from app.vector_store import query_resume


COVER_LETTER_PROMPT = """You are an expert career coach and professional writer.
Write a compelling, personalized cover letter for the job below.

INSTRUCTIONS:
- 3-4 paragraphs, professional but warm tone
- Opening: Hook that shows genuine interest in the role/company
- Middle: 2-3 specific achievements from the resume that match the JD
- Closing: Call to action, confident but not arrogant
- DO NOT use generic filler phrases like "I am writing to express my interest"
- Reference specific skills from the JD by name
- Keep it under 350 words

RESUME HIGHLIGHTS (most relevant to this role):
{resume_context}

JOB DESCRIPTION:
{jd_text}

CANDIDATE NAME (if known, else write "I"):
{candidate_name}

Write the cover letter now:
"""


def generate_cover_letter(
    session_id: str,
    jd_text: str,
    llm_client,
    candidate_name: str = "I",
) -> str:
    """
    Generate a tailored cover letter using RAG over the stored resume.
    """
    # Retrieve most relevant resume sections for this JD
    resume_chunks = query_resume(session_id, jd_text, top_k=6)

    if not resume_chunks:
        return (
            "No resume found for this session. "
            "Please upload your resume first using the file uploader."
        )

    resume_context = "\n\n".join(resume_chunks)

    prompt = COVER_LETTER_PROMPT.format(
        resume_context=resume_context[:2500],
        jd_text=jd_text[:1500],
        candidate_name=candidate_name,
    )

    response = llm_client.invoke(prompt)
    cover_letter = response.content if hasattr(response, "content") else str(response)

    return cover_letter.strip()
