"""
Tool 2: JD Gap Analyzer
- Accepts a job description (text or URL)
- Computes semantic similarity between JD requirements and resume chunks
- Returns missing skills / gaps
"""

from typing import List, Dict
from app.vector_store import query_resume


def _extract_jd_requirements(jd_text: str, llm_client) -> List[str]:
    """Use LLM to pull required skills/qualifications from JD."""
    prompt = f"""Extract all required and preferred skills, tools, technologies, and qualifications 
from this job description. Return ONLY a Python list of strings.
No explanations. No markdown.

JOB DESCRIPTION:
{jd_text[:3000]}
"""
    response = llm_client.invoke(prompt)
    content = response.content if hasattr(response, "content") else str(response)

    try:
        import ast
        requirements = ast.literal_eval(content.strip())
        if isinstance(requirements, list):
            return requirements
    except Exception:
        pass

    return [r.strip() for r in content.split(",") if r.strip()]


def analyze_gap(
    session_id: str,
    jd_text: str,
    llm_client,
) -> Dict:
    """
    Compare JD requirements against stored resume.
    Returns dict with: matched, missing, score, summary.
    """
    # Step 1: Extract JD requirements
    jd_requirements = _extract_jd_requirements(jd_text, llm_client)

    if not jd_requirements:
        return {
            "matched": [],
            "missing": [],
            "score": 0,
            "summary": "Could not parse job description requirements.",
        }

    # Step 2: For each JD requirement, query resume vector store
    matched = []
    missing = []

    for req in jd_requirements:
        resume_chunks = query_resume(session_id, req, top_k=2)
        if resume_chunks:
            # Check if the retrieved chunk is actually relevant (basic threshold)
            combined = " ".join(resume_chunks).lower()
            req_lower = req.lower()
            # Simple keyword check + semantic retrieval combo
            if any(word in combined for word in req_lower.split() if len(word) > 3):
                matched.append(req)
            else:
                missing.append(req)
        else:
            missing.append(req)

    # Step 3: LLM summary of gaps
    score = round(len(matched) / max(len(jd_requirements), 1) * 100)

    summary_prompt = f"""You are a career coach. Based on this analysis, write 3-4 sentences 
summarizing the candidate's fit for the role and what to focus on improving.

Matched skills: {matched}
Missing skills: {missing}
Match score: {score}%
"""
    summary_response = llm_client.invoke(summary_prompt)
    summary = summary_response.content if hasattr(summary_response, "content") else str(summary_response)

    return {
        "matched": matched,
        "missing": missing,
        "score": score,
        "summary": summary.strip(),
    }
