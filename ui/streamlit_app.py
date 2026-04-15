"""
Streamlit Chat UI — Job Application Assistant

Connects to the FastAPI backend at BACKEND_URL (env var or localhost).
"""

import os
import uuid
import requests
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Job Application Assistant",
    page_icon="💼",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stChatMessage { border-radius: 12px; }
    .skill-badge {
        display: inline-block;
        background: #e8f4fd;
        color: #1a6fa3;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 12px;
        margin: 2px;
        font-weight: 500;
    }
    .tool-label {
        font-size: 11px;
        color: #888;
        margin-bottom: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ── Session state init ────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "skills" not in st.session_state:
    st.session_state.skills = []
if "resume_uploaded" not in st.session_state:
    st.session_state.resume_uploaded = False

# Wake up backend
if st.sidebar.button("🔄 Wake up backend", use_container_width=True):
    with st.spinner("Waking up backend (~30s)..."):
        try:
            r = requests.get(f"{BACKEND_URL}/health", timeout=60)
            if r.status_code == 200:
                st.sidebar.success("✅ Backend is live!")
            else:
                st.sidebar.error("Backend returned an error")
        except Exception as e:
            st.sidebar.error(f"Still waking up: {e}")

# ── Sidebar — resume upload ────────────────────────────────────────────────────
with st.sidebar:
    st.title("💼 Job Assistant")
    st.caption("AI-powered job application helper")
    st.divider()

    st.subheader("Step 1: Upload Your Resume")
    candidate_name = st.text_input("Your name (optional)", placeholder="e.g. Jane Smith")
    uploaded_file = st.file_uploader("Upload PDF resume", type=["pdf"])

    if uploaded_file and not st.session_state.resume_uploaded:
        if st.button("📤 Process Resume", type="primary", use_container_width=True):
            with st.spinner("Parsing and indexing your resume…"):
                try:
                    session_id = str(uuid.uuid4())
                    resp = requests.post(
                        f"{BACKEND_URL}/upload-resume",
                        files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
                        data={
                            "session_id": session_id,
                            "candidate_name": candidate_name,
                        },
                        timeout=60,
                    )
                    resp.raise_for_status()
                    data = resp.json()

                    st.session_state.session_id = data["session_id"]
                    st.session_state.skills = data["extracted_skills"]
                    st.session_state.resume_uploaded = True
                    st.session_state.messages = []

                    # Welcome message
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": (
                            f"✅ Resume uploaded! I found **{len(data['extracted_skills'])} skills**.\n\n"
                            "Here's what I can help you with:\n"
                            "1. **Gap Analysis** — paste a job description to see how you match\n"
                            "2. **Cover Letter** — I'll write a tailored one for any role\n"
                            "3. **Interview Prep** — get practice questions for your target job\n\n"
                            "_Paste a job description below to get started!_"
                        ),
                    })
                    st.rerun()

                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to backend. Make sure the FastAPI server is running.")
                except Exception as e:
                    st.error(f"Upload failed: {e}")

    elif st.session_state.resume_uploaded:
        st.success("✅ Resume indexed")
        if st.session_state.skills:
            st.caption("Detected skills:")
            badges_html = "".join(
                f'<span class="skill-badge">{s}</span>'
                for s in st.session_state.skills[:20]
            )
            st.markdown(badges_html, unsafe_allow_html=True)

        st.divider()
        if st.button("🗑 Upload a new resume", use_container_width=True):
            # Clean up server-side session
            try:
                requests.delete(
                    f"{BACKEND_URL}/session/{st.session_state.session_id}",
                    timeout=10,
                )
            except Exception:
                pass
            st.session_state.session_id = None
            st.session_state.resume_uploaded = False
            st.session_state.skills = []
            st.session_state.messages = []
            st.rerun()

    st.divider()
    st.caption("**How to use:**\n\n1. Upload your resume PDF\n2. Paste a job description in chat\n3. Ask for gap analysis, cover letter, or interview prep")


# ── Main chat area ─────────────────────────────────────────────────────────────
st.title("Job Application Assistant 🤖")

if not st.session_state.resume_uploaded:
    st.info("👈 Upload your resume in the sidebar to get started.")
    st.stop()

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Paste a job description or ask me anything…"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call backend
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                # Send last 10 messages as history
                history = st.session_state.messages[-10:-1]

                resp = requests.post(
                    f"{BACKEND_URL}/chat",
                    json={
                        "session_id": st.session_state.session_id,
                        "message": prompt,
                        "history": history,
                    },
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()

                reply = data["reply"]
                tool_used = data.get("tool_used", "")

                # Show tool label
                tool_labels = {
                    "gap_analysis": "🔍 Gap Analysis",
                    "cover_letter": "📝 Cover Letter Generator",
                    "interview_prep": "🎯 Interview Prep",
                    "general": "",
                }
                label = tool_labels.get(tool_used, "")
                if label:
                    st.caption(label)

                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})

            except requests.exceptions.Timeout:
                err = "Request timed out. The LLM took too long. Please try again."
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
            except requests.exceptions.ConnectionError:
                err = "Cannot connect to backend server."
                st.error(err)
            except Exception as e:
                err = f"Something went wrong: {e}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
