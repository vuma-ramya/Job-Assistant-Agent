"""
LangGraph Agent — the core brain of the application.

State graph with 4 tool nodes:
  classify_intent → route → [resume | gap | cover_letter | interview | general]

The agent:
1. Reads the user's message
2. Classifies intent into one of 5 categories
3. Routes to the correct tool node
4. Returns a structured response
"""

import os
from typing import TypedDict, Optional, List, Literal
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from app.tools.gap_tool import analyze_gap
from app.tools.coverletter_tool import generate_cover_letter
from app.tools.interview_tool import generate_interview_prep

load_dotenv()


# ── LLM factory ─────────────────────────────────────────────────────────────

def _build_llm():
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name=model,
            temperature=0.3,
        )
    else:
        return ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=model,
            temperature=0.3,
        )


LLM = _build_llm()


# ── Agent State ──────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    session_id: str
    messages: List[dict]                  # full chat history
    current_user_message: str             # latest user message
    intent: Optional[str]                 # classified intent
    jd_text: Optional[str]               # stored JD (persists across turns)
    tool_output: Optional[str]            # result from whichever tool ran
    tool_used: Optional[str]              # name of tool used


# ── Intent Classifier ────────────────────────────────────────────────────────

INTENT_SYSTEM = """You are an intent classifier for a Job Application Assistant chatbot.
Classify the user's message into EXACTLY ONE of these intents:

- gap_analysis   : user wants to analyze how their resume matches a job description
- cover_letter   : user wants to generate a cover letter
- interview_prep : user wants interview questions or preparation help
- general        : anything else (greetings, questions about the app, follow-ups)

Rules:
- If the message contains a job description or asks about skill gaps → gap_analysis
- If the message mentions "cover letter" or "application letter" → cover_letter
- If the message mentions "interview", "questions", "prepare", "practice" → interview_prep
- Otherwise → general

Respond with ONLY the intent label. No punctuation. No explanation.
"""


def classify_intent(state: AgentState) -> AgentState:
    """Classify the user's intent from their latest message."""
    user_msg = state["current_user_message"]

    response = LLM.invoke([
        SystemMessage(content=INTENT_SYSTEM),
        HumanMessage(content=user_msg),
    ])

    intent = response.content.strip().lower()

    # Validate — default to general if unrecognised
    valid_intents = {"gap_analysis", "cover_letter", "interview_prep", "general"}
    if intent not in valid_intents:
        intent = "general"

    return {**state, "intent": intent}


# ── Router ───────────────────────────────────────────────────────────────────

def route_intent(state: AgentState) -> Literal["gap_analysis", "cover_letter", "interview_prep", "general"]:
    return state.get("intent", "general")


# ── Tool Nodes ───────────────────────────────────────────────────────────────

def _extract_jd_from_message(message: str, existing_jd: Optional[str]) -> str:
    """
    Try to extract a job description from the current message,
    or fall back to previously stored JD.
    """
    # Heuristic: if message is long (>200 chars), treat it as containing the JD
    if len(message) > 200:
        return message

    # Otherwise use the stored JD
    return existing_jd or message


def node_gap_analysis(state: AgentState) -> AgentState:
    jd_text = _extract_jd_from_message(
        state["current_user_message"], state.get("jd_text")
    )

    if not jd_text:
        return {
            **state,
            "tool_output": "Please paste the job description so I can analyze the gap.",
            "tool_used": "gap_analysis",
        }

    result = analyze_gap(state["session_id"], jd_text, LLM)

    output = f"""**Gap Analysis Complete** ✅

**Match Score:** {result['score']}%

**Skills you have ✓**
{chr(10).join(f"• {s}" for s in result['matched']) or "None detected"}

**Skills to work on ✗**
{chr(10).join(f"• {s}" for s in result['missing']) or "Great fit — no major gaps!"}

**Summary**
{result['summary']}

---
_Type "generate cover letter" to create a tailored cover letter for this role._
_Type "interview prep" to get practice questions._
"""
    return {
        **state,
        "jd_text": jd_text,
        "tool_output": output,
        "tool_used": "gap_analysis",
    }


def node_cover_letter(state: AgentState) -> AgentState:
    jd_text = _extract_jd_from_message(
        state["current_user_message"], state.get("jd_text")
    )

    if not jd_text:
        return {
            **state,
            "tool_output": "Please paste the job description so I can write your cover letter.",
            "tool_used": "cover_letter",
        }

    cover_letter = generate_cover_letter(state["session_id"], jd_text, LLM)

    output = f"""**Your Tailored Cover Letter** 📝

{cover_letter}

---
_Feel free to ask me to adjust the tone, make it shorter, or emphasize different skills._
"""
    return {
        **state,
        "jd_text": jd_text,
        "tool_output": output,
        "tool_used": "cover_letter",
    }


def node_interview_prep(state: AgentState) -> AgentState:
    jd_text = _extract_jd_from_message(
        state["current_user_message"], state.get("jd_text")
    )

    if not jd_text:
        return {
            **state,
            "tool_output": "Please paste the job description so I can tailor your interview prep.",
            "tool_used": "interview_prep",
        }

    questions = generate_interview_prep(state["session_id"], jd_text, LLM)

    output = f"""**Interview Prep Questions** 🎯

{questions}

---
_Practice these out loud! Ask me to generate more questions or dive deeper into any topic._
"""
    return {
        **state,
        "jd_text": jd_text,
        "tool_output": output,
        "tool_used": "interview_prep",
    }


def node_general(state: AgentState) -> AgentState:
    """Fallback general conversation node."""
    history = state.get("messages", [])

    # Build message list for LLM
    lc_messages = [
        SystemMessage(content="""You are a friendly, expert Job Application Assistant.
You help users:
1. Analyze how their resume matches a job description (gap analysis)
2. Generate tailored cover letters
3. Prepare for interviews with practice questions

The user has already uploaded their resume. Guide them toward these 3 features.
Keep responses concise and helpful.""")
    ]

    for msg in history[-6:]:  # Last 3 exchanges for context
        if msg["role"] == "user":
            lc_messages.append(HumanMessage(content=msg["content"]))
        else:
            lc_messages.append(AIMessage(content=msg["content"]))

    lc_messages.append(HumanMessage(content=state["current_user_message"]))

    response = LLM.invoke(lc_messages)
    return {
        **state,
        "tool_output": response.content,
        "tool_used": "general",
    }


# ── Build Graph ───────────────────────────────────────────────────────────────

def build_agent() -> StateGraph:
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("classify", classify_intent)
    graph.add_node("gap_analysis", node_gap_analysis)
    graph.add_node("cover_letter", node_cover_letter)
    graph.add_node("interview_prep", node_interview_prep)
    graph.add_node("general", node_general)

    # Entry point
    graph.set_entry_point("classify")

    # Conditional routing after classification
    graph.add_conditional_edges(
        "classify",
        route_intent,
        {
            "gap_analysis": "gap_analysis",
            "cover_letter": "cover_letter",
            "interview_prep": "interview_prep",
            "general": "general",
        },
    )

    # All tool nodes → END
    graph.add_edge("gap_analysis", END)
    graph.add_edge("cover_letter", END)
    graph.add_edge("interview_prep", END)
    graph.add_edge("general", END)

    return graph.compile()


# Compiled agent — imported by main.py
agent = build_agent()


def run_agent(
    session_id: str,
    user_message: str,
    history: list,
    jd_text: Optional[str] = None,
) -> dict:
    """
    Public interface: run the agent and return {reply, tool_used}.
    """
    initial_state: AgentState = {
        "session_id": session_id,
        "messages": history,
        "current_user_message": user_message,
        "intent": None,
        "jd_text": jd_text,
        "tool_output": None,
        "tool_used": None,
    }

    result = agent.invoke(initial_state)

    return {
        "reply": result.get("tool_output", "Sorry, I couldn't process that request."),
        "tool_used": result.get("tool_used", "general"),
    }
