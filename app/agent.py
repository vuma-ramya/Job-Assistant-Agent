"""
LangGraph Agent — 100% Groq powered, no OpenAI dependency.
"""

import os
from typing import TypedDict, Optional, List, Literal
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from app.tools.gap_tool import analyze_gap
from app.tools.coverletter_tool import generate_cover_letter
from app.tools.interview_tool import generate_interview_prep

load_dotenv()

_llm = None

def _build_llm():
    global _llm
    if _llm is not None:
        return _llm
    _llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name=os.getenv("LLM_MODEL", "llama3-8b-8192"),
        temperature=0.3,
    )
    return _llm

class AgentState(TypedDict):
    session_id: str
    messages: List[dict]
    current_user_message: str
    intent: Optional[str]
    jd_text: Optional[str]
    tool_output: Optional[str]
    tool_used: Optional[str]

INTENT_SYSTEM = """You are an intent classifier for a Job Application Assistant chatbot.
Classify the user message into EXACTLY ONE of these intents:
- gap_analysis   : user wants to analyze how their resume matches a job description
- cover_letter   : user wants to generate a cover letter
- interview_prep : user wants interview questions or preparation help
- general        : anything else
Respond with ONLY the intent label. No punctuation. No explanation."""

def classify_intent(state: AgentState) -> AgentState:
    llm = _build_llm()
    response = llm.invoke([
        SystemMessage(content=INTENT_SYSTEM),
        HumanMessage(content=state["current_user_message"]),
    ])
    intent = response.content.strip().lower()
    valid = {"gap_analysis", "cover_letter", "interview_prep", "general"}
    if intent not in valid:
        intent = "general"
    return {**state, "intent": intent}

def route_intent(state: AgentState) -> Literal["gap_analysis", "cover_letter", "interview_prep", "general"]:
    return state.get("intent", "general")

def _extract_jd(message: str, existing_jd: Optional[str]) -> str:
    if len(message) > 200:
        return message
    return existing_jd or message

def node_gap_analysis(state: AgentState) -> AgentState:
    llm = _build_llm()
    jd_text = _extract_jd(state["current_user_message"], state.get("jd_text"))
    if not jd_text:
        return {**state, "tool_output": "Please paste the job description so I can analyze the gap.", "tool_used": "gap_analysis"}
    result = analyze_gap(state["session_id"], jd_text, llm)
    output = f"""**Gap Analysis Complete** ✅\n\n**Match Score:** {result['score']}%\n\n**Skills you have ✓**\n{chr(10).join(f"• {s}" for s in result['matched']) or "None detected"}\n\n**Skills to work on ✗**\n{chr(10).join(f"• {s}" for s in result['missing']) or "Great fit!"}\n\n**Summary**\n{result['summary']}\n\n---\n_Type "generate cover letter" or "interview prep" to continue._"""
    return {**state, "jd_text": jd_text, "tool_output": output, "tool_used": "gap_analysis"}

def node_cover_letter(state: AgentState) -> AgentState:
    llm = _build_llm()
    jd_text = _extract_jd(state["current_user_message"], state.get("jd_text"))
    if not jd_text:
        return {**state, "tool_output": "Please paste the job description so I can write your cover letter.", "tool_used": "cover_letter"}
    letter = generate_cover_letter(state["session_id"], jd_text, llm)
    output = f"""**Your Tailored Cover Letter** 📝\n\n{letter}\n\n---\n_Ask me to adjust tone, length, or emphasis._"""
    return {**state, "jd_text": jd_text, "tool_output": output, "tool_used": "cover_letter"}

def node_interview_prep(state: AgentState) -> AgentState:
    llm = _build_llm()
    jd_text = _extract_jd(state["current_user_message"], state.get("jd_text"))
    if not jd_text:
        return {**state, "tool_output": "Please paste the job description so I can tailor your interview prep.", "tool_used": "interview_prep"}
    questions = generate_interview_prep(state["session_id"], jd_text, llm)
    output = f"""**Interview Prep Questions** 🎯\n\n{questions}\n\n---\n_Ask for more questions or deeper explanations._"""
    return {**state, "jd_text": jd_text, "tool_output": output, "tool_used": "interview_prep"}

def node_general(state: AgentState) -> AgentState:
    llm = _build_llm()
    history = state.get("messages", [])
    lc_messages = [SystemMessage(content="You are a friendly Job Application Assistant. Help users with gap analysis, cover letters, and interview prep. Keep responses concise.")]
    for msg in history[-6:]:
        if msg["role"] == "user":
            lc_messages.append(HumanMessage(content=msg["content"]))
        else:
            lc_messages.append(AIMessage(content=msg["content"]))
    lc_messages.append(HumanMessage(content=state["current_user_message"]))
    response = llm.invoke(lc_messages)
    return {**state, "tool_output": response.content, "tool_used": "general"}

def build_agent():
    graph = StateGraph(AgentState)
    graph.add_node("classify", classify_intent)
    graph.add_node("gap_analysis", node_gap_analysis)
    graph.add_node("cover_letter", node_cover_letter)
    graph.add_node("interview_prep", node_interview_prep)
    graph.add_node("general", node_general)
    graph.set_entry_point("classify")
    graph.add_conditional_edges("classify", route_intent, {
        "gap_analysis": "gap_analysis",
        "cover_letter": "cover_letter",
        "interview_prep": "interview_prep",
        "general": "general",
    })
    graph.add_edge("gap_analysis", END)
    graph.add_edge("cover_letter", END)
    graph.add_edge("interview_prep", END)
    graph.add_edge("general", END)
    return graph.compile()

agent = build_agent()

def run_agent(session_id: str, user_message: str, history: list, jd_text: Optional[str] = None) -> dict:
    result = agent.invoke({
        "session_id": session_id,
        "messages": history,
        "current_user_message": user_message,
        "intent": None,
        "jd_text": jd_text,
        "tool_output": None,
        "tool_used": None,
    })
    return {
        "reply": result.get("tool_output", "Sorry, I could not process that."),
        "tool_used": result.get("tool_used", "general"),
        "jd_text": result.get("jd_text"),
    }
