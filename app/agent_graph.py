# app/agent_graph.py
from __future__ import annotations
import os
from typing import List, TypedDict

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END

from app.tools import TOOLS, TOOLS_BY_NAME

# --- État du graphe ---
class AgentState(TypedDict):
    messages: List[AnyMessage]

def _build_llm():
    load_dotenv()
    model_name = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
    # On "lie" les tools au LLM pour activer les tool-calls
    return ChatAnthropic(model=model_name, temperature=0.2).bind_tools(TOOLS)

# --- Noeud 'model': appelle le LLM ---
def call_model(state: AgentState) -> AgentState:
    llm = _build_llm()
    ai_msg = llm.invoke(state["messages"])
    return {"messages": state["messages"] + [ai_msg]}

# --- Noeud 'tools': exécute les outils demandés par l'IA ---
def call_tools(state: AgentState) -> AgentState:
    messages = state["messages"]
    last = messages[-1]
    tool_results: List[ToolMessage] = []

    # Si l'IA a demandé des tools (tool_calls), on les exécute
    for tc in getattr(last, "tool_calls", []) or []:
        name = tc.get("name")
        args = tc.get("args", {}) or {}
        tool = TOOLS_BY_NAME.get(name)
        if not tool:
            # Tool inconnu: on renvoie une "erreur" au modèle
            tool_results.append(ToolMessage(content=f"Tool '{name}' introuvable", tool_call_id=tc.get("id")))
            continue
        result = tool.invoke(args)  # Les @tool sont des Runnables → .invoke(args)
        tool_results.append(ToolMessage(content=str(result), tool_call_id=tc.get("id")))

    return {"messages": messages + tool_results}

# --- Condition: continuer si l'IA a demandé un outil, sinon terminer ---
def should_continue(state: AgentState):
    last = state["messages"][-1]
    has_tool_calls = bool(getattr(last, "tool_calls", None))
    return "tools" if has_tool_calls else END

# --- Construire et compiler le graphe ---
def build_agent_app():
    workflow = StateGraph(AgentState)
    workflow.add_node("model", call_model)
    workflow.add_node("tools", call_tools)

    workflow.set_entry_point("model")
    workflow.add_conditional_edges("model", should_continue)
    workflow.add_edge("tools", "model")  # boucle: tools -> model (jusqu'à plus de tool_calls)

    return workflow.compile()

# --- Utilitaire: un tour simple (sans session) ---
def agent_answer_once(question: str) -> str:
    app = build_agent_app()
    # état initial: un seul message humain
    state: AgentState = {"messages": [HumanMessage(content=question)]}
    result = app.invoke(state)

    # On récupère le dernier message de l'IA
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage):
            content = msg.content
            return content if isinstance(content, str) else str(content)
    return "(pas de réponse IA)"
