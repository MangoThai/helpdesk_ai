# app/agent_graph.py
from __future__ import annotations

import os
from typing import List, TypedDict

from dotenv import load_dotenv, find_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    SystemMessage,
)
from langgraph.graph import StateGraph, END

from app.tools import TOOLS, TOOLS_BY_NAME

# --- (optionnel) mémoire longue : on tolère l'absence du module ---
try:
    from app.memory import load_profile, system_prefix  # si présent
except Exception:  # pas de mémoire longue -> stubs neutres
    def load_profile():
        return {}

    def system_prefix(_profile: dict) -> str:
        return ""


# ---------- État du graphe ----------
class AgentState(TypedDict):
    messages: List[AnyMessage]


# ---------- LLM + tools ----------
def _build_llm():
    # .env robuste (exécution depuis le CWD)
    load_dotenv(find_dotenv(usecwd=True))
    model_name = os.getenv("ANTHROPIC_MODEL", "claude-3-7-sonnet-20250219")
    # On "lie" les tools au LLM pour activer les tool-calls
    return ChatAnthropic(model=model_name, temperature=0.2).bind_tools(TOOLS)


# ---------- Noeud: appel du modèle ----------
def call_model(state: AgentState) -> AgentState:
    llm = _build_llm()
    messages = state["messages"]

    # Injecter une fois un SystemMessage (profil + consignes outils)
    if not any(getattr(m, "type", "") == "system" for m in messages):
        prof = load_profile()
        parts = []
        if prof:
            parts.append(system_prefix(prof))
        # >>> Consignes importantes, y compris CITATION DES SOURCES <<<
        parts.append(
            "Tu es un assistant Helpdesk IT. Tu peux appeler des outils si nécessaire "
            "(extract_emails, check_host, knowledge_lookup). "
            "Si tu utilises knowledge_lookup, synthétise en t'appuyant STRICTEMENT sur les 'snippets' "
            "et CITE les sources en fin de réponse sous la forme: "
            "Sources: (fichier1.md, fichier2.md). "
            "Après tout tool-call, explique brièvement le résultat avant de conclure. "
            "Si tu n'as pas assez d'informations, dis-le et propose 1–3 questions de clarification."
        )
        messages = [SystemMessage(content="\n\n".join([p for p in parts if p]))] + messages

    ai_msg = llm.invoke(messages)
    return {"messages": messages + [ai_msg]}


# ---------- Noeud: exécution des outils demandés ----------
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
            tool_results.append(
                ToolMessage(content=f"Tool '{name}' introuvable", tool_call_id=tc.get("id"))
            )
            continue
        # Les @tool sont des Runnables -> .invoke(args)
        result = tool.invoke(args)
        tool_results.append(ToolMessage(content=str(result), tool_call_id=tc.get("id")))

    return {"messages": messages + tool_results}


# ---------- Routage: continuer si tool_calls, sinon terminer ----------
def should_continue(state: AgentState):
    last = state["messages"][-1]
    has_tool_calls = bool(getattr(last, "tool_calls", None))
    return "tools" if has_tool_calls else END


# ---------- Construction du graphe ----------
def build_agent_app():
    workflow = StateGraph(AgentState)
    workflow.add_node("model", call_model)
    workflow.add_node("tools", call_tools)

    workflow.set_entry_point("model")
    workflow.add_conditional_edges("model", should_continue)
    workflow.add_edge("tools", "model")  # boucle tools -> model tant qu'il y a des tool_calls

    return workflow.compile()


# ---------- Utilitaire: un tour simple ----------
def agent_answer_once(question: str) -> str:
    app = build_agent_app()
    state: AgentState = {"messages": [HumanMessage(content=question)]}
    result = app.invoke(state)

    # Récupère le dernier message IA textuel
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage):
            content = msg.content
            return content if isinstance(content, str) else str(content)
    return "(pas de réponse IA)"
