# app/routing.py
from __future__ import annotations
from typing import Literal, Optional
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

# --- Schéma de sortie (pydantic) ---
class TicketClassification(BaseModel):
    category: Literal["incident", "demande", "question"] = Field(
        ..., description="Type de ticket IT"
    )
    urgency: Literal["basse", "normale", "haute"] = Field(
        ..., description="Estimation de l'urgence"
    )
    products: list[str] = Field(
        default_factory=list,
        description="Produits / composants détectés (ex: 'VPN', 'Mac', 'Wi-Fi')",
    )
    rationale: str = Field(
        ..., description="Raisonnement concis justifiant la classification"
    )

# --- Chaîne de classification ---
def _build_classifier():
    load_dotenv()
    model_name = os.getenv("ANTHROPIC_MODEL", "claude-3-7-sonnet-20250219")
    model = ChatAnthropic(model=model_name, temperature=0)

    system = (
        "Tu es un routeur de tickets Helpdesk IT. "
        "Classifie le message utilisateur dans l'une des catégories : incident, demande, question. "
        "Déduis également l'urgence (basse, normale, haute) selon l'impact et le blocage. "
        "Liste les produits/technos mentionnés (ex: Mac, VPN, Outlook). "
        "Réponds UNIQUEMENT au format JSON correspondant au schéma."
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Message utilisateur:\n{ticket}\n\nRéponds au format demandé."),
        ]
    )

    # Astuce LangChain : sortie structurée directement en Pydantic
    structured = model.with_structured_output(TicketClassification)
    chain = prompt | structured
    return chain

def classify_ticket(ticket: str) -> TicketClassification:
    chain = _build_classifier()
    return chain.invoke({"ticket": ticket})

# Petite aide pour la suite (affiche une suggestion d'orientation)
def next_step_from_classification(cls: TicketClassification) -> str:
    if cls.category == "incident":
        return "Orienter vers diagnostic rapide (check réseau, statut service, logs)."
    if cls.category == "demande":
        return "Créer une demande standard (process d'approbation, modèles de réponses)."
    return "Réponse informationnelle (FAQ / guide utilisateur)."
