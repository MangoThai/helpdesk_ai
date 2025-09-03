# app/agent.py
from __future__ import annotations

import os
from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def build_helpdesk_agent():
    """
    Construit une chaîne (prompt -> modèle -> parseur) qui prend une 'question'
    et renvoie une réponse texte.
    """
    # 1) Charger les variables d'environnement (.env) une seule fois
    load_dotenv()

    # 2) Vérifier la clé API (message clair si absente)
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY introuvable. Ajoute-la dans .env comme ANTHROPIC_API_KEY=sk-ant-..."
        )

    # 3) Définir le modèle Anthropic via LangChain
    #    Choix pédagogique : modèle 'claude-3-5-sonnet-20240620' (équilibre qualité/coût/latence)
    model = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        temperature=0.2,     # Réponses stables et factuelles
        max_tokens=512       # Limite la longueur de sortie (contrôle des coûts + lisibilité)
    )

    # 4) Définir le “rôle” (contexte) de l’assistant
    system_instructions = (
        "Tu es un assistant Helpdesk IT francophone. "
        "Avant de proposer une solution, vérifie si tu as assez d'informations. "
        "Si nécessaire, pose 1 à 3 questions de clarification, pas plus. "
        "Quand tu proposes une solution, donne des étapes claires, numérotées, "
        "et mentionne les risques éventuels. Si tu n'es pas certain, dis-le."
    )

    # 5) Construire le prompt chat (messages System + Human)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_instructions),
            ("human", "{question}")  # variable 'question' fournie à l'exécution
        ]
    )

    # 6) Chaîner : prompt -> modèle -> parseur de chaîne (renvoie str)
    chain = prompt | model | StrOutputParser()
    return chain


def answer(question: str) -> str:
    """
    Point d'entrée pratique : on construit la chaîne et on l'invoque.
    Séparé pour garder un code testable et clair.
    """
    chain = build_helpdesk_agent()
    return chain.invoke({"question": question})
