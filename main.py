# main.py
import argparse
from langchain_core.messages import HumanMessage, AIMessage

from app.agent import answer
from app.routing import classify_ticket, next_step_from_classification
from app.rag import rag_answer
from app.agent_graph import build_agent_app, agent_answer_once


def build_parser():
    parser = argparse.ArgumentParser(description="Assistant Helpdesk IT (LangChain + Anthropic)")
    sub = parser.add_subparsers(dest="cmd", required=False)

    # ask (par défaut)
    ask_p = sub.add_parser("ask", help="Poser une question libre à l'agent (par défaut).")
    ask_p.add_argument("-q", "--question", default=None)

    # classify
    cls_p = sub.add_parser("classify", help="Classifier un ticket (incident/demande/question + urgence).")
    cls_p.add_argument("-t", "--ticket", default=None)

    # rag
    rag_p = sub.add_parser("rag", help="Question avec RAG (FAQ locales).")
    rag_p.add_argument("-q", "--question", default=None)

    # chat (session multi-tours avec outils)
    sub.add_parser("chat", help="Session interactive multi-tours avec outils (mémoire courte).")

    return parser


def run_chat_session():
    """
    Boucle interactive :
    - On compile une fois le graphe agent.
    - À chaque tour, on ajoute le message humain et on laisse le graphe raisonner (tools si besoin).
    - L'historique 'messages' est conservé en RAM = mémoire courte.
    """
    app = build_agent_app()
    messages = []

    print("Mode chat (multi-tours). Tape '/exit' pour quitter.\n")
    while True:
        user = input("Toi: ").strip()
        if not user:
            continue
        if user.lower() in {"/exit", "exit", "quit", "/q"}:
            print("Fin de la session.")
            break

        messages.append(HumanMessage(content=user))
        result = app.invoke({"messages": messages})
        # On remplace 'messages' par le nouvel historique renvoyé par le graphe
        messages = result["messages"]

        # On cherche le dernier message IA pour l'afficher
        last_ai = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
        if last_ai:
            content = last_ai.content
            print("\nAgent:", content if isinstance(content, str) else str(content), "\n")


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd in (None, "ask"):
        q = getattr(args, "question", None) or input("Ta question helpdesk IT : ")
        print("\n--- Réponse de l'agent ---\n")
        print(answer(q))
        return

    if args.cmd == "classify":
        t = getattr(args, "ticket", None) or input("Contenu du ticket à classifier : ")
        cls = classify_ticket(t)
        print("\n--- Classification ---\n")
        print(f"- Catégorie : {cls.category}")
        print(f"- Urgence   : {cls.urgency}")
        print(f"- Produits  : {', '.join(cls.products) if cls.products else '—'}")
        print(f"- Raison    : {cls.rationale}")
        print(f"- Prochaine étape suggérée : {next_step_from_classification(cls)}")
        return

    if args.cmd == "rag":
        q = getattr(args, "question", None) or input("Question (RAG) : ")
        print("\n--- Réponse (RAG) ---\n")
        print(rag_answer(q))
        return

    if args.cmd == "chat":
        run_chat_session()
        return


if __name__ == "__main__":
    main()
