# main.py
import argparse
from app.agent import answer
from app.routing import classify_ticket, next_step_from_classification

def build_parser():
    parser = argparse.ArgumentParser(description="Assistant Helpdesk IT (LangChain + Anthropic)")
    sub = parser.add_subparsers(dest="cmd", required=False)

    # par défaut: 'ask' (l'agent général)
    ask_p = sub.add_parser("ask", help="Poser une question libre à l'agent (par défaut).")
    ask_p.add_argument("-q", "--question", default=None)

    # nouveau: 'classify'
    cls_p = sub.add_parser("classify", help="Classifier un ticket (incident/demande/question + urgence).")
    cls_p.add_argument("-t", "--ticket", default=None)

    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()

    # Comportement par défaut = ask
    if args.cmd in (None, "ask"):
        q = args.question or input("Ta question helpdesk IT : ")
        print("\n--- Réponse de l'agent ---\n")
        print(answer(q))
        return

    if args.cmd == "classify":
        t = args.ticket or input("Contenu du ticket à classifier : ")
        cls = classify_ticket(t)
        print("\n--- Classification ---\n")
        print(f"- Catégorie : {cls.category}")
        print(f"- Urgence   : {cls.urgency}")
        print(f"- Produits  : {', '.join(cls.products) if cls.products else '—'}")
        print(f"- Raison    : {cls.rationale}")
        print(f"- Prochaine étape suggérée : {next_step_from_classification(cls)}")
        return

if __name__ == "__main__":
    main()
