# Helpdesk AI — LangChain / LangGraph / RAG (macOS)

Assistant Helpdesk IT pédagogique :  
- **ask** : agent simple Anthropic  
- **classify** : routage (incident / demande / question + urgence)  
- **rag** : réponses appuyées sur une FAQ locale (Chroma)  
- **chat** : session multi-tours avec outils (extraction d’emails, check DNS)

## Pile technique
- Python 3.10+ (testé sur macOS)
- LangChain, LangGraph
- Anthropic (Claude) via `langchain-anthropic`
- Chroma + FastEmbed (RAG)
- PyYAML (mémoire longue / profil)

## Démarrage
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # puis mets ta clé
