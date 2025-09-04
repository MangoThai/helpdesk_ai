# app/rag.py
from __future__ import annotations
import os
from typing import List
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings

PERSIST_DIR = "data/vectorstore"

def _build_vectorstore() -> Chroma:
    # Charge tous les .md de data/faq
    loader = DirectoryLoader("data/faq", glob="*.md", loader_cls=TextLoader, show_progress=True)
    docs = loader.load()

    # Embeddings légers
    embeddings = FastEmbedEmbeddings()

    # Chroma persistant (crée ou remplace si besoin)
    vs = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
    )
    vs.persist()
    return vs

def _load_or_build_vs() -> Chroma:
    embeddings = FastEmbedEmbeddings()
    if os.path.exists(PERSIST_DIR):
        return Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    return _build_vectorstore()

def build_rag_chain():
    load_dotenv()
    model_name = os.getenv("ANTHROPIC_MODEL", "claude-3-7-sonnet-20250219")
    llm = ChatAnthropic(model=model_name, temperature=0.1)

    vs = _load_or_build_vs()
    retriever = vs.as_retriever(search_kwargs={"k": 4})

    system = (
        "Tu es un assistant Helpdesk IT. "
        "Utilise STRICTEMENT les extraits fournis comme base de réponse. "
        "Si l'information n'est pas dans les extraits, explique calmement que tu n'as pas la donnée."
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human",
             "Question: {question}\n\n"
             "Extraits:\n{context}\n\n"
             "Réponse (claire, étapes numérotées si pertinent) :")
        ]
    )

    def format_docs(docs):
        # Ajoute le chemin du fichier comme “source”
        lines = []
        for i, d in enumerate(docs, 1):
            src = d.metadata.get("source", "source_inconnue")
            lines.append(f"[{i}] ({src})\n{d.page_content}")
        return "\n\n".join(lines)

    # Chaîne RAG : (question) -> retrieve -> format -> LLM
    chain = (
        {"question": RunnablePassthrough(), "context": retriever | format_docs}
        | prompt
        | llm
    )
    return chain

def rag_answer(question: str) -> str:
    chain = build_rag_chain()
    return chain.invoke(question).content
