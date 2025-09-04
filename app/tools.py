# app/tools.py
from __future__ import annotations
import re
import socket
from typing import List, Dict, Any
from langchain_core.tools import tool

@tool
def extract_emails(text: str) -> List[str]:
    """
    Extrait les adresses e-mail présentes dans 'text'.
    Retourne une liste triée et dédupliquée.
    """
    emails = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text or "")
    return sorted(set(emails))

@tool
def check_host(host: str) -> Dict[str, Any]:
    """
    Vérifie si 'host' se résout via DNS (sans ping réseau).
    Retourne: {'host': ..., 'resolvable': bool, 'ip': 'x.x.x.x' ou None, 'error': str (si échec)}
    """
    try:
        ip = socket.gethostbyname(host)
        return {"host": host, "resolvable": True, "ip": ip, "error": None}
    except Exception as e:
        return {"host": host, "resolvable": False, "ip": None, "error": str(e)}

# Expose une liste d’outils et un mapping par nom (pratique pour l’agent)
TOOLS = [extract_emails, check_host]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}
