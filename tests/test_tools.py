# tests/test_tools.py
from app.tools import extract_emails, check_host

def test_extract_emails_basic():
    txt = "Contacte alice@test.com et bob.smith+it@company.org; sinon vois foo@bar"
    emails = extract_emails.invoke({"text": txt})
    assert "alice@test.com" in emails
    assert "bob.smith+it@company.org" in emails
    # foo@bar ne doit PAS passer (pas de TLD)
    assert all(e.count("@") == 1 and "." in e.split("@")[1] for e in emails)

def test_check_host_runs():
    # On ne vérifie pas l'IP précise, juste que la fonction renvoie bien une structure dict
    result = check_host.invoke({"host": "example.com"})
    assert isinstance(result, dict)
    assert set(result.keys()) == {"host", "resolvable", "ip", "error"}
