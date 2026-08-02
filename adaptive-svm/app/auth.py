"""
auth.py — lightweight role-based authentication for the surveillance dashboard.

Passwords are never stored in plain text — only a salted SHA-256 hash is kept. Two roles:
  • worker      — a health worker: register cases, see monitoring, get the brain's suggestions.
  • supervisor  — full access, including the System Brain and Model pages.

Credentials can be overridden from Streamlit secrets (an [auth] table of username = "role:password");
otherwise the demo accounts below apply. For a production system these would live in a database or
an identity provider — this is a deliberately simple gate for the prototype.
"""
import hashlib

ROLE_LABELS = {"worker": "Health Worker", "supervisor": "Supervisor"}


def _hash(salt, pw):
    return hashlib.sha256((salt + (pw or "")).encode()).hexdigest()


# Demo accounts — CHANGE THESE for any real use.
_DEMO = {
    "worker":     {"name": "Health Worker",  "role": "worker",     "password": "health123"},
    "supervisor": {"name": "Dr. Supervisor", "role": "supervisor", "password": "admin123"},
}


def _users():
    """Merge demo accounts with any [auth] overrides from Streamlit secrets."""
    users = {u: {"name": v["name"], "role": v["role"], "salt": u,
                 "hash": _hash(u, v["password"])} for u, v in _DEMO.items()}
    try:
        import streamlit as st
        for uname, spec in dict(st.secrets.get("auth", {})).items():
            role, _, pw = str(spec).partition(":")
            uname = uname.strip().lower()
            users[uname] = {"name": uname.title(), "role": role.strip() or "worker",
                            "salt": uname, "hash": _hash(uname, pw)}
    except Exception:
        pass
    return users


def verify(username, password):
    """Return a user dict {username, name, role} on success, else None."""
    u = _users().get((username or "").strip().lower())
    if u and _hash(u["salt"], password) == u["hash"]:
        return {"username": (username or "").strip().lower(), "name": u["name"], "role": u["role"]}
    return None
