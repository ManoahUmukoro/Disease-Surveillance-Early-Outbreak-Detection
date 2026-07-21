"""
notifications.py — the alerting layer.

When the Adaptive SVM flags a rising outbreak signal (or a high-risk case), this
module notifies the people in charge. It logs every alert to SQLite (the structured
store) and, if SMTP is configured via environment variables, emails the recipients;
otherwise it degrades gracefully to an in-app log so the prototype runs with zero setup.
"""
import os
import json
import smtplib
import sqlite3
from datetime import datetime
from email.mime.text import MIMEText
from pathlib import Path

DB = Path(__file__).resolve().parents[1] / "data" / "surveillance.db"

# The "people in charge" — in production this would come from a roster table.
RECIPIENTS = [
    {"name": "State Epidemiologist", "role": "Surveillance lead", "email": "state.epi@health.gov.ng"},
    {"name": "NCDC Duty Officer", "role": "National coordination", "email": "duty.officer@ncdc.gov.ng"},
]


def _conn():
    DB.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB)
    c.execute("""CREATE TABLE IF NOT EXISTS notifications(
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, disease TEXT, location TEXT,
        severity TEXT, message TEXT, recipients TEXT, delivered INTEGER)""")
    return c


def _send_email(recipients, subject, body) -> bool:
    host = os.environ.get("SMTP_HOST")
    if not host:                       # no SMTP configured → in-app log only
        return False
    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = os.environ.get("SMTP_FROM", "alerts@surveillance.local")
        msg["To"] = ", ".join(r["email"] for r in recipients)
        with smtplib.SMTP(host, int(os.environ.get("SMTP_PORT", "587"))) as s:
            s.starttls()
            if os.environ.get("SMTP_USER"):
                s.login(os.environ["SMTP_USER"], os.environ["SMTP_PASS"])
            s.send_message(msg)
        return True
    except Exception:
        return False


def send_alert(disease, location, severity, message, recipients=None):
    """Log an alert and notify recipients. Returns a delivery record."""
    recipients = recipients or RECIPIENTS
    ts = datetime.now().isoformat(timespec="seconds")
    subject = f"[{severity}] {disease} outbreak signal — {location}"
    delivered = _send_email(recipients, subject, message)
    c = _conn()
    c.execute("INSERT INTO notifications(ts,disease,location,severity,message,recipients,delivered)"
              " VALUES(?,?,?,?,?,?,?)",
              (ts, disease, location, severity, message,
               json.dumps([r["email"] for r in recipients]), int(delivered)))
    c.commit(); c.close()
    return {"ts": ts, "delivered": delivered, "channel": "email" if delivered else "in-app log",
            "recipients": [r["name"] for r in recipients]}


def check_and_notify(signals):
    """signals: list of {disease, location, severity, message}. Fires on HIGH/MEDIUM
    (probability threshold + epidemiological rule already applied upstream)."""
    fired = []
    for s in signals:
        if s.get("severity") in ("HIGH", "MEDIUM"):
            fired.append({**s, **send_alert(s["disease"], s["location"], s["severity"], s["message"])})
    return fired


def recent(limit=25):
    c = _conn()
    rows = c.execute("SELECT ts,disease,location,severity,message,delivered "
                     "FROM notifications ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    c.close()
    return rows
