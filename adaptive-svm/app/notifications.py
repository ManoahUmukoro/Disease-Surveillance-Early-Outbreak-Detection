"""
notifications.py — the alerting layer.

When the Adaptive SVM flags a rising outbreak signal (or a high-risk case), this module notifies
the people in charge. Every alert is logged to the structured store (SQLite, via store.py) with a
recipient, a delivery method, and a status; if SMTP is configured via environment variables it also
e-mails the recipients, otherwise it degrades gracefully to the in-app dashboard log.
"""
import os
import smtplib
from datetime import datetime
from email.mime.text import MIMEText

import store

# The "people in charge" — in production this would come from a roster table.
RECIPIENTS = [
    {"name": "State Epidemiologist", "role": "Surveillance lead", "email": "state.epi@health.gov.ng"},
    {"name": "NCDC Duty Officer", "role": "National coordination", "email": "duty.officer@ncdc.gov.ng"},
]


def _send_email(recipient, subject, body) -> bool:
    host = os.environ.get("SMTP_HOST")
    if not host:                       # no SMTP configured → dashboard log only
        return False
    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = os.environ.get("SMTP_FROM", "alerts@surveillance.local")
        msg["To"] = recipient["email"]
        with smtplib.SMTP(host, int(os.environ.get("SMTP_PORT", "587"))) as s:
            s.starttls()
            if os.environ.get("SMTP_USER"):
                s.login(os.environ["SMTP_USER"], os.environ["SMTP_PASS"])
            s.send_message(msg)
        return True
    except Exception:
        return False


def send_alert(disease, location, severity, message, recipients=None):
    """Log + deliver one alert per recipient. Returns a delivery record."""
    recipients = recipients or RECIPIENTS
    ts = datetime.now().isoformat(timespec="seconds")
    subject = f"[{severity}] {disease} outbreak signal — {location}"
    smtp_on = bool(os.environ.get("SMTP_HOST"))
    rows = []
    for r in recipients:
        delivered = _send_email(r, subject, message)
        method = "Email" if delivered else "Dashboard"
        status = "Delivered" if delivered else ("Failed" if smtp_on else "Delivered")
        nid = store.log_notification(ts, disease, location, severity, message,
                                     r["name"], method, status)
        rows.append({"id": nid, "recipient": r["name"], "method": method, "status": status})
    return {"ts": ts, "recipients": [r["name"] for r in recipients], "rows": rows,
            "delivered": any(x["status"] == "Delivered" for x in rows)}


def check_and_notify(signals):
    """signals: list of {disease, location, severity, message}. Fires on HIGH/MEDIUM
    (probability threshold + epidemiological rule already applied upstream)."""
    fired = []
    for s in signals:
        if s.get("severity") in ("HIGH", "MEDIUM"):
            fired.append({**s, **send_alert(s["disease"], s["location"], s["severity"], s["message"])})
    return fired
