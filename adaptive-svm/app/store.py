"""
store.py — the STRUCTURED data layer (SQLite) of the Adaptive SVM surveillance system.

In the Chapter 4 architecture this is the "structured / time-series store". It holds:
  • surveillance_events — the time series the Adaptive SVM reads (report_date, state, lga,
    disease, new_cases, deaths, temperature, rainfall, + trend features)
  • cases             — one structured row per registered case (symptoms, demographics,
    environmental factors, and the model's prediction)
  • notifications     — the alert log (status / recipient / method / acknowledged)

Unstructured records (clinical notes, lab reports, uploaded documents) live in MongoDB —
see mongo_store.py. The two are linked by case_id.

No Streamlit import here on purpose, so the data layer is testable on its own.
"""
import json
import math
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd

DB = Path(__file__).resolve().parents[1] / "data" / "surveillance.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS surveillance_events(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_date TEXT, state TEXT, lga TEXT, disease TEXT,
    new_cases INTEGER, deaths INTEGER, temperature REAL, rainfall REAL,
    source TEXT DEFAULT 'historical', created_at TEXT
);
CREATE TABLE IF NOT EXISTS cases(
    case_id TEXT PRIMARY KEY, report_date TEXT, state TEXT, lga TEXT, disease TEXT,
    age_group TEXT, sex TEXT, symptoms TEXT, n_symptoms INTEGER,
    rodent_contact INTEGER, known_contact INTEGER,
    flooding INTEGER, rodent_activity INTEGER, travel INTEGER,
    temperature REAL, rainfall REAL,
    pred_label TEXT, pred_risk REAL, pred_confidence REAL, recommendation TEXT,
    alert_status TEXT, has_documents INTEGER DEFAULT 0, created_at TEXT
);
CREATE TABLE IF NOT EXISTS notifications(
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, disease TEXT, location TEXT,
    severity TEXT, message TEXT, recipient TEXT, method TEXT,
    status TEXT, acknowledged INTEGER DEFAULT 0
);
"""


def conn():
    DB.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(DB))
    # Migrate a legacy notifications table that an earlier deployment may have left behind
    # (its schema predates recipient/method/status/acknowledged). CREATE TABLE IF NOT EXISTS
    # would otherwise keep the old columns and every new query would fail with "no such column".
    try:
        cols = {r[1] for r in c.execute("PRAGMA table_info(notifications)").fetchall()}
        if cols and not {"recipient", "method", "status", "acknowledged"}.issubset(cols):
            c.execute("DROP TABLE notifications")
            c.commit()
    except Exception:
        pass
    c.executescript(SCHEMA)
    return c


def _now():
    return datetime.now().isoformat(timespec="seconds")


# ── illustrative climatology ─────────────────────────────────────────────
# Deterministic month-based values so surveillance_events has plausible weather columns
# without a live feed. Clearly NOT measured data — for operational (registered) cases the
# user enters the real values in the Environmental Factors section of the form.
def climate(month, state=""):
    m = int(month)
    # Nigeria: hotter/drier Dec–Mar, wet Jun–Sep. Small per-state offset for variety.
    off = (sum(map(ord, state)) % 5) - 2
    temp = round(29 + 3 * math.cos(2 * math.pi * (m - 3) / 12) + off * 0.4, 1)
    rain = round(max(0.0, 14 + 12 * math.sin(2 * math.pi * (m - 4) / 12) + off), 1)
    return temp, rain


# ── surveillance_events (time series) ────────────────────────────────────
def events_empty():
    c = conn()
    n = c.execute("SELECT COUNT(*) FROM surveillance_events").fetchone()[0]
    c.close()
    return n == 0


def bootstrap(df):
    """Seed surveillance_events once, from the real SORMAS Lassa line-list (monthly per state),
    plus a small illustrative multi-disease sample so the disease filter is meaningful."""
    if not events_empty():
        return
    d = df.copy()
    d["deceased"] = (d["outcome_case"].astype(str) == "Deceased").astype(int)
    agg = (d.groupby(["State_new", "yr", "mo"])
           .agg(new_cases=("positive", "sum"), deaths=("deceased", "sum")).reset_index())
    rows = []
    for r in agg.itertuples():
        mo = int(r.mo)
        temp, rain = climate(mo, str(r.State_new))
        rows.append((f"{int(r.yr):04d}-{mo:02d}-01", str(r.State_new), None, "Lassa fever",
                     int(r.new_cases), int(r.deaths), temp, rain, "historical", _now()))
    # small illustrative Cholera sample (matches the advisor's example table)
    for day, nc, dth, t, rf in [("2026-01-01", 4, 0, 30, 12), ("2026-01-02", 8, 1, 31, 15),
                                ("2026-01-03", 18, 2, 31, 20)]:
        rows.append((day, "Anambra", "Onitsha North", "Cholera", nc, dth, t, rf, "sample", _now()))
    c = conn()
    c.executemany("INSERT INTO surveillance_events(report_date,state,lga,disease,new_cases,"
                  "deaths,temperature,rainfall,source,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)", rows)
    c.commit(); c.close()


def events_df():
    c = conn()
    df = pd.read_sql_query("SELECT * FROM surveillance_events", c)
    c.close()
    if not df.empty:
        df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
    return df


def add_event(report_date, state, lga, disease, new_cases, deaths, temperature, rainfall,
              source="registration"):
    c = conn()
    c.execute("INSERT INTO surveillance_events(report_date,state,lga,disease,new_cases,deaths,"
              "temperature,rainfall,source,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
              (report_date, state, lga, disease, int(new_cases), int(deaths),
               temperature, rainfall, source, _now()))
    c.commit(); c.close()


# ── cases (structured case records) ──────────────────────────────────────
def insert_case(rec: dict) -> str:
    """rec: the structured fields of a registered case. Returns the generated case_id."""
    case_id = rec.get("case_id") or "C-" + uuid.uuid4().hex[:10].upper()
    c = conn()
    c.execute(
        "INSERT INTO cases(case_id,report_date,state,lga,disease,age_group,sex,symptoms,"
        "n_symptoms,rodent_contact,known_contact,flooding,rodent_activity,travel,temperature,"
        "rainfall,pred_label,pred_risk,pred_confidence,recommendation,alert_status,"
        "has_documents,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (case_id, rec.get("report_date"), rec.get("state"), rec.get("lga"), rec.get("disease"),
         rec.get("age_group"), rec.get("sex"), json.dumps(rec.get("symptoms", [])),
         int(rec.get("n_symptoms", 0)), int(rec.get("rodent_contact", 0)),
         int(rec.get("known_contact", 0)), int(rec.get("flooding", 0)),
         int(rec.get("rodent_activity", 0)), int(rec.get("travel", 0)),
         rec.get("temperature"), rec.get("rainfall"), rec.get("pred_label"),
         rec.get("pred_risk"), rec.get("pred_confidence"), rec.get("recommendation"),
         rec.get("alert_status"), int(rec.get("has_documents", 0)), _now()))
    c.commit(); c.close()
    return case_id


def mark_case_documents(case_id):
    c = conn()
    c.execute("UPDATE cases SET has_documents=1 WHERE case_id=?", (case_id,))
    c.commit(); c.close()


def recent_cases(limit=25):
    c = conn()
    df = pd.read_sql_query(
        "SELECT case_id,report_date,state,lga,disease,pred_label,pred_risk,recommendation,"
        "alert_status,has_documents,created_at FROM cases ORDER BY created_at DESC LIMIT ?",
        c, params=(limit,))
    c.close()
    return df


def case_count():
    c = conn()
    n = c.execute("SELECT COUNT(*) FROM cases").fetchone()[0]
    c.close()
    return n


# ── notifications ────────────────────────────────────────────────────────
def log_notification(ts, disease, location, severity, message, recipient, method, status):
    c = conn()
    cur = c.execute("INSERT INTO notifications(ts,disease,location,severity,message,recipient,"
                    "method,status,acknowledged) VALUES(?,?,?,?,?,?,?,?,0)",
                    (ts, disease, location, severity, message, recipient, method, status))
    nid = cur.lastrowid
    c.commit(); c.close()
    return nid


def recent_notifications(limit=25):
    c = conn()
    df = pd.read_sql_query(
        "SELECT id,ts,disease,location,severity,message,recipient,method,status,acknowledged "
        "FROM notifications ORDER BY id DESC LIMIT ?", c, params=(limit,))
    c.close()
    return df


def acknowledge(nid):
    c = conn()
    c.execute("UPDATE notifications SET acknowledged=1 WHERE id=?", (nid,))
    c.commit(); c.close()
