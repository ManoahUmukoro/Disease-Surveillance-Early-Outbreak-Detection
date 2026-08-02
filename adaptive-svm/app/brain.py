#!/usr/bin/env python3
"""
brain.py — the Autonomous Surveillance Engine (the system "brain").

A headless, self-running control loop that gives the platform autonomy: it runs on a schedule
(GitHub Actions cron) with NO human in the loop and executes one cognitive cycle each time —

    SENSE   read the latest surveillance data (all diseases, all states)
    THINK   score every state x monitored disease with the adaptive SVM (calibrated probability)
    ACT     raise alerts for HIGH-risk, model-predicted outbreaks (with a cooldown, so it never spams)
    LEARN   run the online (partial_fit) update over the data and record current performance

and persists a full run-log so the dashboard's "System Brain" page can show exactly what it did.
This is the classic MAPE-K autonomic-computing loop (Monitor-Analyze-Plan-Execute over a shared
Knowledge base). It is deliberately standalone — no Streamlit import — so it runs anywhere.

Autonomy honestly stated: the engine autonomously MONITORS, DETECTS, ALERTS and LEARNS; a human
still verifies an outbreak in the field before a response is dispatched (human-on-the-loop).
"""
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "app"))
sys.path.insert(0, str(HERE / "scripts"))
import store
import prepare_and_train as P
import notifications
from mongo_store import MongoStore

HIGH_T, MED_T = 0.50, 0.20
LOG_FILE = HERE / "data" / "brain_log.json"
COOLDOWN_HOURS = int(os.environ.get("BRAIN_COOLDOWN_HOURS", "24"))
MAX_ALERTS_PER_RUN = int(os.environ.get("BRAIN_MAX_ALERTS", "8"))


def band(p):
    return "HIGH" if p >= HIGH_T else ("MEDIUM" if p >= MED_T else "LOW")


# ── SENSE ────────────────────────────────────────────────────────────────
def sense():
    store.bootstrap(P.load())
    ev = store.events_df()
    latest = ev.report_date.max()
    return ev, {
        "events": int(len(ev)),
        "diseases": int(ev.disease.nunique()),
        "states": int(ev.state.nunique()),
        "latest_data": (latest.date().isoformat() if latest is not None else None),
        "registered_cases": int(store.case_count()),
    }


# ── THINK ────────────────────────────────────────────────────────────────
def think(bundles, ev):
    monitored = [d for d in sorted(ev.disease.unique()) if d not in ("Other", "Undetermined")]
    b = bundles["outbreak"]
    out = []
    for disease in monitored:
        d = ev[ev.disease == disease].copy()
        if d.empty:
            continue
        d["yr"] = d.report_date.dt.year
        d["mo"] = d.report_date.dt.month
        agg = d.groupby(["state", "yr", "mo"]).agg(confirmed=("new_cases", "sum")).reset_index()
        agg["ord"] = agg.yr * 12 + agg.mo
        agg = agg.sort_values(["state", "ord"])
        if disease == "Lassa fever":                       # the adaptive SVM (genuine prediction)
            g = agg.groupby("state")["confirmed"]
            agg["lag1"] = g.shift(1); agg["lag2"] = g.shift(2)
            agg["roll3"] = g.shift(1).rolling(3, min_periods=1).mean(); agg["trend"] = agg.lag1 - agg.lag2
            agg["hotspot"] = agg.state.isin(P.HOTSPOTS).astype(int)
            agg["sin"] = np.sin(2 * np.pi * agg.mo / 12); agg["cos"] = np.cos(2 * np.pi * agg.mo / 12)
            agg = agg.dropna(subset=["lag1", "lag2", "roll3"])
            if agg.empty:
                continue
            latest = agg.sort_values("ord").groupby("state").tail(1)
            margins = b["model"].decision_function(b["scaler"].transform(latest[b["features"]]))
            for (_, row), m in zip(latest.iterrows(), margins):
                p = P.calibrated_prob(b, float(m))
                out.append({"disease": disease, "state": row.state, "confirmed": int(row.confirmed),
                            "prob": round(p, 3), "level": band(p), "method": "adaptive SVM"})
        else:                                              # descriptive recent-burden rank
            ymax = int(agg.yr.max()); recent = agg[agg.yr >= ymax - 5]
            grp = (recent if not recent.empty else agg).groupby("state").agg(
                confirmed=("confirmed", "sum")).reset_index()
            grp["prob"] = (grp.confirmed.rank(pct=True) * 0.9).round(3)
            for _, row in grp.iterrows():
                out.append({"disease": disease, "state": row.state, "confirmed": int(row.confirmed),
                            "prob": float(row.prob), "level": band(float(row.prob)),
                            "method": "recent-burden"})
    return out


# ── ACT ──────────────────────────────────────────────────────────────────
def act(assessments, mongo):
    """Alert on HIGH, model-PREDICTED outbreaks (adaptive SVM), newest first, respecting a cooldown
    so the same state is not re-alerted every cycle. Burden-ranked diseases are monitored, not auto-
    alerted (their rank is descriptive, not a prediction)."""
    now = datetime.now()
    since = (now - timedelta(hours=COOLDOWN_HOURS)).isoformat(timespec="seconds")
    candidates = sorted([a for a in assessments if a["method"] == "adaptive SVM" and a["level"] == "HIGH"],
                        key=lambda a: a["prob"], reverse=True)
    fired, suppressed = [], 0
    for a in candidates:
        if len(fired) >= MAX_ALERTS_PER_RUN:
            break
        if mongo.alerted_recently(a["disease"], a["state"], since):
            suppressed += 1
            continue
        msg = (f"Autonomous engine: HIGH outbreak risk for {a['disease']} in {a['state']} "
               f"({a['prob']:.0%} calibrated probability; recent confirmed cases={a['confirmed']}). "
               "Recommend NCDC field verification.")
        res = notifications.send_alert(a["disease"], a["state"], "HIGH", msg)
        mongo.record_alert_key(a["disease"], a["state"], now.isoformat(timespec="seconds"))
        fired.append({"disease": a["disease"], "state": a["state"], "severity": "HIGH",
                      "prob": a["prob"], "message": msg, "recipients": res["recipient_str"],
                      "method": res["method"]})
    return {"n_high": len(candidates), "n_fired": len(fired), "suppressed_cooldown": suppressed,
            "alerts": fired}


# ── LEARN ────────────────────────────────────────────────────────────────
def learn(metrics, sense_info):
    m = metrics.get("outbreak", {})
    return {"model": "adaptive SVM (averaged SGD, online partial_fit)",
            "roc_auc": round(float(m.get("auc", float("nan"))), 3),
            "f1": round(float(m.get("f1", float("nan"))), 3),
            "observations_learned": int(m.get("n_eval", 0)),
            "note": "Online update run over the full chronological stream; the model incorporates the "
                    "latest observations each cycle and does not go stale."}


# ── orchestration ────────────────────────────────────────────────────────
def run_once():
    ts = datetime.now().isoformat(timespec="seconds")
    mongo = MongoStore()
    ev, sinfo = sense()
    bundles, metrics = P.build_bundles(P.load())          # online (adaptive) learning happens here
    assessments = think(bundles, ev)
    svm = [a for a in assessments if a["method"] == "adaptive SVM"]          # genuine predictions (Lassa)
    burden = [a for a in assessments if a["method"] == "recent-burden"]      # descriptive monitoring
    counts = {lvl: int(sum(1 for a in svm if a["level"] == lvl)) for lvl in ("HIGH", "MEDIUM", "LOW")}
    action = act(assessments, mongo)
    learning = learn(metrics, sinfo)

    monitored = {}                                                          # highest-burden state per disease
    for a in sorted(burden, key=lambda x: x["prob"], reverse=True):
        monitored.setdefault(a["disease"], {"disease": a["disease"], "top_state": a["state"],
                                            "confirmed": a["confirmed"]})
    predicted_states = sorted(svm, key=lambda a: a["prob"], reverse=True)

    narrative = (f"Adaptive SVM scanned Lassa fever across {len(svm)} states "
                 f"({counts['HIGH']} HIGH, {counts['MEDIUM']} MEDIUM); monitoring {len(monitored)} "
                 f"other diseases by recent burden. Fired {action['n_fired']} alert(s). "
                 f"Model AUC {learning['roc_auc']}.")
    run = {"ts": ts, "status": "ok", "narrative": narrative, "sense": sinfo,
           "think": {"predicted": {"disease": "Lassa fever", "counts": counts, "states": predicted_states},
                     "monitored": list(monitored.values())},
           "act": action, "learn": learning}

    mongo.save_brain_run(run)                              # -> Atlas (dashboard reads this)
    _write_local_log(run)                                  # -> local JSON (dev + fallback)
    print("BRAIN RUN:", narrative)
    print("  atlas:", "written" if mongo.available else f"skipped ({mongo.reason})")
    return run


def _write_local_log(run, keep=50):
    try:
        log = json.loads(LOG_FILE.read_text()) if LOG_FILE.exists() else []
    except Exception:
        log = []
    log.insert(0, run)
    LOG_FILE.write_text(json.dumps(log[:keep], indent=2))


if __name__ == "__main__":
    run_once()
