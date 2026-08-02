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


# ── surge detection (aberration) ─────────────────────────────────────────
def detect_surges(ev, min_cases=200, factor=2.0, recent_months=3):
    """Flag any disease-state whose latest month is well above its own recent baseline — simple
    aberration detection in the spirit of the CDC EARS early-warning algorithms. This works across
    EVERY disease (not just the SVM-modelled Lassa), so the brain is aware of the whole picture. Only
    recent months are considered, so a historical tail is never mistaken for a current surge."""
    gmax = ev.report_date.max()
    cutoff = gmax.to_period("M") - recent_months
    out = []
    for disease in [d for d in sorted(ev.disease.unique()) if d not in ("Other", "Undetermined")]:
        d = ev[ev.disease == disease]
        ts = (d.assign(ym=d.report_date.dt.to_period("M")).groupby(["state", "ym"])
              .new_cases.sum().reset_index())
        for st, g in ts.groupby("state"):
            g = g.sort_values("ym")
            if len(g) < 4 or g.ym.iloc[-1] < cutoff:
                continue
            latest = int(g.new_cases.iloc[-1]); base = g.new_cases.iloc[:-1].tail(6)
            mu = float(base.mean()) if len(base) else 0.0
            if latest >= min_cases and latest >= factor * max(mu, 1) and latest > base.max():
                out.append({"disease": disease, "state": st, "latest": latest,
                            "baseline": round(mu, 1), "ratio": round(latest / max(mu, 1), 1)})
    return sorted(out, key=lambda x: x["latest"], reverse=True)


# ── ACT ──────────────────────────────────────────────────────────────────
def act(candidates, mongo):
    """Autonomously alert the highest-priority outbreak signals — detected SURGES (any disease) plus
    adaptive-SVM HIGH predictions — biggest first, respecting a cooldown so the same state is not
    re-alerted every cycle."""
    now = datetime.now()
    since = (now - timedelta(hours=COOLDOWN_HOURS)).isoformat(timespec="seconds")
    fired, suppressed = [], 0
    for a in candidates:
        if len(fired) >= MAX_ALERTS_PER_RUN:
            break
        if mongo.alerted_recently(a["disease"], a["state"], since):
            suppressed += 1
            continue
        if a.get("kind") == "surge":
            msg = (f"Autonomous engine: SURGE detected for {a['disease']} in {a['state']} — "
                   f"{a['confirmed']:,} cases this month (~{a['ratio']}x the recent baseline). "
                   "Recommend immediate NCDC field verification.")
        else:
            msg = (f"Autonomous engine: HIGH outbreak risk for {a['disease']} in {a['state']} "
                   f"({a['prob']:.0%} calibrated probability; recent cases={a['confirmed']}). "
                   "Recommend NCDC field verification.")
        res = notifications.send_alert(a["disease"], a["state"], "HIGH", msg)
        mongo.record_alert_key(a["disease"], a["state"], now.isoformat(timespec="seconds"))
        fired.append({"disease": a["disease"], "state": a["state"], "severity": "HIGH",
                      "prob": a.get("prob"), "confirmed": a.get("confirmed"), "kind": a.get("kind", "svm"),
                      "message": msg, "recipients": res["recipient_str"], "method": res["method"]})
    return {"n_candidates": len(candidates), "n_fired": len(fired),
            "suppressed_cooldown": suppressed, "alerts": fired}


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

    surges = detect_surges(ev)                                              # aberration across ALL diseases
    surge_sig = [{"disease": s["disease"], "state": s["state"], "confirmed": s["latest"],
                  "prob": min(0.95, 0.55 + 0.05 * s["ratio"]), "ratio": s["ratio"], "kind": "surge"}
                 for s in surges]
    svm_high = [{"disease": a["disease"], "state": a["state"], "confirmed": a["confirmed"],
                 "prob": a["prob"], "kind": "svm"} for a in svm if a["level"] == "HIGH"]
    candidates = sorted(surge_sig + svm_high, key=lambda x: x["confirmed"], reverse=True)
    action = act(candidates, mongo)
    learning = learn(metrics, sinfo)

    monitored = {}                                                          # highest-burden state per disease
    for a in sorted(burden, key=lambda x: x["prob"], reverse=True):
        monitored.setdefault(a["disease"], {"disease": a["disease"], "top_state": a["state"],
                                            "confirmed": a["confirmed"]})
    predicted_states = sorted(svm, key=lambda a: a["prob"], reverse=True)

    if surges:
        s0 = surges[0]
        headline = (f"🚨 Detected a surge — {s0['disease']} in {s0['state']}: {s0['latest']:,} cases "
                    f"this month (~{s0['ratio']}x baseline).")
    else:
        headline = f"Adaptive SVM: {counts['HIGH']} HIGH / {counts['MEDIUM']} MEDIUM for Lassa fever."
    narrative = (f"{headline} Scanned {sinfo['diseases']} diseases across {sinfo['states']} states · "
                 f"fired {action['n_fired']} alert(s) · model AUC {learning['roc_auc']}.")
    run = {"ts": ts, "status": "ok", "narrative": narrative, "sense": sinfo,
           "think": {"predicted": {"disease": "Lassa fever", "counts": counts, "states": predicted_states},
                     "monitored": list(monitored.values()), "surges": surges},
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
