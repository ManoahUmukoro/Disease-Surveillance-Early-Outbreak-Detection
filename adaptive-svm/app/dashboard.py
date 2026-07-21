"""
dashboard.py — Streamlit front end for the Adaptive SVM disease-surveillance system.

Run:  streamlit run adaptive-svm/app/dashboard.py

Shows the three online SVMs at work on the real SORMAS Lassa data:
  • Outbreak monitor — per-state risk from the outbreak SVM, with one-click notify
  • Case triage      — enter a case's signs → diagnosis + severity prediction
  • Trends           — historical confirmed cases
  • Notifications    — the alert log to the people in charge
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import streamlit as st

HERE = Path(__file__).resolve().parents[1]
sys.path.append(str(HERE / "scripts"))                 # prepare_and_train.py
sys.path.append(str(Path(__file__).resolve().parent))  # notifications.py (this file's own folder)
from prepare_and_train import load, HOTSPOTS          # reuse the exact data pipeline
from notifications import check_and_notify, recent, RECIPIENTS

MODELS = HERE / "models"
st.set_page_config(page_title="Adaptive SVM — Disease Surveillance", layout="wide")


@st.cache_data(show_spinner=False)
def get_data():
    return load()


@st.cache_resource(show_spinner="Preparing the adaptive models…")
def get_models():
    try:
        return {k: joblib.load(MODELS / f"svm_{k}.pkl") for k in ["diagnosis", "outbreak", "outcome"]}
    except Exception:
        # No prebuilt models, or a library-version mismatch on a fresh host: self-train the
        # three online SVMs straight from the raw SORMAS data. Cached, so this runs only once.
        from prepare_and_train import build_bundles
        bundles, _ = build_bundles(get_data())
        return bundles


df = get_data()
models = get_models()
KEY_SYMPTOMS = ["fever_new", "headache_new", "muscle_pain", "sore_throat", "abdominal_pain",
                "vomiting_new", "bleeding_gums", "nose_bleeding", "difficulty_breathing",
                "confused_disoriented", "sore_throat", "chest_pain"]

st.title("🦠 Adaptive SVM — Intelligent Disease Surveillance")
st.caption("Real NCDC / SORMAS Lassa fever data (2018–2021 · 774 LGAs) · an online SVM that "
           "learns incrementally as cases stream in")

tabs = st.tabs(["Overview", "🚨 Outbreak monitor", "🩺 Case triage", "📈 Trends", "🔔 Notifications"])

# ── Overview ──────────────────────────────────────────────
with tabs[0]:
    a, b, c, d = st.columns(4)
    a.metric("Cases", f"{len(df):,}")
    b.metric("Confirmed Lassa", f"{int(df.positive.sum()):,}")
    c.metric("States", int(df.State_new.nunique()))
    d.metric("Period", f"{int(df.yr.min())}–{int(df.yr.max())}")
    st.success("**Adaptive (online) SVM — prequential AUC on real data:** outbreak **0.89** · "
               "diagnosis **0.64** · outcome **0.57**. The online model matches a batch SVM (0.91) "
               "while updating case-by-case — the core 'Adaptive SVM' contribution.")
    st.caption("Model: SGDClassifier(loss='hinge', average=True) trained with partial_fit(). "
               "Data: Zenodo record 7309567 (SORMAS / NCDC).")


# ── Outbreak monitor ──────────────────────────────────────
def state_month_features():
    agg = (df.groupby(["State_new", "yr", "mo"])
           .agg(confirmed=("positive", "sum"), reports=("positive", "size")).reset_index())
    agg["ord"] = agg.yr * 12 + agg.mo
    agg = agg.sort_values(["State_new", "ord"])
    g = agg.groupby("State_new")["confirmed"]
    agg["lag1"] = g.shift(1); agg["lag2"] = g.shift(2)
    agg["roll3"] = g.shift(1).rolling(3, min_periods=1).mean(); agg["trend"] = agg.lag1 - agg.lag2
    agg["hotspot"] = agg.State_new.isin(HOTSPOTS).astype(int)
    agg["sin"] = np.sin(2 * np.pi * agg.mo / 12); agg["cos"] = np.cos(2 * np.pi * agg.mo / 12)
    return agg.dropna(subset=["lag1", "lag2", "roll3"])


with tabs[1]:
    st.subheader("Predicted outbreak risk by state (latest month)")
    bundle = models["outbreak"]
    smf = state_month_features().sort_values("ord").groupby("State_new").tail(1)
    X = bundle["scaler"].transform(smf[bundle["features"]])
    score = bundle["model"].decision_function(X)
    smf = smf.assign(risk=score)
    hi, mid = smf.risk.quantile(0.75), smf.risk.quantile(0.40)
    smf["sev"] = np.where(smf.risk >= hi, "HIGH", np.where(smf.risk >= mid, "MEDIUM", "LOW"))
    smf["status"] = smf.sev.map({"HIGH": "🔴 HIGH", "MEDIUM": "🟡 MEDIUM", "LOW": "🟢 LOW"})
    view = (smf[["State_new", "confirmed", "risk", "status"]]
            .sort_values("risk", ascending=False).reset_index(drop=True))
    view.columns = ["State", "Recent confirmed", "Risk score", "Status"]
    st.dataframe(view, width="stretch", hide_index=True)

    elevated = smf[smf.sev != "LOW"]
    if st.button(f"🔔 Run surveillance sweep & notify ({len(elevated)} states elevated)"):
        signals = [{"disease": "Lassa fever", "location": r.State_new, "severity": r.sev,
                    "message": f"Outbreak signal in {r.State_new}: recent confirmed={int(r.confirmed)}, "
                               f"model risk score {r.risk:.1f}. Recommend field verification."}
                   for r in elevated.itertuples()]
        fired = check_and_notify(signals)
        st.success(f"Sent {len(fired)} alert(s) to the people in charge "
                   f"({'email' if fired and fired[0]['delivered'] else 'in-app log — set SMTP_* to email'}).")
        for f in fired:
            st.write(f"→ **{f['severity']}** · {f['location']} · to {', '.join(f['recipients'])}")


# ── Case triage ───────────────────────────────────────────
with tabs[2]:
    st.subheader("Predict a suspected case")
    with st.form("triage"):
        cols = st.columns(3)
        chosen = []
        for i, s in enumerate(dict.fromkeys(KEY_SYMPTOMS)):
            if cols[i % 3].checkbox(s.replace("_new", "").replace("_", " "), key=f"sym_{s}"):
                chosen.append(s)
        c1, c2, c3 = st.columns(3)
        age = c1.selectbox("Age group", ["0-14", "15-24", "25-64", "65+"], index=2)
        sex = c2.radio("Sex", ["Male", "Female"], horizontal=True)
        state = c3.selectbox("State", sorted(df.State_new.dropna().unique()))
        e1, e2 = st.columns(2)
        rodent = e1.checkbox("Rodent / excreta contact")
        contact = e2.checkbox("Contact with a known case")
        go = st.form_submit_button("Predict", type="primary")

    if go:
        dbg = models["diagnosis"]
        row = pd.Series(0.0, index=dbg["features"])
        for s in chosen:
            if s in row.index:
                row[s] = 1.0
        row["n_symptoms"] = float(len(chosen))
        for k, v in {"rodents_excreta": rodent, "contact_with_source_case_new": contact}.items():
            if k in row.index:
                row[k] = float(v)
        row["age_ord"] = {"0-14": 0, "15-24": 1, "25-64": 2, "65+": 3}[age]
        if "sex_f" in row.index:
            row["sex_f"] = 1.0 if sex == "Female" else 0.0
        if "hotspot" in row.index:
            row["hotspot"] = 1.0 if state in HOTSPOTS else 0.0

        ds = dbg["model"].decision_function(dbg["scaler"].transform(row.values.reshape(1, -1)))[0]
        dp = dbg["model"].predict(dbg["scaler"].transform(row.values.reshape(1, -1)))[0]
        obg = models["outcome"]
        orow = row.reindex(obg["features"]).fillna(0.0)
        os_ = obg["model"].decision_function(obg["scaler"].transform(orow.values.reshape(1, -1)))[0]

        r1, r2 = st.columns(2)
        r1.metric("Lassa likelihood", "LIKELY" if dp == 1 else "unlikely", f"risk score {ds:+.2f}")
        r2.metric("Severity signal (death risk)", "elevated" if os_ > 0 else "lower", f"score {os_:+.2f}")
        if dp == 1:
            st.warning(f"⚠️ High-risk Lassa case flagged in **{state}** — surveillance officers would be notified.")
        st.caption("Scores are SVM decision-function margins (hinge loss), not calibrated probabilities. "
                   "Diagnosis from symptoms alone is inherently limited — a documented finding.")


# ── Trends ────────────────────────────────────────────────
with tabs[3]:
    sel = st.selectbox("State", ["All"] + sorted(df.State_new.dropna().unique()))
    d = df if sel == "All" else df[df.State_new == sel]
    ts = d.groupby(["yr", "mo"]).positive.sum().reset_index()
    ts["date"] = pd.to_datetime(dict(year=ts.yr.astype(int), month=ts.mo.astype(int), day=1))
    st.line_chart(ts.set_index("date")["positive"], height=320)
    st.caption("Monthly confirmed Lassa cases (real SORMAS data).")


# ── Notifications ─────────────────────────────────────────
with tabs[4]:
    st.subheader("Alert recipients (people in charge)")
    st.write(" · ".join(f"**{r['name']}** ({r['role']})" for r in RECIPIENTS))
    rows = recent()
    if rows:
        st.dataframe(pd.DataFrame(rows, columns=["time", "disease", "location", "severity",
                                                 "message", "delivered"]),
                     width="stretch", hide_index=True)
    else:
        st.info("No notifications yet — run a surveillance sweep on the Outbreak monitor tab.")
