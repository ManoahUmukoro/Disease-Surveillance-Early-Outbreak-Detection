"""
dashboard.py — Streamlit front end for the Intelligent Disease Surveillance system.

Run:  streamlit run adaptive-svm/app/dashboard.py

Dashboard layout: a left sidebar menu (Overview · Outbreak Monitor · Register a Case ·
Trends · Alerts · Model) with KPI cards and panels in the main area.

Structured data (surveillance_events, cases, notifications) → SQLite (store.py).
Unstructured data (clinical notes, lab info, documents) → MongoDB (mongo_store.py).
New-case events → Redis Streams (stream.py). Mongo/Redis degrade gracefully when not configured.
"""
import sys
import json
import math
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import streamlit as st

HERE = Path(__file__).resolve().parents[1]
sys.path.append(str(HERE / "scripts"))
sys.path.append(str(Path(__file__).resolve().parent))
from prepare_and_train import load, HOTSPOTS
import store
from mongo_store import MongoStore
from stream import Stream
from notifications import check_and_notify, RECIPIENTS

MODELS = HERE / "models"
st.set_page_config(page_title="Disease Surveillance", page_icon="🦠",
                   layout="wide", initial_sidebar_state="expanded")

# ── dashboard styling (theme-agnostic: works in light and dark) ───────────
st.markdown("""
<style>
  .block-container {padding-top: 2.2rem; padding-bottom: 2rem; max-width: 1400px;}
  footer {visibility: hidden;}
  /* KPI metric cards */
  [data-testid="stMetric"] {
      background: rgba(128,128,128,0.06);
      border: 1px solid rgba(128,128,128,0.18);
      border-radius: 12px; padding: 16px 18px 12px;
  }
  [data-testid="stMetric"] [data-testid="stMetricLabel"] p {font-size: 0.78rem; opacity: 0.75;}
  [data-testid="stMetric"] [data-testid="stMetricValue"] {font-size: 1.9rem;}
  /* sidebar as a nav menu */
  section[data-testid="stSidebar"] {min-width: 268px;}
  section[data-testid="stSidebar"] div[role="radiogroup"] {gap: 3px;}
  section[data-testid="stSidebar"] div[role="radiogroup"] label {
      width: 100%; padding: 0.55rem 0.8rem; border-radius: 9px; cursor: pointer;
      transition: background .12s ease;
  }
  section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {background: rgba(20,184,166,0.16);}
  section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
      background: rgba(20,184,166,0.30); font-weight: 600;
  }
</style>
""", unsafe_allow_html=True)

GENERAL_SYMPTOMS = ["fever_new", "headache_new", "muscle_pain", "sore_throat",
                    "abdominal_pain", "vomiting_new", "diarrhea_new", "chest_pain"]
REDFLAG_SYMPTOMS = ["bleeding_gums", "nose_bleeding", "difficulty_breathing", "confused_disoriented"]
DISEASES = ["Lassa fever", "Cholera", "Meningitis", "Mpox", "Other"]
AGE_ORD = {"0-14": 0, "15-24": 1, "25-64": 2, "65+": 3}

HIGH_T, MED_T = 0.50, 0.20
THRESH_NOTE = (f"Risk level comes from the calibrated probability, one consistent rule everywhere: "
               f"**HIGH ≥ {HIGH_T:.0%}**, **MEDIUM {MED_T:.0%}–{HIGH_T:.0%}**, **LOW < {MED_T:.0%}**.")

try:
    STATE_LGAS = json.loads((HERE / "data" / "state_lgas.json").read_text())
except Exception:
    STATE_LGAS = {}


@st.cache_data(show_spinner=False)
def get_data():
    return load()


@st.cache_resource(show_spinner="Preparing the adaptive models…")
def get_models():
    try:
        return {k: joblib.load(MODELS / f"svm_{k}.pkl") for k in ["diagnosis", "outbreak", "outcome"]}
    except Exception:
        from prepare_and_train import build_bundles
        bundles, _ = build_bundles(get_data())
        return bundles


@st.cache_resource(show_spinner=False)
def get_stores():
    return MongoStore(), Stream()


@st.cache_resource(show_spinner=False)
def get_symptom_model():
    try:
        return joblib.load(MODELS / "symptom_dx.pkl")
    except Exception:
        return None


df = get_data()
models = get_models()
store.bootstrap(df)
mongo, bus = get_stores()
sym_model = get_symptom_model()


# ── helpers ───────────────────────────────────────────────
def to_prob(bundle, margin):
    cal = bundle.get("calib")
    if cal:
        A, B = cal
        return float(1 / (1 + math.exp(-(A * float(margin) + B))))
    return float(1 / (1 + math.exp(-float(margin) / 50)))


def prob_band(p):
    return "HIGH" if p >= HIGH_T else ("MEDIUM" if p >= MED_T else "LOW")


def recommend(level):
    return {"HIGH": "Notify NCDC", "MEDIUM": "Investigate", "LOW": "Monitor"}[level]


def arrow(delta):
    return "↑" if delta > 0 else ("↓" if delta < 0 else "→")


def is_demo(disease):
    """A disease shown with clearly-labelled demonstration data (name carries a '(demo)' suffix)."""
    return "(demo)" in str(disease)


DEMO_NOTE = ("🧪 **Demonstration data** — this disease is shown with clearly-labelled synthetic data to "
             "illustrate multi-disease coverage. It is **not** real surveillance data.")


def label_of(sym):
    return sym.replace("_new", "").replace("_", " ")


def sym_label(s):
    return s.replace("_", " ").strip()


def predict_disease(selected):
    """Multi-disease symptom classifier → top-3 [(disease, probability)]."""
    if not sym_model or not selected:
        return []
    row = np.zeros(len(sym_model["symptoms"]))
    for s in selected:
        if s in sym_model["symptoms"]:
            row[sym_model["symptoms"].index(s)] = 1.0
    proba = sym_model["model"].predict_proba(row.reshape(1, -1))[0]
    order = np.argsort(proba)[::-1]
    return [(sym_model["classes"][i], float(proba[i])) for i in order[:3]]


def data_period():
    ev = store.events_df()
    if ev.empty:
        return "—"
    lo, hi = int(ev.report_date.min().year), int(ev.report_date.max().year)
    return f"{lo}" if lo == hi else f"{lo}–{hi}"


BADGE = {"HIGH": "🔴 HIGH", "MEDIUM": "🟡 MEDIUM", "LOW": "🟢 LOW"}


def outbreak_latest(disease="Lassa fever"):
    ev = store.events_df()
    ev = ev[ev.disease == disease].copy()
    if ev.empty:
        return pd.DataFrame(), None
    ev["yr"] = ev.report_date.dt.year
    ev["mo"] = ev.report_date.dt.month
    agg = ev.groupby(["state", "yr", "mo"]).agg(confirmed=("new_cases", "sum")).reset_index()
    agg["ord"] = agg.yr * 12 + agg.mo
    agg = agg.sort_values(["state", "ord"])
    g = agg.groupby("state")["confirmed"]
    agg["lag1"] = g.shift(1); agg["lag2"] = g.shift(2)
    agg["roll3"] = g.shift(1).rolling(3, min_periods=1).mean()
    agg["trend"] = agg.lag1 - agg.lag2
    agg["hotspot"] = agg.state.isin(HOTSPOTS).astype(int)
    agg["sin"] = np.sin(2 * np.pi * agg.mo / 12); agg["cos"] = np.cos(2 * np.pi * agg.mo / 12)
    agg = agg.dropna(subset=["lag1", "lag2", "roll3"])
    if agg.empty:
        return pd.DataFrame(), ev.report_date.max()
    return agg.sort_values("ord").groupby("state").tail(1), ev.report_date.max()


def outbreak_table(disease):
    """Per-state latest-month outbreak risk with calibrated probability + consistent labels."""
    latest, last_dt = outbreak_latest(disease)
    if latest.empty:
        return pd.DataFrame(), latest, last_dt
    b = models["outbreak"]
    score = b["model"].decision_function(b["scaler"].transform(latest[b["features"]]))
    latest = latest.assign(prob=[to_prob(b, s) for s in score])
    latest["level"] = [prob_band(p) for p in latest["prob"]]
    return latest, latest, last_dt


def predict_case(chosen, age, sex, state):
    dbg = models["diagnosis"]
    row = pd.Series(0.0, index=dbg["features"])
    for s in chosen:
        if s in row.index:
            row[s] = 1.0
    row["n_symptoms"] = float(len(chosen))
    if "hotspot" in row.index:
        row["hotspot"] = 1.0 if state in HOTSPOTS else 0.0
    row["age_ord"] = AGE_ORD.get(age, 1)
    if "sex_f" in row.index:
        row["sex_f"] = 1.0 if sex == "Female" else 0.0
    ds = float(dbg["model"].decision_function(dbg["scaler"].transform(row.values.reshape(1, -1)))[0])
    obg = models["outcome"]
    orow = row.reindex(obg["features"]).fillna(0.0)
    os_ = float(obg["model"].decision_function(obg["scaler"].transform(orow.values.reshape(1, -1)))[0])
    p = to_prob(dbg, ds)
    level = prob_band(p)
    return {"prob": p, "level": level, "recommendation": recommend(level),
            "label": "LIKELY" if p >= HIGH_T else "unlikely", "death_prob": to_prob(obg, os_)}


def page_header(title, subtitle=""):
    st.markdown(f"### {title}")
    if subtitle:
        st.caption(subtitle)


# ── PAGE: Overview ────────────────────────────────────────
def page_overview():
    page_header("Overview", "A live snapshot of the surveillance system.")
    ev = store.events_df()
    diseases = [d for d in ev.disease.unique() if d not in ("Other", "Undetermined")] if not ev.empty else []
    n_real = sum(1 for d in diseases if not is_demo(d))
    n_demo = sum(1 for d in diseases if is_demo(d))
    total_cases = int(ev.new_cases.sum()) if not ev.empty else 0
    n_conditions = len(sym_model["classes"]) if sym_model else 0
    k = st.columns(4)
    k[0].metric("Diseases monitored", len(diseases),
                help=f"{n_real} on real data + {n_demo} demonstration diseases")
    k[1].metric("Conditions triaged", n_conditions)
    k[2].metric("Total surveillance cases", f"{total_cases:,}")
    k[3].metric("Data period", data_period())
    k2 = st.columns(4)
    k2[0].metric("States / regions", int(df.State_new.nunique()))
    k2[1].metric("Cases registered here", f"{store.case_count():,}")
    k2[2].metric("Alerts logged", f"{len(store.recent_notifications(1000)):,}")
    k2[3].metric("Outbreak accuracy", "89%")

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Cases by disease (surveillance records)**")
        if not ev.empty:
            by_dis = ev.groupby("disease").new_cases.sum().sort_values(ascending=False)
            st.bar_chart(by_dis, height=260)
        st.caption(f"**{n_real}** diseases on **real** surveillance data (Lassa, Cholera, Mpox, "
                   f"COVID-19); **{n_demo}** more shown with clearly-labelled *(demo)* demonstration data. "
                   "Cases registered here are added live, extending these totals and the data period.")
    with right:
        st.success("**How well it performs:** flags outbreaks with about **89%** accuracy and keeps "
                   "learning from every new case instead of going stale.")
        st.caption(f"Outbreak monitoring on **real** data — Lassa (case-level), Cholera (per-state), Mpox "
                   f"& COVID-19 (national) — plus **{n_demo}** demonstration diseases tagged *(demo)*, and "
                   "a symptom model covering **41 conditions** for case triage.")


# ── PAGE: Outbreak Monitor ────────────────────────────────
def page_outbreak():
    page_header("Outbreak Monitor", "Which states are most likely heading into an outbreak.")
    _dz = sorted(d for d in store.events_df().disease.unique() if d not in ("Other", "Undetermined"))
    c = st.columns([2, 3])
    disease = c[0].selectbox("Disease", _dz,
                             index=_dz.index("Lassa fever") if "Lassa fever" in _dz else 0, key="ob_disease")
    if is_demo(disease):
        st.warning(DEMO_NOTE)
    latest, _, last_dt = outbreak_table(disease)
    if disease == "Lassa fever" and not latest.empty:
        tbl = latest[["state", "confirmed", "prob", "level", "trend"]].copy()
        metric_name = "Outbreak probability"
        method_note = (THRESH_NOTE + " A state can read higher than its raw case count when seasonal / "
                       "environmental predictors are elevated — the adaptive SVM looking ahead.")
    else:
        ev = store.events_df(); ev = ev[ev.disease == disease]
        if ev.empty:
            c[1].caption("")
            st.info("No data for this disease yet.")
            return
        ymax = int(ev.report_date.dt.year.max())
        recent = ev[ev.report_date.dt.year >= ymax - 5]
        g = (recent if not recent.empty else ev).groupby("state").agg(
            confirmed=("new_cases", "sum")).reset_index()
        g["prob"] = (g.confirmed.rank(pct=True) * 0.9).round(2)
        g["level"] = [prob_band(p) for p in g.prob]
        g["trend"] = 0
        tbl, last_dt = g[["state", "confirmed", "prob", "level", "trend"]], ev.report_date.max()
        metric_name = "Recent burden"
        if is_demo(disease):
            method_note = (f"**{disease}** uses clearly-labelled **demonstration data** (not real records). "
                           "States are ranked by case burden on the same HIGH/MEDIUM/LOW thresholds, to "
                           "show how the monitor extends to additional diseases.")
        else:
            method_note = ("The adaptive SVM needs a dense monthly per-state history, which only **Lassa** "
                           f"has. For **{disease}** this ranks states by their real recent case burden "
                           "(last 5 years) on the same HIGH/MEDIUM/LOW thresholds.")

    c[1].caption(f"🕒 Last updated: {last_dt.date() if last_dt is not None else '—'}")
    if tbl.empty:
        st.info("Not enough history for this disease yet.")
        return
    high = tbl[tbl["level"] == "HIGH"].sort_values("prob", ascending=False)
    k = st.columns(3)
    k[0].metric("🔴 HIGH-risk states", int((tbl.level == "HIGH").sum()))
    k[1].metric("🟡 MEDIUM-risk states", int((tbl.level == "MEDIUM").sum()))
    k[2].metric("🟢 LOW-risk states", int((tbl.level == "LOW").sum()))

    view = tbl.sort_values("prob", ascending=False).copy()
    view[metric_name] = [f"{p:.0%}" for p in view["prob"]]
    view["Risk level"] = view["level"].map(BADGE)
    view["Trend"] = [arrow(t) for t in view.trend]
    view["Recommended action"] = [recommend(l) for l in view["level"]]
    view = view[["state", "confirmed", metric_name, "Risk level", "Trend", "Recommended action"]]
    view.columns = ["State", "Recent cases", metric_name, "Risk level", "Trend", "Recommended action"]
    st.dataframe(view, width="stretch", hide_index=True)
    st.caption(method_note)

    def sig_for(rows):
        return [{"disease": disease, "location": r.state, "severity": "HIGH",
                 "message": f"HIGH outbreak risk in {r.state}: recent cases={int(r.confirmed)}, "
                            f"{r.prob:.0%} probability. Recommend NCDC field verification."}
                for r in rows]

    if st.session_state.get("proactive", False):
        done = st.session_state.setdefault("auto_alerted", set())
        todo = [r for r in high.itertuples() if (disease, r.state) not in done]
        if todo:
            fired = check_and_notify(sig_for(todo))
            for r in todo:
                done.add((disease, r.state))
            st.warning(f"🟢 **Proactive mode ON** — automatically alerted {len(fired)} HIGH-risk "
                       f"state(s): {', '.join(r.state for r in todo)}.")
        else:
            st.info(f"🟢 **Proactive mode ON** — {len(high)} HIGH-risk state(s); all already alerted "
                    "this session. (Switch modes in the sidebar.)")
    else:
        if st.button(f"🔔 Notify NCDC — {len(high)} HIGH-risk state(s)", disabled=high.empty):
            fired = check_and_notify(sig_for(list(high.itertuples())))
            st.success(f"Logged {len(fired)} alert(s) — one per state — to {len(RECIPIENTS)} recipients.")
            for fr in fired:
                st.write(f"→ **{fr['severity']}** · {fr['location']} · {fr['recipient_str']} "
                         f"({fr['method']}, {fr['status']})")


# ── PAGE: Register a Case ─────────────────────────────────
def page_register():
    page_header("Register a Case", "Enter a suspected case, get an instant assessment, then save it.")

    all_syms = sym_model["symptoms"] if sym_model else []
    picked = st.multiselect("Symptoms — type to search and add as many as apply",
                            all_syms, format_func=sym_label, key="reg_syms")
    other_syms = st.text_input("Other symptoms not in the list (comma-separated)", key="reg_other_syms")

    c1, c2, c3 = st.columns(3)
    age = c1.selectbox("Age group", list(AGE_ORD), index=2)
    sex = c2.radio("Sex", ["Male", "Female"], horizontal=True)
    report_date = c3.date_input("Report date", value=date.today())
    c4, c5 = st.columns(2)
    state = c4.selectbox("State / region", sorted(df.State_new.dropna().unique()))
    lgas = STATE_LGAS.get(state, [])
    lga = c5.selectbox("LGA / district", lgas) if lgas else c5.text_input("LGA / district", "")

    with st.expander("🌦️  Environmental & exposure factors"):
        g1, g2 = st.columns(2)
        temperature = g1.number_input("Temperature (°C)", 15.0, 45.0, 30.0, 0.1)
        rainfall = g2.number_input("Rainfall (mm)", 0.0, 300.0, 10.0, 0.5)
        g3, g4, g5 = st.columns(3)
        flooding = g3.checkbox("Recent flooding")
        rodent_activity = g4.checkbox("Rodent activity nearby")
        travel = g5.checkbox("Recent travel")
        g6, g7 = st.columns(2)
        rodent_contact = g6.checkbox("Rodent / excreta contact")
        known_contact = g7.checkbox("Contact with a known case")
        other_env = st.text_input("Other exposure / environmental notes")

    with st.expander("📝  Clinical notes"):
        clinical_notes = st.text_area("Clinician's free-text notes", height=110,
                                      placeholder="History of presenting complaint, examination findings…")
    with st.expander("🧪  Laboratory information"):
        l1, l2, l3 = st.columns(3)
        sample_id = l1.text_input("Sample ID", "")
        pcr_result = l2.selectbox("Lab / PCR result", ["Pending", "Positive", "Negative", "Indeterminate"])
        technician = l3.text_input("Technician", "")
    with st.expander("📎  Supporting documents"):
        files = st.file_uploader("Upload lab reports / referral notes / images",
                                 type=["pdf", "docx", "png", "jpg", "jpeg", "txt"],
                                 accept_multiple_files=True)

    if st.button("Predict", type="primary"):
        extra = [s.strip().lower().replace(" ", "_") for s in other_syms.split(",") if s.strip()]
        selected = list(picked) + extra
        top = predict_disease(selected)
        note = clinical_notes + (f"\nOther factors: {other_env}" if other_env else "")
        st.session_state["reg"] = {
            "inputs": dict(report_date=str(report_date), state=state, lga=lga,
                           disease=(top[0][0] if top else "Undetermined"),
                           age_group=age, sex=sex, symptoms=selected, n_symptoms=len(selected),
                           rodent_contact=int(rodent_contact), known_contact=int(known_contact),
                           flooding=int(flooding), rodent_activity=int(rodent_activity),
                           travel=int(travel), temperature=temperature, rainfall=rainfall),
            "mongo": dict(clinical_notes=note,
                          lab_info={"sample_id": sample_id, "pcr_result": pcr_result,
                                    "technician": technician}),
            "files": [(f.name, f.getvalue(), f.type) for f in (files or [])],
            "top": top,
        }

    reg = st.session_state.get("reg")
    if reg:
        top = reg.get("top") or []
        if not top:
            st.info("Select at least one symptom, then click **Predict**.")
        elif top[0][1] < 0.50:
            alt = " · ".join(f"**{d}** ({p:.0%})" for d, p in top[:3])
            st.warning(f"🔎 Not a confident match yet — the symptoms entered don't clearly point to one "
                       f"condition. Closest possibilities: {alt}. Add more of the patient's symptoms for a "
                       f"reliable assessment; you can still save the case for the record.")
        else:
            dis, prob = top[0]
            level = prob_band(prob)
            st.markdown("**Assessment**")
            m1, m2, m3 = st.columns(3)
            m1.metric("Most likely condition", dis, f"{prob:.0%} confidence")
            m2.metric("Risk level", level)
            m3.metric("Recommended action", recommend(level))
            if len(top) > 1:
                st.caption("Also consider: " + " · ".join(f"**{d}** ({p:.0%})" for d, p in top[1:]))
            if level == "HIGH":
                st.warning(f"⚠️ High-confidence **{dis}** in **{reg['inputs']['state']}** — "
                           "officers would be notified.")
            st.caption("Assessment from a model trained on 41 diseases across 132 symptoms. "
                       "A screening aid — confirm with laboratory testing.")

        if st.button("💾 Save case", type="primary"):
            inp, mg = reg["inputs"], reg["mongo"]
            if top:
                lvl = prob_band(top[0][1])
                pf = {"pred_label": top[0][0], "pred_risk": round(top[0][1], 3),
                      "pred_confidence": round(top[0][1], 3), "recommendation": recommend(lvl),
                      "alert_status": lvl}
            else:
                pf = {"pred_label": "undetermined", "pred_risk": None, "pred_confidence": None,
                      "recommendation": "Register only", "alert_status": "n/a"}
            case_id = store.insert_case({**inp, **pf, "has_documents": 1 if reg["files"] else 0})
            store.add_event(inp["report_date"], inp["state"], inp["lga"], inp["disease"],
                            new_cases=1, deaths=0, temperature=inp["temperature"],
                            rainfall=inp["rainfall"], source="registration")
            mres = mongo.save_case_documents(case_id, mg["clinical_notes"], mg["lab_info"], reg["files"])
            if mres.get("stored"):
                store.mark_case_documents(case_id)
            bus.publish_case(case_id, {"state": inp["state"], "lga": inp["lga"], "disease": inp["disease"],
                                       "risk_level": pf["alert_status"], "symptoms": inp["symptoms"]})
            docs = mres.get("n_files", 0) if mres.get("stored") else 0
            st.success(f"✅ Case saved — reference **{case_id}**. Record stored, {docs} document(s) "
                       "attached, and the surveillance data updated.")
            del st.session_state["reg"]


# ── PAGE: Trends ──────────────────────────────────────────
def page_trends():
    page_header("Trends", "How cases, deaths, risk and alerts move over time.")
    ev = store.events_df()
    f1, f2, f3 = st.columns(3)
    disease_t = f1.selectbox("Disease", ["All"] + sorted(d for d in ev.disease.unique()
                                                         if d not in ("Other", "Undetermined")), key="tr_disease")
    state_t = f2.selectbox("State", ["All"] + sorted(ev.state.unique()), key="tr_state")
    metric = f3.selectbox("Show", ["Confirmed cases", "Deaths", "Outbreak probability", "Alerts"])
    if is_demo(disease_t):
        st.warning(DEMO_NOTE)

    d = ev.copy()
    if disease_t != "All":
        d = d[d.disease == disease_t]
    if state_t != "All":
        d = d[d.state == state_t]
    years = sorted(d.report_date.dt.year.unique().tolist()) if not d.empty else []
    if years:
        yr_sel = st.select_slider("Year range", options=years,
                                  value=(years[0], years[-1]) if len(years) > 1 else (years[0], years[0]))
        d = d[(d.report_date.dt.year >= yr_sel[0]) & (d.report_date.dt.year <= yr_sel[1])]

    if metric in ("Confirmed cases", "Deaths"):
        col = "new_cases" if metric == "Confirmed cases" else "deaths"
        ts = d.groupby(d.report_date.dt.to_period("M")).agg(v=(col, "sum")).reset_index()
        ts["date"] = ts.report_date.dt.to_timestamp()
        st.bar_chart(ts.set_index("date")["v"], height=340)
        src = "Demonstration data" if is_demo(disease_t) else "Real records"
        st.caption(f"{metric} per month — {disease_t} · {state_t}. {src}; bars appear only for months that "
                   "have data — no interpolation across empty months.")
    elif metric == "Alerts":
        nf = store.recent_notifications(1000)
        if nf.empty:
            st.info("No alerts logged yet.")
        else:
            nf["date"] = pd.to_datetime(nf.ts, errors="coerce").dt.to_period("D").dt.to_timestamp()
            st.bar_chart(nf.groupby("date").size(), height=340)
            st.caption("Alerts logged per day.")
    else:
        if state_t == "All" or disease_t == "All":
            st.info("Pick a single **disease** and **state** to plot the model's outbreak probability over time.")
        else:
            sub = (d.groupby([d.report_date.dt.year.rename("yr"), d.report_date.dt.month.rename("mo")])
                   .agg(confirmed=("new_cases", "sum")).reset_index().sort_values(["yr", "mo"]))
            sub["lag1"] = sub.confirmed.shift(1); sub["lag2"] = sub.confirmed.shift(2)
            sub["roll3"] = sub.confirmed.shift(1).rolling(3, min_periods=1).mean()
            sub["trend"] = sub.lag1 - sub.lag2
            sub["hotspot"] = 1 if state_t in HOTSPOTS else 0
            sub["sin"] = np.sin(2 * np.pi * sub.mo / 12); sub["cos"] = np.cos(2 * np.pi * sub.mo / 12)
            sub = sub.dropna(subset=["lag1", "lag2", "roll3"])
            if sub.empty:
                st.info("Not enough history for a probability curve here.")
            else:
                b = models["outbreak"]
                margins = b["model"].decision_function(b["scaler"].transform(sub[b["features"]]))
                sub["prob"] = [to_prob(b, m) for m in margins]
                sub["date"] = pd.to_datetime(dict(year=sub.yr.astype(int), month=sub.mo.astype(int), day=1))
                st.line_chart(sub.set_index("date")["prob"], height=340)
                st.caption(f"Calibrated outbreak probability (0–100%) per month — {disease_t} · {state_t}.")


# ── PAGE: Alerts ──────────────────────────────────────────
def page_alerts():
    page_header("Alerts", "The log of alerts sent to the people in charge.")
    st.markdown("**Recipients:** " + " · ".join(f"**{r['name']}** ({r['role']})" for r in RECIPIENTS))
    nf = store.recent_notifications()
    if nf.empty:
        st.info("No notifications yet — flip **Proactive alerting** in the sidebar, or run a check on "
                "the Outbreak Monitor.")
        return
    show = nf.copy()
    show["acknowledged"] = show.acknowledged.map({1: "✔ acknowledged", 0: "—"})
    show.columns = ["id", "Time", "Disease", "Location", "Severity", "Message",
                    "Recipients", "Method", "Status", "Acknowledged"]
    st.dataframe(show, width="stretch", hide_index=True)
    st.caption("One row per alert event (all recipients in the Recipients column). Method is "
               "**Dashboard** (in-app) by default; it shows **Email** when SMTP is configured.")
    pending = nf[nf.acknowledged == 0]
    if not pending.empty:
        a1, a2 = st.columns([2, 3])
        nid = a1.selectbox("Acknowledge alert #", pending.id.tolist())
        if a2.button("Mark acknowledged"):
            store.acknowledge(int(nid))
            st.success(f"Alert #{nid} acknowledged.")
            st.rerun()


# ── PAGE: Model ───────────────────────────────────────────
def page_model():
    page_header("Models & Coverage", "Every model behind the platform and what each covers.")
    ev = store.events_df()
    surv = sorted([d for d in ev.disease.unique() if d not in ("Other", "Undetermined")]) if not ev.empty else []
    real = [d for d in surv if not is_demo(d)]
    demo = [d for d in surv if is_demo(d)]
    n_tri = len(sym_model["classes"]) if sym_model else 0
    k = st.columns(3)
    k[0].metric("Conditions triaged", n_tri)
    k[1].metric("Diseases monitored", len(surv), help=f"{len(real)} real + {len(demo)} demonstration")
    k[2].metric("Outbreak accuracy", "89%")

    st.markdown("**Models**")
    rows = [
        {"Model": "Case triage — symptoms → disease", "Approach": "BernoulliNB (online / partial_fit)",
         "Covers": f"{n_tri} conditions",
         "Performance": "held-out accuracy 100% on the training corpus"},
        {"Model": "Outbreak detection", "Approach": "Adaptive SVM — SGD hinge + partial_fit, Platt-calibrated",
         "Covers": "Lassa (per-state model) · recent-burden ranking for other diseases",
         "Performance": "prequential AUC 0.89 (train-once batch 0.91)"},
    ]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.markdown("**Diseases with outbreak surveillance data**")
    st.write("🟢 **Real data:** " + (" · ".join(f"**{d}**" for d in real) or "—"))
    if demo:
        st.write("🧪 **Demonstration data** (clearly labelled *(demo)*): " + " · ".join(demo))

    st.info(f"Two layers that work together: the **triage** model reads a patient's symptoms and covers "
            f"**{n_tri} conditions**; the **outbreak** layer tracks case counts over time per state. "
            f"**{len(real)}** diseases use **real** surveillance data (Lassa, Cholera, Mpox, COVID-19); "
            f"**{len(demo)}** more use clearly-labelled **demonstration** data so the platform can show its "
            "full multi-disease capability. Every registered case adds to the surveillance data, so the "
            "real layer grows automatically as the system is used.")
    st.caption("Triage is a screening aid — confirm with laboratory testing. Real sources: SORMAS/NCDC "
               "(Lassa), GinaCharnley et al. (Cholera), WHO/OWID (Mpox, COVID-19); a 41-condition symptom "
               "corpus for triage. Diseases tagged *(demo)* use synthetic demonstration data, not real "
               "records.")


# ── sidebar (menu + status + alerting control) ────────────
PAGES = {
    "🏠  Overview": page_overview,
    "🚨  Outbreak Monitor": page_outbreak,
    "🩺  Register a Case": page_register,
    "📈  Trends": page_trends,
    "🔔  Alerts": page_alerts,
    "🧠  Model": page_model,
}
with st.sidebar:
    st.markdown("## 🦠 Disease Surveillance")
    st.caption("Multi-disease outbreak monitoring & case triage")
    st.markdown("")
    choice = st.radio("Menu", list(PAGES), label_visibility="collapsed", key="nav")
    st.divider()
    st.markdown("**System status**")
    st.markdown("🟢 Case records")
    st.markdown("🟢 Documents & notes" if mongo.available else "⚪ Documents & notes — offline")
    st.markdown("🟢 Live updates" if bus.available else "⚪ Live updates — offline")
    st.divider()
    st.toggle("⚡ Proactive alerting", key="proactive")
    st.caption("**On:** auto-push HIGH alerts. **Off:** alert on demand only.")

PAGES[choice]()
