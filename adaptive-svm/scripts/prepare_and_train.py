#!/usr/bin/env python3
"""
prepare_and_train.py — Adaptive SVM (online / incremental learning) on the REAL
SORMAS Lassa fever dataset (Zenodo 7309567: 20,062 cases, 2018-2021, 774 LGAs).

Each task is a linear SVM trained by averaged SGD on the hinge loss —
SGDClassifier(loss='hinge', average=True) updated with partial_fit() as cases stream
in chronologically (a genuinely *adaptive* model). For each task we also fit a BATCH
linear SVM (LinearSVC) as a benchmark, so the thesis can compare adaptive vs. batch.

  1. DIAGNOSIS  — is a suspected case confirmed Lassa?  (symptoms + demographics + exposure)
  2. OUTBREAK   — is a state-month an outbreak?          (lagged counts + seasonality)
  3. OUTCOME    — will a case survive or die?            (clinical severity)

Online scoring is PREQUENTIAL (predict each batch, then train on it) — the honest way
to evaluate an online learner.
"""
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import pyreadstat
import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")
np.random.seed(42)

HERE = Path(__file__).resolve().parents[1]
DATA = HERE / "data"
MODELS = HERE / "models"
MODELS.mkdir(exist_ok=True)

# Broad symptom set (all Yes/No clinical flags present in the SORMAS export)
SYMPTOMS = ["fever_new", "headache_new", "abdominal_pain", "vomiting_new", "diarrhea_new",
            "sore_throat", "muscle_pain", "fatigue_weakness", "chest_pain", "cough_new",
            "difficulty_breathing", "jaundice_new", "bleeding_gums", "nose_bleeding",
            "blood_in_stool", "bleeding_bruising", "conjunctivities_red_eyes",
            "anorexia_loss_of_appetite", "joint_pain_arthritis", "backache_new",
            "bleeding_injection_site", "bleeding_from_eyes", "bleeding_from_vagina",
            "blood_urine", "bloody_black_stool", "chills_sweats", "confused_disoriented",
            "coughing_out_blood", "dark_urine", "digested_blood_coffee", "fresh_blood_vomit",
            "hiccups_new", "malaise_new", "nausea_new", "oedema_face_neck", "pain_behind_eyes",
            "side_pain", "acute_hearing_loss", "alteered_consciousness"]
EXPOSURE = ["rodents_excreta", "burial_of_case", "contact_with_source_case_new",
            "direct_contact_probable_case"]
HOTSPOTS = ["Edo", "Ondo", "Ebonyi", "Bauchi", "Taraba", "Plateau"]


def yesbin(s):
    return s.astype(str).str.strip().str.upper().eq("YES").astype(int)


def load():
    df, _ = pyreadstat.read_sav(DATA / "lassa_sormas.sav", apply_value_formats=True)
    df["yr"] = pd.to_numeric(df["date_report_years"], errors="coerce")
    df["mo"] = pd.to_numeric(df["date_report_months"], errors="coerce").fillna(6).clip(1, 12)
    df = df.dropna(subset=["yr"]).copy()
    df["ord"] = df["yr"] * 12 + df["mo"]
    df = df.sort_values("ord").reset_index(drop=True)
    df["positive"] = (df["case_classification_recode"].astype(str).str.lower() == "positive").astype(int)
    return df


def case_features(df):
    X = pd.DataFrame(index=df.index)
    present = [c for c in SYMPTOMS if c in df.columns]
    for c in present:
        X[c] = yesbin(df[c])
    X["n_symptoms"] = X[present].sum(axis=1)
    X["n_hemorrhagic"] = X[[c for c in present if "bleed" in c or "blood" in c]].sum(axis=1)
    for c in EXPOSURE:
        X[c] = yesbin(df[c]) if c in df.columns else 0
    X["age_ord"] = df["age_group2"].map({"0-14": 0, "15-24": 1, "25-64": 2, "65+": 3}).fillna(1)
    X["sex_f"] = (pd.to_numeric(df["sex_new2"], errors="coerce") == 2).astype(int)
    at = df["area_type_new"] if "area_type_new" in df.columns else pd.Series("", index=df.index)
    X["urban"] = at.astype(str).str.upper().str.startswith("U").astype(int)
    X["hotspot"] = df["State_new"].isin(HOTSPOTS).astype(int)
    # geopolitical zone one-hot (small cardinality, adds geographic signal)
    if "geopolitical_zone" in df.columns:
        z = pd.get_dummies(df["geopolitical_zone"].astype(str).str[:12], prefix="zone")
        X = pd.concat([X, z.astype(int)], axis=1)
    X["month_sin"] = np.sin(2 * np.pi * df["mo"] / 12)
    X["month_cos"] = np.cos(2 * np.pi * df["mo"] / 12)
    return X.astype(float)


def online_svm(X, y, warm=0.15, batch=100):
    """Prequential averaged-SGD online SVM. Returns (metrics, model, scaler)."""
    X = X.reset_index(drop=True); y = y.reset_index(drop=True).astype(int)
    n = len(X); w = max(int(n * warm), 60)
    cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=y.values)
    scaler = StandardScaler().fit(X.iloc[:w])
    clf = SGDClassifier(loss="hinge", average=True, class_weight={0: cw[0], 1: cw[1]},
                        alpha=1e-4, random_state=42)
    for _ in range(5):                       # a few warm-up passes to initialise
        clf.partial_fit(scaler.transform(X.iloc[:w]), y.iloc[:w], classes=np.array([0, 1]))
    preds, acts, scores = [], [], []
    i = w
    while i < n:
        j = min(i + batch, n)
        Xb = scaler.transform(X.iloc[i:j]); yb = y.iloc[i:j].values
        preds.extend(clf.predict(Xb)); scores.extend(clf.decision_function(Xb)); acts.extend(yb)
        clf.partial_fit(Xb, yb)
        i = j
    acts, preds, scores = np.array(acts), np.array(preds), np.array(scores)
    p, r, f, _ = precision_recall_fscore_support(acts, preds, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(acts, scores)
    except Exception:
        auc = float("nan")
    return (dict(acc=accuracy_score(acts, preds), prec=p, rec=r, f1=f, auc=auc,
                 n_eval=len(acts), pos_rate=float(acts.mean())), clf, scaler)


def batch_svm(X, y):
    """Batch linear SVM benchmark (temporal 70/30 split)."""
    X = X.reset_index(drop=True); y = y.reset_index(drop=True).astype(int)
    tr = int(len(X) * 0.70)
    sc = StandardScaler().fit(X.iloc[:tr])
    clf = LinearSVC(class_weight="balanced", C=0.1, max_iter=5000).fit(sc.transform(X.iloc[:tr]), y.iloc[:tr])
    s = clf.decision_function(sc.transform(X.iloc[tr:])); yte = y.iloc[tr:]
    pr = (s > 0).astype(int)
    p, r, f, _ = precision_recall_fscore_support(yte, pr, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(yte, s)
    except Exception:
        auc = float("nan")
    return dict(f1=f, auc=auc)


OUTBREAK_FEATURES = ["lag1", "lag2", "roll3", "trend", "hotspot", "sin", "cos"]


def _outbreak_frame(df):
    """State-month aggregation with lagged counts, seasonality and the outbreak label."""
    agg = (df.groupby(["State_new", "yr", "mo"])
           .agg(confirmed=("positive", "sum"), reports=("positive", "size")).reset_index())
    agg["ord"] = agg.yr * 12 + agg.mo
    agg = agg.sort_values(["State_new", "ord"]).reset_index(drop=True)
    g = agg.groupby("State_new")["confirmed"]
    agg["lag1"] = g.shift(1); agg["lag2"] = g.shift(2)
    agg["roll3"] = g.shift(1).rolling(3, min_periods=1).mean(); agg["trend"] = agg.lag1 - agg.lag2
    thr = agg.groupby("State_new")["confirmed"].transform(lambda s: max(s.quantile(0.70), 3))
    agg["outbreak"] = (agg.confirmed > thr).astype(int)
    agg["hotspot"] = agg.State_new.isin(HOTSPOTS).astype(int)
    agg["sin"] = np.sin(2 * np.pi * agg.mo / 12); agg["cos"] = np.cos(2 * np.pi * agg.mo / 12)
    return agg.dropna(subset=["lag1", "lag2", "roll3"]).sort_values("ord").reset_index(drop=True)


def build_bundles(df):
    """Train all three online SVMs, returning ({task: {model, scaler, features}}, {task: metrics}).

    Shared by main() (which persists the .pkl bundles) and by the dashboard, which calls this as a
    no-pickle fallback — so the app self-trains from the raw SORMAS data if a prebuilt model can't
    be loaded (e.g. a scikit-learn version change on a fresh deployment host)."""
    Xc = case_features(df)
    bundles, metrics = {}, {}

    mask = df["case_classification_recode"].notna()
    m1, c1, s1 = online_svm(Xc[mask], df["positive"][mask])
    bundles["diagnosis"] = {"model": c1, "scaler": s1, "features": list(Xc.columns)}
    metrics["diagnosis"] = m1

    oc = df["outcome_case"].astype(str); om = oc.isin(["Recovered", "Deceased"])
    m2, c2, s2 = online_svm(Xc[om], (oc[om] == "Deceased").astype(int))
    bundles["outcome"] = {"model": c2, "scaler": s2, "features": list(Xc.columns)}
    metrics["outcome"] = m2

    agg = _outbreak_frame(df)
    m3, c3, s3 = online_svm(agg[OUTBREAK_FEATURES], agg["outbreak"])
    bundles["outbreak"] = {"model": c3, "scaler": s3, "features": OUTBREAK_FEATURES}
    metrics["outbreak"] = m3
    return bundles, metrics


def main():
    df = load()
    Xc = case_features(df)
    bundles, metrics = build_bundles(df)
    for k in ["diagnosis", "outbreak", "outcome"]:
        joblib.dump(bundles[k], MODELS / f"svm_{k}.pkl")

    # Batch (train-once) benchmarks for the adaptive-vs-batch comparison in the thesis.
    mask = df["case_classification_recode"].notna()
    b1 = batch_svm(Xc[mask], df["positive"][mask])
    oc = df["outcome_case"].astype(str); om = oc.isin(["Recovered", "Deceased"])
    b2 = batch_svm(Xc[om], (oc[om] == "Deceased").astype(int))
    ob = _outbreak_frame(df)
    b3 = batch_svm(ob[OUTBREAK_FEATURES], ob["outbreak"])

    print("\n=== Adaptive SVM on REAL SORMAS data — online (adaptive) vs batch benchmark ===")
    print(f"{'task':30s} | online AUC / F1 | batch AUC / F1 | (n, base_rate)")
    for name, m, b in [("1. DIAGNOSIS (confirmed Lassa)", metrics["diagnosis"], b1),
                       ("2. OUTBREAK (state-month)", metrics["outbreak"], b3),
                       ("3. OUTCOME (death)", metrics["outcome"], b2)]:
        print(f"{name:30s} |   {m['auc']:.2f} / {m['f1']:.2f}   |   {b['auc']:.2f} / {b['f1']:.2f}   "
              f"| (n={m['n_eval']}, +{m['pos_rate']:.0%})")


if __name__ == "__main__":
    main()
