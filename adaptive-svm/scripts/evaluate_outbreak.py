#!/usr/bin/env python3
"""
evaluate_outbreak.py — reproduce the Adaptive SVM outbreak-detection evaluation and save
publication-quality figures for Chapter 4.5 (Performance Evaluation).

It re-runs the exact prequential (test-then-train) loop used to train the model on the REAL
SORMAS Lassa data, capturing the per-batch predictions so we can plot a confusion matrix, an
ROC curve, a prequential learning curve, and a metrics summary.
"""
from pathlib import Path
import json
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, roc_curve, roc_auc_score, accuracy_score,
                             precision_recall_fscore_support)
from sklearn.utils.class_weight import compute_class_weight

import sys
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
import prepare_and_train as P

warnings.filterwarnings("ignore")
OUT = HERE / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
TEAL, TEAL_DK, AMBER, INK, GREY = "#0F766E", "#0B5850", "#E0A100", "#1F2933", "#6B7280"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": "#9AA5B1",
                     "axes.linewidth": 0.8, "figure.dpi": 150})


def prequential_outbreak(warm=0.15, batch=100):
    """Replicates prepare_and_train.online_svm for the OUTBREAK task, returning the
    captured actuals / predictions / margins plus a running-accuracy trace."""
    df = P.load()
    agg = P._outbreak_frame(df)
    X = agg[P.OUTBREAK_FEATURES].reset_index(drop=True)
    y = agg["outbreak"].reset_index(drop=True).astype(int)
    n = len(X); w = max(int(n * warm), 60)
    cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=y.values)
    scaler = StandardScaler().fit(X.iloc[:w])
    clf = SGDClassifier(loss="hinge", average=True, class_weight={0: cw[0], 1: cw[1]},
                        alpha=1e-4, random_state=42)
    for _ in range(5):
        clf.partial_fit(scaler.transform(X.iloc[:w]), y.iloc[:w], classes=np.array([0, 1]))
    acts, preds, scores, curve = [], [], [], []
    i = w
    while i < n:
        j = min(i + batch, n)
        Xb = scaler.transform(X.iloc[i:j]); yb = y.iloc[i:j].values
        pb = clf.predict(Xb)
        preds.extend(pb); scores.extend(clf.decision_function(Xb)); acts.extend(yb)
        curve.append(accuracy_score(acts, preds))   # running prequential accuracy
        clf.partial_fit(Xb, yb)
        i = j
    return (np.array(acts), np.array(preds), np.array(scores), np.array(curve),
            dict(n_total=n, n_warm=w, n_eval=len(acts)))


def main():
    acts, _raw_preds, scores, curve, meta = prequential_outbreak()
    # Calibration-consistency (reviewer Point 1): ROC-AUC is computed on the CALIBRATED (Platt-scaled)
    # outbreak probabilities. Because AUC depends only on ranking and Platt scaling is monotonic, this
    # is identical to the raw-margin AUC. The threshold-dependent metrics (Accuracy, Precision, Recall,
    # F1) are reported at the classifier's NATIVE decision boundary (the SVM margin) — the model's true
    # operating point — while the calibrated probability is used to communicate risk on the dashboard
    # (HIGH/MEDIUM/LOW bands), not to re-threshold the classifier.
    lr = LogisticRegression(max_iter=1000).fit(scores.reshape(-1, 1), acts)
    A, B = float(lr.coef_[0][0]), float(lr.intercept_[0])
    probs_raw = 1.0 / (1.0 + np.exp(-(A * scores + B)))   # calibrated probabilities (AUC/ROC + display)
    preds = (scores > 0).astype(int)                      # SVM native operating point
    p, r, f, _ = precision_recall_fscore_support(acts, preds, average="binary", zero_division=0)
    acc = accuracy_score(acts, preds)
    auc = roc_auc_score(acts, probs_raw)                  # == raw-margin AUC (Platt is monotonic)
    metrics = {"accuracy": round(acc, 4), "precision": round(p, 4), "recall": round(r, 4),
               "f1": round(f, 4), "roc_auc": round(auc, 4), "auc_on": "calibrated_probabilities",
               "class_metrics_at": "svm_decision_boundary",
               "n_evaluated": int(meta["n_eval"]), "n_total": int(meta["n_total"]),
               "positive_rate": round(float(acts.mean()), 4)}
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("OUTBREAK metrics:", json.dumps(metrics))

    # ── 1. Confusion matrix ──────────────────────────────
    cm = confusion_matrix(acts, preds)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(cm, cmap="BuGn")
    labels = ["No outbreak", "Outbreak"]
    ax.set_xticks([0, 1], labels); ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted", fontweight="bold"); ax.set_ylabel("Actual", fontweight="bold")
    ax.set_title("Confusion Matrix — Outbreak Detection", fontweight="bold", color=TEAL_DK, pad=12)
    thr = cm.max() / 2
    names = [["True Negative", "False Positive"], ["False Negative", "True Positive"]]
    for a in range(2):
        for bb in range(2):
            ax.text(bb, a, f"{cm[a, bb]:,}\n{names[a][bb]}", ha="center", va="center",
                    color="white" if cm[a, bb] > thr else INK, fontsize=11, fontweight="bold")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(OUT / "confusion_matrix.png", bbox_inches="tight"); plt.close(fig)

    # ── 2. ROC curve ─────────────────────────────────────
    fpr, tpr, _ = roc_curve(acts, probs_raw)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot(fpr, tpr, color=TEAL, lw=2.4, label=f"Adaptive SVM (AUC = {auc:.2f})")
    ax.fill_between(fpr, tpr, alpha=0.12, color=TEAL)
    ax.plot([0, 1], [0, 1], "--", color=GREY, lw=1.2, label="Chance (AUC = 0.50)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate", fontweight="bold")
    ax.set_ylabel("True Positive Rate (Recall)", fontweight="bold")
    ax.set_title("ROC Curve — Outbreak Detection", fontweight="bold", color=TEAL_DK, pad=12)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout(); fig.savefig(OUT / "roc_auc.png", bbox_inches="tight"); plt.close(fig)

    # ── 3. Prequential learning curve ────────────────────
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    xs = np.arange(1, len(curve) + 1)
    ax.plot(xs, curve, color=TEAL, lw=2.2)
    ax.axhline(acc, ls="--", color=AMBER, lw=1.3, label=f"Final accuracy = {acc:.2f}")
    ax.set_xlabel("Streaming batch (chronological)", fontweight="bold")
    ax.set_ylabel("Running accuracy", fontweight="bold")
    ax.set_ylim(min(0.5, curve.min() - 0.05), 1.0)
    ax.set_title("Adaptive Learning Curve (prequential)", fontweight="bold", color=TEAL_DK, pad=12)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout(); fig.savefig(OUT / "learning_curve.png", bbox_inches="tight"); plt.close(fig)

    # ── 4. Metrics summary bar ───────────────────────────
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    names = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    vals = [acc, p, r, f, auc]
    bars = ax.bar(names, vals, color=[TEAL, TEAL, TEAL, TEAL, TEAL_DK], width=0.62)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", fontweight="bold")
    ax.set_ylim(0, 1.08); ax.set_ylabel("Score", fontweight="bold")
    ax.set_title("Performance Summary — Outbreak Detection", fontweight="bold", color=TEAL_DK, pad=12)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "metrics_summary.png", bbox_inches="tight"); plt.close(fig)

    print("saved 4 figures to", OUT)


if __name__ == "__main__":
    main()
