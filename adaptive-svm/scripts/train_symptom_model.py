#!/usr/bin/env python3
"""
train_symptom_model.py — train the MULTI-DISEASE symptom classifier used by Case Registration.

Data: a global disease–symptom dataset (41 diseases × 132 symptoms, 4,920 records) — well beyond
Nigeria — from the widely-used Kaggle "Disease Prediction" corpus
(kaushil268/disease-prediction-using-machine-learning), mirrored at
github.com/sohamvsonar/Disease-Prediction-and-Medical-Recommendation-System. Saved locally as
data/disease_symptoms.csv.

Model: an ADAPTIVE online classifier — SGDClassifier(loss='modified_huber') (SVM-family loss with
calibrated probabilities) trained with partial_fit(), consistent with the project's adaptive theme.
It predicts the most likely disease (and a differential) from the symptoms a clinician selects.
Saved to models/symptom_dx.pkl as {model, symptoms, classes}.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

HERE = Path(__file__).resolve().parents[1]


def main():
    d = pd.read_csv(HERE / "data" / "disease_symptoms.csv")
    d = d.loc[:, ~d.columns.str.contains("^Unnamed")]
    symptoms = [c for c in d.columns if c != "prognosis"]
    X = d[symptoms].values.astype(float)
    y = d["prognosis"].astype(str).str.strip().values
    classes = np.unique(y)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    clf = SGDClassifier(loss="modified_huber", alpha=1e-4, random_state=42)
    rng = np.random.RandomState(0)
    for epoch in range(20):                      # online (partial_fit) training over epochs
        idx = rng.permutation(len(Xtr))
        for i in range(0, len(idx), 64):
            b = idx[i:i + 64]
            clf.partial_fit(Xtr[b], ytr[b], classes=classes)

    acc = accuracy_score(yte, clf.predict(Xte))
    print(f"multi-disease symptom classifier — test accuracy: {acc:.3f} "
          f"| {len(classes)} diseases | {len(symptoms)} symptoms")
    joblib.dump({"model": clf, "symptoms": symptoms, "classes": list(classes)},
                HERE / "models" / "symptom_dx.pkl")
    print("saved models/symptom_dx.pkl")


if __name__ == "__main__":
    main()
