# Adaptive SVM — Intelligent Disease Surveillance System

An intelligent disease-surveillance dashboard for Nigeria, built around an **adaptive
(online) Support Vector Machine** that keeps learning from every new case instead of being
trained once and left to go stale. It reads real Lassa fever surveillance records, predicts
which states are heading into an outbreak, triages suspected cases, and alerts the officials
in charge.

This folder is the reference prototype for the thesis
**“An Adaptive SVM Model for Intelligent Disease Surveillance System.”** It lives alongside
the earlier LSTM/React forecasting system in the same repository ("keep both").

> **Live demo:** https://disease-surveillance-early-outbreak-detection-9eggg9qvphqsz7z6.streamlit.app/

---

## What it does

| Job | What it means |
|---|---|
| **Diagnose** | Given a suspected patient's symptoms, exposure and location, estimate whether it is really Lassa fever. |
| **Predict outbreaks** | For each state, forecast whether the coming period looks like an outbreak. |
| **Assess severity** | For a confirmed case, estimate whether it is high-risk (elevated death risk). |
| **Send alerts** | Automatically notify the people in charge the moment risk is high (logged to SQLite; e-mail if SMTP is configured). |
| **Show a dashboard** | A live screen where officials see risk by state, triage cases, view trends, and read the alert log. |

## The model — why it's *adaptive*

Each task is a linear SVM trained by **averaged stochastic gradient descent on the hinge loss**
— `SGDClassifier(loss="hinge", average=True)` — and updated with **`partial_fit()`** as cases
stream in chronologically. The hinge loss makes it an SVM; `partial_fit` is what lets it keep
learning on the fly. Online performance is measured **prequentially** (predict each incoming
batch, *then* train on it) — the honest way to score an online learner. For comparison, each
task also fits a **batch** `LinearSVC` (train once, on a 70/30 temporal split).

## Results on real data (prequential AUC)

| Task | Adaptive (online) | Batch (train-once) |
|---|---|---|
| **Outbreak** (state-month) | **0.89** | 0.91 |
| **Diagnosis** (confirmed Lassa) | 0.64 | — |
| **Outcome** (death) | 0.57 | — |

**Headline finding:** the always-learning model (0.89) essentially matches the traditional
train-once model (0.91) — continuous learning at almost no cost in accuracy. Diagnosis and
outcome are honestly modest: early Lassa symptoms resemble malaria/flu, so symptoms alone are
a genuinely limited signal — a documented result, not a bug.

## Data

Real **SORMAS / NCDC Lassa fever line-list**, published open-access on Zenodo
(**record 7309567**): 20,062 cases, 2018–2021, across 774 local government areas in 37 states
(3,162 laboratory-confirmed). `data/lassa_sormas.sav` is the SPSS export, read with `pyreadstat`.
This is public, de-identified surveillance data.

## Layout

```
adaptive-svm/
  app/dashboard.py          Streamlit app (the 5-tab dashboard) — the deploy entrypoint
  app/notifications.py      Alerting layer — logs to SQLite, e-mails if SMTP_* env vars are set
  app/requirements.txt      Deploy manifest Streamlit Cloud installs (dashboard subset, pinned)
  scripts/prepare_and_train.py   Feature engineering + the three online SVMs (and batch benchmark)
  data/lassa_sormas.sav     Real SORMAS Lassa dataset (Zenodo 7309567)
  models/svm_*.pkl          Prebuilt model bundles (the app self-trains if these can't be loaded)
  docs/                     Plain-English guide (.docx / .pdf)
```

## Run it locally

```bash
cd adaptive-svm
pip install -r requirements.txt
streamlit run app/dashboard.py
```

Then open http://localhost:8501. To retrain the models from scratch:

```bash
python scripts/prepare_and_train.py
```

## Deploy (Streamlit Community Cloud)

1. Go to **share.streamlit.io** → **Create app** → pick this repository.
2. Set **Main file path** to `adaptive-svm/app/dashboard.py`.
3. Under **Advanced settings**, choose **Python 3.12** (matches the pinned libraries).
4. Deploy. Cloud installs from `adaptive-svm/app/requirements.txt`. First build ~1–2 min; then cached.

## Notes & honest limitations

- A single-laptop prototype: Redis Streams / Pandas / MongoDB + SQLite / Streamlit stand in for
  the heavier production stack (Kafka / Spark / Cassandra) described in the thesis.
- SVM decision scores are hinge-loss margins, not calibrated probabilities.
- E-mail alerting degrades gracefully to an in-app log when no `SMTP_*` variables are set, so the
  demo runs with zero setup.

---

*Prepared for the thesis “An Adaptive SVM Model for Intelligent Disease Surveillance System.”*
