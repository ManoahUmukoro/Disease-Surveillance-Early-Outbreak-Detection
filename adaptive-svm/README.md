# Adaptive SVM — Intelligent Disease Surveillance System

An intelligent disease-surveillance dashboard for Nigeria, built around an **adaptive
(online) Support Vector Machine** that keeps learning from every new case instead of being
trained once and left to go stale. It reads real Lassa fever surveillance records, registers and
triages new cases, predicts which states are heading into an outbreak, and alerts the officials
in charge.

This folder is the reference prototype for the thesis
**“An Adaptive SVM Model for Intelligent Disease Surveillance System.”** It lives alongside
the earlier LSTM/React forecasting system in the same repository ("keep both").

> **Live demo:** https://disease-surveillance-early-outbreak-detection-9eggg9qvphqsz7z6.streamlit.app/

---

## What it does

| Job | What it means |
|---|---|
| **Register & triage** | Enter a suspected case (symptoms, environment, clinical notes, lab info, documents) → the SVM returns a risk level, confidence and recommended action → save it. |
| **Predict outbreaks** | For each state, forecast whether the coming period looks like an outbreak — with risk level, confidence, trend and a recommended action. |
| **Assess severity** | For a confirmed case, estimate whether it is high-risk (elevated death risk). |
| **Send alerts** | Notify the people in charge the moment risk is high (logged with recipient, method and status; e-mail if SMTP is configured). |
| **Monitor over time** | Trends by disease / state / period, switching between cases, deaths, risk score and alerts. |

The dashboard has six tabs: **Overview · Outbreak monitor · Case Registration & Triage · Trends ·
Notifications · Model Status**.

## Architecture — two databases + a streaming bus

A surveillance record splits into two shapes of data, which is why the system uses **two stores**,
linked by a single `case_id`:

- **SQLite** — the *structured / time-series* store the SVM reads: `surveillance_events`
  (date, state, LGA, disease, new_cases, deaths, temperature, rainfall), structured case records,
  and the alert log. (`store.py`)
- **MongoDB** — the *unstructured* store: clinical notes, laboratory information, and uploaded
  documents (PDF / DOCX / images via GridFS). (`mongo_store.py`)
- **Redis Streams** — the *ingestion bus*: each saved case is published to `surveillance:cases`
  and consumed by a streaming loop (`stream.py`, `scripts/stream_consumer.py`).

MongoDB and Redis **degrade gracefully** — if they aren't configured the app runs SQLite-only and
shows the exact payload it would have written/published, so the demo always runs. See
[`docs/CHAPTER4_ARCHITECTURE.md`](docs/CHAPTER4_ARCHITECTURE.md) for the full design and the
production-to-laptop technology substitution.

## The model — why it's *adaptive*

Each task is a linear SVM trained by **averaged stochastic gradient descent on the hinge loss**
— `SGDClassifier(loss="hinge", average=True)` — and updated with **`partial_fit()`** as cases
stream in chronologically. Online performance is measured **prequentially** (predict each incoming
batch, *then* train on it). For comparison, each task also fits a **batch** `LinearSVC`.

## Results on real data (prequential AUC)

| Task | Adaptive (online) | Batch (train-once) |
|---|---|---|
| **Outbreak** (state-month) | **0.89** | 0.91 |
| **Diagnosis** (confirmed Lassa) | 0.64 | — |
| **Outcome** (death) | 0.57 | — |

**Headline finding:** the always-learning model (0.89) essentially matches the traditional
train-once model (0.91) — continuous learning at almost no cost in accuracy.

## Data

Real **SORMAS / NCDC Lassa fever line-list**, published open-access on Zenodo
(**record 7309567**): 20,062 cases, 2018–2021, across 774 local government areas in 37 states
(3,162 laboratory-confirmed). `data/lassa_sormas.sav` is read with `pyreadstat`.

## Layout

```
adaptive-svm/
  app/dashboard.py          Streamlit app (6 tabs) — the deploy entrypoint
  app/store.py              SQLite — structured store (events, cases, notifications)
  app/mongo_store.py        MongoDB — unstructured store (notes, lab, documents); graceful
  app/stream.py             Redis Streams — ingestion bus publisher; graceful
  app/notifications.py      Alerting layer (recipient / method / status / acknowledge)
  app/requirements.txt      Deploy manifest Streamlit Cloud installs (pinned)
  scripts/prepare_and_train.py   Feature engineering + the three online SVMs (+ batch benchmark)
  scripts/stream_consumer.py     Redis Streams processing loop (the streaming consumer)
  data/lassa_sormas.sav     Real SORMAS Lassa dataset (Zenodo 7309567)
  models/svm_*.pkl          Prebuilt model bundles (app self-trains if these can't be loaded)
  docs/                     Plain-English guide + CHAPTER4_ARCHITECTURE.md
  Dockerfile · docker-compose.yml · .env.example    Full local stack
```

## Run it locally (single process, SQLite-only)

```bash
cd adaptive-svm
pip install -r requirements.txt
streamlit run app/dashboard.py        # http://localhost:8501
python scripts/prepare_and_train.py   # (optional) retrain the models
```

## Run the full stack (Docker Compose)

Brings up the app with **real MongoDB and Redis** plus the streaming consumer — the complete
Chapter 4 architecture on one machine:

```bash
cd adaptive-svm
docker compose up --build             # http://localhost:8501
```

## Deploy (Streamlit Community Cloud)

1. **share.streamlit.io** → **Create app** → pick this repository.
2. **Main file path:** `adaptive-svm/app/dashboard.py` · **Advanced → Python 3.12**.
3. To activate the unstructured store on the hosted demo, add **`MONGODB_URI`** in the app's
   **Secrets** (reuse your MongoDB Atlas cluster). Optionally add `REDIS_URL`. Without them the app
   runs SQLite-only and shows the payloads it would have written — it still works.

## Notes & honest limitations

- A single-laptop prototype: Redis Streams / Python loop / MongoDB + SQLite / Streamlit stand in for
  the production stack (Kafka / Spark / Cassandra) — see the architecture doc.
- The prediction model is Lassa-specific; the structured store carries a `disease` field so the
  system is multi-disease-ready, but predictions for other diseases are indicative only.
- SVM decision scores are hinge-loss margins, not calibrated probabilities.

---

*Prepared for the thesis “An Adaptive SVM Model for Intelligent Disease Surveillance System.”*
