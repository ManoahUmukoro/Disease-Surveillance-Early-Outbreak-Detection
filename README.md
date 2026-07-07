# Disease Surveillance & Early Outbreak Detection System

A big-data, ML-driven disease surveillance prototype for Nigeria. It monitors four
priority diseases — **Lassa fever, cholera, cerebrospinal meningitis (CSM), and
mpox** — forecasts case counts four weeks ahead with LSTM networks, flags
anomalous outbreak signals with Isolation Forests, and presents everything on a
real-time dashboard. Demonstrated on a calibrated synthetic benchmark **and**
validated against **real NCDC / WHO data** (see notebook 06).

> Postgraduate research prototype. A streamlined three-tier architecture
> (MongoDB + FastAPI + React) that demonstrates the full research pipeline at an
> appropriate scale.

---

## Architecture

```
NCDC / WHO weekly data (real) + calibrated synthetic ┐
Open-Meteo weather (real) ───────────────────────────┤
                                     ▼
                        Jupyter notebooks (01–06)
        EDA · preprocessing · LSTM · Isolation Forest · evaluation · real-world validation
                                     │  (train models, precompute outputs)
                                     ▼
                            MongoDB  ── disease_cases
                                     ├─ forecast_results
                                     └─ outbreak_alerts
                                     ▼
                   FastAPI backend (8 endpoints, read-only)
                                     ▼
                React dashboard (Vite · Tailwind · Recharts · React Query)
```

**Key design choice:** model outputs (forecasts and anomaly alerts) are
**precomputed in the notebooks and stored in MongoDB**. The web tier only reads
from MongoDB and **never loads TensorFlow**, so the backend stays light enough for
a free hosting tier.

---

## Results

**Controlled benchmark — calibrated synthetic data (the deployed demo):** targets met.

| Model | Metric | Target | Result |
|-------|--------|--------|--------|
| LSTM forecasting | MAPE < 25% | ≥ 3 of 4 | **3/4** — Lassa 22%, cholera ~25%, meningitis 15% |
| Isolation Forest | F1 > 0.65 | ≥ 2 of 4 | **2/4** — cholera 0.70, meningitis 0.86 |

**Real-world validation — real NCDC/WHO data (notebook 06):** honest and harder.

| Disease | Real source | LSTM error | Anomaly F1 |
|---|---|---|---|
| Lassa fever | NCDC weekly, 2020–2025 | MAPE ≈ 37% | ≈ 0.48 |
| Mpox | WHO / OWID, 2022+ | MAE ≈ 4 cases\* | ≈ 0.61 |

\* Mpox counts are so low (~7/week) that MAPE is not meaningful; MAE is reported instead.
Real, sparse LMIC surveillance data is genuinely hard to forecast — consistent with the
published literature, and documented transparently rather than hidden behind the benchmark.

---

## Tech stack

- **Backend:** Python 3.11, FastAPI, Motor/PyMongo, Uvicorn
- **ML / data:** TensorFlow/Keras (LSTM), scikit-learn (Isolation Forest), pandas, NumPy
- **Database:** MongoDB 7 (local, or MongoDB Atlas free tier)
- **Frontend:** React 18, Vite, Tailwind CSS, Recharts, Axios, TanStack React Query
- **Notebooks:** Jupyter (authored as jupytext `.py`, executed to `.ipynb`)

---

## Prerequisites

- Python 3.11+
- Node.js 18+
- MongoDB 7 running locally (`brew install mongodb-community@7.0` → `brew services start mongodb-community@7.0`)
  — or a MongoDB Atlas connection string.

---

## Quick start (from scratch, ~15–20 min)

```bash
# 1. Configure environment
cp .env.example .env            # defaults to local MongoDB and API port 8000

# 2. Python environment + dependencies
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt      # backend (serving)
pip install -r requirements-ml.txt            # notebooks / data / training

# 3. Build the data, train the models, and precompute outputs (benchmark)
python backend/scripts/generate_synthetic_data.py        # weekly case CSVs
python backend/scripts/fetch_weather.py                  # Open-Meteo weather (free, no key)
jupytext --to ipynb --execute notebooks/02_data_preprocessing.py -o notebooks/02_data_preprocessing.ipynb
jupytext --to ipynb --execute notebooks/03_lstm_training.py       -o notebooks/03_lstm_training.ipynb
jupytext --to ipynb --execute notebooks/04_isolation_forest.py    -o notebooks/04_isolation_forest.ipynb

# 4. Seed MongoDB (loads cases, forecasts, and alerts)
python backend/scripts/seed_database.py --drop

# 5. Run the backend (http://localhost:8000, docs at /docs)
uvicorn main:app --app-dir backend --reload --port 8000

# 6. Run the frontend (separate terminal) — http://localhost:5173
cd frontend && npm install && npm run dev

# (optional) Real-world validation on real NCDC/WHO data:
python backend/scripts/build_real_data.py                # assembles national_<disease>.csv
jupytext --to ipynb --execute notebooks/06_real_world_validation.py -o notebooks/06_real_world_validation.ipynb
```

> **Note on ports:** the default is `8000`. If `8000` is occupied, set `API_PORT`
> (and `VITE_API_BASE_URL` / the Vite proxy target) to a free port such as `8001`.

---

## API endpoints

Base URL: `http://localhost:8000` · interactive docs at `/docs`.

| Endpoint | Params | Returns |
|----------|--------|---------|
| `GET /api/health` | — | `{ "status": "ok" }` |
| `GET /api/diseases` | — | `["lassa","cholera","meningitis","mpox"]` |
| `GET /api/states` | — | 37 state names |
| `GET /api/cases` | `disease, state, period` | weekly `{ date, week, cases, confirmed, suspected, deaths, cfr }` |
| `GET /api/forecast` | `disease, weeks` | 4-week `{ horizon, week, predicted, lower_ci, upper_ci }` + metrics |
| `GET /api/anomalies` | `disease, state` | weekly `{ date, week, cases, anomaly_score, is_anomaly, severity }` |
| `GET /api/alerts` | `disease, state, limit` | recent `{ date, disease, state, severity, description }` |
| `GET /api/summary` | `disease, state` | `{ total_cases, active_alerts, states_affected, cfr }` |

---

## Notebooks (`notebooks/`)

Authored as reviewable jupytext `.py` sources; execute to `.ipynb` with embedded
outputs. Examiners can read the `.ipynb` files directly.

1. `01_data_exploration` — EDA, data quality, seasonality, geographic burden
2. `02_data_preprocessing` — cleaning, feature engineering, train/val/test split, scalers
3. `03_lstm_training` — LSTM forecasting (national, log-stabilised), 4-week forecasts
4. `04_isolation_forest` — anomaly detection, severity mapping, F1 tuning
5. `05_model_evaluation` — reloads saved models and consolidates all benchmark metrics
6. `06_real_world_validation` — re-runs the models on **real** NCDC/WHO data (Lassa, mpox) and reports the honest results vs the synthetic benchmark

---

## Deployment (all free tiers)

- **Database:** MongoDB Atlas (512 MB free) — set `MONGODB_URI` to the SRV string and re-run the seed script.
- **Backend:** Render — Blueprint (`render.yaml`); set `MONGODB_URI`. Runs `uvicorn main:app --host 0.0.0.0 --port $PORT`.
- **Frontend:** Vercel — Root Directory `frontend`; set `VITE_API_BASE_URL` to the deployed backend URL.
- **Model training:** Google Colab or Kaggle (free GPU) if retraining at scale.

---

## Data & validation strategy

A **two-track** data strategy, both fully documented and clearly labeled in-app:

1. **Controlled benchmark — calibrated synthetic data.** The deployed demo runs on
   synthetic weekly data generated from real epidemiological parameters (case-fatality
   rates, seasonality, documented outbreak years) via
   `backend/scripts/generate_synthetic_data.py`. This gives clean, complete, per-state
   coverage for demonstrating the full pipeline, and is where the target metrics are
   met. Weather is real (Open-Meteo historical archive).

2. **Real-world validation — real published data.** `backend/scripts/build_real_data.py`
   assembles genuine national weekly series and notebook `06` re-runs the models on them:
   - **Lassa fever** — NCDC Weekly Epidemiological Reports, 2020–2025
     (`github.com/EmmanuelNiyi/ncdc-lassa-fever-timeseries-2020-2025`)
   - **Mpox** — WHO / Our World in Data, 2022+ (`github.com/owid/monkeypox`)
   - **Cholera & meningitis** — no clean weekly public dataset exists (only NCDC PDF
     situation reports and a 1971–2021 literature compilation), so these remain
     **modeled** and are clearly flagged everywhere.

   On real data the models are markedly harder (Lassa MAPE ~37%; mpox barely forecastable
   at ~7 cases/week) — an honest reflection of sparse LMIC surveillance data.

---

## Project structure

```
disease-surveillance-system/
├── backend/
│   ├── main.py                 # FastAPI app
│   ├── routers/                # cases, forecast, anomalies, alerts, summary
│   ├── services/               # db_service (Motor)
│   ├── schemas/                # response models
│   ├── scripts/                # generate_synthetic_data, fetch_weather, build_real_data, seed_database
│   ├── data/                   # generated + assembled CSVs (gitignored)
│   ├── models/                 # trained .h5 / .pkl (gitignored)
│   └── requirements.txt
├── frontend/                   # Vite + React + Tailwind dashboard
├── notebooks/                  # 6 Jupyter notebooks (.py sources + .ipynb)
├── render.yaml                 # Render deployment blueprint
├── requirements-ml.txt
├── .env.example
└── README.md
```
