# Chapter 4 — System Architecture (Adaptive SVM Disease Surveillance)

This document explains the implemented architecture of *An Adaptive SVM Model for Intelligent
Disease Surveillance System*, how each component maps to the production design proposed in
Chapter 3, and how the running demonstration proves that every component is necessary.

---

## 1. Technology substitution (production → laptop-buildable)

The Chapter 3 design specifies an enterprise big-data stack. Chapter 4 substitutes each heavy
component with a lightweight equivalent that runs on a single machine while preserving the same
*role* in the architecture.

| Production component (Ch 3) | Role | Substitute (Ch 4) | Why it is faithful |
|---|---|---|---|
| Apache Kafka | Ingestion bus | **Redis Streams** | Same publish/consume "as data arrives" semantics, one process |
| Apache Spark Structured Streaming | Stream processing | **Python streaming loop** (`stream_consumer.py`) | Genuine streaming logic; single-process instead of a cluster |
| MongoDB | Unstructured / semi-structured store | **MongoDB** (Atlas or local) | Kept — already lightweight |
| Apache Cassandra | Structured / time-series store | **SQLite** | Preserves the two-database design without JVM overhead |
| TensorFlow / PyTorch | Modelling | **scikit-learn** (`SGDClassifier`) | The contribution is the *adaptive SVM*, not deep learning |

The result runs three ways from the same code: **Streamlit Community Cloud** (hosted demo,
Atlas for MongoDB), **Docker Compose** (full local stack), or a **single `streamlit run`**
(SQLite-only, everything else degrades gracefully).

---

## 2. The two-database design (the core of the demo)

A single disease surveillance record is naturally split into two shapes of data, which is why the
architecture needs **two** stores:

| Store | Data shape | Holds | Read by |
|---|---|---|---|
| **SQLite** | structured / time-series | `surveillance_events` (date, state, LGA, disease, new_cases, deaths, temperature, rainfall), structured `cases`, and the `notifications` log | the **Adaptive SVM** and every chart |
| **MongoDB** | unstructured / semi-structured | clinical notes (free text), laboratory information (sample ID, PCR result, technician), and uploaded documents (PDF / DOCX / images, via GridFS) | clinicians / reviewers, on demand |

The two are linked by a single **`case_id`**. The SVM only ever needs the clean, uniform
structured rows — every `surveillance_events` row has exactly the same columns — which is precisely
why that data lives in SQLite, while the free-form clinical material lives in MongoDB.

---

## 3. End-to-end data flow (Register → Predict → Save)

The **Case Registration & Triage** page demonstrates the whole architecture on one screen:

```
             ┌─────────────────────── Streamlit dashboard ───────────────────────┐
 user enters │ Basic info · Environmental │ Clinical notes · Lab info · Documents │
   a case    └──────────────┬──────────────┴───────────────────┬─────────────────┘
                            │ Predict (Adaptive SVM)            │
                            ▼                                   │
                  risk · confidence · action                   │
                            │ Save case (case_id)              │
             ┌──────────────┼───────────────────┬──────────────┘
             ▼              ▼                    ▼
        ┌─────────┐   ┌───────────┐      ┌──────────────┐
        │ SQLite  │   │  MongoDB  │      │ Redis Streams│
        │ cases + │   │ notes,lab,│      │ surveillance:│
        │ events  │   │ documents │      │   cases      │
        └─────────┘   └───────────┘      └──────┬───────┘
                                                ▼
                                        stream_consumer.py
                                   (partial_fit / aggregation)
```

1. The user fills the accordion form and clicks **Predict** — the Lassa-trained diagnosis and
   outcome SVMs return a risk level, a confidence, and a recommended action.
2. **Save case** writes the structured fields and a new `surveillance_events` row to **SQLite**,
   the notes/lab/documents to **MongoDB**, and publishes a new-case event to **Redis Streams** —
   all tied together by one `case_id`.
3. The **consumer** process reads the stream and reacts to the case (the seam for online learning).

If MongoDB or Redis is not configured, the app says so plainly and shows the exact payload it would
have written/published, so the flow stays transparent and the demo still runs.

---

## 4. The adaptive model

- **Learner:** `SGDClassifier(loss='hinge', average=True)` — a linear SVM trained by averaged
  stochastic gradient descent and updated online with `partial_fit()` as cases stream in.
- **Evaluation:** prequential (predict each incoming batch, then train on it) — the honest way to
  score an online learner.
- **Result on real SORMAS data (20,062 Lassa cases, 2018–2021):** outbreak **0.89**, diagnosis
  0.64, outcome 0.57 (prequential AUC). The adaptive model (0.89) matches a train-once batch SVM
  (0.91) — **continuous learning at almost no cost in accuracy**, which is the thesis's contribution.

---

## 5. Running it

**Hosted demo (Streamlit Community Cloud).** Deploy `adaptive-svm/app/dashboard.py` (Python 3.12).
Set `MONGODB_URI` in the app's Secrets (reuse the Atlas cluster) to activate the unstructured store;
without it the app runs SQLite-only.

**Full local stack (Docker Compose).**
```bash
cd adaptive-svm
docker compose up --build      # dashboard at http://localhost:8501
```
This starts the app, a real MongoDB, a real Redis, and the streaming consumer — the complete
Chapter 4 architecture on one machine.

**Single process (development).**
```bash
cd adaptive-svm
pip install -r requirements.txt
streamlit run app/dashboard.py
```

---

## 6. Why each component is justified (defense summary)

- **SQLite** — the SVM needs uniform, structured, time-series rows; that is what it reads.
- **MongoDB** — real cases carry free-text notes, lab records, and documents that do not fit a
  fixed schema.
- **Redis Streams** — surveillance is continuous; new cases must be processed as they arrive, not
  in nightly batches.
- **Adaptive SVM** — an online learner keeps up with drift without being taken offline to retrain,
  and this system proves it does so without sacrificing accuracy.
