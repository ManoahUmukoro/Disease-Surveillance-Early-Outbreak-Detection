"""Disease Surveillance API — FastAPI app.

Serves precomputed model outputs (forecasts, anomaly alerts) and historical case
data from MongoDB. The web tier reads results only — it never loads TensorFlow —
so it stays light enough for a free hosting tier.

Run locally:  uvicorn main:app --reload --app-dir backend
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import cases, forecast, anomalies, alerts, summary

app = FastAPI(
    title="Disease Surveillance API",
    description="Big Data–Driven Disease Surveillance & Early Outbreak Detection (Nigeria)",
    version="1.0.0",
)

origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", "http://localhost:5173").split(",")]
app.add_middleware(
    # Public read-only API (no cookies/auth), so any origin may read it and a
    # wildcard is safe. allow_credentials must be False for a "*" origin to work.
    CORSMiddleware, allow_origins=origins, allow_credentials=False,
    allow_methods=["*"], allow_headers=["*"],
)


@app.get("/api/health", tags=["meta"])
async def health():
    return {"status": "ok"}


@app.get("/", tags=["meta"])
async def root():
    return {"name": "Disease Surveillance API", "docs": "/docs", "health": "/api/health"}


for module in (cases, forecast, anomalies, alerts, summary):
    app.include_router(module.router)
