#!/usr/bin/env python3
"""
seed_database.py

Load the generated/real surveillance CSVs into MongoDB.

Collections (per the PRD):
  - disease_cases     : raw weekly case data (loaded here)
  - forecast_results  : precomputed LSTM 4-week forecasts (written by notebook 3)
  - outbreak_alerts   : precomputed Isolation Forest anomalies (written by notebook 4)

The model notebooks export forecasts_<disease>.csv / alerts_<disease>.csv into
backend/data/; if present, this script loads them too, so a single seed run
reproduces the full database the API serves.

Usage:
  python seed_database.py            # load from CSVs in ../data
  python seed_database.py --drop     # drop collections first (clean reseed)
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from pymongo import ASCENDING, MongoClient

DISEASES = ["lassa", "cholera", "meningitis", "mpox"]
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_csv(path: Path, disease: str) -> list[dict]:
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    docs = df.to_dict("records")
    for d in docs:
        d["disease"] = disease
    return docs


def seed(drop: bool) -> None:
    load_dotenv(REPO_ROOT / ".env")
    uri = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
    dbname = os.environ.get("MONGODB_DB", "disease_surveillance")
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    db = client[dbname]
    client.admin.command("ping")          # fail fast if Mongo isn't reachable
    print(f"Connected to {uri} / db={dbname}")

    if drop:
        for c in ("disease_cases", "forecast_results", "outbreak_alerts"):
            db[c].drop()
        print("Dropped existing collections.")

    # disease_cases — required (raw weekly data)
    total = 0
    for disease in DISEASES:
        path = DATA_DIR / f"ncdc_{disease}.csv"
        if not path.exists():
            print(f"  ! missing {path.name} — run generate_synthetic_data.py first")
            continue
        docs = _load_csv(path, disease)
        if docs:
            db.disease_cases.insert_many(docs)
            total += len(docs)
            print(f"  {disease:11s} {len(docs):6d} weekly records")
    db.disease_cases.create_index(
        [("disease", ASCENDING), ("state", ASCENDING), ("date", ASCENDING)]
    )
    print(f"disease_cases: {total} docs total")

    # forecast_results / outbreak_alerts — optional (produced by the model notebooks)
    for coll, prefix in (("forecast_results", "forecasts_"), ("outbreak_alerts", "alerts_")):
        loaded = 0
        for disease in DISEASES:
            path = DATA_DIR / f"{prefix}{disease}.csv"
            if path.exists():
                docs = _load_csv(path, disease)
                if docs:
                    db[coll].insert_many(docs)
                    loaded += len(docs)
        if loaded:
            db[coll].create_index([("disease", ASCENDING), ("state", ASCENDING)])
            print(f"{coll}: {loaded} docs")
        else:
            print(f"{coll}: none yet (run the model notebooks, then reseed)")

    client.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Seed MongoDB from surveillance CSVs.")
    ap.add_argument("--drop", action="store_true", help="drop collections before loading")
    args = ap.parse_args()
    seed(args.drop)


if __name__ == "__main__":
    main()
