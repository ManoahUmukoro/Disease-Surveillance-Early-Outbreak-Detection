"""Isolation Forest anomaly series (per state, or aggregated nationally)."""
from fastapi import APIRouter, Query

from services.db_service import get_db, iso

router = APIRouter(prefix="/api", tags=["surveillance"])


@router.get("/anomalies")
async def get_anomalies(disease: str = Query(...), state: str = "all"):
    match = {"disease": disease}
    if state and state.lower() != "all":
        match["state"] = state
    pipeline = [
        {"$match": match},
        {"$group": {"_id": "$date", "week": {"$first": "$week"},
                    "cases": {"$sum": "$confirmed"},
                    "score": {"$min": "$anomaly_score"},
                    "n_anom": {"$sum": {"$cond": ["$is_anomaly", 1, 0]}},
                    "n_high": {"$sum": {"$cond": [{"$eq": ["$severity", "HIGH"]}, 1, 0]}},
                    "n_med": {"$sum": {"$cond": [{"$eq": ["$severity", "MEDIUM"]}, 1, 0]}}}},
        {"$sort": {"_id": 1}},
    ]
    out = []
    async for r in get_db().outbreak_alerts.aggregate(pipeline):
        severity = "HIGH" if r["n_high"] else ("MEDIUM" if r["n_med"] else "NORMAL")
        out.append({"date": iso(r["_id"]), "week": r["week"], "cases": r["cases"],
                    "anomaly_score": round(r["score"], 4), "is_anomaly": bool(r["n_anom"]),
                    "severity": severity})
    return out
