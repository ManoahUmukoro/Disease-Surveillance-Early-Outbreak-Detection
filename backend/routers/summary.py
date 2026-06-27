"""Dashboard summary statistics."""
from fastapi import APIRouter, Query

from services.db_service import get_db

router = APIRouter(prefix="/api", tags=["surveillance"])
LATEST_YEAR = 2024


@router.get("/summary")
async def get_summary(disease: str = Query(...), state: str = "all"):
    db = get_db()
    match = {"disease": disease}
    if state and state.lower() != "all":
        match["state"] = state
    agg = await db.disease_cases.aggregate([
        {"$match": match},
        {"$group": {"_id": None, "confirmed": {"$sum": "$confirmed"}, "deaths": {"$sum": "$deaths"}}},
    ]).to_list(1)
    confirmed = agg[0]["confirmed"] if agg else 0
    deaths = agg[0]["deaths"] if agg else 0
    states_affected = len(await db.disease_cases.distinct("state", {**match, "confirmed": {"$gt": 0}}))
    active_alerts = await db.outbreak_alerts.count_documents(
        {**match, "is_anomaly": True, "year": LATEST_YEAR})
    return {"disease": disease, "state": state, "total_cases": confirmed,
            "active_alerts": active_alerts, "states_affected": states_affected,
            "cfr": round(deaths / confirmed, 4) if confirmed else 0.0}
