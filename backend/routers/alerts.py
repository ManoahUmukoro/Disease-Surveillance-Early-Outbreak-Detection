"""Recent outbreak alerts (most recent flagged anomalies)."""
from fastapi import APIRouter

from services.db_service import get_db, iso

router = APIRouter(prefix="/api", tags=["surveillance"])


@router.get("/alerts")
async def get_alerts(disease: str | None = None, state: str | None = None, limit: int = 20):
    match = {"is_anomaly": True}
    if disease:
        match["disease"] = disease
    if state and state.lower() != "all":
        match["state"] = state
    cur = get_db().outbreak_alerts.find(
        match, {"_id": 0, "date": 1, "disease": 1, "state": 1, "severity": 1, "description": 1}
    ).sort("date", -1).limit(max(1, min(limit, 200)))
    return [{"date": iso(r["date"]), "disease": r["disease"], "state": r["state"],
             "severity": r["severity"], "description": r["description"]} async for r in cur]
