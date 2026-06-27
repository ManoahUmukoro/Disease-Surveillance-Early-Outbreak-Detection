"""Reference data + historical case series."""
from fastapi import APIRouter, Query

from services.db_service import get_db, iso

router = APIRouter(prefix="/api", tags=["surveillance"])
DISEASES = ["lassa", "cholera", "meningitis", "mpox"]


@router.get("/diseases")
async def list_diseases():
    return DISEASES


@router.get("/states")
async def list_states():
    states = await get_db().disease_cases.distinct("state")
    return sorted(states)


@router.get("/cases")
async def get_cases(disease: str = Query(...), state: str = "all", period: str = "all"):
    """Weekly case series for a disease. state='all' aggregates nationally;
    period can be a 4-digit year to restrict the range."""
    match = {"disease": disease}
    if state and state.lower() != "all":
        match["state"] = state
    if period and period.isdigit():
        match["year"] = int(period)
    pipeline = [
        {"$match": match},
        {"$group": {"_id": "$date", "week": {"$first": "$week"},
                    "confirmed": {"$sum": "$confirmed"}, "suspected": {"$sum": "$suspected"},
                    "deaths": {"$sum": "$deaths"}}},
        {"$sort": {"_id": 1}},
    ]
    return [
        {"date": iso(r["_id"]), "week": r["week"], "cases": r["confirmed"],
         "confirmed": r["confirmed"], "suspected": r["suspected"], "deaths": r["deaths"],
         "cfr": round(r["deaths"] / r["confirmed"], 4) if r["confirmed"] else 0.0}
        async for r in get_db().disease_cases.aggregate(pipeline)
    ]
