"""LSTM 4-week forecast (national per disease) + model metrics."""
from fastapi import APIRouter, Query

from services.db_service import get_db

router = APIRouter(prefix="/api", tags=["surveillance"])


@router.get("/forecast")
async def get_forecast(disease: str = Query(...), state: str = "all", weeks: int = 4):
    rows = [r async for r in get_db().forecast_results.find(
        {"disease": disease}, {"_id": 0}).sort("horizon", 1)]
    if rows:
        rows = rows[:max(1, min(weeks, len(rows)))]
        metrics = {"mae": rows[0].get("mae"), "rmse": rows[0].get("rmse"), "mape": rows[0].get("mape")}
    else:
        metrics = {}
    return {
        "disease": disease,
        "national": True,   # forecasts are national-level weekly counts
        "forecast": [{"horizon": r["horizon"], "week": r["week"], "predicted": r["predicted"],
                      "lower_ci": r["lower_ci"], "upper_ci": r["upper_ci"]} for r in rows],
        "metrics": metrics,
    }
