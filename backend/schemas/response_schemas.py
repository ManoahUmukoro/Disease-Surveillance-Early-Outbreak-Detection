"""Pydantic response models — reference / OpenAPI documentation for the API shapes."""
from pydantic import BaseModel


class CasePoint(BaseModel):
    date: str
    week: int
    cases: int
    confirmed: int
    suspected: int
    deaths: int
    cfr: float


class ForecastPoint(BaseModel):
    horizon: int
    week: int
    predicted: float
    lower_ci: float
    upper_ci: float


class AnomalyPoint(BaseModel):
    date: str
    week: int
    cases: int
    anomaly_score: float
    is_anomaly: bool
    severity: str


class Alert(BaseModel):
    date: str
    disease: str
    state: str
    severity: str
    description: str


class Summary(BaseModel):
    disease: str
    state: str
    total_cases: int
    active_alerts: int
    states_affected: int
    cfr: float
