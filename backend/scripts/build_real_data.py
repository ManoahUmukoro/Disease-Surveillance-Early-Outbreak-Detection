#!/usr/bin/env python3
"""
build_real_data.py

Assembles NATIONAL weekly surveillance series for the four target diseases, using
REAL public data where a clean weekly dataset exists, and clearly-flagged modeled
data where it does not (no clean public weekly CSV is available):

  Lassa fever  REAL   NCDC weekly epidemiological reports, 2020-2025
                      (github.com/EmmanuelNiyi/ncdc-lassa-fever-timeseries-2020-2025)
  Mpox         REAL   WHO / Our World in Data, 2022+  (github.com/owid/monkeypox)
  Cholera      MODEL  calibrated to NCDC cholera parameters (2021/2024 epidemics)
  Meningitis   MODEL  calibrated to NCDC CSM parameters (dry-season, epidemic year)

Output: backend/data/national_<disease>.csv, unified schema:
  date, year, week, confirmed, suspected, deaths, cfr, is_outbreak, is_modeled, source
"""
from __future__ import annotations

import io
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import httpx

DATA = Path(__file__).resolve().parents[1] / "data"
DATA.mkdir(parents=True, exist_ok=True)

LASSA_URL = ("https://raw.githubusercontent.com/EmmanuelNiyi/"
             "ncdc-lassa-fever-timeseries-2020-2025/main/data%20files/"
             "lassa_fever_timeseries_full.csv")
MPOX_URL = "https://raw.githubusercontent.com/owid/monkeypox/main/owid-monkeypox-data.csv"

COLS = ["date", "year", "week", "confirmed", "suspected", "deaths",
        "cfr", "is_outbreak", "is_modeled", "source"]


def _get(url: str) -> str:
    r = httpx.get(url, timeout=90, follow_redirects=True)
    r.raise_for_status()
    return r.text


def _label_outbreaks(confirmed: pd.Series) -> pd.Series:
    """Operational outbreak flag: a week is anomalous if confirmed rises well above
    its local rolling level (documented statistical definition, used as ground truth
    for the anomaly-detection evaluation)."""
    base = confirmed.rolling(9, center=True, min_periods=3).median()
    resid = (confirmed - base)
    thr = max(resid.mean() + 1.5 * resid.std(), 3)
    return (resid > thr).astype(int)


def _finalize(df: pd.DataFrame, modeled: bool, source: str) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True)
    df["cfr"] = np.where(df["confirmed"] > 0, (df["deaths"] / df["confirmed"]).round(4), 0.0)
    df["is_outbreak"] = _label_outbreaks(df["confirmed"])
    df["is_modeled"] = modeled
    df["source"] = source
    return df[COLS]


def build_lassa() -> pd.DataFrame:
    d = pd.read_csv(io.StringIO(_get(LASSA_URL))).dropna(subset=["epi_year", "epi_week"])
    out = pd.DataFrame({
        "year": d["epi_year"].astype(int),
        "week": d["epi_week"].astype(int),
        "confirmed": d["confirmed_cases"].fillna(0).astype(int),
        "suspected": d["suspected_cases"].fillna(0).astype(int),
        "deaths": d["deaths"].fillna(0).astype(int),
    })
    out = out[out.week.between(1, 52)]
    out["date"] = [date.fromisocalendar(int(y), int(w), 1).isoformat()
                   for y, w in zip(out.year, out.week)]
    return _finalize(out, False, "NCDC weekly epidemiological reports (2020-2025)")


def build_mpox() -> pd.DataFrame:
    d = pd.read_csv(io.StringIO(_get(MPOX_URL)))
    d = d[d["location"] == "Nigeria"].copy()
    d["date"] = pd.to_datetime(d["date"])
    iso = d["date"].dt.isocalendar()
    d["year"], d["week"] = iso["year"], iso["week"]
    g = (d.groupby(["year", "week"])
         .agg(confirmed=("new_cases", "sum"), deaths=("new_deaths", "sum"))
         .reset_index())
    g["confirmed"] = g["confirmed"].fillna(0).clip(lower=0).astype(int)
    g["deaths"] = g["deaths"].fillna(0).clip(lower=0).astype(int)
    g["suspected"] = 0
    g = g[g.week.between(1, 52)]
    g["date"] = [date.fromisocalendar(int(y), int(w), 1).isoformat()
                 for y, w in zip(g.year, g.week)]
    return _finalize(g, False, "WHO / Our World in Data mpox dataset (2022+)")


def _model_series(start_year, end_year, baseline, peak_week, width, amp,
                  cfr, confirm_rate, outbreaks, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for year in range(start_year, end_year + 1):
        trend = 1.0 + 0.03 * (year - start_year)
        for week in range(1, 53):
            dist = min(abs(week - peak_week), 52 - abs(week - peak_week))
            sf = 1 + amp * np.exp(-(dist ** 2) / (2 * width ** 2))
            om = 1.0
            for (oy, w0, w1, mult) in outbreaks:
                if year == oy and w0 <= week <= w1:
                    om = mult
            lam = baseline * sf * trend * om
            suspected = int(rng.poisson(max(lam, 0.0)))
            confirmed = int(rng.binomial(suspected, confirm_rate)) if suspected else 0
            deaths = int(rng.binomial(confirmed, cfr)) if confirmed else 0
            rows.append({"year": year, "week": week,
                         "date": date.fromisocalendar(year, week, 1).isoformat(),
                         "confirmed": confirmed, "suspected": suspected, "deaths": deaths})
    return pd.DataFrame(rows)


def build_cholera() -> pd.DataFrame:
    d = _model_series(2018, 2024, baseline=110, peak_week=31, width=7.0, amp=3.0,
                      cfr=0.032, confirm_rate=0.35,
                      outbreaks=[(2021, 18, 46, 7.0), (2024, 22, 40, 4.0)], seed=101)
    return _finalize(d, True, "Modeled — calibrated to NCDC cholera parameters (no clean weekly public dataset)")


def build_meningitis() -> pd.DataFrame:
    d = _model_series(2018, 2024, baseline=85, peak_week=10, width=7.0, amp=3.2,
                      cfr=0.085, confirm_rate=0.30,
                      outbreaks=[(2018, 4, 18, 3.2)], seed=202)
    return _finalize(d, True, "Modeled — calibrated to NCDC meningitis parameters (no clean weekly public dataset)")


def main() -> None:
    builders = {"lassa": build_lassa, "cholera": build_cholera,
                "meningitis": build_meningitis, "mpox": build_mpox}
    for name, fn in builders.items():
        df = fn()
        df.to_csv(DATA / f"national_{name}.csv", index=False)
        tag = "MODELED" if bool(df.is_modeled.iloc[0]) else "REAL   "
        print(f"{name:11s} {tag} weeks={len(df):4d}  {df.date.min()} -> {df.date.max()}  "
              f"confirmed_total={int(df.confirmed.sum()):6d}  outbreak_weeks={int(df.is_outbreak.sum())}")


if __name__ == "__main__":
    main()
