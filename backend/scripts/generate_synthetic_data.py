#!/usr/bin/env python3
"""
generate_synthetic_data.py

Generate calibrated synthetic weekly disease-surveillance data for Nigeria
(2015–2024, per state) for the four target diseases: Lassa fever, cholera,
cerebrospinal meningitis (CSM), and mpox.

The series are calibrated to published NCDC/WHO epidemiology — case-fatality
rates, seasonality, regional burden, and known outbreak years — so the
downstream LSTM forecasting and Isolation Forest anomaly-detection models have
realistic signal to learn from. Known outbreak weeks are labelled (is_outbreak)
to serve as ground truth for anomaly-detection evaluation.

This produces the baseline series and the series for diseases without clean
public CSVs (CSM, mpox). Real NCDC situation-report data for Lassa and cholera
is layered in separately (see ingest_ncdc.py) under the hybrid data strategy.

Output: one CSV per disease in backend/data/ with columns:
    date, year, week, state, suspected, confirmed, deaths, cfr, is_outbreak
"""
from __future__ import annotations

import argparse
import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

# ── Geography ──────────────────────────────────────────────────────────────
# 36 states + FCT grouped into the six geopolitical zones, so disease burden
# can be weighted by region (Lassa in the south, meningitis in the north, etc.).
ZONES = {
    "NW": ["Jigawa", "Kaduna", "Kano", "Katsina", "Kebbi", "Sokoto", "Zamfara"],
    "NE": ["Adamawa", "Bauchi", "Borno", "Gombe", "Taraba", "Yobe"],
    "NC": ["Benue", "Kogi", "Kwara", "Nasarawa", "Niger", "Plateau", "FCT"],
    "SW": ["Ekiti", "Lagos", "Ogun", "Ondo", "Osun", "Oyo"],
    "SE": ["Abia", "Anambra", "Ebonyi", "Enugu", "Imo"],
    "SS": ["Akwa Ibom", "Bayelsa", "Cross River", "Delta", "Edo", "Rivers"],
}
STATES = [s for states in ZONES.values() for s in states]          # 37 total
STATE_ZONE = {s: z for z, states in ZONES.items() for s in states}

# ── Disease calibration ────────────────────────────────────────────────────
# baseline      : mean weekly suspected cases for an "average" state
# cfr           : case-fatality rate (deaths / confirmed)
# confirm_rate  : fraction of suspected cases that get lab-confirmed
# peak_week     : ISO week of the seasonal peak
# season_*      : width and amplitude of the (circular) seasonal bump
# trend         : multiplicative yearly trend (spread / increasing burden)
# start_year    : first year the disease has meaningful data
# zone_weight   : relative burden multiplier per geopolitical zone
# outbreaks     : injected epidemic windows (year, start_week, end_week, mult)
DISEASES = {
    "lassa": {
        "baseline": 0.8, "cfr": 0.189, "confirm_rate": 0.22,
        "peak_week": 6, "season_width": 6.0, "season_amp": 3.5,
        "trend": 1.04, "start_year": 2015,
        "zone_weight": {"SS": 2.2, "SW": 1.8, "NE": 1.6, "NC": 1.0, "SE": 1.3, "NW": 0.6},
        "outbreaks": [(2018, 1, 14, 2.2), (2020, 1, 16, 2.4), (2023, 1, 15, 2.3)],
    },
    "cholera": {
        "baseline": 3.0, "cfr": 0.032, "confirm_rate": 0.45,
        "peak_week": 31, "season_width": 8.0, "season_amp": 2.5,
        "trend": 1.03, "start_year": 2015,
        "zone_weight": {"NE": 2.4, "NW": 2.0, "NC": 1.2, "SS": 1.0, "SW": 0.9, "SE": 0.8},
        "outbreaks": [(2021, 18, 46, 5.0), (2024, 22, 40, 2.5)],
    },
    "meningitis": {
        "baseline": 2.5, "cfr": 0.085, "confirm_rate": 0.18,
        "peak_week": 10, "season_width": 7.0, "season_amp": 3.2,
        "trend": 1.01, "start_year": 2015,
        "zone_weight": {"NW": 2.6, "NE": 2.0, "NC": 1.3, "SW": 0.5, "SE": 0.4, "SS": 0.4},
        "outbreaks": [(2017, 4, 20, 6.0)],   # major Serogroup C epidemic
    },
    "mpox": {
        "baseline": 1.0, "cfr": 0.028, "confirm_rate": 0.38,
        "peak_week": 26, "season_width": 14.0, "season_amp": 0.6,
        "trend": 1.05, "start_year": 2017,    # re-emerged in Nigeria Sep 2017
        "zone_weight": {"SS": 2.4, "SW": 1.6, "SE": 1.1, "NC": 0.9, "NW": 0.6, "NE": 0.6},
        "outbreaks": [(2017, 36, 52, 3.0), (2022, 20, 52, 3.5),
                      (2023, 1, 52, 3.0), (2024, 1, 52, 2.8)],
    },
}

YEARS = list(range(2015, 2025))   # 2015–2024 inclusive
WEEKS = list(range(1, 53))        # ISO weeks 1–52


def seasonal_factor(week: int, peak_week: int, width: float, amp: float) -> float:
    """Circular Gaussian bump on a 52-week cycle (handles dry-season wraparound)."""
    d = abs(week - peak_week)
    d = min(d, 52 - d)
    return 1.0 + amp * math.exp(-(d * d) / (2 * width * width))


def outbreak_factor(params: dict, year: int, week: int) -> tuple[float, int]:
    """Return (multiplier, is_outbreak) for any injected epidemic window."""
    for (oy, ws, we, mult) in params["outbreaks"]:
        if year == oy and ws <= week <= we:
            return mult, 1
    return 1.0, 0


def generate_disease(params: dict, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for year in YEARS:
        if year < params["start_year"]:
            continue
        trend = params["trend"] ** (year - params["start_year"])
        for week in WEEKS:
            sf = seasonal_factor(week, params["peak_week"],
                                 params["season_width"], params["season_amp"])
            omult, is_out = outbreak_factor(params, year, week)
            d = date.fromisocalendar(year, week, 1)   # Monday of the ISO week
            for state in STATES:
                zw = params["zone_weight"].get(STATE_ZONE[state], 1.0)
                lam = params["baseline"] * zw * sf * trend * omult
                suspected = int(rng.poisson(max(lam, 0.0)))
                confirmed = int(rng.binomial(suspected, params["confirm_rate"])) if suspected else 0
                deaths = int(rng.binomial(confirmed, params["cfr"])) if confirmed else 0
                cfr = round(deaths / confirmed, 4) if confirmed else 0.0
                # Outbreak label is per state: flag only when an epidemic window
                # materially elevates an affected (higher-burden) zone, so the label
                # marks a genuinely detectable spike rather than the whole country-week.
                is_out_state = int(is_out and zw >= 1.5)
                rows.append({
                    "date": d.isoformat(), "year": year, "week": week, "state": state,
                    "suspected": suspected, "confirmed": confirmed, "deaths": deaths,
                    "cfr": cfr, "is_outbreak": is_out_state,
                })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic NCDC-style surveillance data.")
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parents[1] / "data"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    for name, params in DISEASES.items():
        df = generate_disease(params, rng)
        df.to_csv(out_dir / f"ncdc_{name}.csv", index=False)
        print(f"{name:11s} rows={len(df):6d}  states={df.state.nunique()}  "
              f"years={df.year.min()}-{df.year.max()}  "
              f"confirmed/yr~{int(df.confirmed.sum() / df.year.nunique()):5d}  "
              f"outbreak-weeks={int(df.is_outbreak.sum())}")
    print(f"\nWrote {len(DISEASES)} CSVs to {out_dir}")


if __name__ == "__main__":
    main()
