#!/usr/bin/env python3
"""
fetch_weather.py

Pull historical weekly weather (mean temperature, total rainfall) per Nigerian
state from the free Open-Meteo archive API (no key required) and save to
backend/data/weather.csv. Used as exogenous features for the LSTM forecaster —
rainfall drives cholera, the dry season drives Lassa and meningitis.

Weeks are ISO weeks, so the file joins cleanly to the case data on (year, week,
state). Output columns: year, week, state, temp_mean, rainfall_sum
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import httpx
import pandas as pd

# Approximate state centroids (lat, lon) — 36 states + FCT.
STATE_COORDS = {
    "Jigawa": (12.23, 9.35), "Kaduna": (10.52, 7.44), "Kano": (11.50, 8.52),
    "Katsina": (12.99, 7.60), "Kebbi": (11.49, 4.23), "Sokoto": (13.06, 5.24),
    "Zamfara": (12.17, 6.66), "Adamawa": (9.33, 12.40), "Bauchi": (10.31, 9.85),
    "Borno": (11.83, 13.15), "Gombe": (10.29, 11.17), "Taraba": (8.00, 10.50),
    "Yobe": (12.00, 11.50), "Benue": (7.33, 8.75), "Kogi": (7.80, 6.74),
    "Kwara": (8.97, 4.57), "Nasarawa": (8.50, 8.20), "Niger": (9.93, 5.60),
    "Plateau": (9.22, 9.52), "FCT": (9.07, 7.49), "Ekiti": (7.72, 5.31),
    "Lagos": (6.52, 3.38), "Ogun": (7.00, 3.35), "Ondo": (7.10, 4.84),
    "Osun": (7.56, 4.52), "Oyo": (8.16, 3.61), "Abia": (5.45, 7.52),
    "Anambra": (6.21, 6.94), "Ebonyi": (6.27, 8.08), "Enugu": (6.46, 7.55),
    "Imo": (5.57, 7.05), "Akwa Ibom": (5.01, 7.85), "Bayelsa": (4.77, 6.07),
    "Cross River": (5.87, 8.60), "Delta": (5.53, 5.90), "Edo": (6.34, 5.62),
    "Rivers": (4.86, 6.91),
}

START, END = "2015-01-01", "2024-12-31"
ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
OUT = Path(__file__).resolve().parents[1] / "data" / "weather.csv"


def fetch_state(client: httpx.Client, state: str, lat: float, lon: float):
    params = {
        "latitude": lat, "longitude": lon, "start_date": START, "end_date": END,
        "daily": "temperature_2m_mean,precipitation_sum", "timezone": "Africa/Lagos",
    }
    for attempt in range(4):
        try:
            r = client.get(ARCHIVE, params=params, timeout=60)
            r.raise_for_status()
            d = r.json()["daily"]
            df = pd.DataFrame({
                "date": pd.to_datetime(d["time"]),
                "temp": d["temperature_2m_mean"],
                "rain": d["precipitation_sum"],
            })
            iso = df["date"].dt.isocalendar()
            df["year"], df["week"] = iso["year"], iso["week"]
            g = (df.groupby(["year", "week"])
                   .agg(temp_mean=("temp", "mean"), rainfall_sum=("rain", "sum"))
                   .reset_index())
            g["state"] = state
            return g
        except Exception as e:                       # noqa: BLE001
            if attempt == 3:
                print(f"  ! {state} failed: {e}")
                return None
            time.sleep(2)


def main() -> None:
    frames = []
    with httpx.Client() as client:
        for i, (state, (lat, lon)) in enumerate(STATE_COORDS.items(), 1):
            g = fetch_state(client, state, lat, lon)
            if g is not None:
                frames.append(g)
                print(f"  [{i:2d}/{len(STATE_COORDS)}] {state:12s} weeks={len(g)}")
            time.sleep(0.5)   # be polite to the free API (avoids 429 rate limits)
    if not frames:
        print("ERROR: no weather data fetched")
        sys.exit(1)
    out = pd.concat(frames, ignore_index=True)
    out = out[out["week"] <= 52]                     # align with 52-week case data
    out = out[["year", "week", "state", "temp_mean", "rainfall_sum"]]
    out.to_csv(OUT, index=False)
    print(f"\nWrote {len(out)} rows ({out.state.nunique()} states) to {OUT}")


if __name__ == "__main__":
    main()
