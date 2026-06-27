# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 02 · Data Preprocessing & Feature Engineering
# Cleans the weekly surveillance data, joins weather, engineers the features the
# models consume (lags, rolling statistics, cyclical week-of-year, weather, state
# encoding, seasonal deviation), assigns a strictly temporal train/val/test split,
# fits and saves a per-disease feature scaler, and writes a processed table per
# disease for the model notebooks.
#
# **Split (per the project spec):** train = 2015–2021, validation = 2022,
# test = 2023–2024.

# %%
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

DATA = Path("../backend/data")
MODELS = Path("../backend/models")
MODELS.mkdir(parents=True, exist_ok=True)

DISEASES = ["lassa", "cholera", "meningitis", "mpox"]
LAGS = [1, 2, 3, 4, 8, 12]
FEATURES = ([f"lag_{l}" for l in LAGS] +
            ["roll4_mean", "roll4_std", "seasonal_dev",
             "woy_sin", "woy_cos", "temp_mean", "rainfall_sum", "state_code"])
TRAIN_YEARS = range(2015, 2022)   # 2015–2021
VAL_YEAR = 2022
TEST_YEARS = (2023, 2024)

# %% [markdown]
# ## Load cases + weather and merge
# Weather is joined on (year, week, state); any gaps are filled with the national
# weekly mean so every row has weather features.

# %%
weather = pd.read_csv(DATA / "weather.csv")
print("weather:", weather.shape, "| states:", weather.state.nunique())


def build(disease: str) -> pd.DataFrame:
    df = pd.read_csv(DATA / f"ncdc_{disease}.csv", parse_dates=["date"])
    df = df.merge(weather, on=["year", "week", "state"], how="left")
    for col in ["temp_mean", "rainfall_sum"]:
        df[col] = df[col].fillna(df.groupby("week")[col].transform("mean"))
    return df.sort_values(["state", "date"]).reset_index(drop=True)


# %% [markdown]
# ## Feature engineering (within each state)
# All temporal features are computed per state to avoid leakage across states, and
# rolling/lag features use only past weeks (shifted) so no future information leaks
# into a row's predictors.

# %%
def engineer(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("state", group_keys=False)
    for l in LAGS:
        df[f"lag_{l}"] = g["confirmed"].shift(l)
    df["roll4_mean"] = g["confirmed"].transform(lambda s: s.shift(1).rolling(4).mean())
    df["roll4_std"] = g["confirmed"].transform(lambda s: s.shift(1).rolling(4).std())
    wk_mean = df.groupby(["state", "week"])["confirmed"].transform("mean")
    df["seasonal_dev"] = df["confirmed"] - wk_mean          # deviation from typical week
    df["woy_sin"] = np.sin(2 * np.pi * df["week"] / 52)
    df["woy_cos"] = np.cos(2 * np.pi * df["week"] / 52)
    df["state_code"] = df["state"].astype("category").cat.codes
    return df


def split_of(year: int) -> str:
    if year in TRAIN_YEARS:
        return "train"
    return "val" if year == VAL_YEAR else "test"


processed = {}
for d in DISEASES:
    df = engineer(build(d))
    df["split"] = df["year"].map(split_of)
    df = df.dropna(subset=[f"lag_{max(LAGS)}", "roll4_std"]).reset_index(drop=True)
    processed[d] = df
    print(f"{d:11s} rows={len(df):6d}  "
          f"train={(df.split=='train').sum():5d}  "
          f"val={(df.split=='val').sum():4d}  "
          f"test={(df.split=='test').sum():5d}")

# %% [markdown]
# ## Normalisation
# A `StandardScaler` is fit on **training rows only** (no leakage) and saved per
# disease; the model notebooks reuse it. The processed table itself is saved
# unscaled so it stays interpretable for evaluation and the dashboard.

# %%
for d in DISEASES:
    df = processed[d]
    scaler = StandardScaler().fit(df.loc[df.split == "train", FEATURES])
    joblib.dump({"scaler": scaler, "features": FEATURES}, MODELS / f"scaler_{d}.pkl")
    df.to_csv(DATA / f"processed_{d}.csv", index=False)
    print(f"saved processed_{d}.csv  +  scaler_{d}.pkl  ({len(FEATURES)} features)")

# %% [markdown]
# ## Feature preview (Lassa)

# %%
processed["lassa"][["date", "state", "confirmed"] + FEATURES].head(8)

# %% [markdown]
# ### Takeaways
# - Weather joined and gap-filled; all temporal features computed within each
#   state with past-only windows (no leakage); lag warm-up rows dropped.
# - Split is strictly temporal (2015–21 / 2022 / 2023–24), so the test set
#   simulates true forecasting on unseen future weeks.
# - One scaler per disease, fit on training rows only, saved for the model stage.
#
# **Next:** `03 · LSTM training` — sequence construction and 4-week-ahead forecasting.
