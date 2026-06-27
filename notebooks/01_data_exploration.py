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
# # 01 · Data Exploration
# **Big Data–Driven Intelligent Disease Surveillance & Early Outbreak Detection (Nigeria)**
#
# Exploratory analysis of weekly surveillance data for the four target diseases —
# **Lassa fever, cholera, cerebrospinal meningitis (CSM), and mpox** — covering
# 2015–2024 across Nigeria's 36 states + FCT. We examine dataset structure, data
# quality / missing values, temporal trends, seasonality, geographic burden, and
# the labelled outbreak weeks that serve as ground truth for anomaly detection.

# %%
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (11, 4)

DATA = Path("../backend/data")
DISEASES = ["lassa", "cholera", "meningitis", "mpox"]
LABELS = {"lassa": "Lassa fever", "cholera": "Cholera",
          "meningitis": "Meningitis (CSM)", "mpox": "Mpox"}


def load(disease: str) -> pd.DataFrame:
    df = pd.read_csv(DATA / f"ncdc_{disease}.csv", parse_dates=["date"])
    df["disease"] = disease
    return df


cases = pd.concat([load(d) for d in DISEASES], ignore_index=True)
print("Combined shape:", cases.shape)
cases.head()

# %% [markdown]
# ## Dataset overview
# Each row is one (disease, state, ISO-week) observation.

# %%
for d in DISEASES:
    sub = cases[cases.disease == d]
    print(f"{LABELS[d]:18s} rows={len(sub):6d}  "
          f"{sub.date.min().date()} -> {sub.date.max().date()}  "
          f"states={sub.state.nunique()}")

# %% [markdown]
# ## Data quality — missing values & range checks
# Real NCDC data carries gaps and inconsistent case definitions; here we confirm
# completeness and that the case hierarchy (deaths ≤ confirmed ≤ suspected) holds.

# %%
print("Missing values per column:")
print(cases.isnull().sum())
print("\nIntegrity checks (counts of violations, expect 0):")
print("  suspected < 0       :", int((cases.suspected < 0).sum()))
print("  confirmed > suspected:", int((cases.confirmed > cases.suspected).sum()))
print("  deaths > confirmed  :", int((cases.deaths > cases.confirmed).sum()))

# %% [markdown]
# ## Burden summary by disease
# Totals over the full period, with the case-fatality rate (CFR).

# %%
summary = (cases.groupby("disease")
           .agg(suspected=("suspected", "sum"),
                confirmed=("confirmed", "sum"),
                deaths=("deaths", "sum"))
           .reindex(DISEASES))
summary["CFR_%"] = (100 * summary.deaths / summary.confirmed).round(1)
summary.index = [LABELS[d] for d in summary.index]
summary

# %% [markdown]
# ## National weekly trend (2015–2024)
# Confirmed cases per epidemiological week, summed across all states. Outbreak
# years stand out clearly — e.g. the 2021 cholera epidemic and the 2022 mpox surge.

# %%
fig, axes = plt.subplots(4, 1, figsize=(12, 11))
for ax, d in zip(axes, DISEASES):
    ts = (cases[cases.disease == d].groupby("date")
          .agg(confirmed=("confirmed", "sum"), outbreak=("is_outbreak", "max")))
    ax.plot(ts.index, ts.confirmed, lw=0.9, color="#0f172a")
    ax.fill_between(ts.index, 0, ts.confirmed.max(),
                    where=ts.outbreak.astype(bool), color="red", alpha=0.08, step="mid")
    ax.set_title(f"{LABELS[d]} — national weekly confirmed (red = labelled outbreak)")
    ax.set_ylabel("confirmed")
plt.tight_layout()
plt.savefig("figures/national_trends.png", dpi=110, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Seasonality
# Mean weekly confirmed by ISO week (normalised per disease) reveals each
# disease's seasonal signature: Lassa and meningitis peak in the dry season,
# cholera in the rains.

# %%
fig, ax = plt.subplots(figsize=(12, 4))
for d in DISEASES:
    wk = cases[cases.disease == d].groupby("week").confirmed.mean()
    ax.plot(wk.index, wk / wk.max(), marker="o", ms=3, label=LABELS[d])
ax.set(xlabel="ISO week", ylabel="normalised mean confirmed",
       title="Seasonality by disease")
ax.legend()
plt.tight_layout()
plt.savefig("figures/seasonality.png", dpi=110, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Geographic burden — Lassa fever, top 10 states

# %%
top = (cases[cases.disease == "lassa"].groupby("state").confirmed.sum()
       .sort_values(ascending=False).head(10))
fig, ax = plt.subplots(figsize=(10, 4))
sns.barplot(x=top.values, y=top.index, ax=ax, color="#b91c1c")
ax.set(xlabel="total confirmed (2015–2024)", title="Lassa fever — top 10 states")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Weather alignment (cholera vs rainfall)
# Open-Meteo weekly rainfall, overlaid on cholera incidence, to motivate weather
# as an exogenous feature for the forecaster.

# %%
wpath = DATA / "weather.csv"
if wpath.exists():
    w = pd.read_csv(wpath)
    nat_rain = w.groupby(["year", "week"]).rainfall_sum.mean().reset_index()
    chol = (cases[cases.disease == "cholera"].groupby(["year", "week"])
            .confirmed.sum().reset_index())
    m = chol.merge(nat_rain, on=["year", "week"])
    m["t"] = m.year + m.week / 52
    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(m.t, m.confirmed, color="#1d4ed8", lw=0.9)
    ax1.set_ylabel("cholera confirmed", color="#1d4ed8")
    ax2 = ax1.twinx()
    ax2.plot(m.t, m.rainfall_sum, color="#059669", lw=0.7, alpha=0.6)
    ax2.set_ylabel("mean weekly rainfall (mm)", color="#059669")
    ax1.set_title("Cholera vs national rainfall")
    plt.tight_layout()
    plt.show()
    print("Correlation (cholera confirmed vs rainfall):",
          round(m.confirmed.corr(m.rainfall_sum), 3))
else:
    print("weather.csv not present yet — run backend/scripts/fetch_weather.py")

# %% [markdown]
# ### Takeaways
# - Series are complete (no missing weeks) with a valid case hierarchy.
# - Outbreak years are visually distinct → strong signal for anomaly detection.
# - Pronounced, disease-specific seasonality → week-of-year is a key feature.
# - Cholera tracks rainfall → justifies weather as an exogenous LSTM input.
#
# **Next:** `02 · Data preprocessing` — cleaning, normalisation, feature
# engineering (lags, week-of-year, weather), and the train/val/test split.
