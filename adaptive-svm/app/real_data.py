"""
real_data.py — parse the additional REAL disease sources into surveillance-event rows, so the
system covers more than one disease on genuinely real data (no fabricated records):

  • Cholera — GinaCharnley et al., peer-reviewed sub-national (per-state) Nigeria compilation
              (github.com/GinaCharnley/cholera_data_drc_nga), 1971–2021.
  • Mpox    — Our World in Data (OWID) national monthly series for Nigeria, 2022–present.

Returns a DataFrame with the surveillance_events columns so store.bootstrap can load it alongside
the SORMAS Lassa data.
"""
import pandas as pd
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
_COLS = ["report_date", "state", "lga", "disease", "new_cases", "deaths",
         "temperature", "rainfall", "source"]
_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
_STATE_FIX = {"Akwa": "Akwa Ibom", "Fct": "FCT", "Abuja": "FCT", "Nassarawa": "Nasarawa"}


def _num(x):
    try:
        return int(str(x).replace(",", "").strip())
    except Exception:
        return 0


def _month_of(dstr):
    if pd.isna(dstr):
        return 1
    s = str(dstr).lower()
    for k, v in _MONTHS.items():
        if k in s:
            return v
    return 1


def cholera_events():
    f = DATA / "cholera_nigeria.csv"
    if not f.exists():
        return pd.DataFrame(columns=_COLS)
    c = pd.read_csv(f)
    rows = []
    for _, r in c.iterrows():
        if pd.isna(r.get("Year")) or pd.isna(r.get("State")):
            continue
        ca, de = _num(r.get("Cases")), _num(r.get("Deaths"))
        if ca == 0 and de == 0:
            continue
        st = str(r["State"]).strip().title()
        st = _STATE_FIX.get(st, st)
        yr, mo = int(r["Year"]), _month_of(r.get("Date"))
        rows.append((f"{yr:04d}-{mo:02d}-01", st, None, "Cholera", ca, de, None, None, "real:cholera"))
    df = pd.DataFrame(rows, columns=_COLS)
    if df.empty:
        return df
    return (df.groupby(["report_date", "state", "lga", "disease", "temperature", "rainfall", "source"],
                       dropna=False)[["new_cases", "deaths"]].sum().reset_index())[_COLS]


def mpox_events():
    f = DATA / "mpox_nigeria.csv"
    if not f.exists():
        return pd.DataFrame(columns=_COLS)
    m = pd.read_csv(f)
    m["date"] = pd.to_datetime(m["date"], errors="coerce")
    ts = m.dropna(subset=["date"]).set_index("date")["new_cases"].fillna(0).resample("MS").sum()
    ts = ts[ts > 0]
    rows = [(d.strftime("%Y-%m-01"), "Nigeria (national)", None, "Mpox", int(v), 0, None, None, "real:mpox")
            for d, v in ts.items()]
    return pd.DataFrame(rows, columns=_COLS)


def covid_events():
    """Real COVID-19 national monthly series for Nigeria (WHO / OWID, 2020-2023)."""
    f = DATA / "covid_nigeria.csv"
    if not f.exists():
        return pd.DataFrame(columns=_COLS)
    m = pd.read_csv(f)
    m["date"] = pd.to_datetime(m["date"], errors="coerce")
    ts = m.dropna(subset=["date"]).set_index("date")["new_cases"].fillna(0).resample("MS").sum()
    ts = ts[ts > 0]
    rows = [(d.strftime("%Y-%m-01"), "Nigeria (national)", None, "COVID-19", int(v), 0, None, None,
             "real:covid") for d, v in ts.items()]
    return pd.DataFrame(rows, columns=_COLS)


# ── clearly-labelled DEMONSTRATION data (NOT real) ───────────────────────
# For diseases without a clean public real time-series, so the platform can demonstrate its full
# multi-disease capability. Every row is tagged source='demo' and the disease name carries a
# "(demo)" suffix, so it is never mistaken for real surveillance data. Not for epidemiological use.
NIGERIA_STATES = [
    "Abia", "Adamawa", "Akwa Ibom", "Anambra", "Bauchi", "Bayelsa", "Benue", "Borno", "Cross River",
    "Delta", "Ebonyi", "Edo", "Ekiti", "Enugu", "FCT", "Gombe", "Imo", "Jigawa", "Kaduna", "Kano",
    "Katsina", "Kebbi", "Kogi", "Kwara", "Lagos", "Nasarawa", "Niger", "Ogun", "Ondo", "Osun", "Oyo",
    "Plateau", "Rivers", "Sokoto", "Taraba", "Yobe", "Zamfara"]
DEMO_DISEASES = ["Malaria", "Typhoid", "Measles", "Yellow fever", "Meningitis (CSM)",
                 "Tuberculosis", "Pneumonia", "Diphtheria"]


def demo_events():
    import numpy as np
    rng = np.random.RandomState(7)
    rows = []
    for di, dis in enumerate(DEMO_DISEASES):
        base = 3 + (di % 5) * 4
        phase = (di * 3) % 12
        for st in NIGERIA_STATES:
            sf = 0.3 + rng.rand() * 1.4
            for year in range(2022, 2027):
                for mo in range(1, 13):
                    if year == 2026 and mo > 3:
                        continue
                    seas = 1 + 0.7 * np.sin(2 * np.pi * (mo - phase) / 12)
                    n = int(rng.poisson(max(0.1, base * seas * sf)))
                    if n <= 0:
                        continue
                    rows.append((f"{year}-{mo:02d}-01", st, None, f"{dis} (demo)", n,
                                 int(rng.binomial(n, 0.02)), None, None, "demo"))
    # A staged, clearly-labelled DEMO outbreak so the platform always shows a fresh, high-case CURRENT
    # event that the autonomous engine can detect and alert on — a Measles surge in Kano, early 2026.
    for ym, cases, deaths in [("2026-01", 1200, 18), ("2026-02", 2450, 41), ("2026-03", 4100, 73)]:
        rows.append((f"{ym}-01", "Kano", None, "Measles (demo)", cases, deaths, None, None, "demo"))
    return pd.DataFrame(rows, columns=_COLS)


def extra_events():
    """Real non-Lassa diseases + clearly-labelled demonstration diseases, combined."""
    parts = [cholera_events(), mpox_events(), covid_events(), demo_events()]
    parts = [p for p in parts if not p.empty]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=_COLS)
