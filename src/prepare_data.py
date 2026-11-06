"""
prepare_data.py: builds the clean, analysis-ready files from the raw OECD download.

Input (raw download, unchanged):
  OECD:      OECD.SDD.NAD.SEEA:DSD_MARITIME_TRANSPORT@DF_MARITIME_TRANSPORT(2.0)
             "Maritime transport CO2 emissions (experimental)", complete CSV,
             last updated 2026-10-14, downloaded 2026-10-31 (src/download_data.py).

Outputs (written to OUT_DIR):
  oecd_maritime_co2_monthly_v1.csv   tidy monthly panel (one row per area x vessel x component x month)
  oecd_areas_v1.csv                  area lookup (code, name, type, OECD membership)
  oecd_series_summary_v1.csv         one row per series: coverage, gaps, size, core-sample flag

Usage:
  python src/prepare_data.py --oecd <oecd_complete.csv> --out <out_dir>
"""
import argparse
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--oecd", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()
OECD_RAW, OUT_DIR = args.oecd, args.out

# ---------------------------------------------------------------- constants
OECD_MEMBERS = [  # 38 members; verified against the residence-based OECD aggregate (members sum to ~99%)
    "AUS", "AUT", "BEL", "CAN", "CHL", "COL", "CRI", "CZE", "DNK", "EST", "FIN", "FRA", "DEU",
    "GRC", "HUN", "ISL", "IRL", "ISR", "ITA", "JPN", "KOR", "LVA", "LTU", "LUX", "MEX", "NLD",
    "NZL", "NOR", "POL", "PRT", "SVK", "SVN", "ESP", "SWE", "CHE", "TUR", "GBR", "USA"]
AGGREGATES = {"W": "world", "OECD": "group_aggregate", "W_O": "other_countries"}
COMPONENTS = {  # code: (letter, basis, short label)
    "RES_DOM_IN":    ("A", "component", "Domestic voyages in country, resident operators"),
    "RES_DOM_OUT":   ("B", "component", "Domestic voyages in other countries, resident operators"),
    "NRES_DOM_IN":   ("C", "component", "Domestic voyages in country, non-resident operators"),
    "RES_INT_FROM":  ("D", "component", "International voyages departing country, resident operators"),
    "RES_INT_TO":    ("E", "component", "International voyages arriving in country, resident operators"),
    "RES_INT_OUT":   ("F", "component", "International voyages between other countries, resident operators"),
    "NRES_INT_FROM": ("G", "component", "International voyages departing country, non-resident operators"),
    "RES_TOTAL":     ("A+B+D+E+F", "residence", "SEEA air emission accounts: maritime transport (ISIC H50)"),
    "TER_DOM":       ("A+C", "territory", "Domestic navigation (UNFCCC basis)"),
    "TER_INT":       ("D+G", "territory", "International navigation, departing country (UNFCCC memo item)"),
}
FULL_START, FULL_END = "2019-01", "2026-06"   # window in which all components are published
CORE_MIN_2025_T = 1e5                          # 0.1 Mt CO2 in 2025
SEP2025 = "2025-09"                            # broad-based one-month spike, likely processing artefact

# ---------------------------------------------------------------- OECD
raw = pd.read_csv(OECD_RAW, dtype=str, keep_default_na=False)
m = raw[(raw.FREQ == "M") & (raw.VESSEL != "ALL_VESSELSXMISC_FISH")].copy()
m["co2_t"] = pd.to_numeric(m["OBS_VALUE"])
m = m.rename(columns={"REF_AREA": "area_code", "VESSEL": "vessel",
                      "VESSEL_EMISSIONS_SOURCE": "component", "TIME_PERIOD": "period"})
m = m[["area_code", "Reference area", "vessel", "component", "period", "co2_t"]]

# area lookup
areas = (m[["area_code", "Reference area"]].drop_duplicates()
         .rename(columns={"Reference area": "area_name"}).sort_values("area_code"))
areas["area_type"] = areas.area_code.map(AGGREGATES).fillna("economy")
areas["oecd_member"] = areas.area_code.isin(OECD_MEMBERS)
m = m.drop(columns="Reference area")

# calendar
dt = pd.to_datetime(m.period, format="%Y-%m")
m["days_in_month"] = dt.dt.days_in_month
m["co2_t_per_day"] = m.co2_t / m.days_in_month

# component metadata
m["component_letter"] = m.component.map({k: v[0] for k, v in COMPONENTS.items()})
m["basis"] = m.component.map({k: v[1] for k, v in COMPONENTS.items()})

# flags
m["flag_sep2025"] = m.period == SEP2025
last_comp_month = (m[m.component.str.startswith(("RES_DOM", "RES_INT", "NRES"))]
                   .groupby("area_code").period.max())
m["flag_after_component_coverage"] = (m.component == "RES_TOTAL") & (
    m.period > m.area_code.map(last_comp_month).fillna("0000-00"))

# accounting identity check for the residence total (A+B+D+E+F), all-vessel series
allv = m[m.vessel == "ALL_VESSELS"].pivot_table(index=["area_code", "period"], columns="component",
                                                 values="co2_t", aggfunc="sum")
parts = ["RES_DOM_IN", "RES_DOM_OUT", "RES_INT_FROM", "RES_INT_TO", "RES_INT_OUT"]
have = allv.reindex(columns=parts).notna().any(axis=1) & allv.get("RES_TOTAL").notna()
rel = ((allv["RES_TOTAL"] - allv.reindex(columns=parts).fillna(0).sum(axis=1)).abs()
       / allv["RES_TOTAL"].replace(0, np.nan))
bad = rel[have & (rel > 0.01)].index
m["flag_identity_mismatch"] = False
key = pd.MultiIndex.from_frame(m[["area_code", "period"]])
m.loc[(m.vessel == "ALL_VESSELS").values & (m.component == "RES_TOTAL").values & key.isin(bad), "flag_identity_mismatch"] = True

m = m.sort_values(["area_code", "vessel", "component", "period"])
cols = ["area_code", "vessel", "component", "component_letter", "basis", "period", "days_in_month",
        "co2_t", "co2_t_per_day", "flag_sep2025", "flag_after_component_coverage", "flag_identity_mismatch"]
m[cols].to_csv(f"{OUT_DIR}/oecd_maritime_co2_monthly_v1.csv", index=False, float_format="%.3f")
areas.to_csv(f"{OUT_DIR}/oecd_areas_v1.csv", index=False)

# series summary
full = pd.period_range(FULL_START, FULL_END, freq="M").astype(str)
def summarise(g):
    p = set(g.period)
    span = pd.period_range(min(p), max(p), freq="M").astype(str)
    missing_in_span = [x for x in span if x not in p]
    longest = run = 0
    for x in span:
        run = run + 1 if x not in p else 0
        longest = max(longest, run)
    return pd.Series({
        "start": min(p), "end": max(p), "n_months": len(p),
        "n_missing_within_span": len(missing_in_span), "longest_gap_months": longest,
        "complete_2019_01_to_2026_06": all(x in p for x in full),
        "co2_t_2025": g.loc[g.period.str.startswith("2025"), "co2_t"].sum()})
s = m.groupby(["area_code", "vessel", "component"]).apply(summarise, include_groups=False).reset_index()
s["core_series"] = s.complete_2019_01_to_2026_06 & (s.co2_t_2025 > CORE_MIN_2025_T)
s.to_csv(f"{OUT_DIR}/oecd_series_summary_v1.csv", index=False, float_format="%.1f")

print("rows:", len(m), "| series:", len(s), "| core series:", int(s.core_series.sum()))
