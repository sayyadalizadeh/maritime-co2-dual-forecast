"""
vintage.py: data vintage comparison and the paired real-time comparison at origin 2026-06.

Part 1. Vintages. The October 2025 vintage (data/external/oecd_maritime_v1_2025-10-06.xlsx, supplied by the authors;
  sheet "Table": the published residence total, all vessels, monthly tonnes, January 2022 to June 2025, areas by
  name) is read with pandas + python-calamine (openpyxl fails on the export's stylesheet). Areas are matched by their
  OECD names to the October 2026 vintage (data/clean/oecd_areas_v1.csv, built from the raw CSV of 31 October 2026);
  the 2026 values are the published residence totals (component RES_TOTAL, ALL_VESSELS, co2_t) of
  data/clean/oecd_maritime_co2_monthly_v1.csv. Revision of a series-month = 100 x (2026 value - 2025 value) /
  2025 value; statistics of its absolute value (median, 95th percentile, maximum) for World, for economies, and by
  year; the series-months revised most. The vintage files carry no release dates.
Part 2. Paired comparison at origin 2026-06 (targets 2026-07 ... 2026-09, h 1-3): per-economy MASE of the published
  residence total (the 110 system economies), FM track against STAT track (reconciled point forecasts), from
  results/eval_realtime_exact_ext_nested/res_total_pub_cells.csv. Per-economy MASE = mean over the three horizons of
  |y_pub - forecast| / scale_pub (scale_pub = mean |y_t - y_t-12| of the published total up to the origin).
  Difference = MASE_FM - MASE_STAT; Wilcoxon signed-rank test (two-sided), the share of economies where FM is better,
  and a percentile bootstrap 95% interval of the median difference (2,000 resamples of economies, seed 42).
  Both with the (1 - d) correction and with d = 0. Definition of d (evaluate_realtime.py): for each
  economy u, d_u = mean over the 6 months up to and including the origin of (published - component sum) / published
  (months with a zero published value skipped; d = 0 if none is left); positive when the published total exceeds
  the component sum; one value per economy and origin, the same for every horizon; forecast of the published total =
  RES_TOTAL_SYS forecast / (1 - d_u). The base forecasts (TimesFM 2.5 against AutoETS) are compared the same way.

Outputs: results/vintage/
Usage:
  python src/vintage.py
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from runlog import RunLog, out_path

XLSX = "data/external/oecd_maritime_v1_2025-10-06.xlsx"
SEED, REPS = 42, 2000
MONTHS = {m: i + 1 for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])}


def read_vintage_2025(path):
    """Long frame area_name, period, co2_t from the October 2025 export (sheet Table)."""
    t = pd.read_excel(path, sheet_name="Table", header=None, engine="calamine")
    hdr = int(t.index[t[1].astype(str).str.strip() == "Time period"][0])
    cols = {}
    for j, v in t.iloc[hdr].items():
        m = re.fullmatch(r"(\d{4})-([A-Za-z]{3})", str(v).strip())
        if m:
            cols[j] = f"{m.group(1)}-{MONTHS[m.group(2)]:02d}"
    body = t.iloc[hdr + 2:]
    body = body[body[1].notna() & body[list(cols)].notna().any(axis=1)]
    names = body[1].astype(str).str.replace("·", "", regex=False).str.strip()
    w = body[list(cols)].rename(columns=cols)
    w.index = names
    long = w.stack().rename("co2_t_2025").reset_index()
    long.columns = ["area_name", "period", "co2_t_2025"]
    long["co2_t_2025"] = pd.to_numeric(long.co2_t_2025, errors="coerce")
    return long.dropna(), sorted(cols.values())


def stats(v):
    a = v.abs()
    return {"n": int(len(a)), "median_abs_pct": a.median(), "p95_abs_pct": a.quantile(0.95), "max_abs_pct": a.max(),
            "mean_pct": v.mean(), "share_exactly_equal": float((v == 0).mean()), "share_abs_gt_1pct": float((a > 1).mean())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean", default="data/clean")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/vintage")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    log = RunLog("vintage.py", vars(args))
    RES, OUT, CL = Path(args.results), Path(args.out), Path(args.clean)

    def save(df, name):
        df.to_csv(out_path(OUT / name, args.overwrite), index=False, float_format="%.6g", encoding="utf-8")

    # ---------------------------------------------------------------- part 1: vintage comparison
    v25, months25 = read_vintage_2025(XLSX)
    areas = pd.read_csv(CL / "oecd_areas_v1.csv", encoding="utf-8")
    name_map = dict(zip(areas.area_name.str.strip(), areas.area_code))
    v25["area_code"] = v25.area_name.map(name_map)
    match = v25.drop_duplicates("area_name")[["area_name", "area_code"]].merge(
        areas[["area_code", "area_type"]], on="area_code", how="left")
    save(match, "area_matching.csv")
    p = pd.read_csv(CL / "oecd_maritime_co2_monthly_v1.csv",
                    usecols=["area_code", "vessel", "component", "period", "co2_t"])
    p = p[(p.component == "RES_TOTAL") & (p.vessel == "ALL_VESSELS")][["area_code", "period", "co2_t"]]
    m = v25.dropna(subset=["area_code"]).merge(p.rename(columns={"co2_t": "co2_t_2026"}), on=["area_code", "period"],
                                               how="left")
    m = m.merge(areas[["area_code", "area_type"]], on="area_code", how="left")
    miss = m[m.co2_t_2026.isna()]
    m = m.dropna(subset=["co2_t_2026"])
    m = m[m.co2_t_2025 != 0]
    m["revision_pct"] = 100 * (m.co2_t_2026 - m.co2_t_2025) / m.co2_t_2025
    m["year"] = m.period.str[:4]
    save(m.sort_values(["area_code", "period"]), "revisions_series_months.csv")
    rows = []
    for grp, d in [("World", m[m.area_code == "W"]), ("economies", m[m.area_type == "economy"]),
                   ("other aggregates", m[(m.area_type != "economy") & (m.area_code != "W")])]:
        rows.append({"group": grp, "year": "all", **stats(d.revision_pct)})
        for y, dy in d.groupby("year"):
            rows.append({"group": grp, "year": y, **stats(dy.revision_pct)})
    rs = pd.DataFrame(rows)
    save(rs, "revision_stats.csv")
    top = m[m.area_type == "economy"].assign(abs_pct=lambda d: d.revision_pct.abs()).sort_values("abs_pct", ascending=False)
    save(top.head(30), "most_revised_economy_months.csv")
    wm = m[m.area_code == "W"].assign(abs_pct=lambda d: d.revision_pct.abs()).sort_values("abs_pct", ascending=False)
    save(wm, "world_revisions.csv")
    cov = pd.DataFrame([{"areas_in_2025_file": v25.area_name.nunique(), "months_in_2025_file": len(months25),
                         "first_month": months25[0], "last_month": months25[-1],
                         "areas_matched_by_name": int(match.area_code.notna().sum()),
                         "areas_not_matched": "; ".join(match[match.area_code.isna()].area_name),
                         "series_months_compared": len(m), "series_months_without_2026_value": len(miss),
                         "economies_compared": m[m.area_type == "economy"].area_code.nunique()}])
    save(cov, "coverage.csv")
    print(cov.T.to_string())
    print(rs[rs.year == "all"].to_string(index=False))
    print(rs[(rs.group == "economies")].to_string(index=False))

    # ---------------------------------------------------------------- part 2: paired comparison at origin 2026-06
    c = pd.read_csv(RES / "eval_realtime_exact_ext_nested" / "res_total_pub_cells.csv")
    c = c[(c.origin == "2026-06") & (c.unit_type == "economy")]
    rng = np.random.default_rng(SEED)
    rows, per = [], []
    for dv in ("d", "d0"):
        cc = c[c.d_variant == dv]
        for kind, col in [("track (reconciled MinT)", "MASE"), ("base (TimesFM 2.5 vs AutoETS)", "MASE_base")]:
            e = cc.pivot_table(index="unit", columns="track", values=col, aggfunc="mean").dropna()
            nh = cc.groupby(["unit", "track"]).horizon.nunique().unstack()
            diff = e["FM"] - e["STAT"]
            w = wilcoxon(e["FM"], e["STAT"], alternative="two-sided")
            idx = rng.integers(0, len(diff), size=(REPS, len(diff)))
            bm = np.median(diff.to_numpy()[idx], axis=1)
            lo, hi = np.percentile(bm, [2.5, 97.5])
            rows.append({"d_variant": "(1 - d) correction" if dv == "d" else "d = 0", "forecasts": kind,
                         "n_economies": len(diff), "horizons_per_economy": int(nh.min().min()),
                         "median_MASE_FM": e["FM"].median(), "median_MASE_STAT": e["STAT"].median(),
                         "median_diff_FM_minus_STAT": diff.median(), "boot95_lo": lo, "boot95_hi": hi,
                         "mean_diff": diff.mean(), "share_FM_better": float((diff < 0).mean()),
                         "n_ties": int((diff == 0).sum()), "wilcoxon_stat": float(w.statistic), "wilcoxon_p": float(w.pvalue)})
            per.append(e.assign(diff=diff, d_variant=dv, forecasts=kind).reset_index())
            dvals = cc[cc.track == "FM"].drop_duplicates("unit").d
            rows[-1].update({"d_median": dvals.median(), "d_min": dvals.min(), "d_max": dvals.max()})
    pr = pd.DataFrame(rows)
    save(pr, "paired_2026-06.csv")
    save(pd.concat(per, ignore_index=True), "paired_2026-06_per_economy.csv")
    pd.set_option("display.width", 250)
    print(pr.to_string(index=False))
    log.finish("vintage comparison")


if __name__ == "__main__":
    main()
