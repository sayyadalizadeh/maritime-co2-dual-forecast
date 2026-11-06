"""
build_panel.py: builds the modelling panels of the forecasting system (panel v2).

Rules applied:
  structural zeros (option --structural-zeros, on by default): a missing component
      value is set to zero only when every published total that contains it is published
      in that month and equals the sum of the other published components within 0.01% or
      1 t. Every imputed zero is logged.
  residence total: inside the system the residence total is RES_TOTAL_SYS = A+B+D+E+F for every unit
      (economies, Residual, World). The published total is kept as RES_TOTAL_PUB for
      reporting; it is not part of the constraint system.
  discrepancy node: the World-level series DELTA_INT = E + F - G closes the international world balance.

The system consists of
  - the system economies: economies whose seven components (A to G, all vessels) are
    available in every month from 2019-01 to 2026-06 after the structural-zero rule;
  - the Residual node: World minus the sum over system economies, for every measure;
  - the World node (area `W`), plus DELTA_INT.
Economies outside the system enter through the Residual. Their core series (DATA_RULES.md
Section 5) of RES_TOTAL_PUB, TER_DOM and TER_INT are kept in a separate outside-system panel.

Inputs (data/clean, written by src/prepare_data.py):
  oecd_maritime_co2_monthly_v1.csv, oecd_areas_v1.csv, oecd_series_summary_v1.csv
Outputs (SFX = --suffix, default v2):
  results/system_economies_SFX.csv             system economies, 2025 shares of World, structural-zero status
  results/coverage_2025_SFX.csv                2025 share of World covered by the system, by measure
  results/structural_zeros.csv                 every zero imputed under the structural-zero rule
  results/res_total_discrepancy.csv            RES_TOTAL_PUB vs RES_TOTAL_SYS, every system unit and month
  data/clean/panel_system_SFX.parquet          long system panel (incl. RES_TOTAL_PUB rows), 2019-01 to 2026-06
  data/clean/panel_system_wide_SFX.parquet     wide panel of the constraint-system series (co2_t_per_day)
  data/clean/panel_outside_SFX.parquet         outside-system core series, long, 2019-01 to 2026-06
  data/clean/realtime_res_total_pub_2026-07_2026-09_SFX.parquet
                                               RES_TOTAL_PUB for July to September 2026 (real-time test)

Values per day are computed as co2_t / days_in_month (not taken from the 3-decimal rounded
column of the clean file), so that identities hold to floating-point precision.

Usage:
  python src/build_panel.py [--structural-zeros | --no-structural-zeros] [--suffix v2] [--overwrite]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from runlog import RunLog, out_path
from structural_zeros import COMPONENTS, PUB_TOTALS, A, B, C, D, E, F, G, structural_zero_mask

# ---------------------------------------------------------------- constants
SYS_MEASURES = COMPONENTS + ["RES_TOTAL_SYS", "TER_DOM", "TER_INT"]       # measures of every system unit
LETTER = dict(zip(SYS_MEASURES + ["RES_TOTAL_PUB", "DELTA_INT"],
                  list("ABCDEFG") + ["A+B+D+E+F", "A+C", "D+G", "published", "E+F-G"]))
BASIS = {**{k: "component" for k in COMPONENTS}, "RES_TOTAL_SYS": "residence", "RES_TOTAL_PUB": "residence",
         "TER_DOM": "territory", "TER_INT": "territory", "DELTA_INT": "discrepancy"}
START, END = "2019-01", "2026-06"            # panel window (all components published)
RT_START, RT_END = "2026-07", "2026-09"      # RES_TOTAL_PUB only: real-time test months
SEP2025 = "2025-09"
WORLD, RESIDUAL = "W", "RESIDUAL"

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--clean", default="data/clean")
ap.add_argument("--results", default="results")
ap.add_argument("--suffix", default="v2", help="version suffix of the output files")
ap.add_argument("--structural-zeros", action=argparse.BooleanOptionalAction, default=True,
                help="apply the structural-zero rule (default: on)")
ap.add_argument("--overwrite", action="store_true", help="allow overwriting existing outputs")
args = ap.parse_args()
log = RunLog("build_panel.py", vars(args))
CLEAN, RESULTS, SFX = Path(args.clean), Path(args.results), args.suffix

# ---------------------------------------------------------------- load
m = pd.read_csv(CLEAN / "oecd_maritime_co2_monthly_v1.csv", dtype={"period": str})
areas = pd.read_csv(CLEAN / "oecd_areas_v1.csv")
summary = pd.read_csv(CLEAN / "oecd_series_summary_v1.csv")
m = m[m.vessel == "ALL_VESSELS"].drop(columns="vessel")
months = pd.period_range(START, END, freq="M").astype(str).tolist()
days = pd.Series(pd.PeriodIndex(months, freq="M").days_in_month, index=months)
economies = sorted(areas.loc[areas.area_type == "economy", "area_code"])

# Full grid economy x month (window) with the seven components and the three published totals
# (co2_t, tonnes per month). Absent rows become NaN.
cols_all = COMPONENTS + list(PUB_TOTALS)
X = (m[m.area_code.isin(economies) & m.period.isin(months) & m.component.isin(cols_all)]
     .pivot(index=["area_code", "period"], columns="component", values="co2_t")
     .reindex(index=pd.MultiIndex.from_product([economies, months], names=["area_code", "period"]),
              columns=cols_all))
missing_before = X[COMPONENTS].isna()

# ---------------------------------------------------------------- structural zeros
# The rule is in src/structural_zeros.py, so that it can be unit-tested.
zero_mask, present_sum, contained_in = structural_zero_mask(X)
if not args.structural_zeros:
    zero_mask[:] = False
Xi = X.copy()
Xi[COMPONENTS] = Xi[COMPONENTS].mask(zero_mask, 0.0)

# ---------------------------------------------------------------- system economies
complete = Xi[COMPONENTS].notna().groupby(level="area_code").all().all(axis=1)
complete_before = (~missing_before).groupby(level="area_code").all().all(axis=1)
system = sorted(complete[complete].index)
joined = sorted(set(system) - set(complete_before[complete_before].index))
print(f"system economies: {len(system)} (joined through structural zeros: {joined})")

# Published territory totals of system economies must be complete (they are system series).
ter_gaps = Xi.loc[system, ["TER_DOM", "TER_INT"]].isna().groupby(level="area_code").sum()
ter_gaps = ter_gaps[(ter_gaps > 0).any(axis=1)]
if len(ter_gaps):
    raise SystemExit(f"Published TER_DOM/TER_INT incomplete for system economies:\n{ter_gaps}")

# Log of imputed zeros: one row per (zero, confirming total).
zlog = []
for k in COMPONENTS:
    for (a, p) in zero_mask.index[zero_mask[k]]:
        for T in contained_in[k]:
            zlog.append({"unit": a, "measure": k, "letter": LETTER[k], "period": p, "confirming_total": T,
                         "total_published_t": X.at[(a, p), T], "sum_other_components_t": present_sum[T][(a, p)],
                         "in_system": a in system})
zlog = pd.DataFrame(zlog, columns=["unit", "measure", "letter", "period", "confirming_total",
                                   "total_published_t", "sum_other_components_t", "in_system"])
zlog["abs_diff_t"] = (zlog.total_published_t - zlog.sum_other_components_t).abs()
zlog = zlog.sort_values(["in_system", "unit", "measure", "period"], ascending=[False, True, True, True])
zlog.to_csv(out_path(RESULTS / "structural_zeros.csv", args.overwrite), index=False, float_format="%.4f")
n_z = zlog.drop_duplicates(["unit", "measure", "period"])
print(f"structural zeros imputed: {len(n_z)} cells ({int(n_z.in_system.sum())} in system economies)")

# Why is Switzerland (or any other large economy) not in the system? Count remaining gaps.
gaps_after = Xi[COMPONENTS].isna().groupby(level="area_code").sum()

# ---------------------------------------------------------------- unit-level panels (co2_t)
def unit_frame(df):
    """Add RES_TOTAL_SYS and rename the published total, for a frame with component columns."""
    out = df[COMPONENTS].copy()
    out["RES_TOTAL_SYS"] = df[[A, B, D, E, F]].sum(axis=1, min_count=5)
    out["TER_DOM"], out["TER_INT"] = df["TER_DOM"], df["TER_INT"]
    out["RES_TOTAL_PUB"] = df["RES_TOTAL"]
    return out


eco = unit_frame(Xi.loc[system])                                             # index (area, period)
wld_raw = (m[(m.area_code == WORLD) & m.period.isin(months) & m.component.isin(cols_all)]
           .pivot(index="period", columns="component", values="co2_t").reindex(index=months, columns=cols_all))
if wld_raw[COMPONENTS + ["TER_DOM", "TER_INT"]].isna().any().any():
    raise SystemExit("World series incomplete in the panel window")
wld = unit_frame(wld_raw)
# Residual = World - sum over system economies (RES_TOTAL_PUB only where every system economy publishes it).
eco_sum = eco.groupby(level="period").sum(min_count=len(system))
res = wld - eco_sum.reindex(months)
res["RES_TOTAL_PUB"] = wld["RES_TOTAL_PUB"] - eco.groupby(level="period").RES_TOTAL_PUB.sum(min_count=len(system))
delta = (wld[E] + wld[F] - wld[G]).rename("DELTA_INT")


def to_long(frame, unit_type):
    """frame: index (unit, period), measure columns (co2_t) -> long rows."""
    lg = frame.stack(future_stack=True).rename("co2_t").reset_index()
    lg.columns = ["unit", "period", "measure", "co2_t"]
    lg["unit_type"] = unit_type
    return lg


eco_long = to_long(eco, "economy")
res_long = to_long(res.assign(unit=RESIDUAL).set_index("unit", append=True).swaplevel(), "residual")
wld_long = to_long(wld.assign(unit=WORLD).set_index("unit", append=True).swaplevel(), "world")
dlt_long = to_long(delta.to_frame().assign(unit=WORLD).set_index("unit", append=True).swaplevel(), "world")
panel = pd.concat([eco_long, res_long, wld_long, dlt_long], ignore_index=True)
panel["days_in_month"] = panel.period.map(days)
panel["co2_t_per_day"] = panel.co2_t / panel.days_in_month
panel["letter"] = panel.measure.map(LETTER)
panel["basis"] = panel.measure.map(BASIS)
panel["in_constraints"] = panel.measure != "RES_TOTAL_PUB"
panel["flag_sep2025"] = panel.period == SEP2025
zkeys = set(zip(zlog.unit, zlog.measure, zlog.period))
panel["flag_structural_zero"] = [(u, k, p) in zkeys for u, k, p in zip(panel.unit, panel.measure, panel.period)]
unit_order = {u: i for i, u in enumerate(system + [RESIDUAL, WORLD])}
meas_order = {k: i for i, k in enumerate(SYS_MEASURES + ["DELTA_INT", "RES_TOTAL_PUB"])}
panel = (panel.assign(_u=panel.unit.map(unit_order), _k=panel.measure.map(meas_order))
         .sort_values(["_u", "_k", "period"]).drop(columns=["_u", "_k"]).reset_index(drop=True))
panel = panel[["unit", "unit_type", "measure", "letter", "basis", "in_constraints", "period", "days_in_month",
               "co2_t", "co2_t_per_day", "flag_sep2025", "flag_structural_zero"]]
sys_rows = panel[panel.in_constraints]
assert sys_rows.co2_t_per_day.notna().all(), "missing values in constraint-system series"
assert len(sys_rows) == ((len(system) + 2) * len(SYS_MEASURES) + 1) * len(months)

panel.to_parquet(out_path(CLEAN / f"panel_system_{SFX}.parquet", args.overwrite), index=False)
wide = sys_rows.assign(col=sys_rows.unit + "|" + sys_rows.measure).pivot(
    index="period", columns="col", values="co2_t_per_day")
wide = wide[[f"{u}|{k}" for u in unit_order for k in SYS_MEASURES] + [f"{WORLD}|DELTA_INT"]]
wide.columns.name = None
wide.to_parquet(out_path(CLEAN / f"panel_system_wide_{SFX}.parquet", args.overwrite))
neg = sys_rows[sys_rows.co2_t < 0]
print("negative values in system series:", "none" if neg.empty else neg.groupby(["unit", "measure"]).size())

# ---------------------------------------------------------------- system economies and coverage
y25 = panel[panel.period.str.startswith("2025")].groupby(["unit", "measure"]).co2_t.sum().unstack()
y25["TER_TOTAL"] = y25.TER_DOM + y25.TER_INT
w25 = y25.loc[WORLD]
se = pd.DataFrame({"area_code": system}).merge(
    areas[["area_code", "area_name", "oecd_member"]], on="area_code", how="left")
s = y25.loc[system]
se["res_total_sys_2025_t"] = s.RES_TOTAL_SYS.values
se["ter_total_2025_t"] = s.TER_TOTAL.values
se["share_world_residence_2025"] = (s.RES_TOTAL_SYS / w25.RES_TOTAL_SYS).values
se["share_world_territory_2025"] = (s.TER_TOTAL / w25.TER_TOTAL).values
se["share_world_ter_dom_2025"] = (s.TER_DOM / w25.TER_DOM).values
se["share_world_ter_int_2025"] = (s.TER_INT / w25.TER_INT).values
se["joined_through_structural_zeros"] = se.area_code.isin(joined)
se["n_structural_zeros"] = se.area_code.map(n_z[n_z.in_system].groupby("unit").size()).fillna(0).astype(int)
se = se.sort_values("share_world_residence_2025", ascending=False).reset_index(drop=True)
se.to_csv(out_path(RESULTS / f"system_economies_{SFX}.csv", args.overwrite), index=False, float_format="%.6g")

cov = (y25.loc[system].sum() / w25).rename("share_2025").to_frame()
cov = cov.loc[SYS_MEASURES + ["TER_TOTAL", "RES_TOTAL_PUB"]]
cov.index.name = "measure"
cov.to_csv(out_path(RESULTS / f"coverage_2025_{SFX}.csv", args.overwrite), float_format="%.6f")
print("2025 coverage of World by measure:\n", cov.share_2025.round(4).to_string())

# ---------------------------------------------------------------- discrepancy RES_TOTAL_PUB vs SYS
disc = panel[panel.measure.isin(["RES_TOTAL_PUB", "RES_TOTAL_SYS"])].pivot(
    index=["unit", "unit_type", "period"], columns="measure", values="co2_t").reset_index()
disc = disc.rename(columns={"RES_TOTAL_PUB": "published_t", "RES_TOTAL_SYS": "component_sum_t"})
disc["abs_diff_t"] = disc.published_t - disc.component_sum_t
disc["rel_diff"] = disc.abs_diff_t / disc.published_t
disc["abs_diff_t"] = disc.abs_diff_t.abs()
disc = disc.assign(_u=disc.unit.map(unit_order)).sort_values(["_u", "period"]).drop(columns="_u")
disc.to_csv(out_path(RESULTS / "res_total_discrepancy.csv", args.overwrite), index=False, float_format="%.6g")

# ---------------------------------------------------------------- outside-system panel
core = summary[(summary.vessel == "ALL_VESSELS") & summary.core_series & summary.area_code.isin(economies)
               & ~summary.area_code.isin(system) & summary.component.isin(["RES_TOTAL", "TER_DOM", "TER_INT"])]
out = m.merge(core[["area_code", "component"]], on=["area_code", "component"])
out = out[out.period.isin(months)].copy()
out["measure"] = out.component.replace({"RES_TOTAL": "RES_TOTAL_PUB"})
out["days_in_month"] = out.period.map(days)
out["co2_t_per_day"] = out.co2_t / out.days_in_month
out = out.rename(columns={"area_code": "unit"}).assign(unit_type="outside", basis=lambda d: d.measure.map(BASIS),
                                                      flag_sep2025=lambda d: d.period == SEP2025)
out = out[["unit", "unit_type", "measure", "basis", "period", "days_in_month", "co2_t", "co2_t_per_day",
           "flag_sep2025"]].sort_values(["unit", "measure", "period"]).reset_index(drop=True)
n_out = out.groupby(["unit", "measure"]).size()
assert (n_out == len(months)).all() and out.co2_t.notna().all()
assert ("CHE", "RES_TOTAL_PUB") in n_out.index, "Switzerland RES_TOTAL_PUB must be an outside-system series"
out.to_parquet(out_path(CLEAN / f"panel_outside_{SFX}.parquet", args.overwrite), index=False)
print(f"outside-system series: {len(n_out)} "
      f"({n_out.index.get_level_values('measure').value_counts().to_dict()})")

# ---------------------------------------------------------------- real-time hold-out (RES_TOTAL_PUB)
rt_months = pd.period_range(RT_START, RT_END, freq="M").astype(str).tolist()
rt = m[(m.component == "RES_TOTAL") & m.period.isin(rt_months)]
outside_res = sorted(core.loc[core.component == "RES_TOTAL", "area_code"])
parts = [rt[rt.area_code.isin(system)].assign(unit_type="economy"),
         rt[rt.area_code == WORLD].assign(unit_type="world"),
         rt[rt.area_code.isin(outside_res)].assign(unit_type="outside")]
rt_eco = parts[0]
if rt_eco.groupby("area_code").co2_t.count().reindex(system).fillna(0).eq(len(rt_months)).all():
    r = (parts[1].set_index("period").co2_t - rt_eco.groupby("period").co2_t.sum()).rename("co2_t").reset_index()
    parts.append(r.assign(area_code=RESIDUAL, unit_type="residual"))
else:
    print("RES_TOTAL_PUB incomplete for some system economies in the real-time months; Residual not built")
rtp = pd.concat(parts, ignore_index=True)[["area_code", "unit_type", "period", "co2_t"]].rename(
    columns={"area_code": "unit"})
rtp.insert(2, "measure", "RES_TOTAL_PUB")
rtp["days_in_month"] = pd.PeriodIndex(rtp.period, freq="M").days_in_month
rtp["co2_t_per_day"] = rtp.co2_t / rtp.days_in_month
rtp.to_parquet(out_path(CLEAN / f"realtime_res_total_pub_{RT_START}_{RT_END}_{SFX}.parquet", args.overwrite),
               index=False)

# ---------------------------------------------------------------- console summary for the report
big_out = (summary[(summary.vessel == "ALL_VESSELS") & (summary.component == "RES_TOTAL")
                   & summary.area_code.isin(economies) & ~summary.area_code.isin(system)]
           .nlargest(8, "co2_t_2025").area_code)
print("remaining component gaps (months, after structural zeros) for the largest excluded economies:\n",
      gaps_after.loc[big_out].to_string())
print(f"panel rows: {len(panel)} | wide: {wide.shape} | outside rows: {len(out)} | real-time rows: {len(rtp)}")
log.finish(f"{len(system)} system economies ({len(joined)} via structural zeros); {len(n_z)} structural zeros")
