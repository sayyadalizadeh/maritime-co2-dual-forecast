"""
constraints.py: linear constraint matrix C of the forecasting system.

The vector y_t stacks, for month t, the ten measures (A to G, the residence total,
TER_DOM, TER_INT) of every unit of the system: the system economies, the Residual node
and World, in the column order of data/clean/panel_system_wide_<suffix>.parquet. From
panel v2 on, the residence total is RES_TOTAL_SYS (the component sum A+B+D+E+F) and the vector
ends with the World-level discrepancy node DELTA_INT (E+F-G at World level, so that the World balance holds
exactly). C is built so that C @ y_t = 0 encodes the accounting identities:

  within      for every unit u:   RES_TOTAL(_SYS) - (A+B+D+E+F) = 0
                                   TER_DOM   - (A+C)       = 0
                                   TER_INT   - (D+G)       = 0
  aggregation for every measure k: k(World) - sum_c k(c) - k(Residual) = 0
  world_bal   at World level:      B - C = 0   and   E + F - G (- DELTA_INT) = 0

Redundancy: the within-unit rows of World follow from the within-unit rows of the
economies and the Residual plus the aggregation rows. They are kept in C (so that
every identity can be checked on the data) and flagged in the row legend; a
full-row-rank version without the flagged rows is saved as well, for methods that
need C to have full row rank.

Outputs (results/, SFX = --suffix; v1 outputs carry no suffix):
  C_matrix_SFX.npz, C_matrix_rows_SFX.csv, C_matrix_cols_SFX.csv   C and its legends
  C_matrix_fullrank_SFX.npz                         C without the redundant rows (full row rank)
  constraint_residuals_history_SFX.csv              C @ y_t for every month and row
  constraint_residuals_summary_SFX.csv              median / max abs and rel residual by type
  identity_violations_gt1pct_SFX.csv                units whose within-unit identities miss
                                                    by more than 1% in any month

Usage:
  python src/constraints.py [--suffix v2] [--clean data/clean] [--results results] [--overwrite]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from runlog import RunLog, out_path

COMPONENTS = ["RES_DOM_IN", "RES_DOM_OUT", "NRES_DOM_IN", "RES_INT_FROM",
              "RES_INT_TO", "RES_INT_OUT", "NRES_INT_FROM"]
A, B, C_, D, E, F, G = COMPONENTS
WORLD, RESIDUAL = "W", "RESIDUAL"
DELTA = "DELTA_INT"
REL_THRESHOLD = 0.01        # 1%: threshold for listing identity misses
SEP2025 = "2025-09"

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--suffix", default="v2", help="panel version (v1 = first panel version)")
ap.add_argument("--clean", default="data/clean")
ap.add_argument("--results", default="results")
ap.add_argument("--overwrite", action="store_true", help="allow overwriting existing outputs")
args = ap.parse_args()
log = RunLog("constraints.py", vars(args))
CLEAN, RESULTS, SFX = Path(args.clean), Path(args.results), args.suffix
# v1 outputs were written without a suffix
tag = "" if SFX == "v1" else f"_{SFX}"


# ---------------------------------------------------------------- series (columns of C)
wide = pd.read_parquet(CLEAN / f"panel_system_wide_{SFX}.parquet")
cols = pd.DataFrame([c.split("|") for c in wide.columns], columns=["unit", "measure"])
cols.insert(0, "col", range(len(cols)))
units = list(dict.fromkeys(cols.unit))                 # system economies, RESIDUAL, W
RES = "RES_TOTAL_SYS" if "RES_TOTAL_SYS" in set(cols.measure) else "RES_TOTAL"
MEASURES = COMPONENTS + [RES, "TER_DOM", "TER_INT"]
WITHIN = {  # total: (parts, row type)
    RES: ([A, B, D, E, F], f"within_{RES}"),
    "TER_DOM": ([A, C_], "within_TER_DOM"),
    "TER_INT": ([D, G], "within_TER_INT"),
}
has_delta = (WORLD, DELTA) in set(zip(cols.unit, cols.measure))
economies = [u for u in units if u not in (RESIDUAL, WORLD)]
idx = {(u, k): i for i, u, k in cols[["col", "unit", "measure"]].itertuples(index=False)}
cols["unit_type"] = np.where(cols.unit == WORLD, "world", np.where(cols.unit == RESIDUAL, "residual", "economy"))

# ---------------------------------------------------------------- rows of C
rows, entries = [], []   # entries: (row, col, value)


def add_row(kind, unit, measure, terms, scale_col, text):
    """terms: list of (col, coefficient). scale_col: column whose |value| scales the relative residual."""
    r = len(rows)
    rows.append({"row": r, "type": kind, "unit": unit, "measure": measure,
                 "scale_series": cols.loc[scale_col, "unit"] + "|" + cols.loc[scale_col, "measure"],
                 "_scale_col": scale_col, "equation": text})
    entries.extend((r, c, v) for c, v in terms)


for u in units:
    for total, (parts, kind) in WITHIN.items():
        terms = [(idx[u, total], 1.0)] + [(idx[u, p], -1.0) for p in parts]
        add_row(kind, u, total, terms, idx[u, total], f"{total} - ({' + '.join(parts)}) = 0")
for k in MEASURES:
    terms = [(idx[WORLD, k], 1.0)] + [(idx[u, k], -1.0) for u in economies + [RESIDUAL]]
    add_row("aggregation", WORLD, k, terms, idx[WORLD, k], f"{k}(W) - sum_c {k}(c) - {k}(Residual) = 0")
add_row("world_balance_B_C", WORLD, f"{B},{C_}", [(idx[WORLD, B], 1.0), (idx[WORLD, C_], -1.0)],
        idx[WORLD, B], "B(W) - C(W) = 0")
if has_delta:
    add_row("world_balance_EF_G", WORLD, f"{E},{F},{G},{DELTA}",
            [(idx[WORLD, E], 1.0), (idx[WORLD, F], 1.0), (idx[WORLD, G], -1.0), (idx[WORLD, DELTA], -1.0)],
            idx[WORLD, G], "E(W) + F(W) - G(W) - DELTA_INT = 0")
else:
    add_row("world_balance_EF_G", WORLD, f"{E},{F},{G}",
            [(idx[WORLD, E], 1.0), (idx[WORLD, F], 1.0), (idx[WORLD, G], -1.0)],
            idx[WORLD, G], "E(W) + F(W) - G(W) = 0")

rows = pd.DataFrame(rows)
r_, c_, v_ = zip(*entries)
Cmat = sp.csr_matrix((v_, (r_, c_)), shape=(len(rows), len(cols)))

# ---------------------------------------------------------------- rank and redundant rows
# Greedy pass: a row is redundant if it does not increase the rank of the rows kept so far.
# World within-unit rows are tested last, so that they (and not aggregation rows) are the ones flagged.
dense = Cmat.toarray()
rank = int(np.linalg.matrix_rank(dense))
world_within = rows.type.str.startswith("within") & (rows.unit == WORLD)
order = list(rows.index[~world_within]) + list(rows.index[world_within])
kept, cur = [], 0
for r in order:
    rk = int(np.linalg.matrix_rank(dense[kept + [r]]))
    if rk > cur:
        kept.append(r)
        cur = rk
rows["redundant"] = ~rows.row.isin(kept)
assert cur == rank
sv = np.linalg.svd(dense, compute_uv=False)
print(f"series: {Cmat.shape[1]} | constraints: {Cmat.shape[0]} | rank: {rank} | "
      f"redundant rows: {rows.redundant.sum()} -> {list(rows.loc[rows.redundant, 'equation'] + ' [' + rows.loc[rows.redundant, 'unit'] + ']')}")
print(f"singular values: largest {sv.max():.3f}, smallest nonzero {sv[sv > 1e-10].min():.3f}")

C_full = Cmat[sorted(kept)]
sp.save_npz(out_path(RESULTS / f"C_matrix{tag}.npz", args.overwrite), Cmat)
sp.save_npz(out_path(RESULTS / f"C_matrix_fullrank{tag}.npz", args.overwrite), C_full)
rows.drop(columns="_scale_col").to_csv(out_path(RESULTS / f"C_matrix_rows{tag}.csv", args.overwrite), index=False)
cols.to_csv(out_path(RESULTS / f"C_matrix_cols{tag}.csv", args.overwrite), index=False)

# ---------------------------------------------------------------- residuals on the historical data
Y = wide.to_numpy()                                  # months x series (t/day)
R = Y @ Cmat.T.toarray()                             # months x constraints
S = np.abs(Y[:, rows._scale_col.to_numpy()])         # scale for relative residuals
with np.errstate(divide="ignore", invalid="ignore"):
    REL = np.where(S > 0, np.abs(R) / S, np.where(R == 0, 0.0, np.inf))

hist = pd.DataFrame({
    "period": np.repeat(wide.index.to_numpy(), len(rows)),
    "row": np.tile(rows.row.to_numpy(), len(wide)),
    "residual_t_per_day": R.ravel(), "scale_t_per_day": S.ravel(), "rel_residual": REL.ravel()})
hist = hist.merge(rows[["row", "type", "unit", "measure"]], on="row")
hist["flag_sep2025"] = hist.period == SEP2025
hist.to_csv(out_path(RESULTS / f"constraint_residuals_history{tag}.csv", args.overwrite), index=False, float_format="%.6g")


def summarise(h):
    a, rel = h.residual_t_per_day.abs(), h.rel_residual
    return pd.Series({
        "n_cells": len(h),
        "median_abs_t_per_day": a.median(), "max_abs_t_per_day": a.max(),
        "median_rel": rel.median(), "max_rel": rel.max(),
        "n_rel_gt_1e-4": int((rel > 1e-4).sum()), "n_rel_gt_1pct": int((rel > REL_THRESHOLD).sum())})


hist["unit_group"] = np.where(hist.unit == WORLD, "world", np.where(hist.unit == RESIDUAL, "residual", "economies"))
summ = []
for sample, h0 in [("all_months", hist), ("excl_2025-09", hist[~hist.flag_sep2025])]:
    g = h0.groupby(["type", "unit_group"]).apply(summarise, include_groups=False).reset_index()
    summ.append(g.assign(sample=sample))
summ = pd.concat(summ)[["sample", "type", "unit_group", "n_cells", "median_abs_t_per_day", "max_abs_t_per_day",
                        "median_rel", "max_rel", "n_rel_gt_1e-4", "n_rel_gt_1pct"]]
summ.to_csv(out_path(RESULTS / f"constraint_residuals_summary{tag}.csv", args.overwrite), index=False, float_format="%.4g")
with pd.option_context("display.width", 200, "display.max_columns", 20):
    print(summ[summ["sample"] == "all_months"].to_string(index=False))

# ---------------------------------------------------------------- within-unit misses > 1%
w = hist[hist.type.str.startswith("within") & (hist.rel_residual > REL_THRESHOLD)]
viol = (w.groupby(["unit", "type"])
        .agg(n_months=("period", "size"), max_rel=("rel_residual", "max"),
             max_abs_t_per_day=("residual_t_per_day", lambda x: x.abs().max()),
             months=("period", lambda x: " ".join(sorted(x))))
        .reset_index().sort_values(["unit", "type"]))
viol.to_csv(out_path(RESULTS / f"identity_violations_gt1pct{tag}.csv", args.overwrite), index=False, float_format="%.4g")
print(f"units with a within-unit miss > 1%: {viol.unit.nunique()}")
with pd.option_context("display.width", 250, "display.max_colwidth", 120):
    print(viol.to_string(index=False))

# ---------------------------------------------------------------- rounding check
# A residual counts as rounding if it is within the structural-zero tolerance: 1 t per month (expressed per
# day with the shortest month, 1/28 t/day) or 0.01% of the row's total.
ABS_TOL, REL_TOL = 1 / 28, 1e-4
hist["beyond_rounding"] = (hist.residual_t_per_day.abs() > ABS_TOL) & (hist.rel_residual > REL_TOL)
n_fail = hist.groupby("type").beyond_rounding.sum()
print("cells beyond rounding by constraint type:\n", n_fail.to_string())
log.finish(f"C {Cmat.shape}, rank {rank}; cells beyond rounding: {int(n_fail.sum())}")
if n_fail.sum() > 0:
    sys.exit(f"{int(n_fail.sum())} constraint cells fail beyond rounding: stop and report.")
