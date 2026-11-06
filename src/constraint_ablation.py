"""
constraint_ablation.py: constraint ablation of the MinT exact-QP reconciliation.

Constraint sets (rows of the full v2 matrix results/C_matrix_v2.npz, DELTA_INT column dropped as in reconcile.py):
  1  full dual allocation: the paper's system (stored runs, results/reconciled_v2_burnin12 and reconciled_v2_fmownw)
  2  without the two world-balance rows (world_balance_B_C, world_balance_EF_G)
  3  aggregation only: the 10 aggregation rows, one per measure (each measure reconciled to the World separately;
     no within-unit identities, no balance)
For sets 2 and 3 the selected rows are reduced to independent rows (pivoted QR, reconcile.independent_rows) and the
dead-series rows are added per origin, exactly as reconcile.origin_constraints does for set 1. Everything else is
reconcile.py unchanged (it is called with its own main(); only load_system and origin_constraints are replaced):
MinT-shrink W (12-month burn-in for AutoETS; own one-step errors for TimesFM 2.5), the exact non-negative QP with
quadprog, 31 main origins, h 1-12.

Mode --run SET --base BASE   reconcile one base under one set (output results/ablation/reconciled_set<SET>/...)
Mode --score                 MASE (main window) of economies (median and weighted, 2022 weights) and of every World series,
                             sets 2 and 3 against set 1; violations of the omitted identities in the set-2/3
                             forecasts: the World residence-territory gap as in T3c,
                             100 x (TER_DOM + TER_INT + r G - RES_TOTAL_SYS) / RES_TOTAL_SYS (World), mean |gap| by
                             horizon; the B(W) = C(W) row; and the within-unit identities (relative violation
                             |k - sum of components| / k, per unit and month, for the three totals).
Usage:
  python src/constraint_ablation.py --run 2 --base AutoETS
  python src/constraint_ablation.py --run 3 --base TimesFM25
  python src/constraint_ablation.py --score
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

import reconcile as R
from evalcore import WINDOWS, load_history, origin_scales
from runlog import RunLog, out_path

OUT = Path("results/ablation")
SETS = {"2": lambda t: ~t.str.startswith("world_balance"), "3": lambda t: t == "aggregation"}
TRACKS = {"FM": ("TimesFM25", "v2_fmownw"), "STAT": ("AutoETS", "v2_burnin12")}
ORIGINS = pd.period_range("2022-12", "2025-06", freq="M").astype(str).tolist()
TOTAL_PARTS = {"RES_TOTAL_SYS": ["RES_DOM_IN", "RES_DOM_OUT", "RES_INT_FROM", "RES_INT_TO", "RES_INT_OUT"],
               "TER_DOM": ["RES_DOM_IN", "NRES_DOM_IN"], "TER_INT": ["RES_INT_FROM", "NRES_INT_FROM"]}


def patched(set_id):
    """load_system / origin_constraints replacements for constraint set 2 or 3."""
    load0 = R.load_system

    def load_system(results, clean):
        s = load0(results, clean)
        RES = Path(results)
        C = sp.load_npz(RES / "C_matrix_v2.npz").toarray()
        cols = pd.read_csv(RES / "C_matrix_cols_v2.csv")
        rows = pd.read_csv(RES / "C_matrix_rows_v2.csv")
        uids_all = (cols.unit + "|" + cols.measure).tolist()
        keep = [uids_all.index(u) for u in s["uids"]]
        sel = SETS[set_id](rows.type).to_numpy()
        C0 = C[sel][:, keep]
        C0 = C0[R.independent_rows(C0)]
        s.update(C0=C0, ef_row=None, set_rows=int(sel.sum()), set_rank=C0.shape[0])
        print(f"constraint set {set_id}: {int(sel.sum())} rows selected, {C0.shape[0]} independent", flush=True)
        return s

    def origin_constraints(sysd, o):
        n, hist = sysd["n"], sysd["hist"]
        pos = hist.index.get_loc(o)
        r = float(sysd["ratio"].iloc[pos - R.R_WINDOW + 1: pos + 1].mean())
        dead = np.flatnonzero((sysd["Yh"].iloc[pos - R.DEAD_WINDOW + 1: pos + 1] == 0).all().to_numpy())
        Co = sysd["C0"].copy()
        A = R.with_zero_rows(Co, dead, n)
        E = np.zeros((len(dead), n))
        E[np.arange(len(dead)), dead] = 1.0
        info = {"origin": o, "r": r, "n_dead": len(dead), "dead_series": " ".join(sysd["uids"][i] for i in dead),
                "rows_C": Co.shape[0], "rows_after_dead": Co.shape[0] + len(dead), "rows_independent": A.shape[0]}
        return A, np.vstack([Co, E]), r, dead, info

    return load_system, origin_constraints


def run(set_id, base, origins, overwrite, out=OUT):
    R.load_system, R.origin_constraints = patched(set_id)
    argv = ["reconcile.py", "--bases", base, "--methods", "MinT", "--tag", f"set{set_id}", "--out", str(out)]
    if base == "TimesFM25":
        argv += ["--fm-w", "own"]
    if origins:
        argv += ["--origins", *origins]
    if overwrite:
        argv.append("--overwrite")
    sys.argv = argv
    R.main()


def mase_summary(f, y_long, sc, meta):
    d = f.merge(y_long, on=["unique_id", "target"]).merge(sc, on=["unique_id", "origin"])
    d = d[WINDOWS["main"](d.target)]
    d["sae"] = (d.y - d.value).abs() / d.scale1
    d.loc[~np.isfinite(d.sae), "sae"] = np.nan
    s = d.groupby("unique_id").sae.mean()
    e = s[s.index.isin(meta.index[meta.unit_type == "economy"])].dropna()
    by_h = d[d.unique_id.isin(e.index)].groupby(["horizon", "unique_id"]).sae.mean().reset_index()
    by_h["w"] = meta.loc[by_h.unique_id, "weight_2022"].to_numpy()
    wh = by_h.groupby("horizon").apply(lambda x: np.average(x.sae, weights=x.w), include_groups=False)
    return s, e, wh


def score(results, clean, overwrite):
    RES = Path(results)
    meta = pd.read_parquet(RES / "base_forecasts_v2" / "series_meta.parquet").set_index("unique_id")
    Y = load_history(clean)
    sc = origin_scales(Y)[["unique_id", "origin", "scale1"]]
    y_long = Y.stack().rename("y").reset_index()
    y_long.columns = ["target", "unique_id", "y"]
    sysd = R.load_system(results, clean)
    ratio = sysd["ratio"]
    r_o = {o: float(ratio.loc[:o].tail(12).mean()) for o in ORIGINS}
    flt = [("method", "==", "MinT"), ("variant", "==", "nn_qp")]
    cols = ["unique_id", "origin", "target", "horizon", "value"]
    summ, world, byh, gap, within, coh = [], [], [], [], [], []
    for tr, (base, tag) in TRACKS.items():
        recs = {"1": pd.read_parquet(RES / f"reconciled_{tag}" / f"base={base}" / "reconciled.parquet", filters=flt)[cols]}
        for s_ in SETS:
            recs[s_] = pd.read_parquet(OUT / f"reconciled_set{s_}" / f"base={base}" / "reconciled.parquet", filters=flt)[cols]
        recs = {k: v[(v.unique_id != "W|DELTA_INT") & v.origin.isin(ORIGINS)] for k, v in recs.items()}
        s1, e1, wh1 = mase_summary(recs["1"], y_long, sc, meta)
        for s_, f in recs.items():
            s, e, wh = mase_summary(f, y_long, sc, meta)
            w = meta.loc[e.index, "weight_2022"]
            summ.append({"track": tr, "set": s_, "median_MASE": e.median(), "weighted_MASE": np.average(e, weights=w),
                         "median_rel_to_set1": e.median() / e1.median(),
                         "weighted_rel_to_set1": np.average(e, weights=w) / np.average(e1, weights=meta.loc[e1.index, "weight_2022"]),
                         "n_series": len(e), "n_cells": len(f)})
            for u in sorted(x for x in s.index if x.startswith("W|")):
                world.append({"track": tr, "set": s_, "unique_id": u, "MASE": s[u], "MASE_set1": s1[u], "rel_to_set1": s[u] / s1[u]})
            byh += [{"track": tr, "set": s_, "horizon": int(h), "weighted_MASE": v, "weighted_MASE_set1": wh1[h]}
                    for h, v in wh.items()]
            # ---- omitted identities
            wv = f[f.unique_id.str.startswith("W|")].pivot_table(index=["origin", "horizon"], columns="unique_id",
                                                                 values="value").reset_index()
            r = wv.origin.map(r_o)
            dev = (wv["W|TER_DOM"] + wv["W|TER_INT"] + r * wv["W|NRES_INT_FROM"] - wv["W|RES_TOTAL_SYS"]) / wv["W|RES_TOTAL_SYS"] * 100
            bc = (wv["W|RES_DOM_OUT"] - wv["W|NRES_DOM_IN"]) / wv["W|RES_TOTAL_SYS"] * 100
            g = pd.DataFrame({"horizon": wv.horizon, "gap": dev.abs(), "bc": bc.abs()}).groupby("horizon").agg(
                mean_abs_res_ter_gap_pct=("gap", "mean"), max_abs_res_ter_gap_pct=("gap", "max"),
                mean_abs_B_minus_C_pct_of_world=("bc", "mean"))
            gap += [{"track": tr, "set": s_, "horizon": int(h), **row} for h, row in g.to_dict("index").items()]
            p = f.assign(unit=f.unique_id.str.split("|").str[0], measure=f.unique_id.str.split("|").str[1]) \
                .pivot_table(index=["unit", "origin", "horizon"], columns="measure", values="value")
            for k, parts in TOTAL_PARTS.items():
                v = (p[k] - p[parts].sum(axis=1)).abs()
                rel = v / p[k].abs().clip(lower=1.0)
                within.append({"track": tr, "set": s_, "total": k, "n_cells": int(v.notna().sum()),
                               "mean_abs_t_per_day": v.mean(), "max_abs_t_per_day": v.max(),
                               "median_rel_pct": 100 * rel.median(), "mean_rel_pct": 100 * rel.mean(),
                               "p95_rel_pct": 100 * rel.quantile(0.95), "max_rel_pct": 100 * rel.max(),
                               "share_cells_rel_gt_1pct": float((rel > 0.01).mean()),
                               "world_mean_rel_pct": 100 * rel.xs("W", level="unit").mean()})
        for s_ in SETS:
            c = pd.read_csv(OUT / f"reconcile_set{s_}" / f"coherence_{base}.csv")
            c = c[(c.method == "MinT") & (c.variant == "nn_qp")]
            coh.append({"track": tr, "set": s_, "max_abs_Cy_own_set": c.max_abs_Cy.max(), "min_y": c.min_y.min()})
    O = OUT
    for name, rows in [("mase_summary.csv", summ), ("mase_world_series.csv", world), ("mase_by_horizon.csv", byh),
                       ("violations_world_gap_by_horizon.csv", gap), ("violations_within_unit.csv", within),
                       ("coherence_own_set.csv", coh)]:
        pd.DataFrame(rows).to_csv(out_path(O / name, overwrite), index=False, float_format="%.6g")
    pd.set_option("display.width", 220)
    print(pd.DataFrame(summ).to_string(index=False))
    print(pd.DataFrame(gap).groupby(["track", "set"])[["mean_abs_res_ter_gap_pct"]].mean().to_string())
    print(pd.DataFrame(within).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean", default="data/clean")
    ap.add_argument("--results", default="results")
    ap.add_argument("--run", choices=list(SETS))
    ap.add_argument("--base", choices=["AutoETS", "TimesFM25"])
    ap.add_argument("--origins", nargs="+", default=None, help="subset of origins (timing only)")
    ap.add_argument("--out", default=str(OUT), help="output root of --run (timing runs: a scratch folder)")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.run:
        run(args.run, args.base, args.origins, args.overwrite, args.out)
    if args.score:
        log = RunLog("constraint_ablation.py", vars(args))
        score(args.results, args.clean, args.overwrite)
        log.finish("score")


if __name__ == "__main__":
    main()
