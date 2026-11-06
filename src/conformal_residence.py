"""
conformal_residence.py: coverage of the residence totals under pooled split conformal, next to ACI.

ACI is the method; the under-coverage of the residence totals (FM ACI 80%: 0.751, 95%: 0.873; supplementary_analyses.py,
coverage_block_bootstrap.csv of supplementary_analyses.py) is reported, and the coverage of pooled split conformal for the residence totals is a robustness result.
Nothing is refitted: the conformal intervals are those of evaluate_prob.py, rebuilt
per cell from the stored half-widths (results/eval_prob_exact_nested/conformal_halfwidths.csv, per origin, horizon and
measure group; residence = RES_TOTAL_SYS), the stored track point forecasts (MinT, exact QP) and the seasonal-naive
scale at the origin: [point - q scale, point + q scale], lower bound clipped at zero. The ACI (gamma 0.05) and
uncalibrated intervals are the stored ones (results/prob_main_exact/track=*/aci_intervals.parquet, quantiles.parquet). All
intervals nested (conformal and sample-quantile intervals are nested by construction).
Cells: the 110 RES_TOTAL_SYS series of the system economies, calibration origins 2023-12 ... 2025-06, horizons 1-12,
main window (targets to 2025-12 without 2025-09).
Coverage two ways:
  pooled   hits / cells, with a 95% circular moving-block bootstrap interval over consecutive origins (block 6,
           2,000 resamples, seed 42), as in coverage_block_bootstrap.csv (supplementary_analyses.py; the source of the figures 0.751 and 0.873)
  series   mean over series of the per-series coverage, as in T4; checked against the stored per-series coverage
           (results/eval_prob_exact_nested/intervals_per_series.parquet)
Also by horizon, mean interval width relative to ACI, and the World residence total.

Outputs: results/conformal_residence/coverage_residence_totals.csv, coverage_by_horizon.csv, check_stored.csv
--prob-dir and --eval-dir read the exact evaluation samples (results/prob_main_exact,
results/eval_prob_exact_nested); the conformal intervals do not depend on the samples.

Usage:
  python src/conformal_residence.py
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from evalcore import WINDOWS, load_history, origin_scales
from evaluate_prob import CAL_START, GROUP, nest, track_points
from runlog import RunLog, out_path

SEED, BLOCK, REPS = 42, 6, 2000
INTERVALS = ["uncal_rec", "aci_g0.05", "conformal"]


def mbb_indices(n, block, rng):
    """Circular moving-block bootstrap indices (as in supplementary_analyses.py)."""
    starts = rng.integers(0, n, size=int(np.ceil(n / block)))
    return np.concatenate([np.arange(s, s + block) % n for s in starts])[:n]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean", default="data/clean")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/conformal_residence")
    ap.add_argument("--prob-dir", default=None, help="evaluation samples' intervals (default results/prob_main_exact)")
    ap.add_argument("--eval-dir", default=None, help="evaluate_prob.py --nest output (default results/eval_prob_exact_nested)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    log = RunLog("conformal_residence.py", vars(args))
    RES, OUT = Path(args.results), Path(args.out)

    def save(df, name):
        df.to_csv(out_path(OUT / name, args.overwrite), index=False, float_format="%.6g")

    meta = pd.read_parquet(RES / "base_forecasts_v2" / "series_meta.parquet")
    eco = meta[(meta.unit_type == "economy") & (meta.measure == "RES_TOTAL_SYS")].unique_id.tolist()
    keep = eco + ["W|RES_TOTAL_SYS"]
    Y = load_history(args.clean)
    y_long = Y.stack().rename("y").reset_index().rename(columns={"period": "target", "uid": "unique_id"})
    scales = origin_scales(Y)[["unique_id", "origin", "scale1"]]
    EV = Path(args.eval_dir) if args.eval_dir else RES / "eval_prob_exact_nested"
    hw = pd.read_csv(EV / "conformal_halfwidths.csv")
    stored = pd.read_parquet(EV / "intervals_per_series.parquet")

    cells = []
    for tr in ["FM", "STAT"]:
        P = (Path(args.prob_dir) if args.prob_dir else RES / "prob_main_exact") / f"track={tr}"
        pts = track_points(RES, tr)
        pts = pts[pts.unique_id.isin(keep) & (pts.origin >= CAL_START) & (pts.horizon <= 12)]
        a = pd.read_parquet(P / "aci_intervals.parquet", filters=[("gamma", "==", 0.05), ("unique_id", "in", keep)])
        q = pd.read_parquet(P / "quantiles.parquet", columns=["unique_id", "origin", "horizon", "target", "kind",
                                                             "q0.025", "q0.1", "q0.9", "q0.975"],
                            filters=[("kind", "==", "rec"), ("unique_id", "in", keep)])
        fr = []
        for lv, (ql, qh) in {80: ("q0.1", "q0.9"), 95: ("q0.025", "q0.975")}.items():
            fr.append(q[["unique_id", "origin", "horizon", "target", ql, qh]].rename(columns={ql: "lo", qh: "hi"})
                      .assign(level=lv, interval="uncal_rec"))
            fr.append(a[a.level == lv][["unique_id", "origin", "horizon", "target", "lo", "hi"]]
                      .assign(level=lv, interval="aci_g0.05"))
            h = hw[(hw.track == tr) & (hw.level == lv) & (hw.group == GROUP["RES_TOTAL_SYS"])][["origin", "horizon", "q"]]
            cf = pts.merge(h, on=["origin", "horizon"]).merge(scales, on=["unique_id", "origin"])
            cf["lo"], cf["hi"] = cf.value - cf.q * cf.scale1, cf.value + cf.q * cf.scale1
            fr.append(cf[["unique_id", "origin", "horizon", "target", "lo", "hi"]].assign(level=lv, interval="conformal"))
        iv = pd.concat(fr, ignore_index=True)
        iv = iv[iv.origin >= CAL_START]
        iv["lo"] = iv.lo.clip(lower=0)
        iv = nest(iv, ["unique_id", "origin", "horizon", "interval"])
        iv = iv.merge(y_long, on=["unique_id", "target"])
        iv = iv[WINDOWS["main"](iv.target)]
        iv["hit"] = ((iv.y >= iv.lo) & (iv.y <= iv.hi)).astype(float)
        iv["width"] = iv.hi - iv.lo
        cells.append(iv.assign(track=tr))
    C = pd.concat(cells, ignore_index=True)

    # ---- check: per-series coverage against the stored values of evaluate_prob.py --nest
    E = C[C.unique_id.isin(eco)]
    ps = E.groupby(["track", "interval", "level", "unique_id"]).hit.mean().reset_index()
    st = stored[(stored.window == "main") & (stored.horizon == "all") & stored.unique_id.isin(eco)]
    chk = []
    for lv in (80, 95):
        m = ps[ps.level == lv].merge(st[["track", "interval", "unique_id", f"cov{lv}"]], on=["track", "interval", "unique_id"])
        chk.append({"level": lv, "n": len(m), "max_abs_diff": float((m.hit - m[f"cov{lv}"]).abs().max())})
    chk = pd.DataFrame(chk)
    save(chk, "check_stored.csv")
    assert (chk.max_abs_diff < 1e-9).all() and (chk.n == 2 * len(INTERVALS) * len(eco)).all(), chk

    # ---- coverage of the residence totals: pooled with block bootstrap, and mean over series
    rng = np.random.default_rng(SEED)
    aci_w = E[E.interval == "aci_g0.05"].set_index(["track", "level", "unique_id", "origin", "horizon"]).width
    rows = []
    for (tr, it, lv), d in E.groupby(["track", "interval", "level"]):
        agg = d.groupby("origin").hit.agg(["sum", "size"]).sort_index()
        s, n = agg["sum"].to_numpy(), agg["size"].to_numpy()
        T = len(agg)
        reps = np.empty(REPS)
        for r in range(REPS):
            ii = mbb_indices(T, min(BLOCK, T), rng)
            reps[r] = s[ii].sum() / n[ii].sum()
        dw = d.set_index(["track", "level", "unique_id", "origin", "horizon"]).width
        ratio = (dw / aci_w.loc[dw.index]).replace([np.inf, -np.inf], np.nan)
        rows.append({"track": tr, "interval": it, "level": lv, "n_series": d.unique_id.nunique(), "n_origins": T,
                     "n_cells": len(d), "coverage_pooled": s.sum() / n.sum(), "boot_lo95": np.percentile(reps, 2.5),
                     "boot_hi95": np.percentile(reps, 97.5), "coverage_mean_over_series": d.groupby("unique_id").hit.mean().mean(),
                     "median_width_ratio_to_aci": float(ratio.median())})
    w = C[C.unique_id == "W|RES_TOTAL_SYS"].groupby(["track", "interval", "level"]).hit.agg(["mean", "size"])
    out = pd.DataFrame(rows).merge(w.rename(columns={"mean": "world_coverage", "size": "world_cells"}).reset_index(),
                                   on=["track", "interval", "level"])
    save(out, "coverage_residence_totals.csv")
    bh = E.groupby(["track", "interval", "level", "horizon"]).hit.agg(["mean", "size"]).reset_index() \
        .rename(columns={"mean": "coverage_pooled", "size": "n_cells"})
    save(bh, "coverage_by_horizon.csv")
    pd.set_option("display.width", 220)
    print(chk.to_string(index=False))
    print(out.round(3).to_string(index=False))
    log.finish("conformal coverage of the residence totals")


if __name__ == "__main__":
    main()
