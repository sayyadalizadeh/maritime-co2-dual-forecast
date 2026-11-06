"""
evaluate_fm.py: base-model comparison including the zero-shot foundation models.

Models: the six models of base_forecasts.py (results/base_forecasts_v2) and the foundation models
Chronos-Bolt, TimesFM (3.0) and TimesFM25 (results/base_forecasts_fm). Output with all models:
results/eval_fm_v2; without TimesFM25 (--models without TimesFM25 --out results/eval_fm): results/eval_fm.
Point forecast: the mean of the statsforecast models and LightGBM, the median of the foundation models.

Metrics per series, model, window and horizon (evalcore; scales from the in-sample seasonal-naive
errors up to each origin):
  MASE, RMSSE
  sCRPS3   mean pinball loss over the quantiles shared by all probabilistic models (0.1, 0.5, 0.9),
           scaled like MASE. Statistical models: 0.1 = lo_80, 0.9 = hi_80, 0.5 = point forecast
           (their Gaussian intervals are symmetric). LightGBM has no quantiles.
  cov80    share of targets inside [q0.1, q0.9]
Summaries are weighted within one level at a time (evalcore.summarise_levels).

Tests (main window), AutoETS as the reference:
  Diebold-Mariano (HLN), absolute-error loss, h = 1, 3, 6, 12, per series; shares of series
  significantly better / worse at 5%, for all series and by level.
  Model Confidence Set (90%) over all models, loss per origin = mean scaled absolute error,
  per horizon group; computed over all series and over system economies.

Outputs: results/eval_fm/
Usage:
  python src/evaluate_fm.py [--overwrite]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from arch.bootstrap import MCS

from evalcore import (HORIZON_GROUPS, WINDOWS, add_scaled_errors, dm_hln, load_history, origin_scales,
                      per_series_metrics, summarise_levels)
from runlog import RunLog, out_path

STAT = ["SeasonalNaive", "RandomWalkWithDrift", "AutoETS", "AutoTheta", "AutoARIMA", "LightGBM"]
FM = ["ChronosBolt", "TimesFM", "TimesFM25"]
ALL_MODELS = STAT + FM
REF = "AutoETS"
QS = {0.1: "q10", 0.5: "q50", 0.9: "q90"}
SEED = 42


def load_model(res, m):
    """Forecasts of one model with columns unique_id, origin, target, horizon, y, mean, q10, q50, q90."""
    if m in FM:
        f = pd.read_parquet(res / "base_forecasts_fm" / f"model={m}" / "forecasts.parquet")
        f = f.rename(columns={"q0.1": "q10", "q0.5": "q50", "q0.9": "q90"})
        f["mean"] = f["q50"]                                     # point forecast = median
    else:
        f = pd.read_parquet(res / "base_forecasts_v2" / f"model={m}" / "forecasts.parquet")
        f["q10"], f["q90"], f["q50"] = f["lo_80"], f["hi_80"], f["mean"]
    return f[["unique_id", "origin", "target", "horizon", "y", "mean", "q10", "q50", "q90"]].assign(model=m)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean", default="data/clean")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/eval_fm_v2", help="without TimesFM25: results/eval_fm")
    ap.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    ap.add_argument("--mcs-reps", type=int, default=10000)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    log = RunLog("evaluate_fm.py", vars(args))
    RES, OUT = Path(args.results), Path(args.out)
    MODELS = args.models

    def save(df, name):
        df.to_csv(out_path(OUT / name, args.overwrite), index=False, float_format="%.6g")

    meta = pd.read_parquet(RES / "base_forecasts_v2" / "series_meta.parquet")
    sc = origin_scales(load_history(args.clean))
    ps_all, loss = [], []
    for m in MODELS:
        f = add_scaled_errors(load_model(RES, m), sc)
        pin = sum(np.maximum(q * (f.y - f[c]), (q - 1) * (f.y - f[c])) for q, c in QS.items())
        f["scrps3"] = pin / len(QS) / f.scale1
        f["cov80"] = ((f.y >= f.q10) & (f.y <= f.q90)).astype(float).where(f.q10.notna())
        ps = per_series_metrics(f, ["model"])
        extra = []
        for wname, wfun in WINDOWS.items():
            fw = f[wfun(f.target)]
            for keys, hcol in [(["unique_id", "model"], None), (["unique_id", "model", "horizon"], "horizon")]:
                e = fw.groupby(keys).agg(sCRPS3=("scrps3", "mean"), cov80=("cov80", "mean")).reset_index()
                e["horizon"] = e.horizon.astype(str) if hcol else "all"
                extra.append(e.assign(window=wname))
        ps = ps.merge(pd.concat(extra), on=["unique_id", "model", "window", "horizon"])
        ps_all.append(ps)
        loss.append(f.loc[WINDOWS["main"](f.target), ["model", "unique_id", "origin", "horizon", "ae", "sae"]])
        print(f"{m}: {len(f)} forecasts scored")
    ps = pd.concat(ps_all, ignore_index=True).merge(meta, on="unique_id")
    ps.to_parquet(out_path(OUT / "metrics_per_series.parquet", args.overwrite), index=False)
    summ = summarise_levels(ps, ["model"], metrics=("MASE", "RMSSE", "sCRPS3", "cov80"))
    save(summ, "summary_weighted.csv")
    save(summarise_levels(ps, ["model"], metrics=("MASE", "cov80"),
                          horizons=[str(h) for h in range(1, 13)]), "summary_weighted_by_horizon.csv")

    # ---- Diebold-Mariano against AutoETS
    L = pd.concat(loss, ignore_index=True).merge(meta[["unique_id", "unit_type"]], on="unique_id")
    P = L[L.horizon.isin([1, 3, 6, 12])].pivot_table(index=["unique_id", "horizon", "origin"], columns="model",
                                                    values="ae")
    rows = []
    for (uid, h), g in P.groupby(level=["unique_id", "horizon"]):
        for m in MODELS:
            if m == REF:
                continue
            stat, p = dm_hln(g[m] - g[REF], int(h))
            rows.append({"unique_id": uid, "horizon": int(h), "model": m, "dm_stat": stat, "p_value": p,
                         "mean_loss_diff": (g[m] - g[REF]).mean()})
    dm = pd.DataFrame(rows).merge(meta[["unique_id", "unit_type"]], on="unique_id")
    save(dm, "dm_vs_autoets_per_series.csv")
    dm["better"] = (dm.p_value < 0.05) & (dm.mean_loss_diff < 0)
    dm["worse"] = (dm.p_value < 0.05) & (dm.mean_loss_diff > 0)
    s_all = dm.groupby(["model", "horizon"]).agg(n=("unique_id", "size"), share_better=("better", "mean"),
                                                 share_worse=("worse", "mean")).reset_index().assign(units="all")
    s_lvl = dm.groupby(["unit_type", "model", "horizon"]).agg(n=("unique_id", "size"), share_better=("better", "mean"),
                                                              share_worse=("worse", "mean")).reset_index()
    save(pd.concat([s_all, s_lvl.rename(columns={"unit_type": "units"})]), "dm_vs_autoets_summary.csv")

    # ---- Model Confidence Set over the eight models
    mcs_rows = []
    for scope, d0 in [("all_series", L), ("system_economies", L[L.unit_type == "economy"])]:
        for hg in HORIZON_GROUPS:
            d = d0 if hg == "all" else d0[d0.horizon == int(hg)]
            lo = d.pivot_table(index="origin", columns="model", values="sae", aggfunc="mean")[MODELS]
            mcs = MCS(lo.to_numpy(), size=0.10, reps=args.mcs_reps, method="R", bootstrap="stationary", seed=SEED)
            mcs.compute()
            pv = mcs.pvalues.reindex(range(len(MODELS)))
            for i, m in enumerate(MODELS):
                mcs_rows.append({"scope": scope, "horizon_group": hg, "model": m, "n_origins": len(lo),
                                 "mean_loss": lo[m].mean(), "mcs_pvalue": float(pv.iloc[i, 0]),
                                 "in_mcs_90": i in mcs.included})
    mcs_df = pd.DataFrame(mcs_rows)
    save(mcs_df, "mcs_90.csv")
    print(mcs_df[mcs_df.horizon_group == "all"].to_string(index=False))
    log.finish(f"{len(MODELS)} models")


if __name__ == "__main__":
    main()
