"""
evaluate_base.py: evaluation of the statistical base forecasts (base_forecasts.py).

Windows (target months):
  main   up to 2025-12, excluding 2025-09
  robA   up to 2025-12, including 2025-09
  robB   2026-01 to 2026-06 (stress test; only horizons 7 to 12 reach these months)

Metrics per series, model and horizon (scaled by the in-sample seasonal-naive errors of the
training data up to each origin, m = 12):
  MASE   mean of |e| / mean|y_t - y_{t-12}|
  RMSSE  sqrt(mean of e^2 / mean (y_t - y_{t-12})^2)
  sCRPS  mean pinball loss over the stored quantiles (0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975;
         the point forecast is used as the 0.5 quantile), scaled like MASE. This is the CRPS
         proxy of the protocol; it equals about half the CRPS.
  cov80, cov95  share of targets inside the 80% and 95% intervals
  sWinkler95    Winkler score of the 95% interval, scaled like MASE
LightGBM has point forecasts only.

Summaries: median and mean across series and a weighted mean (weights: each series' mean
value per day in 2022), by group (all series; residence / territory / components /
discrepancy; measure; unit type) and horizon (1, 3, 6, 12, all).

Tests (main window):
  Diebold-Mariano with the Harvey-Leybourne-Newbold correction, absolute-error loss, per series,
  each model against AutoETS at horizons 1, 3, 6, 12; share of series significantly better and
  worse at 5% (two-sided).
  Model Confidence Set (90%, arch, stationary bootstrap, range statistic), loss per origin =
  mean scaled absolute error across series, per horizon group (1, 3, 6, 12, all).

Outputs: results/eval_base_v2/*.csv and metrics_per_series.parquet

Usage:
  python src/evaluate_base.py [--forecasts results/base_forecasts_v2] [--out results/eval_base_v2] [--overwrite]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from arch.bootstrap import MCS
from scipy import stats

from runlog import RunLog, out_path

SEASON = 12
QUANTILES = {0.025: "lo_95", 0.1: "lo_80", 0.25: "lo_50", 0.5: "mean", 0.75: "hi_50", 0.9: "hi_80", 0.975: "hi_95"}
HORIZONS = [1, 3, 6, 12]
BENCH = "AutoETS"
SEED = 42
WINDOWS = {
    "main": lambda t: (t <= "2025-12") & (t != "2025-09"),
    "robA": lambda t: t <= "2025-12",
    "robB": lambda t: (t >= "2026-01") & (t <= "2026-06"),
}
MEASURE_GROUP = {"RES_TOTAL_SYS": "residence", "RES_TOTAL_PUB": "residence", "TER_DOM": "territory",
                 "TER_INT": "territory", "DELTA_INT": "discrepancy"}

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--clean", default="data/clean")
ap.add_argument("--suffix", default="v2")
ap.add_argument("--forecasts", default="results/base_forecasts_v2")
ap.add_argument("--out", default="results/eval_base_v2")
ap.add_argument("--mcs-reps", type=int, default=10000)
ap.add_argument("--overwrite", action="store_true")
args = ap.parse_args()
log = RunLog("evaluate_base.py", vars(args))
FC, OUT = Path(args.forecasts), Path(args.out)


def save(df, name, **kw):
    df.to_csv(out_path(OUT / name, args.overwrite), index=False, float_format="%.5g", **kw)


# ---------------------------------------------------------------- data
meta = pd.read_parquet(FC / "series_meta.parquet")
meta["group"] = meta.measure.map(MEASURE_GROUP).fillna("component")
f = pd.concat([pd.read_parquet(p).assign(model=p.parent.name.split("=", 1)[1])   # one file per model
               for p in sorted(FC.glob("model=*/forecasts.parquet"))], ignore_index=True)
models = sorted(f.model.unique(), key=lambda m: ["SeasonalNaive", "RandomWalkWithDrift", "AutoETS", "AutoTheta",
                                                 "AutoARIMA", "LightGBM"].index(m))
print("models:", models, "| rows:", len(f))

# Actual history (per day) of every series, wide: months x series.
sys_long = pd.read_parquet(Path(args.clean) / f"panel_system_{args.suffix}.parquet")
sys_long = sys_long[sys_long.in_constraints]
out_long = pd.read_parquet(Path(args.clean) / f"panel_outside_{args.suffix}.parquet")
hist = pd.concat([sys_long.assign(uid=sys_long.unit + "|" + sys_long.measure),
                  out_long.assign(uid="OUT:" + out_long.unit + "|" + out_long.measure)])
Y = hist.pivot(index="period", columns="uid", values="co2_t_per_day")[meta.unique_id]

# ---------------------------------------------------------------- in-sample scales per origin
# scale1[o] = mean |y_t - y_{t-12}| over t <= o; scale2[o] = mean (y_t - y_{t-12})^2.
dif = (Y - Y.shift(SEASON)).iloc[SEASON:]
cnt = np.arange(1, len(dif) + 1)[:, None]
scale1 = pd.DataFrame(dif.abs().cumsum().to_numpy() / cnt, index=dif.index, columns=dif.columns)
scale2 = pd.DataFrame((dif ** 2).cumsum().to_numpy() / cnt, index=dif.index, columns=dif.columns)
sc = (scale1.stack().rename("scale1").to_frame().join(scale2.stack().rename("scale2"))
      .reset_index().rename(columns={"period": "origin", "uid": "unique_id"}))
f = f.merge(sc, on=["unique_id", "origin"], how="left")
assert f.scale1.notna().all()
n_zero_scale = int((f.drop_duplicates(["unique_id", "origin"]).scale1 == 0).sum())
print("series-origins with zero scale:", n_zero_scale)

# ---------------------------------------------------------------- pointwise scores
e = f.y - f["mean"]
f["ae"] = e.abs()
f["sae"] = f.ae / f.scale1
f["sse"] = e ** 2 / f.scale2
has_int = f.lo_95.notna()
pin = np.zeros(len(f))
for q, c in QUANTILES.items():
    u = f.y - f[c]
    pin += np.maximum(q * u, (q - 1) * u)
f["scrps"] = np.where(has_int, pin / len(QUANTILES) / f.scale1, np.nan)
for lv in (80, 95):
    f[f"cov{lv}"] = np.where(has_int, ((f.y >= f[f"lo_{lv}"]) & (f.y <= f[f"hi_{lv}"])).astype(float), np.nan)
alpha = 0.05
wink = (f.hi_95 - f.lo_95) + (2 / alpha) * ((f.lo_95 - f.y).clip(lower=0) + (f.y - f.hi_95).clip(lower=0))
f["swinkler95"] = np.where(has_int, wink / f.scale1, np.nan)

# ---------------------------------------------------------------- per-series metrics
METRICS = ["MASE", "RMSSE", "sCRPS", "cov80", "cov95", "sWinkler95"]


def per_series(d):
    g = d.groupby(["unique_id", "model"])
    r = g.agg(MASE=("sae", "mean"), RMSSE=("sse", "mean"), sCRPS=("scrps", "mean"), cov80=("cov80", "mean"),
              cov95=("cov95", "mean"), sWinkler95=("swinkler95", "mean"), n=("sae", "size"))
    r["RMSSE"] = np.sqrt(r.RMSSE)
    return r.reset_index()


rows = []
for wname, wfun in WINDOWS.items():
    fw = f[wfun(f.target)]
    for h in sorted(fw.horizon.unique()):
        rows.append(per_series(fw[fw.horizon == h]).assign(window=wname, horizon=str(h)))
    rows.append(per_series(fw).assign(window=wname, horizon="all"))
ps = pd.concat(rows, ignore_index=True).merge(meta, on="unique_id")
ps.to_parquet(out_path(OUT / "metrics_per_series.parquet", args.overwrite), index=False)

# ---------------------------------------------------------------- summaries
def summarise(d, by):
    out = []
    for keys, g in d.groupby(by, observed=True):
        rec = dict(zip(by, keys if isinstance(keys, tuple) else (keys,)))
        w = g.weight_2022
        for mtr in METRICS:
            v = g[mtr]
            ok = v.notna()
            rec_m = {**rec, "metric": mtr, "n_series": int(ok.sum())}
            if ok.any():
                rec_m.update(median=v[ok].median(), mean=v[ok].mean(),
                             weighted=np.average(v[ok], weights=w[ok]))
            out.append(rec_m)
    return pd.DataFrame(out)


hsel = ps[ps.horizon.isin([str(h) for h in HORIZONS] + ["all"])]
summ = pd.concat([
    summarise(hsel.assign(grouping="all", grp="all"), ["window", "grouping", "grp", "model", "horizon"]),
    summarise(hsel.assign(grouping="group", grp=hsel.group), ["window", "grouping", "grp", "model", "horizon"]),
    summarise(hsel.assign(grouping="measure", grp=hsel.measure), ["window", "grouping", "grp", "model", "horizon"]),
    summarise(hsel.assign(grouping="unit_type", grp=hsel.unit_type), ["window", "grouping", "grp", "model", "horizon"]),
], ignore_index=True)
save(summ, "summary.csv")
# Full horizon profile 1..12 (all series) for the figures.
save(summarise(ps.assign(grouping="all", grp="all"), ["window", "grouping", "grp", "model", "horizon"]),
     "summary_by_horizon_all.csv")


# ---------------------------------------------------------------- Diebold-Mariano (HLN)
def dm_hln(d, h):
    """d: loss differential series (model - benchmark). Returns (statistic, p-value)."""
    d = np.asarray(d, float)
    n = len(d)
    if n < 3 or np.allclose(d, 0):
        return np.nan, np.nan
    dbar = d.mean()
    dc = d - dbar
    gam = [np.dot(dc[k:], dc[:n - k]) / n for k in range(h)]
    v = (gam[0] + 2 * sum(gam[1:])) / n
    if v <= 0:                                                    # fall back to the lag-0 variance
        v = gam[0] / n
    stat = dbar / np.sqrt(v) * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return stat, 2 * stats.t.sf(abs(stat), df=n - 1)


fm = f[WINDOWS["main"](f.target) & f.horizon.isin(HORIZONS)]
L = fm.pivot_table(index=["unique_id", "horizon", "origin"], columns="model", values="ae")
dm_rows = []
for (uid, h), g in L.groupby(level=["unique_id", "horizon"]):
    for mdl in models:
        if mdl == BENCH:
            continue
        stat, p = dm_hln(g[mdl] - g[BENCH], int(h))
        dm_rows.append({"unique_id": uid, "horizon": int(h), "model": mdl, "n": len(g), "dm_stat": stat,
                        "p_value": p, "mean_loss_diff": (g[mdl] - g[BENCH]).mean()})
dm = pd.DataFrame(dm_rows).merge(meta[["unique_id", "unit_type", "group"]], on="unique_id")
save(dm, "dm_tests_per_series.csv")
dm["better"] = (dm.p_value < 0.05) & (dm.mean_loss_diff < 0)
dm["worse"] = (dm.p_value < 0.05) & (dm.mean_loss_diff > 0)
dm_sum = (dm.groupby(["model", "horizon"])
          .agg(n_series=("unique_id", "size"), share_better=("better", "mean"), share_worse=("worse", "mean"),
               n_untestable=("p_value", lambda p: int(p.isna().sum())))
          .reset_index())
save(dm_sum, "dm_summary.csv")
print(dm_sum.to_string(index=False))

# ---------------------------------------------------------------- Model Confidence Set
fmain = f[WINDOWS["main"](f.target)]
mcs_rows = []
for hg in [str(h) for h in HORIZONS] + ["all"]:
    d = fmain if hg == "all" else fmain[fmain.horizon == int(hg)]
    loss = d.pivot_table(index="origin", columns="model", values="sae", aggfunc="mean")[models]
    mcs = MCS(loss.to_numpy(), size=0.10, reps=args.mcs_reps, method="R", bootstrap="stationary", seed=SEED)
    mcs.compute()
    pv = mcs.pvalues.reindex(range(len(models)))
    for i, mdl in enumerate(models):
        mcs_rows.append({"horizon_group": hg, "model": mdl, "n_origins": len(loss),
                         "mean_loss": loss[mdl].mean(), "mcs_pvalue": float(pv.iloc[i, 0]),
                         "in_mcs_90": i in mcs.included})
mcs_df = pd.DataFrame(mcs_rows)
save(mcs_df, "mcs_90.csv")
print(mcs_df.to_string(index=False))

# ---------------------------------------------------------------- console overview
ov = summ[(summ.grouping == "all") & (summ.metric == "MASE") & (summ.horizon == "all")]
print(ov.pivot(index="model", columns="window", values=["median", "weighted"]).round(3).to_string())
log.finish(f"{len(models)} models, {meta.shape[0]} series")
