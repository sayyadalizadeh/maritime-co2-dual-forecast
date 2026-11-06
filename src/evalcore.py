"""
evalcore.py: evaluation helpers shared by the evaluation scripts.

- load_history: actual values per day of every modelled series (wide, months x series).
- origin_scales: in-sample seasonal-naive scales up to each origin (MASE and RMSSE denominators).
- per_series_metrics: MASE and RMSSE per series, window, horizon and model key.
- summarise_levels: summaries - medians and weighted means within one
  level at a time (system economies; outside-system series), World and Residual series one by one.
- dm_hln: Diebold-Mariano test with the Harvey-Leybourne-Newbold correction.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SEASON = 12
WINDOWS = {
    "main": lambda t: (t <= "2025-12") & (t != "2025-09"),     # main window: up to 2025-12, without 2025-09
    "robA": lambda t: t <= "2025-12",                          # including 2025-09
    "robB": lambda t: (t >= "2026-01") & (t <= "2026-06"),     # stress test (after the 2026 reclassification)
}
HORIZON_GROUPS = ["1", "3", "6", "12", "all"]
MEASURE_GROUP = {"RES_TOTAL_SYS": "residence", "RES_TOTAL_PUB": "residence", "TER_DOM": "territory",
                 "TER_INT": "territory", "DELTA_INT": "discrepancy"}
LEVEL = {"economy": "economies", "outside": "outside", "world": "world", "residual": "residual"}


def load_history(clean="data/clean", suffix="v2"):
    """Wide frame of values per day: index period (YYYY-MM), one column per unique_id."""
    clean = Path(clean)
    s = pd.read_parquet(clean / f"panel_system_{suffix}.parquet")
    s = s[s.in_constraints]
    o = pd.read_parquet(clean / f"panel_outside_{suffix}.parquet")
    h = pd.concat([s.assign(uid=s.unit + "|" + s.measure), o.assign(uid="OUT:" + o.unit + "|" + o.measure)])
    return h.pivot(index="period", columns="uid", values="co2_t_per_day")


def origin_scales(Y):
    """Long frame unique_id, origin, scale1 (mean |y_t - y_t-12|), scale2 (mean squared) over t <= origin."""
    dif = (Y - Y.shift(SEASON)).iloc[SEASON:]
    cnt = np.arange(1, len(dif) + 1)[:, None]
    s1 = pd.DataFrame(dif.abs().cumsum().to_numpy() / cnt, index=dif.index, columns=dif.columns)
    s2 = pd.DataFrame((dif ** 2).cumsum().to_numpy() / cnt, index=dif.index, columns=dif.columns)
    out = s1.stack().rename("scale1").to_frame().join(s2.stack().rename("scale2")).reset_index()
    return out.rename(columns={"period": "origin", "uid": "unique_id"})


def add_scaled_errors(f, scales, pred="mean"):
    """Add ae, sae (|e| / scale1) and sse (e^2 / scale2) to a forecast frame with y and `pred`."""
    f = f.merge(scales, on=["unique_id", "origin"], how="left")
    e = f["y"] - f[pred]
    f["ae"] = e.abs()
    f["sae"] = f.ae / f.scale1
    f["sse"] = e ** 2 / f.scale2
    return f


def per_series_metrics(f, keys):
    """MASE and RMSSE per series for every window and horizon (1..12 and 'all').
    keys: the columns identifying a forecasting method (e.g. ['base', 'method'])."""
    rows = []
    for wname, wfun in WINDOWS.items():
        fw = f[wfun(f.target)]
        g_all = fw.groupby(["unique_id"] + keys)
        r = g_all.agg(MASE=("sae", "mean"), RMSSE=("sse", "mean"), n=("sae", "size")).reset_index()
        rows.append(r.assign(window=wname, horizon="all"))
        r = fw.groupby(["unique_id", "horizon"] + keys).agg(
            MASE=("sae", "mean"), RMSSE=("sse", "mean"), n=("sae", "size")).reset_index()
        rows.append(r.assign(window=wname, horizon=r.horizon.astype(str)))
    ps = pd.concat(rows, ignore_index=True)
    ps["RMSSE"] = np.sqrt(ps.RMSSE)
    return ps


def summarise_levels(ps, keys, metrics=("MASE", "RMSSE"), horizons=HORIZON_GROUPS):
    """Summaries within one level at a time, weighted by 2022 per-day emissions. ps: per-series metrics joined with meta (unit_type, measure, weight_2022).
    Pooled levels (system economies, outside-system series): median, mean and weighted mean,
    for all series, by measure group and by measure. World and Residual: one row per series."""
    ps = ps[ps.horizon.isin(horizons)].copy()
    ps["level"] = ps.unit_type.map(LEVEL)
    ps["group"] = ps.measure.map(MEASURE_GROUP).fillna("component")
    recs, frames = [], []
    by = ["window", "horizon"] + keys
    pooled = ps[ps.level.isin(["economies", "outside"])]
    for grouping, col in [("all", None), ("group", "group"), ("measure", "measure")]:
        d = pooled.assign(grp="all" if col is None else pooled[col])
        for k, g in d.groupby(["level", "grp"] + by, observed=True):
            rec = dict(zip(["level", "grp"] + by, k))
            for m in metrics:
                v, w = g[m], g.weight_2022
                ok = v.notna() & np.isfinite(v)
                recs.append({**rec, "grouping": grouping, "metric": m, "n_series": int(ok.sum()),
                            "median": v[ok].median(), "mean": v[ok].mean(),
                            "weighted": np.average(v[ok], weights=w[ok]) if ok.any() else np.nan})
    single = ps[ps.level.isin(["world", "residual"])]
    for m in metrics:
        d = single[["level", "measure"] + by + [m]].rename(columns={"measure": "grp", m: "median"})
        d["mean"] = d["weighted"] = d["median"]
        frames.append(d.assign(grouping="series", metric=m, n_series=1))
    res = pd.concat([pd.DataFrame(recs)] + frames, ignore_index=True)
    return res[["window", "level", "grouping", "grp"] + keys + ["horizon", "metric", "n_series",
                                                                 "median", "mean", "weighted"]]


def dm_hln(d, h):
    """Diebold-Mariano with the HLN small-sample correction. d: loss differential series.
    Returns (statistic, two-sided p-value) with a t(n-1) reference; the lag-0 variance is used
    when the long-run variance estimate is not positive."""
    d = np.asarray(d, float)
    n = len(d)
    if n < 3 or np.allclose(d, 0):
        return np.nan, np.nan
    dbar = d.mean()
    dc = d - dbar
    gam = [np.dot(dc[k:], dc[:n - k]) / n for k in range(h)]
    v = (gam[0] + 2 * sum(gam[1:])) / n
    if v <= 0:
        v = gam[0] / n
    stat = dbar / np.sqrt(v) * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return stat, 2 * stats.t.sf(abs(stat), df=n - 1)
