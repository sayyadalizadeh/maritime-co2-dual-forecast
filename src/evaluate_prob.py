"""
evaluate_prob.py: probabilistic evaluation and calibration of the two tracks.

Inputs: results/prob_main/track=<FM|STAT>/ (prob_samples.py) and the tracks' exact non-negative point
forecasts (MinT, nn_qp): FM = reconciled_v2_fmownw/base=TimesFM25, STAT = reconciled_v2_burnin12/base=AutoETS.

Interval types (80% and 95%):
  uncal_base   base sample quantiles (0.1/0.9, 0.025/0.975)
  uncal_rec    reconciled sample quantiles
  aci_g0.05    adaptive conformal on the reconciled sample quantiles, gamma 0.05 (main)
  aci_g0.01    the same, gamma 0.01 (robustness)
  conformal    pooled split conformal: at origin o, for horizon h and measure group (residence:
               RES_TOTAL_SYS; territory: TER_DOM, TER_INT; components A-G), the scaled absolute errors
               |y - point| / scale of the reconciled point forecasts of all system series made at earlier
               origins o' with o' + h <= o (outcome known at o); half-width = their empirical quantile at
               level min(1, ceil((N + 1)(1 - alpha)) / N) (split-conformal finite-sample level), times the
               series' scale at o. Lower bounds of every interval are clipped at zero (emissions are
               non-negative; coverage of non-negative outcomes is unaffected).
Calibration is evaluated on origins 2023-12 ... 2025-06 (2022-12 ... 2023-11 is the warm-up).

Metrics (scaled like MASE by the in-sample seasonal-naive scale at the origin; series with zero scale
are excluded by evalcore's finite filter):
  sCRPS      sample CRPS (prob_samples.py), base and reconciled; all 31 origins
  cov80, cov95, sWinkler95 (and sWinkler80) per interval type; origins from 2023-12
  ES, VS     energy and variogram scores of the scaled vectors (per origin and horizon); summarised by
             the mean and, because single explosive base intervals dominate the mean, the median over cells
Summaries are weighted within one level at a time (evalcore.summarise_levels); windows main, robA, robB.
Tests (main window): Diebold-Mariano (HLN) on CRPS per series, h = 1, 3, 6, 12: reconciled vs base for
each track, FM vs STAT (reconciled); on ES per horizon (series over origins).

With --nest every 80% interval is clipped to lie inside the 95% interval of the same type,
series, origin and horizon (nesting) before scoring; written to results/eval_prob_nested. Only the ACI
intervals can change (sample quantiles and conformal half-widths are nested by construction).

Outputs: results/eval_prob/
Usage:
  python src/evaluate_prob.py [--overwrite]
  python src/evaluate_prob.py --nest --out results/eval_prob_nested
  python src/evaluate_prob.py --nest --prob-dir results/prob_main_exact --out results/eval_prob_exact_nested
                                                    (the exactly reconciled samples)
"""
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from evalcore import HORIZON_GROUPS, WINDOWS, dm_hln, load_history, origin_scales, summarise_levels
from runlog import RunLog, out_path

TRACKS = {"FM": ("reconciled_v2_fmownw", "TimesFM25"), "STAT": ("reconciled_v2_burnin12", "AutoETS")}
CAL_START = "2023-12"
GROUP = {"RES_TOTAL_SYS": "residence", "TER_DOM": "territory", "TER_INT": "territory"}
ALPHA = {80: 0.20, 95: 0.05}


def track_points(res, track, origins=None):
    tag, base = TRACKS[track]
    p = pd.read_parquet(Path(res) / tag / f"base={base}" / "reconciled.parquet",
                        filters=[("method", "==", "MinT"), ("variant", "==", "nn_qp")])
    p = p[p.unique_id != "W|DELTA_INT"]
    return p if origins is None else p[p.origin.isin(origins)]


def conformal_halfwidths(pts, y_long, scales, at_origins, h_max=12):
    """Pooled split-conformal half-widths (scaled) per measure group, horizon and origin.
    pts: past point forecasts (unique_id, origin, target, horizon, value); returns frame
    (origin, horizon, group, level, n_pool, q). Horizons above h_max use the pool of h_max."""
    e = pts.merge(y_long, on=["unique_id", "target"]).merge(scales, on=["unique_id", "origin"])
    e["se"] = (e.y - e.value).abs() / e.scale1
    e = e[np.isfinite(e.se)]
    e["group"] = e.unique_id.str.split("|").str[1].map(GROUP).fillna("components")
    rows = []
    for (o, h_req) in at_origins:
        h = min(h_req, h_max)
        d = e[(e.horizon == h) & (e.target <= o)]
        for g, dg in d.groupby("group"):
            v = np.sort(dg.se.to_numpy())
            N = len(v)
            for lv, a in ALPHA.items():
                k = min(N, math.ceil((N + 1) * (1 - a)))
                rows.append({"origin": o, "horizon": h_req, "group": g, "level": lv, "n_pool": N,
                             "q": v[k - 1] if N else np.nan})
    return pd.DataFrame(rows)


def nest(iv, keys):
    """Nesting: clip each 80% interval to lie inside the 95% interval with the same keys."""
    w95 = iv[iv.level == 95][keys + ["lo", "hi"]].rename(columns={"lo": "lo95", "hi": "hi95"})
    m = iv.merge(w95, on=keys, how="left", validate="many_to_one")
    is80 = m.level == 80
    m.loc[is80, "lo"] = np.fmax(m.loc[is80, "lo"], m.loc[is80, "lo95"])
    m.loc[is80, "hi"] = np.fmin(m.loc[is80, "hi"], m.loc[is80, "hi95"])
    return m.drop(columns=["lo95", "hi95"])


def interval_scores(d, lv):
    """Coverage and scaled Winkler score of interval columns lo, hi against y."""
    a = ALPHA[lv]
    cov = ((d.y >= d.lo) & (d.y <= d.hi)).astype(float)
    wink = (d.hi - d.lo) + (2 / a) * (d.lo - d.y).clip(lower=0) + (2 / a) * (d.y - d.hi).clip(lower=0)
    return cov, wink / d.scale1


def per_series(f, keys, cols):
    """Mean of cols per series, window and horizon (1..12 and 'all')."""
    out = []
    for wname, wfun in WINDOWS.items():
        fw = f[wfun(f.target)]
        a = fw.groupby(["unique_id"] + keys)[cols].mean().reset_index().assign(window=wname, horizon="all")
        b = fw.groupby(["unique_id", "horizon"] + keys)[cols].mean().reset_index().assign(window=wname)
        b["horizon"] = b.horizon.astype(str)
        out += [a, b]
    return pd.concat(out, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean", default="data/clean")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/eval_prob")
    ap.add_argument("--nest", action="store_true", help="clip 80%% intervals inside the 95%% intervals")
    ap.add_argument("--prob-dir", default=None, help="samples-based inputs (default results/prob_main; exact samples: "
                                                     "results/prob_main_exact)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    log = RunLog("evaluate_prob.py", vars(args))
    RES, OUT = Path(args.results), Path(args.out)

    def save(df, name):
        df.to_csv(out_path(OUT / name, args.overwrite), index=False, float_format="%.6g")

    meta = pd.read_parquet(RES / "base_forecasts_v2" / "series_meta.parquet")
    Y = load_history(args.clean)
    y_long = Y.stack().rename("y").reset_index().rename(columns={"period": "target", "uid": "unique_id"})
    scales = origin_scales(Y)[["unique_id", "origin", "scale1"]]
    hor = [str(h) for h in range(1, 13)]

    crps_ps, int_ps, joint, crps_cells, conf_all = [], [], [], [], []
    for tr in TRACKS:
        P = (Path(args.prob_dir) if args.prob_dir else RES / "prob_main") / f"track={tr}"
        # ---- CRPS
        c = pd.read_parquet(P / "crps.parquet").merge(scales, on=["unique_id", "origin"])
        c["sCRPS"] = c.crps / c.scale1
        c = c.assign(track=tr)
        crps_cells.append(c[WINDOWS["main"](c.target)][["track", "kind", "unique_id", "origin", "horizon", "sCRPS"]])
        crps_ps.append(per_series(c, ["track", "kind"], ["sCRPS"]))
        # ---- intervals
        q = pd.read_parquet(P / "quantiles.parquet", columns=["unique_id", "origin", "horizon", "target", "kind",
                                                             "q0.025", "q0.1", "q0.9", "q0.975"])
        aci = pd.read_parquet(P / "aci_intervals.parquet")
        pts = track_points(RES, tr)
        cal_origins = sorted(o for o in pts.origin.unique() if o >= CAL_START)
        hw = conformal_halfwidths(pts, y_long, scales, [(o, h) for o in cal_origins for h in range(1, 13)])
        conf_all.append(hw.assign(track=tr))
        frames = []
        for lv, (ql, qh) in {80: ("q0.1", "q0.9"), 95: ("q0.025", "q0.975")}.items():
            for kind in ["base", "rec"]:
                d = q[q.kind == kind][["unique_id", "origin", "horizon", "target", ql, qh]]
                frames.append(d.rename(columns={ql: "lo", qh: "hi"}).assign(level=lv, interval=f"uncal_{kind}"))
            for g in [0.05, 0.01]:
                d = aci[(aci.level == lv) & (aci.gamma == g)][["unique_id", "origin", "horizon", "target", "lo", "hi"]]
                frames.append(d.assign(level=lv, interval=f"aci_g{g}"))
            cf = pts.assign(group=pts.unique_id.str.split("|").str[1].map(GROUP).fillna("components")) \
                .merge(hw[hw.level == lv], on=["origin", "horizon", "group"]) \
                .merge(scales, on=["unique_id", "origin"])
            cf["lo"], cf["hi"] = cf.value - cf.q * cf.scale1, cf.value + cf.q * cf.scale1
            frames.append(cf[["unique_id", "origin", "horizon", "target", "lo", "hi"]].assign(level=lv, interval="conformal"))
        iv = pd.concat(frames, ignore_index=True)
        iv = iv[iv.origin >= CAL_START]
        iv["lo"] = iv.lo.clip(lower=0)
        if args.nest:
            iv = nest(iv, ["unique_id", "origin", "horizon", "interval"])
        iv = iv.merge(y_long, on=["unique_id", "target"]).merge(scales, on=["unique_id", "origin"])
        rows = []
        for lv in ALPHA:
            d = iv[iv.level == lv].copy()
            d[f"cov{lv}"], d[f"sWinkler{lv}"] = interval_scores(d, lv)
            rows.append(per_series(d.assign(track=tr), ["track", "interval"], [f"cov{lv}", f"sWinkler{lv}"]))
        int_ps.append(rows[0].merge(rows[1], on=["unique_id", "track", "interval", "window", "horizon"]))
        # ---- joint scores
        j = pd.read_csv(P / "joint_scores.csv").assign(track=tr)
        joint.append(j)
        print(f"{tr}: CRPS {len(c)} cells, intervals {len(iv)} rows")

    # ---- summaries (weighted within one level at a time)
    cps = pd.concat(crps_ps, ignore_index=True).merge(meta, on="unique_id")
    s_crps = summarise_levels(cps, ["track", "kind"], metrics=("sCRPS",), horizons=HORIZON_GROUPS + hor)
    save(s_crps, "summary_crps.csv")
    ips = pd.concat(int_ps, ignore_index=True).merge(meta, on="unique_id")
    s_int = summarise_levels(ips, ["track", "interval"], metrics=("cov80", "cov95", "sWinkler80", "sWinkler95"),
                             horizons=HORIZON_GROUPS + hor)
    save(s_int, "summary_intervals.csv")
    cps.to_parquet(out_path(OUT / "crps_per_series.parquet", args.overwrite), index=False)
    ips.to_parquet(out_path(OUT / "intervals_per_series.parquet", args.overwrite), index=False)
    save(pd.concat(conf_all, ignore_index=True), "conformal_halfwidths.csv")

    J = pd.concat(joint, ignore_index=True)
    js = []
    for wname, wfun in WINDOWS.items():
        jw = J[wfun(J.target)]
        for hg in HORIZON_GROUPS:
            d = jw if hg == "all" else jw[jw.horizon == int(hg)]
            g = d.groupby(["track", "kind"]).agg(ES=("es", "mean"), VS=("vs", "mean"), ES_median=("es", "median"),
                                                 VS_median=("vs", "median"), n_cells=("es", "size"))
            js.append(g.reset_index().assign(window=wname, horizon=hg))
    save(pd.concat(js, ignore_index=True), "summary_joint.csv")

    # ---- Diebold-Mariano on CRPS (main window)
    Cc = pd.concat(crps_cells, ignore_index=True)
    Cc = Cc[Cc.horizon.isin([1, 3, 6, 12])]
    Cc["combo"] = Cc.track + "+" + Cc.kind
    Pv = Cc.pivot_table(index=["unique_id", "horizon", "origin"], columns="combo", values="sCRPS")
    comps = [("FM+rec", "FM+base"), ("STAT+rec", "STAT+base"), ("FM+rec", "STAT+rec"), ("FM+base", "STAT+base")]
    dm = []
    for (uid, h), g in Pv.groupby(level=["unique_id", "horizon"]):
        for a, b in comps:
            dd = (g[a] - g[b]).dropna()
            stat, p = dm_hln(dd, int(h))
            dm.append({"comparison": f"{a} vs {b}", "unique_id": uid, "horizon": int(h), "n": len(dd),
                       "dm_stat": stat, "p_value": p, "mean_diff": dd.mean()})
    dm = pd.DataFrame(dm).merge(meta[["unique_id", "unit_type"]], on="unique_id")
    dm["better"] = (dm.p_value < 0.05) & (dm.mean_diff < 0)
    dm["worse"] = (dm.p_value < 0.05) & (dm.mean_diff > 0)
    save(dm.groupby(["comparison", "unit_type", "horizon"]).agg(n=("unique_id", "size"), share_better=("better", "mean"),
                                                               share_worse=("worse", "mean")).reset_index(), "dm_crps_summary.csv")
    dm.to_parquet(out_path(OUT / "dm_crps_per_series.parquet", args.overwrite), index=False)

    # ---- Diebold-Mariano on the energy and variogram scores, per horizon (series over origins)
    Jm = J[WINDOWS["main"](J.target)].assign(combo=lambda d: d.track + "+" + d.kind)
    dmj = []
    for score in ["es", "vs"]:
        Pj = Jm.pivot_table(index=["horizon", "origin"], columns="combo", values=score)
        for h, g in Pj.groupby(level="horizon"):
            for a, b in comps:
                dd = (g[a] - g[b]).dropna()
                stat, p = dm_hln(dd, int(h))
                dmj.append({"score": score, "comparison": f"{a} vs {b}", "horizon": int(h), "n": len(dd),
                            "mean_a": g[a].mean(), "mean_b": g[b].mean(), "dm_stat": stat, "p_value": p})
    save(pd.DataFrame(dmj), "dm_joint.csv")
    log.finish("tracks FM, STAT")


if __name__ == "__main__":
    main()
