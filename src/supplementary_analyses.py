"""
supplementary_analyses.py: supplementary tables.

Everything is compiled from stored results: no model is refitted and no QP is solved. Cheap new statistics are
computed from stored forecasts, samples and intervals where an item asks for them (DM tests for the two tracks,
Benjamini-Hochberg adjustment, Friedman/Nemenyi with OLS, Wilson intervals, a block bootstrap over origins).
Outputs: results/supplementary/ (CSV), described in supplementary.md (written by this script).

Items (output files in results/supplementary/):
  1   robustness windows: main, robA (main incl. 2025-09), robB (2026-01 ... 2026-06)
  2   OLS reconciliation (benchmark only)
  3   Friedman and Nemenyi per base over reconciliation methods
  4   Diebold-Mariano tests (point and CRPS), BH-adjusted counts
  5   Model Confidence Set: settings and p-values
  6   calibration variants (ACI gamma 0.01, pooled split conformal) and joint scores of base samples
  7   origins behind T4
  8   ACI diagnostics
  9   outside-system series
  10  the STAT reconciled CRPS at h = 11 (Fig. 4b)
  11  foundation-model inference settings
  13  data notes (international share by calendar month; Switzerland)

The sample-based items (1, 4, 6, 7, 8, 10) read the exactly reconciled samples by default (results/prob_main_exact,
results/eval_prob_exact_nested); with --prob-dir results/prob_main --eval-dir results/eval_prob_nested --out
results/supplementary_zeroed they are rerun on the samples with negatives set to zero, for the comparison of the two
(exact_samples.py --part compare); the point-based parts of items 1 and 4 come out unchanged.

Usage:
  python src/supplementary_analyses.py [--items 1 2 ...] [--overwrite]
  python src/supplementary_analyses.py --items 1 4 6 7 8 10 --prob-dir results/prob_main
         --eval-dir results/eval_prob_nested --out results/supplementary_zeroed
An item whose first output exists is skipped unless --overwrite is given (resume after an interruption).
"""
import argparse
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from evalcore import WINDOWS, load_history, origin_scales
from evaluate_prob import nest
from runlog import RunLog, out_path

RES = Path("results")
OUT = RES / "supplementary"
PROB = RES / "prob_main_exact"                                               # samples-based inputs (--prob-dir)
EVALP = RES / "eval_prob_exact_nested"                                       # evaluate_prob.py --nest output (--eval-dir)
STAT_B = ["AutoETS", "AutoTheta", "AutoARIMA", "SeasonalNaive"]
FM_B = ["ChronosBolt", "TimesFM", "TimesFM25"]
TRACKS = {"FM": ("TimesFM25", "MinT"), "STAT": ("AutoETS", "MinT")}          # FM and STAT tracks
CAL_START = "2023-12"                                                        # calibration origins (evaluate_prob.py)
ALPHA_CLIP = (0.001, 0.999)                                                  # prob_samples.py
SEED, BLOCK, REPS = 42, 6, 2000                                              # block bootstrap of item 8
Q05_NEMENYI = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850}            # Demsar (2006), alpha 0.05
BASIS_GROUP = {"RES_TOTAL_SYS": "residence total", "TER_DOM": "territory measures", "TER_INT": "territory measures"}
OVERWRITE = False
DOC = {}                                                                     # file -> description (supplementary.md)


def save(df, name, desc, float_format="%.6g"):
    df.to_csv(out_path(OUT / name, OVERWRITE), index=False, float_format=float_format)
    if PROB != RES / "prob_main" or EVALP != RES / "eval_prob_nested":     # name the sources actually read
        desc = (desc.replace("results/prob_main", PROB.as_posix()).replace("results/eval_prob_nested", EVALP.as_posix())
                .replace("(negative values set to zero)", "(exact non-negative QP)"))
    DOC[name] = desc
    print("saved", name, df.shape)


def meta():
    return pd.read_parquet(RES / "base_forecasts_v2" / "series_meta.parquet")


def bh(p):
    """Benjamini-Hochberg adjusted p-values (NaN kept)."""
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    q = p[ok]
    n = len(q)
    if n == 0:
        return out
    order = np.argsort(q)
    adj = q[order] * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    res = np.empty(n)
    res[order] = np.minimum(adj, 1.0)
    out[ok] = res
    return out


def dm_hln_flag(d, h):
    """evalcore.dm_hln plus a flag for the lag-0 fallback (identical statistic)."""
    d = np.asarray(d, float)
    n = len(d)
    if n < 3 or np.allclose(d, 0):
        return np.nan, np.nan, False
    dc = d - d.mean()
    gam = [np.dot(dc[k:], dc[:n - k]) / n for k in range(h)]
    v = (gam[0] + 2 * sum(gam[1:])) / n
    fb = v <= 0
    if fb:
        v = gam[0] / n
    stat = d.mean() / np.sqrt(v) * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return stat, 2 * stats.t.sf(abs(stat), df=n - 1), fb


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    hw = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return c - hw, c + hw


# ------------------------------------------------------------------ cell-level point losses (items 1 and 4)
def point_cells(m):
    """Absolute errors (t/day) of every base forecast and every reconciled nn_qp forecast, all windows, with series
    meta. Combos: <base>+base for the nine base models, <base>+<method> for OLS/WLS/MinT (FM: own W)."""
    Y = load_history("data/clean")
    yl = Y.stack().rename("y").reset_index().rename(columns={"period": "target", "uid": "unique_id"})
    frames = []
    for b in STAT_B + ["LightGBM", "RandomWalkWithDrift"] + FM_B:
        d = "base_forecasts_fm" if b in FM_B else "base_forecasts_v2"
        f = pd.read_parquet(RES / d / f"model={b}" / "forecasts.parquet",
                            columns=["unique_id", "origin", "target", "horizon", "y", "mean"])
        frames.append(f.assign(base=b, method="base").rename(columns={"mean": "value"}))
    for b in STAT_B + FM_B:
        flt = [("variant", "==", "nn_qp")]
        r = pd.read_parquet(RES / "reconciled_v2_burnin12" / f"base={b}" / "reconciled.parquet", filters=flt)
        if b in FM_B:                       # OLS does not depend on W (v2_burnin12); WLS and MinT from own W (v2_fmownw)
            r = pd.concat([r[r.method == "OLS"],
                           pd.read_parquet(RES / "reconciled_v2_fmownw" / f"base={b}" / "reconciled.parquet",
                                           filters=flt + [("method", "!=", "OLS")])])
        r = r[r.unique_id != "W|DELTA_INT"].merge(yl, on=["unique_id", "target"])
        frames.append(r.drop(columns="variant").assign(base=b))
    L = pd.concat(frames, ignore_index=True)
    L["ae"] = (L.y - L.value).abs()
    return L.merge(m, on="unique_id")


# ------------------------------------------------------------------ item 1
def item1(m, cells):
    win = ["main", "robA", "robB"]
    s = pd.read_csv(RES / "eval_fm_w" / "summary.csv")
    s = s[s.variant.isin(["base", "nn_qp"]) & s.horizon.isin(["all", "1", "3", "6", "12"])
          & (s.grouping.isin(["all", "group"]) | s.level.isin(["world"]))]
    b9 = pd.read_csv(RES / "eval_fm_v2" / "summary_weighted.csv")
    b9 = b9[b9.horizon.isin(["all", "1", "3", "6", "12"]) & (b9.grouping.isin(["all", "group"]) | (b9.level == "world"))
            & b9.metric.isin(["MASE", "RMSSE"])]
    b9 = b9.rename(columns={"model": "base"}).assign(method="base", variant="base")
    b9 = b9[b9.base.isin(["LightGBM", "RandomWalkWithDrift"])]          # the other seven are in eval_fm_w
    pt = pd.concat([s, b9], ignore_index=True)
    save(pt, "point_by_window.csv",
         "MASE and RMSSE (median, mean, weighted with 2022 weights) of every base model (base forecasts) and every "
         "reconciliation method (exact QP, nn_qp; FM bases with own W; OLS benchmark), windows main / robA (incl. "
         "2025-09) / robB (2026-01..06, stress test), horizons all, 1, 3, 6, 12; system economies (all and by "
         "measure group) and World series one by one. Sources: results/eval_fm_w/summary.csv, "
         "results/eval_fm_v2/summary_weighted.csv (LightGBM, Drift).")
    # ranks of all combinations (no OLS) by weighted and median MASE, economies, all horizons
    e = pt[(pt.level == "economies") & (pt.grouping == "all") & (pt.horizon == "all") & (pt.metric == "MASE")
           & (pt.method != "OLS") & ~pt.method.str.contains(r"\[ETS-W\]")]
    e = e.assign(combo=e.base + "+" + e.method)
    rk = []
    for w in win:
        x = e[e.window == w].copy()
        x["rank_weighted"] = x.weighted.rank()
        x["rank_median"] = x["median"].rank()
        rk.append(x[["window", "combo", "weighted", "median", "rank_weighted", "rank_median"]])
    rk = pd.concat(rk).pivot(index="combo", columns="window")
    rk.columns = [f"{a}_{b}" for a, b in rk.columns]
    rk = rk.reset_index().sort_values("rank_weighted_main")
    save(rk, "method_ranks.csv",
         "Rank of every base/method combination (no OLS; FM with own W) by weighted and by median MASE over system "
         "economies, all horizons, in each window (1 = best). Spearman correlations between windows in rank_correlation.csv.")
    sp = []
    for a, b in [("main", "robA"), ("main", "robB"), ("robA", "robB")]:
        for st in ["weighted", "median"]:
            r = stats.spearmanr(rk[f"rank_{st}_{a}"], rk[f"rank_{st}_{b}"])
            sp.append({"window_1": a, "window_2": b, "statistic": st, "spearman_rho": r.statistic,
                       "p_value": r.pvalue, "n_combos": len(rk)})
    save(pd.DataFrame(sp), "rank_correlation.csv", "Spearman correlation of the ranks of method_ranks.csv between windows.")
    # probabilistic scores of the tracks by window
    c = pd.read_csv(EVALP / "summary_crps.csv")
    c = c[c.horizon.astype(str).isin(["all", "1", "3", "6", "12"]) & ((c.grouping == "all") | (c.level == "world"))]
    iv = pd.read_csv(EVALP / "summary_intervals.csv")
    iv = iv[iv.horizon.astype(str).isin(["all", "1", "3", "6", "12"]) & ((iv.grouping == "all") | (iv.level == "world"))]
    save(pd.concat([c.assign(table="sCRPS", interval=c.kind), iv.assign(table="intervals")], ignore_index=True),
         "prob_tracks_by_window.csv",
         "Both tracks: scaled CRPS (base and reconciled samples, all 31 origins) and coverage / scaled Winkler of the "
         "80% and 95% intervals (uncal_base, uncal_rec, aci_g0.05 = main, aci_g0.01, conformal; origins 2023-12 .. "
         "2025-06; 80% nested inside 95%), windows main / robA / robB; system economies (grouping all) and World "
         "series. Coverage: use the column 'mean' (mean over series of the per-series coverage, as in T4). Source: "
         "results/eval_prob_nested/summary_crps.csv, summary_intervals.csv.")
    pb = []
    for src, metrics in [("eval_base_v2", ["sCRPS", "cov80", "cov95", "sWinkler95"]), ("eval_fm_v2", ["sCRPS3", "cov80"])]:
        x = pd.read_csv(RES / src / "summary_weighted.csv")
        x = x[x.metric.isin(metrics) & x.horizon.isin(["all", "1", "3", "6", "12"])
              & (((x.level == "economies") & (x.grouping == "all")) | (x.level == "world"))]
        pb.append(x.assign(source=src))
    save(pd.concat(pb, ignore_index=True), "prob_base_models_by_window.csv",
         "Base models' own probabilistic scores, windows main / robA / robB. eval_base_v2 (statistical models "
         "and LightGBM where available): sCRPS = mean pinball loss over the quantiles 0.025, 0.1, 0.25, 0.5, 0.75, 0.9, "
         "0.975, cov80, cov95, sWinkler95; eval_fm_v2 (all nine models): sCRPS3 = mean pinball loss over 0.1, 0.5, 0.9 "
         "(the quantiles shared by all models), cov80. All origins, no calibration.")
    # weighted MAE in kt per day (system economies) and World MAE
    rows = []
    for w in win:
        x = cells[WINDOWS[w](cells.target)]
        ps = x.groupby(["base", "method", "unique_id", "unit_type", "basis", "weight_2022"], observed=True).ae.mean() \
            .reset_index()
        for (b, mth), g in ps.groupby(["base", "method"]):
            ec = g[g.unit_type == "economy"]
            rec = {"window": w, "base": b, "method": mth, "n_series": len(ec),
                   "wMAE_kt_per_day": np.average(ec.ae, weights=ec.weight_2022) / 1e3,
                   "median_MAE_kt_per_day": ec.ae.median() / 1e3}
            for bs in ["residence", "territory", "component"]:
                eb = ec[ec.basis == bs]
                rec[f"wMAE_{bs}_kt_per_day"] = np.average(eb.ae, weights=eb.weight_2022) / 1e3
            for u in ["W|RES_TOTAL_SYS", "W|TER_DOM", "W|TER_INT"]:
                v = g.loc[g.unique_id == u, "ae"]
                rec[f"MAE_{u.replace('|', '_')}_kt_per_day"] = v.iloc[0] / 1e3 if len(v) else np.nan
            rows.append(rec)
    save(pd.DataFrame(rows), "weighted_MAE_kt.csv",
         "Mean absolute error in kt CO2 per day: weighted mean over system economy series of each series' MAE (2022 "
         "weights), also by basis (residence = RES_TOTAL_SYS; territory = TER_DOM, TER_INT; components A-G), the "
         "median over series, and the MAE of the World totals. All horizons 1-12 pooled, per window. Base forecasts of "
         "all nine models; reconciled forecasts exact QP (nn_qp), FM with own W. Computed from the stored forecasts.")


# ------------------------------------------------------------------ item 2
def item2():
    s = pd.read_csv(RES / "eval_fm_w" / "summary.csv")
    keep = s.metric.isin(["MASE", "RMSSE"]) & s.horizon.isin(["all", "1", "3", "6", "12"]) & \
        s.method.isin(["base", "OLS", "WLS", "MinT"]) & s.variant.isin(["base", "nn_qp", "free"]) & \
        (((s.level == "economies") & s.grouping.isin(["all", "group"])) | s.level.isin(["world", "residual"]))
    x = s[keep]
    save(x[x.base.isin(["AutoETS", "TimesFM25"])], "OLS_tracks.csv",
         "OLS reconciliation (W = I; benchmark only) for the bases of both tracks (STAT: AutoETS, FM: TimesFM25), "
         "next to base, WLS and MinT (FM: own W). Variants nn_qp (exact non-negative QP) and free (projection without "
         "non-negativity). MASE and RMSSE, windows main/robA/robB, horizons all/1/3/6/12, system economies (all, by "
         "group), World and Residual series. Source: results/eval_fm_w/summary.csv.")
    e = x[(x.level == "economies") & (x.grouping == "all") & (x.horizon == "all") & (x.metric == "MASE")
          & x.variant.isin(["base", "nn_qp"])]
    p = e.pivot_table(index=["window", "base"], columns="method", values=["median", "weighted"]).reset_index()
    p.columns = ["_".join([c for c in col if c]) for col in p.columns]
    save(p, "OLS_all_bases_summary.csv",
         "MASE over system economies (median and weighted), all horizons, for every base: base, OLS, WLS, MinT (exact "
         "QP).")


# ------------------------------------------------------------------ item 3
def item3():
    fr = pd.read_csv(RES / "eval_fm_w" / "friedman_nemenyi.csv").assign(treatments="base, WLS, MinT")
    ps = pd.read_parquet(RES / "eval_fm_w" / "metrics_per_series.parquet",
                         filters=[("window", "==", "main"), ("unit_type", "==", "economy")])
    ps = ps[ps.variant.isin(["base", "nn_qp"]) & ps.method.isin(["base", "OLS", "WLS", "MinT"])]
    rows = []
    meths = ["base", "OLS", "WLS", "MinT"]
    for (b, h), g in ps.groupby(["base", "horizon"]):
        if h not in ["all", "1", "3", "6", "12"]:
            continue
        for tset in [["base", "WLS", "MinT"], meths]:
            d = g.pivot(index="unique_id", columns="method", values="MASE")[tset].dropna()
            chi2, p = stats.friedmanchisquare(*[d[t] for t in tset])
            rk = d.rank(axis=1).mean()
            k, N = len(tset), len(d)
            cd = Q05_NEMENYI[k] * np.sqrt(k * (k + 1) / (6 * N))
            rec = {"base": b, "horizon": h, "treatments": ", ".join(tset), "n_series": N, "friedman_chi2": chi2,
                   "friedman_p": p, "nemenyi_cd": cd}
            rec.update({f"mean_rank_{t}": rk[t] for t in tset})
            for i, t1 in enumerate(tset):
                for t2 in tset[i + 1:]:
                    rec[f"sig_{t1}_vs_{t2}"] = bool(abs(rk[t1] - rk[t2]) > cd)
            rows.append(rec)
    out = pd.DataFrame(rows)
    # the stored test (results/eval_fm_w) must be reproduced exactly
    chk = out[(out.horizon == "all") & (out.treatments == "base, WLS, MinT")].set_index("base").friedman_chi2
    st = fr.drop_duplicates("base").set_index("base").friedman_chi2
    assert np.allclose(chk.loc[st.index], st, rtol=1e-5), "Friedman statistics differ from results/eval_fm_w"
    save(out, "friedman_nemenyi.csv",
         "Friedman test over reconciliation methods, per base: blocks = system economy series, treatments base / WLS / "
         "MinT (FM with own W) and, as a supplement, base / OLS / WLS / MinT; loss = MASE of each "
         "series, main window, all horizons and h = 1, 3, 6, 12; reconciled = exact QP. Mean ranks (1 = best), "
         "Nemenyi critical difference CD = q_0.05 sqrt(k(k+1)/(6N)) with q from Demsar (2006); sig_* = |rank "
         "difference| > CD. The 'all' rows with three treatments reproduce results/eval_fm_w/friedman_nemenyi.csv.")


# ------------------------------------------------------------------ item 4
def item4(cells):
    m = cells[WINDOWS["main"](cells.target) & cells.horizon.isin([1, 3, 6, 12])]
    m = m[m.method.isin(["base", "MinT"]) & m.base.isin(["TimesFM25", "AutoETS"])]
    m = m.assign(combo=m.base.map({"TimesFM25": "FM", "AutoETS": "STAT"}) + "+" + m.method.map({"base": "base", "MinT": "rec"}))
    P = m.pivot_table(index=["unique_id", "unit_type", "basis", "horizon", "origin"], columns="combo", values="ae")
    comps = [("FM+rec", "STAT+rec"), ("FM+rec", "FM+base"), ("STAT+rec", "STAT+base"), ("FM+base", "STAT+base")]
    rows = []
    for (uid, ut, bs, h), g in P.groupby(level=["unique_id", "unit_type", "basis", "horizon"]):
        for a, b in comps:
            dd = (g[a] - g[b]).dropna().to_numpy()
            st, p, fb = dm_hln_flag(dd, int(h))
            rows.append({"loss": "absolute error (t/day)", "comparison": f"{a} vs {b}", "unique_id": uid,
                         "unit_type": ut, "basis": bs, "horizon": int(h), "n": len(dd), "lag_length": int(h) - 1,
                         "dm_stat": st, "p_value": p, "mean_loss_diff": dd.mean(), "lag0_fallback": fb})
    pt = pd.DataFrame(rows)
    cr = pd.read_parquet(EVALP / "dm_crps_per_series.parquet")
    cr = cr.merge(meta()[["unique_id", "basis"]], on="unique_id").rename(columns={"mean_diff": "mean_loss_diff"})
    cr = cr.assign(loss="scaled CRPS (sample CRPS / in-sample seasonal-naive MAE at the origin)",
                   lag_length=cr.horizon - 1, lag0_fallback=np.nan)
    D = pd.concat([pt, cr[pt.columns]], ignore_index=True)
    D["test_family"] = D.loss.str.split(" ").str[0] + " | " + D.comparison + " | h" + D.horizon.astype(str) + " | " + D.unit_type
    D["p_bh"] = np.nan
    for fam, g in D.groupby("test_family"):
        D.loc[g.index, "p_bh"] = bh(g.p_value)
    D.to_parquet(out_path(OUT / "dm_per_series.parquet", OVERWRITE), index=False)
    DOC["dm_per_series.parquet"] = "DM tests per series (input of dm_counts.csv)."
    D["better"] = (D.p_value < 0.05) & (D.mean_loss_diff < 0)
    D["worse"] = (D.p_value < 0.05) & (D.mean_loss_diff > 0)
    D["better_bh"] = (D.p_bh < 0.05) & (D.mean_loss_diff < 0)
    D["worse_bh"] = (D.p_bh < 0.05) & (D.mean_loss_diff > 0)
    s = D.groupby(["loss", "comparison", "unit_type", "horizon"]).agg(
        n_tests=("unique_id", "size"), n_untestable=("p_value", lambda p: int(p.isna().sum())),
        n_lag0_fallback=("lag0_fallback", lambda v: np.nan if v.isna().all() else int(np.nansum(v.astype(float)))),
        n_better=("better", "sum"), n_worse=("worse", "sum"), n_better_bh=("better_bh", "sum"),
        n_worse_bh=("worse_bh", "sum"), mean_n_origins=("n", "mean")).reset_index()
    save(s, "dm_counts.csv",
         "Diebold-Mariano tests, main window, h = 1, 3, 6, 12, one test per series and horizon, the two tracks (FM = "
         "TimesFM 2.5 + MinT own W, STAT = AutoETS + MinT; reconciled = exact QP). Point loss: absolute error in t "
         "CO2/day; CRPS loss: scaled sample CRPS (stored in results/eval_prob_nested). Loss differential d_t = loss(a) - "
         "loss(b) over the origins whose target is in the main window. Statistic: HLN-corrected DM (Harvey, Leybourne "
         "and Newbold 1997): long-run variance = gamma_0 + 2 sum_{k=1}^{h-1} gamma_k (rectangular kernel, lag length "
         "h - 1), statistic times sqrt((n + 1 - 2h + h(h-1)/n) / n), two-sided p from t(n - 1); if the variance estimate "
         "is not positive, gamma_0 is used (n_lag0_fallback; not recorded for the stored CRPS tests). better/worse: p < 0.05 and mean d < 0 / > 0 (a better "
         "than b). BH: Benjamini-Hochberg at 5% within each family = loss x comparison x horizon x unit type "
         "(system economies: 1,100 tests per family). Point DM in results/eval_fm_w/dm_tracks.csv used "
         "TimesFM 3.0 as the FM base and is superseded here.")
    dmb = pd.read_csv(RES / "eval_fm_v2" / "dm_vs_autoets_per_series.csv")
    dmb["p_bh"] = np.nan
    for _, g in dmb.groupby(["model", "horizon", "unit_type"]):
        dmb.loc[g.index, "p_bh"] = bh(g.p_value)
    dmb = dmb.assign(better=(dmb.p_value < .05) & (dmb.mean_loss_diff < 0), worse=(dmb.p_value < .05) & (dmb.mean_loss_diff > 0),
                     better_bh=(dmb.p_bh < .05) & (dmb.mean_loss_diff < 0), worse_bh=(dmb.p_bh < .05) & (dmb.mean_loss_diff > 0))
    sb = dmb.groupby(["model", "unit_type", "horizon"]).agg(
        n_tests=("unique_id", "size"), n_better=("better", "sum"), n_worse=("worse", "sum"),
        n_better_bh=("better_bh", "sum"), n_worse_bh=("worse_bh", "sum")).reset_index()
    save(sb, "dm_base_models_vs_ETS.csv",
         "Base models against AutoETS (DM of evaluate_fm.py, absolute error, HLN, h = 1, 3, 6, 12, main window, all 1,170 "
         "series), counts better / worse than ETS, unadjusted and BH-adjusted within model x horizon x unit type. "
         "Source: results/eval_fm_v2/dm_vs_autoets_per_series.csv.")


# ------------------------------------------------------------------ item 5
def item5():
    rows = []
    for src, key, scope_col in [("eval_fm_w", "combo", None), ("eval_fm_v2", "model", "scope"), ("eval_base_v2", "model", None)]:
        x = pd.read_csv(RES / src / "mcs_90.csv").rename(columns={key: "candidate"})
        x["scope"] = x[scope_col] if scope_col else "system_economies" if src == "eval_fm_w" else "all_series"
        x["source"] = src
        x["block_size"] = np.sqrt(x.n_origins).astype(int)
        rows.append(x[["source", "scope", "horizon_group", "candidate", "n_origins", "block_size", "mean_loss",
                       "mcs_pvalue", "in_mcs_90"]])
    save(pd.concat(rows, ignore_index=True), "mcs_pvalues.csv",
         "Model Confidence Set (Hansen, Lunde and Nason 2011) as run by arch 8.0.0 (arch.bootstrap.MCS): size 0.10 "
         "(90% MCS), statistic 'R' (range of the standardised pairwise loss differences, elimination by the worst "
         "model), stationary bootstrap (Politis-Romano) with mean block length int(sqrt(T)) = block_size, T = number "
         "of origins (31 for 'all' and h = 1; 30 for h = 3, 6; 24 for h = 12), 10,000 replications, seed 42. Loss: one "
         "value per origin = mean over series and over the horizon group of the scaled absolute error |y - yhat| / "
         "(in-sample seasonal-naive MAE at the origin), main-window targets only. Candidates: eval_fm_w = every base "
         "x {base, WLS, MinT} (FM also with the AutoETS W), exact QP, system economies (the set used to select the "
         "tracks); eval_fm_v2 = the nine base models, scope all_series (1,170 series) and system_economies; eval_base_v2 "
         "= the nine base models, all series. mcs_pvalue = MCS p-value; in_mcs_90 = p > 0.10.")


# ------------------------------------------------------------------ item 6
def item6():
    iv = pd.read_csv(EVALP / "summary_intervals.csv")
    iv = iv[(iv.window == "main") & (iv.level == "economies") & (iv.grouping == "all")]
    iv = iv.pivot_table(index=["track", "interval", "horizon"], columns="metric", values=["mean", "median", "weighted"])
    iv.columns = [f"{b}_{a}" for a, b in iv.columns]
    keep = ["cov80_mean", "cov95_mean", "sWinkler80_weighted", "sWinkler80_median", "sWinkler95_weighted", "sWinkler95_median"]
    iv = iv[keep].reset_index()
    iv["h"] = pd.to_numeric(iv.horizon, errors="coerce").fillna(99)
    save(iv.sort_values(["track", "interval", "h"]).drop(columns="h"), "calibration_variants.csv",
         "Coverage (mean over system economy series of the per-series coverage) and scaled Winkler score (weighted, "
         "median) by horizon, main window, origins 2023-12 .. 2025-06, both tracks, for every interval type: "
         "uncal_base, uncal_rec (sample quantiles), aci_g0.05 (main), aci_g0.01, conformal (pooled split "
         "conformal per horizon and measure group, finite-sample level). 80% nested inside 95% (only ACI can "
         "change). Source: results/eval_prob_nested/summary_intervals.csv.")
    j = pd.read_csv(EVALP / "summary_joint.csv")
    j = j[j.window == "main"]
    j["h"] = pd.to_numeric(j.horizon, errors="coerce").fillna(99)
    save(j.sort_values(["track", "kind", "h"]).drop(columns="h"), "joint_scores.csv",
         "Energy score (ES) and variogram score (VS, p = 0.5, on scaled values) of the base and the reconciled "
         "samples (negative values set to zero), both tracks, main window, all 31 origins: median over origin x "
         "horizon cells (primary) and mean; n_cells = cells. Source: results/eval_prob_nested/summary_joint.csv.")


# ------------------------------------------------------------------ item 7
def item7():
    rows = []
    for tr in ["FM", "STAT"]:
        P = PROB / f"track={tr}"
        c = pd.read_parquet(P / "crps.parquet", columns=["origin", "horizon", "target"]).drop_duplicates()
        a = pd.read_parquet(P / "aci_intervals.parquet", columns=["origin", "horizon", "target", "gamma"],
                            filters=[("gamma", "==", 0.05), ("level", "==", 80)]).drop_duplicates()
        for name, d, first in [("sCRPS, ES, VS", c, None), ("coverage and Winkler (uncal and ACI)", a, CAL_START)]:
            d = d[WINDOWS["main"](d.target)]
            if first:
                d = d[d.origin >= first]
            for h, g in d.groupby("horizon"):
                rows.append({"track": tr, "metric": name, "horizon": int(h), "n_origins": g.origin.nunique(),
                             "first_origin": g.origin.min(), "last_origin": g.origin.max(),
                             "first_target": g.target.min(), "last_target": g.target.max()})
    save(pd.DataFrame(rows), "T4_origins.csv",
         "Origins behind T4 (main window: targets up to 2025-12, without 2025-09). CRPS and the joint scores use all "
         "31 origins 2022-12 .. 2025-06. Both coverage columns of T4, uncalibrated (reconciled sample quantiles) and "
         "ACI, use the same calibration origins 2023-12 .. 2025-06 (19 origins); the origins 2022-12 .. 2023-11 are "
         "the ACI warm-up (alpha is updated from 2023-01 on, but not scored). n_origins per horizon falls because "
         "targets after 2025-12 and 2025-09 are excluded.")


# ------------------------------------------------------------------ item 8
def mbb_indices(n, block, rng):
    """Circular moving-block bootstrap: indices of one resample of a series of length n (blocks of consecutive
    origins, wrapped at the end so that every origin has the same probability of being drawn)."""
    starts = rng.integers(0, n, size=int(np.ceil(n / block)))
    return np.concatenate([np.arange(s, s + block) % n for s in starts])[:n]


def item8(m):
    Y = load_history("data/clean")
    yl = Y.stack().rename("y").reset_index().rename(columns={"period": "target", "uid": "unique_id"})
    eco = m[m.unit_type == "economy"].copy()
    eco["tercile"] = pd.qcut(eco.weight_2022.rank(method="first"), 3, labels=["small", "medium", "large"])
    eco["bgroup"] = eco.measure.map(BASIS_GROUP).fillna("components")
    alpha_rows, width_rows, cells_all = [], [], []
    for tr in ["FM", "STAT"]:
        P = PROB / f"track={tr}"
        a = pd.read_parquet(P / "aci_intervals.parquet")
        # alpha_t: state used at each origin (all origins incl. the warm-up), system economies
        ae = a[a.unique_id.isin(eco.unique_id)]
        for (g, lv, o), d in ae.groupby(["gamma", "level", "origin"]):
            al = d.alpha.to_numpy()
            alpha_rows.append({"track": tr, "gamma": g, "level": lv, "origin": o, "calibration_origin": o >= CAL_START,
                               "n_cells": len(al), **{f"alpha_p{q}": np.percentile(al, q) for q in [5, 25, 50, 75, 95]},
                               "share_at_floor": float((al <= ALPHA_CLIP[0] + 1e-12).mean()),
                               "share_at_ceiling": float((al >= ALPHA_CLIP[1] - 1e-12).mean())})
        # interval cells of the calibration origins: ACI (gamma 0.05) and uncalibrated (reconciled quantiles)
        q = pd.read_parquet(P / "quantiles.parquet", columns=["unique_id", "origin", "horizon", "target", "kind",
                                                             "q0.025", "q0.1", "q0.9", "q0.975"],
                            filters=[("kind", "==", "rec")])
        fr = []
        for lv, (ql, qh) in {80: ("q0.1", "q0.9"), 95: ("q0.025", "q0.975")}.items():
            fr.append(q[["unique_id", "origin", "horizon", "target", ql, qh]].rename(columns={ql: "lo", qh: "hi"})
                      .assign(level=lv, interval="uncal_rec"))
            fr.append(a[(a.gamma == 0.05) & (a.level == lv)][["unique_id", "origin", "horizon", "target", "lo", "hi"]]
                      .assign(level=lv, interval="aci_g0.05"))
        iv = pd.concat(fr, ignore_index=True)
        iv = iv[(iv.origin >= CAL_START) & iv.unique_id.isin(eco.unique_id)]
        iv["lo"] = iv.lo.clip(lower=0)
        iv = nest(iv, ["unique_id", "origin", "horizon", "interval"])          # nested, as in evaluate_prob.py
        iv = iv.merge(yl, on=["unique_id", "target"])
        iv = iv[WINDOWS["main"](iv.target)]
        iv["hit"] = ((iv.y >= iv.lo) & (iv.y <= iv.hi)).astype(float)
        iv["width"] = iv.hi - iv.lo
        iv = iv.merge(eco[["unique_id", "bgroup", "tercile"]], on="unique_id")
        iv["track"] = tr
        # width ratio ACI / uncal per cell
        wd = iv.pivot_table(index=["unique_id", "origin", "horizon", "level", "bgroup"], columns="interval",
                            values="width").reset_index()
        wd = wd[wd.uncal_rec > 0]
        wd["ratio"] = wd["aci_g0.05"] / wd.uncal_rec
        for (lv, h), d in pd.concat([wd, wd.assign(horizon=0)]).groupby(["level", "horizon"]):
            for bg, dd in [("all", d)] + list(d.groupby("bgroup")):
                r = dd.ratio
                width_rows.append({"track": tr, "level": lv, "horizon": "all" if h == 0 else int(h), "group": bg,
                                   "n_cells": len(r), "ratio_median": r.median(), "ratio_p25": r.quantile(.25),
                                   "ratio_p75": r.quantile(.75), "ratio_p95": r.quantile(.95),
                                   "share_ratio_gt_2": float((r > 2).mean()), "share_ratio_lt_1": float((r < 1).mean())})
        cells_all.append(iv[["track", "interval", "level", "unique_id", "origin", "horizon", "bgroup", "tercile", "hit"]])
    save(pd.DataFrame(alpha_rows), "aci_alpha_by_origin.csv",
         "ACI state alpha_t by origin (the alpha used for the interval made at that origin), system economy series x "
         "horizons 1-12: percentiles over cells, and the shares at the floor (alpha = 0.001, the widest interval) and "
         "the ceiling (alpha = 0.999). All origins 2022-12 .. 2025-06; calibration_origin marks 2023-12 on. Both "
         "gammas, both levels. Source: results/prob_main/track=*/aci_intervals.parquet.")
    save(pd.DataFrame(width_rows), "aci_width_ratio.csv",
         "Ratio of the ACI interval width (gamma 0.05, nested, lower bound clipped at 0) to the uncalibrated width "
         "(reconciled sample quantiles, clipped at 0), per cell; calibration origins 2023-12 .. 2025-06, main window, "
         "system economies; by level, horizon and measure group. Cells with zero uncalibrated width are excluded.")
    C = pd.concat(cells_all, ignore_index=True)
    rows = []
    groupings = [("horizon", ["horizon"]), ("horizon x basis", ["horizon", "bgroup"]),
                 ("horizon x tercile", ["horizon", "tercile"]), ("basis x tercile", ["bgroup", "tercile"]),
                 ("all", [])]
    for gname, keys in groupings:
        for k, d in C.groupby(["track", "interval", "level"] + keys, observed=True):
            n, kk = len(d), d.hit.sum()
            lo, hi = wilson(kk, n)
            rec = dict(zip(["track", "interval", "level"] + keys, k))
            rows.append({"grouping": gname, **rec, "n_cells": n, "n_series": d.unique_id.nunique(),
                         "n_origins": d.origin.nunique(), "coverage": kk / n, "wilson_lo95": lo, "wilson_hi95": hi})
    W = pd.DataFrame(rows)
    save(W, "coverage_wilson.csv",
         "Pooled cell coverage (hits / cells) with Wilson 95% intervals, by horizon, by horizon x measure group "
         "(residence total = RES_TOTAL_SYS; territory measures = TER_DOM, TER_INT; components A-G), by horizon x size "
         "tercile (terciles of the 1,100 system economy series by 2022 weight) and by group x tercile; uncal_rec and "
         "aci_g0.05, nested, calibration origins, main window. Wilson intervals treat cells as independent, which "
         "they are not (same origin, correlated series), so they are too narrow; coverage_block_bootstrap.csv gives block-bootstrap intervals. "
         "Pooled coverage differs slightly from T4, which averages the per-series coverage.")
    rng = np.random.default_rng(SEED)
    bt = []
    for keys in [["horizon"], ["bgroup"], []]:
        for k, d in C.groupby(["track", "interval", "level"] + keys, observed=True):
            agg = d.groupby("origin").hit.agg(["sum", "size"]).sort_index()
            s, n = agg["sum"].to_numpy(), agg["size"].to_numpy()
            T = len(agg)
            blk = min(BLOCK, T)
            est = s.sum() / n.sum()
            reps = np.empty(REPS)
            for r in range(REPS):
                ii = mbb_indices(T, blk, rng)
                reps[r] = s[ii].sum() / n[ii].sum()
            rec = dict(zip(["track", "interval", "level"] + keys, k))
            bt.append({**rec, "n_origins": T, "block_length": blk, "reps": REPS, "coverage": est,
                       "boot_lo95": np.percentile(reps, 2.5), "boot_hi95": np.percentile(reps, 97.5)})
    save(pd.DataFrame(bt), "coverage_block_bootstrap.csv",
         "Pooled coverage with 95% percentile intervals from a circular moving-block bootstrap over consecutive origins (blocks "
         "of 6 origins, or all origins if fewer; 2,000 resamples; seed 42; each resample keeps whole origins, so the "
         "cross-sectional dependence within an origin is preserved). By horizon, by measure group and overall; "
         "uncal_rec and aci_g0.05, calibration origins, main window, system economies.")


# ------------------------------------------------------------------ item 9
def item9(m):
    ps = pd.read_parquet(RES / "eval_fm_v2" / "metrics_per_series.parquet")
    o = ps[ps.unit_type == "outside"]
    rows = []
    for (w, h, mdl), g in o.groupby(["window", "horizon", "model"]):
        if h not in ["all", "1", "3", "6", "12"]:
            continue
        for grp, d in [("all", g)] + list(g.groupby("measure")):
            ok = d.MASE.notna() & np.isfinite(d.MASE)
            d = d[ok]
            rows.append({"window": w, "horizon": h, "model": mdl, "measure": grp, "n_series": len(d),
                         "n_economies": d.unit.nunique(), "MASE_median": d.MASE.median(),
                         "MASE_weighted": np.average(d.MASE, weights=d.weight_2022),
                         "RMSSE_median": d.RMSSE.median(), "RMSSE_weighted": np.average(d.RMSSE, weights=d.weight_2022),
                         "sCRPS3_median": d.sCRPS3.median(), "cov80_mean": d.cov80.mean()})
    save(pd.DataFrame(rows), "outside_system_summary.csv",
         "Accuracy of the outside-system series (published totals of economies outside the system, forecast as "
         "separate series and not reconciled): MASE, RMSSE (median, and weighted with 2022 weights within the outside "
         "level), sCRPS3 median, cov80, by model (nine base models), window, horizon and measure. Source: "
         "results/eval_fm_v2/metrics_per_series.parquet.")
    x = o[(o.window == "main") & (o.horizon == "all")].pivot_table(index=["unique_id", "unit", "measure", "weight_2022"],
                                                                   columns="model", values="MASE").reset_index()
    save(x.sort_values("weight_2022", ascending=False), "outside_system_series.csv",
         "MASE per outside-system series and model, main window, all horizons, sorted by 2022 weight (t/day).")


# ------------------------------------------------------------------ item 10
def item10(m):
    Y = load_history("data/clean")
    sc = origin_scales(Y)[["unique_id", "origin", "scale1"]]
    c = pd.read_parquet(PROB / "track=STAT" / "crps.parquet").merge(sc, on=["unique_id", "origin"])
    c["sCRPS"] = c.crps / c.scale1
    c = c[WINDOWS["main"](c.target) & np.isfinite(c.sCRPS)].merge(m, on="unique_id")
    c = c[c.unit_type == "economy"]
    ps = c.groupby(["kind", "horizon", "unique_id", "weight_2022"]).sCRPS.mean().reset_index()
    tot = ps.groupby(["kind", "horizon"]).apply(lambda d: np.average(d.sCRPS, weights=d.weight_2022)).rename("weighted")
    save(tot.reset_index(), "stat_crps_by_horizon.csv",
         "STAT track: weighted scaled CRPS over system economies by horizon, base and reconciled samples (reproduces "
         "Fig. 4b / T4).")
    W = ps.groupby(["kind", "horizon"]).weight_2022.transform("sum")
    ps["contribution"] = ps.sCRPS * ps.weight_2022 / W
    rows = []
    for h in [10, 11, 12]:
        d = ps[(ps.kind == "rec") & (ps.horizon == h)].sort_values("contribution", ascending=False).head(10)
        b = ps[(ps.kind == "base") & (ps.horizon == h)].set_index("unique_id").sCRPS
        for _, r in d.iterrows():
            rows.append({"horizon": h, "unique_id": r.unique_id, "weight_2022": r.weight_2022, "sCRPS_rec": r.sCRPS,
                         "sCRPS_base": b.get(r.unique_id, np.nan), "contribution_to_weighted": r.contribution,
                         "share_of_weighted": r.contribution / tot.loc[("rec", h)]})
    save(pd.DataFrame(rows), "stat_h11_top_series.csv",
         "Series with the largest contributions (w_i sCRPS_i / sum w) to the STAT reconciled weighted sCRPS at h = 10, "
         "11 and 12, with the same series' base sCRPS.")
    top = [r["unique_id"] for r in rows if r["horizon"] == 11][:3]
    cc = c[(c.horizon.isin([10, 11, 12])) & c.unique_id.isin(top)]
    cc = cc.pivot_table(index=["unique_id", "horizon", "origin", "target"], columns="kind", values="sCRPS").reset_index()
    save(cc, "stat_h11_top_cells.csv",
         "Origin-level scaled CRPS (base and reconciled) of the three series contributing most at h = 11, at h = 10-12.")
    # by origin: weighted mean over series of the cell sCRPS, and the h-level value without each origin
    wsum = c.groupby(["kind", "horizon", "origin"]).apply(
        lambda d: pd.Series({"weighted_cell_sCRPS": np.average(d.sCRPS, weights=d.weight_2022), "n_series": len(d)}))
    wsum = wsum.reset_index()
    rows = []
    for (k, h), d in c[c.horizon.isin([10, 11, 12])].groupby(["kind", "horizon"]):
        for o in sorted(d.origin.unique()):
            x = d[d.origin != o].groupby(["unique_id", "weight_2022"]).sCRPS.mean().reset_index()
            rows.append({"kind": k, "horizon": h, "origin_left_out": o,
                         "weighted_without_origin": np.average(x.sCRPS, weights=x.weight_2022)})
    lo = pd.DataFrame(rows).merge(wsum.rename(columns={"origin": "origin_left_out"}), on=["kind", "horizon", "origin_left_out"])
    save(lo, "stat_crps_by_origin.csv",
         "STAT, h = 10-12: weighted mean over system economy series of the cell sCRPS at each origin, and the "
         "horizon's weighted sCRPS (mean over origins per series, then weighted over series, as in T4) recomputed "
         "without that origin (leave-one-origin-out).")
    hk = c[(c.unique_id == "HKG|RES_INT_TO") & (c.origin == "2024-09")].pivot_table(index="horizon", columns="kind",
                                                                                     values="sCRPS").reset_index()
    save(hk, "hkg_res_int_to_2024-09.csv",
         "Scaled CRPS of HKG RES_INT_TO at origin 2024-09 (the AutoETS base interval that dominates the joint-score "
         "means), base and reconciled samples, by horizon.")


# ------------------------------------------------------------------ item 11
def item11():
    hub = Path.home() / ".cache" / "huggingface" / "hub"

    def rev(repo):
        p = hub / f"models--{repo.replace('/', '--')}" / "refs" / "main"
        return p.read_text().strip() if p.exists() else "not in cache"
    ctx = ("full history from 2019-01 to the origin month: 48 (2022-12) to 78 (2025-06) months at the rolling "
           "origins, 90 months at the real-time origin 2026-06, 36 to 48 months at the long-horizon origins; one-step "
           "errors (W) 12 to 77 months, 2020-01 .. 2026-06 (fm_onestep.py)")
    rows = [
        {"model": "Chronos-Bolt (base)", "checkpoint": "amazon/chronos-bolt-base", "revision": rev("amazon/chronos-bolt-base"),
         "package": "chronos-forecasting 2.3.2 (BaseChronosPipeline.predict_quantiles)",
         "max_context_of_model": "2048 (config.json context_length)", "context_used": ctx,
         "quantile_head": "direct quantile outputs 0.1, 0.2, ..., 0.9 (config.json); horizon 12 within the native "
                          "prediction_length 64, one pass",
         "point_forecast": "median q0.5 (model mean kept as model_point)", "batch_size": "256 series (CHRONOS_BATCH)",
         "other": "float32, device_map cuda, no covariates, no fine-tuning",
         "source": "src/fm_forecasts.py; HF cache refs/main"},
        {"model": "TimesFM 3.0", "checkpoint": "google/timesfm-3.0-pytorch", "revision": rev("google/timesfm-3.0-pytorch"),
         "package": "timesfm 3.0.2 (TimesFM3Forecaster.predict_batch)",
         "max_context_of_model": "no fixed maximum in the package; context left-padded to whole input patches of 32",
         "context_used": ctx,
         "quantile_head": "quantile outputs 0.1 ... 0.9 (config.json), output patch 64; sort_quantiles=True",
         "point_forecast": "median (forecast output)", "batch_size": "package default per_core_batch_size 4",
         "other": "package defaults: use_symmetric_averaging=False, make_positive=False, use_znorm=False, "
                  "padding_mode='none'; float32; no covariates (no model-card setting differs from these)",
         "source": "src/fm_forecasts.py; timesfm3/torch/timesfm3_forecaster.py"},
        {"model": "TimesFM 2.5 (FM track)", "checkpoint": "google/timesfm-2.5-200m-pytorch",
         "revision": rev("google/timesfm-2.5-200m-pytorch"),
         "package": "timesfm 3.0.2 (TimesFM_2p5_200M_torch.forecast)",
         "max_context_of_model": "max_context=1024 (ForecastConfig of the model card)", "context_used": ctx,
         "quantile_head": "use_continuous_quantile_head=True; deciles 0.1 ... 0.9 plus a mean channel; "
                          "fix_quantile_crossing=True; max_horizon=256 (one pass)",
         "point_forecast": "median (mean channel stored as mean_head)",
         "batch_size": "per_core_batch_size=256",
         "other": "normalize_inputs=True, force_flip_invariance=True, infer_is_positive=True, torch_compile=False "
                  "(speed only), float32; no covariates",
         "source": "src/fm_forecasts.py (TIMESFM25_CONFIG)"},
    ]
    vers = Path("results/fm_environment_packages.txt")
    if vers.exists():
        v = [l.strip() for l in vers.read_text().splitlines() if re.match(r"^(torch|timesfm|chronos-forecasting|transformers|huggingface_hub)==", l.strip())]
        for r in rows:
            r["environment"] = "dl-gpu: " + ", ".join(v)
    save(pd.DataFrame(rows), "fm_settings.csv",
         "Inference settings of the foundation models (zero-shot): checkpoint and exact revision (Hugging Face cache "
         "refs/main), package, context, quantile head, batch size and other "
         "settings. All series of one origin are forecast in one call; seeds 42.")


# ------------------------------------------------------------------ item 13
def item13():
    h = pd.read_parquet("data/clean/panel_system_wide_v2.parquet")
    sh = (h["W|TER_INT"] / (h["W|TER_DOM"] + h["W|TER_INT"]) * 100).rename("share")
    d = sh.reset_index()
    d.columns = ["period", "share"]
    d["year"], d["month"] = d.period.str[:4].astype(int), d.period.str[5:7].astype(int)
    d = d[d.year <= 2025]
    t = d.pivot(index="year", columns="month", values="share")
    t["annual_mean"] = t.mean(axis=1)
    t["jan_minus_mean_of_dec_feb"] = np.nan
    for y in t.index:
        dec = sh.get(f"{y - 1}-12", np.nan)
        t.loc[y, "jan_minus_mean_of_dec_feb"] = t.loc[y, 1] - np.nanmean([dec, t.loc[y, 2]])
    t.columns = [f"m{c:02d}" if isinstance(c, (int, np.integer)) else c for c in t.columns]
    save(t.reset_index(), "intl_share_by_month.csv",
         "World international share of the territory total, TER_INT / (TER_DOM + TER_INT) in %, by year (rows) and "
         "calendar month (m01..m12), 2019-2025, with the annual mean and the January value minus the mean of the "
         "preceding December and the following February (the recurring January dip; December 2018 is not in the "
         "data, so 2019 uses February only). Source: data/clean/panel_system_wide_v2.parquet.")
    # Switzerland: published components by month and why the structural-zero rule cannot fill them
    mm = pd.read_csv("data/clean/oecd_maritime_co2_monthly_v1.csv", dtype={"period": str})
    c = mm[(mm.area_code == "CHE") & (mm.vessel == "ALL_VESSELS")].pivot(index="period", columns="component", values="co2_t")
    months = pd.period_range("2019-01", "2026-06", freq="M").astype(str)
    c = c.reindex(months)
    comp = {"A": "RES_DOM_IN", "B": "RES_DOM_OUT", "C": "NRES_DOM_IN", "D": "RES_INT_FROM", "E": "RES_INT_TO",
            "F": "RES_INT_OUT", "G": "NRES_INT_FROM"}
    totals = {"RES_TOTAL": "ABDEF", "TER_DOM": "AC", "TER_INT": "DG"}
    for k in list(comp.values()) + list(totals):
        if k not in c:
            c[k] = np.nan
    z = pd.read_csv(RES / "structural_zeros.csv")
    z = set(z[z.unit == "CHE"].apply(lambda r: (r.measure, r.period), axis=1))
    rows = []
    for p in months:
        for L, k in comp.items():
            pub = pd.notna(c.at[p, k])
            filled = (k, p) in z
            reason = ""
            if not pub and not filled:
                why = []
                for T, parts in totals.items():
                    if L not in parts:
                        continue
                    tv = c.at[p, T]
                    if pd.isna(tv):
                        why.append(f"{T} not published")
                        continue
                    other = sum(c.at[p, comp[x]] for x in parts if x != L and pd.notna(c.at[p, comp[x]]))
                    miss = [x for x in parts if x != L and pd.isna(c.at[p, comp[x]])]
                    gap = tv - other
                    if abs(gap) > max(1.0, 1e-4 * abs(tv)):
                        why.append(f"{T} published but exceeds the published parts by {gap:,.0f} t"
                                   + (f" (also missing: {''.join(miss)})" if miss else ""))
                reason = "; ".join(why)
            rows.append({"period": p, "letter": L, "component": k, "published": pub, "structural_zero": filled,
                         "missing_after_structural_zeros": not pub and not filled, "why_rule_cannot_fill": reason})
    sw = pd.DataFrame(rows)
    save(sw, "switzerland_components.csv",
         "Switzerland (CHE), 2019-01 .. 2026-06: for each component and month, published or not, set to zero by the "
         "structural-zero rule (results/structural_zeros.csv) or still missing, and the reason the rule cannot fill it: a zero is only allowed "
         "when every published total containing the component (RES_TOTAL for A, B, D, E, F; TER_DOM for A, C; TER_INT "
         "for D, G) is published and equals the sum of its other published parts within 0.01% or 1 t. Source: "
         "data/clean/oecd_maritime_co2_monthly_v1.csv.")
    s = sw.assign(year=sw.period.str[:4]).groupby(["year", "letter"]).agg(
        published=("published", "sum"), structural_zero=("structural_zero", "sum"), missing=("missing_after_structural_zeros", "sum")).reset_index()
    save(s, "switzerland_summary.csv", "switzerland_components.csv counted by year and component (months).")


def write_md():
    """supplementary.md: one entry per output file (descriptions collected in DOC, plus files of earlier runs)."""
    lines = ["# Supplementary analyses", "",
             "Built by `src/supplementary_analyses.py` from stored results; nothing was refitted. "
             "Abbreviations: main = main evaluation window (targets to 2025-12 without 2025-09); robA = main window "
             "including 2025-09; robB = 2026-01 to 2026-06 (stress test). FM = TimesFM 2.5 + MinT (W from its own "
             "one-step errors); STAT = AutoETS + MinT. Weighted = mean weighted by each series' mean per-day emissions "
             "in 2022, within one level. Scaled metrics are divided by the in-sample seasonal-naive MAE at the origin.",
             "", f"Sample-based items read `{PROB.as_posix()}` and `{EVALP.as_posix()}`.", ""]
    old = {}
    mdp = OUT / "supplementary.md"
    if mdp.exists():                                    # keep the entries of items run earlier
        for blk in mdp.read_text(encoding="utf-8").split("\n## ")[1:]:
            name, _, body = blk.partition("\n")
            old[name.strip().strip("`")] = body.strip()
    for f in sorted(p.name for p in OUT.iterdir() if p.suffix in (".csv", ".parquet")):
        lines += [f"## `{f}`", "", DOC.get(f, old.get(f, "")), ""]
    mdp.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main():
    global OVERWRITE, OUT, PROB, EVALP
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13])
    ap.add_argument("--prob-dir", default=None, help="samples-based inputs (default results/prob_main_exact)")
    ap.add_argument("--eval-dir", default=None, help="evaluate_prob.py --nest output (default results/eval_prob_exact_nested)")
    ap.add_argument("--out", default=None, help="output folder (default results/supplementary)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    OVERWRITE = args.overwrite
    OUT = Path(args.out) if args.out else OUT
    PROB = Path(args.prob_dir) if args.prob_dir else PROB
    EVALP = Path(args.eval_dir) if args.eval_dir else EVALP
    log = RunLog("supplementary_analyses.py", vars(args))
    OUT.mkdir(parents=True, exist_ok=True)
    first = {1: "point_by_window.csv", 2: "OLS_tracks.csv", 3: "friedman_nemenyi.csv",
             4: "dm_per_series.parquet", 5: "mcs_pvalues.csv", 6: "calibration_variants.csv",
             7: "T4_origins.csv", 8: "aci_alpha_by_origin.csv", 9: "outside_system_summary.csv",
             10: "stat_crps_by_horizon.csv", 11: "fm_settings.csv",
             13: "intl_share_by_month.csv"}
    m = meta()
    cells = None
    for it in args.items:
        if (OUT / first[it]).exists() and not OVERWRITE:
            print(f"item {it}: outputs exist, skipped")
            continue
        print(f"---- item {it}")
        if it in (1, 4) and cells is None:
            cells = point_cells(m)
        {1: lambda: item1(m, cells), 2: item2, 3: item3, 4: lambda: item4(cells), 5: item5, 6: item6, 7: item7,
         8: lambda: item8(m), 9: lambda: item9(m), 10: lambda: item10(m), 11: item11, 13: item13}[it]()
        write_md()
    log.finish(f"items {args.items}")


if __name__ == "__main__":
    main()
