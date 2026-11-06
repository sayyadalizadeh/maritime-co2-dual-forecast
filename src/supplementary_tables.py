"""
supplementary_tables.py: the additional tables of the paper in one Markdown file.

The paper's Supplementary Tables S1-S45 are in results/paper/SUPPLEMENTARY_TABLES.md, numbered in the order the paper
cites them; that file is part of the repository and is not written by this script. This script renders every table
computed from the stored results; those that appear among S1-S45 (IN_SUPPLEMENT below, with their table number) are
skipped, and the others are written, without numbers, to results/paper/ADDITIONAL_TABLES.md.

Everything is read from stored results; nothing is refitted or recomputed. The script only selects rows and columns,
pivots, rounds and formats. Each table has a one-line caption, its source file(s) and a compact Markdown table
(headline rows and columns; the full tables stay in the source CSVs).

Sources:
  results/supplementary/            supplementary analyses (supplementary_analyses.py; definitions in
                                    results/supplementary/supplementary.md); the sample-based ones use the exactly
                                    reconciled samples
  results/conformal_residence/      coverage of the residence totals, pooled split conformal
  results/paper/T6b_hstar_variants.csv, fig4b_stat_crps_without_2024-09.csv   H* variants; the Fig. 4b diagnostic
  results/w_diagnostics/            diagnostics of W
  results/fm_postrelease/           post-release evaluation of the foundation models
  results/ablation/                 constraint ablation
  results/vintage/                  vintage comparison and paired test at origin 2026-06
  results/sample_coherence/         coherence of the evaluation samples with negatives set to zero
  results/exact_eval/, results/exact_samples/, results/compare/, results/prob_main_exact/
                                    exact non-negative QP of the samples; zeroed against exact samples

Every probabilistic table uses exactly reconciled non-negative samples. The test on later-published data at origin
2026-06 scores the intervals of the final forecasts, data file S1 (ACI state extended with the calibration origins);
the version with the state of the evaluation origins only is shown next to it as robustness.

Output: results/paper/ADDITIONAL_TABLES.md
Usage:
  python src/supplementary_tables.py [--overwrite]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from runlog import RunLog, out_path

WIN = {"main": "main", "robA": "main incl. 2025-09", "robB": "2026-01..06 (stress)"}
MODEL = {"SeasonalNaive": "Seasonal naive", "RandomWalkWithDrift": "Drift", "AutoETS": "ETS", "AutoTheta": "Theta",
         "AutoARIMA": "ARIMA", "LightGBM": "LightGBM", "ChronosBolt": "Chronos-Bolt", "TimesFM": "TimesFM 3.0",
         "TimesFM25": "TimesFM 2.5"}
TRACKS = {"FM": ("TimesFM25", "MinT"), "STAT": ("AutoETS", "MinT")}
INTERVAL = {"uncal_base": "uncalibrated, base samples", "uncal_rec": "uncalibrated, reconciled samples",
            "aci_g0.05": "ACI gamma 0.05 (main)", "aci_g0.01": "ACI gamma 0.01", "conformal": "pooled split conformal"}
H4 = ["1", "3", "6", "12", "all"]
PLAIN = {"year", "horizon", "rank", "h", "origins"}                     # integer columns printed without separators

# Tables rendered here that are among the paper's Supplementary Tables (results/paper/SUPPLEMENTARY_TABLES.md):
# caption -> table number there. They are not written to ADDITIONAL_TABLES.md.
IN_SUPPLEMENT = {
    "Ranks of the 23 base/method combinations by weighted and median MASE in each window (1 = best)": "S14",
    "Spearman correlation of the method ranks between windows": "S15",
    "Friedman test and Nemenyi critical difference over base / WLS / MinT, per base, main window": "S20",
    "Diebold-Mariano tests per system economy series, absolute error loss, main window: counts of significant "
    "differences (5%)": "S16",
    "Diebold-Mariano tests per system economy series, scaled CRPS loss, main window: counts of significant "
    "differences (5%)": "S29",
    "Model Confidence Set p-values, base x reconciliation method, system economies, main window": "S21",
    "Model Confidence Set p-values of the nine base models (system economies and all 1,170 series)": "S11",
    "Number of origins behind each T4 column, by horizon (both tracks)": "S8",
    "Pooled coverage with 95% moving-block bootstrap intervals over origins (block 6), by measure group": "S30",
    "ACI coverage by measure group and size tercile (terciles of the 1,100 series by 2022 weight)": "S31",
    "Coverage of the residence totals (RES_TOTAL_SYS of the 110 system economies): pooled split conformal against "
    "ACI and uncalibrated intervals": "S32",
    "Coverage of the residence totals by horizon": "S33",
    "Inference settings of the foundation models (zero-shot)": "S7",
    "World international share of the territory total (%), by year and month": "S2",
    "Switzerland: months per component published / set to zero by the structural-zero rule / still missing": "S3",
    "H* for every variant of the skill test": "S41",
    "W by origin: shrinkage intensity, condition numbers on a unit diagonal, floored and dead series": "S44",
    "Similarity of the one-step error correlations (W) with the rolling-origin h-step error correlations": "S19",
    "Dead series (fixed at zero at an origin) that later become non-zero": "S45",
    "Post-release evaluation: target months after each model's release available in the stored forecasts": "S12",
    "Post-release evaluation: MASE, median / weighted over system economies, on the same post-release target "
    "months": "S13",
    "Constraint ablation: MASE of MinT exact QP under three constraint sets, system economies, main window": "S22",
    "Constraint ablation: MASE of the World series by constraint set": "S23",
    "Constraint ablation: violations of the within-unit identities left by each set (%)": "S24",
    "Constraint ablation: World residence-territory gap and B(W) - C(W), % of the World total": "S25",
    "October 2025 against October 2026 vintage: coverage of the comparison": "S4",
    "Revisions from the October 2025 to the October 2026 vintage (%)": "S5",
    "The ten most revised economy-months": "S6",
    "Paired comparison of FM and STAT at the real-time origin 2026-06 (published residence totals, 110 economies, "
    "2026-07 to 2026-09)": "S39",
    "Table 4 quantities, samples with negatives set to zero against exactly reconciled samples, all horizons": "S34",
    "FM minus STAT weighted scaled CRPS, origin-block bootstrap, zeroed and exact samples": "S35",
    "Weighted scaled CRPS with and without origin 2024-09 (diagnostic only, no origin is excluded)": "S36",
    "Coverage events with the outcome on an interval bound within 1e-9 t/day": "S37",
}


def fmt(v, d, plain=False):
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else ""
    if isinstance(v, (int, np.integer)):
        return str(v) if plain else f"{v:,}"
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return ""
        if np.isinf(v):
            return "inf"
        if float(v).is_integer() and d == 0:
            return fmt(int(v), d, plain)
        return f"{v:,.{d}f}"
    return str(v).replace("|", "/")


def md(df, digits=3, cols=None):
    """Markdown table; floats rounded to `digits`, or per column with cols = {column: digits}."""
    cols = cols or {}
    out = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    for r in df.itertuples(index=False):
        out.append("| " + " | ".join(fmt(v, cols.get(c, digits), c in PLAIN) for c, v in zip(df.columns, r)) + " |")
    return "\n".join(out)


class Doc:
    """Collects the additional tables. A table among the Supplementary Tables (IN_SUPPLEMENT) is recorded in
    `skipped` and not written; a section heading is written only when one of its tables is."""

    def __init__(self):
        self.parts, self.index, self.n, self.skipped, self.pending = [], [], 0, [], None

    def section(self, title):
        self.pending = title

    def table(self, caption, source, df, digits=3, cols=None, note=None):
        if caption in IN_SUPPLEMENT:
            self.skipped.append(IN_SUPPLEMENT[caption])
            return
        if self.pending:
            self.parts.append(f"## {self.pending}\n")
            self.index.append(f"- {self.pending}")
            self.pending = None
        self.n += 1
        self.index.append(f"  - {caption}")
        src = "; ".join(f"`{s}`" for s in ([source] if isinstance(source, str) else source))
        body = [f"### {caption}", "", f"Source: {src}", ""]
        if note:
            body += [note, ""]
        body += [md(df, digits, cols), ""]
        self.parts.append("\n".join(body))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/paper/ADDITIONAL_TABLES.md")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    log = RunLog("supplementary_tables.py", vars(args))
    RES = Path(args.results)

    def rd(path):
        p = RES / path
        return pd.read_csv(p), f"results/{path}"

    D = Doc()

    # ================================================================== A. point accuracy and tests
    D.section("Point accuracy, robustness windows and tests")
    a, src = rd("supplementary/point_by_window.csv")
    a = a[(a.level == "economies") & (a.grouping == "all") & (a.metric == "MASE") & (a.horizon == "all")]
    keep = a[(a.method == "base") | a.apply(lambda r: (r.base, r.method) in TRACKS.values(), axis=1)].copy()
    keep["forecast"] = [("FM (TimesFM 2.5 + MinT)" if b == "TimesFM25" else "STAT (ETS + MinT)") if m == "MinT"
                        else MODEL[b] for b, m in zip(keep.base, keep.method)]
    p = keep.pivot_table(index="forecast", columns="window", values=["weighted", "median"])
    t = pd.DataFrame({"forecast": p.index})
    for w in WIN:
        t[f"weighted, {WIN[w]}"] = p[("weighted", w)].to_numpy()
    for w in WIN:
        t[f"median, {WIN[w]}"] = p[("median", w)].to_numpy()
    D.table("MASE of the base models and the two tracks by evaluation window, system economies, horizons 1-12 pooled",
            src, t.sort_values(f"weighted, main"),
            note="Weighted = 2022 per-day emission weights. Reconciled = exact non-negative QP. Every method and "
                 "the World series one by one are in the source file.")
    a2, _ = rd("supplementary/point_by_window.csv")
    rows = []
    for tr, (b, m) in TRACKS.items():
        for lvl, uid in [("economies", "all"), ("world", "RES_TOTAL_SYS"), ("world", "TER_INT")]:
            x = a2[(a2.base == b) & (a2.method == m) & (a2.metric == "MASE") & (a2.level == lvl) &
                   ((a2.grouping == "all") if lvl == "economies" else (a2.grp == uid))]
            for w in WIN:
                xw = x[x.window == w].set_index("horizon")["weighted" if lvl == "economies" else "mean"]
                rows.append({"track": tr, "series": "system economies, weighted" if lvl == "economies" else f"World {uid}",
                             "window": WIN[w], **{f"h {h}": xw.get(h, np.nan) for h in H4}})
    D.table("MASE of the two tracks by horizon and window: system economies (weighted) and two World series", src,
            pd.DataFrame(rows), note="The 2026 window has targets 2026-01 ... 2026-06 only, which the origins up to 2025-06 "
                                     "reach from h = 7 on.")
    b, src = rd("supplementary/method_ranks.csv")
    t = b[["combo", "weighted_main", "rank_weighted_main", "rank_weighted_robA", "rank_weighted_robB",
           "rank_median_main", "rank_median_robA", "rank_median_robB"]].copy()
    t[t.columns[2:]] = t[t.columns[2:]].astype(float)
    t.columns = ["base + method", "weighted MASE, main", "rank weighted, main", "rank weighted, incl. 2025-09",
                 "rank weighted, 2026", "rank median, main", "rank median, incl. 2025-09", "rank median, 2026"]
    D.table("Ranks of the 23 base/method combinations by weighted and median MASE in each window (1 = best)", src, t,
            cols={c: 1 for c in t.columns[2:]})
    c, src = rd("supplementary/rank_correlation.csv")
    c["window_1"], c["window_2"] = c.window_1.map(WIN), c.window_2.map(WIN)
    D.table("Spearman correlation of the method ranks between windows", src, c, cols={"p_value": 4})
    f, src = rd("supplementary/weighted_MAE_kt.csv")
    f = f[(f.method == "base") | f.apply(lambda r: (r.base, r.method) in TRACKS.values(), axis=1)].copy()
    f["forecast"] = [("FM" if bb == "TimesFM25" else "STAT") if m == "MinT" else MODEL[bb] for bb, m in zip(f.base, f.method)]
    p = f.pivot_table(index="forecast", columns="window", values=["wMAE_kt_per_day", "MAE_W_RES_TOTAL_SYS_kt_per_day"])
    t = pd.DataFrame({"forecast": p.index})
    for w in WIN:
        t[f"weighted MAE, {WIN[w]}"] = p[("wMAE_kt_per_day", w)].to_numpy()
    for w in WIN:
        t[f"World residence total MAE, {WIN[w]}"] = p[("MAE_W_RES_TOTAL_SYS_kt_per_day", w)].to_numpy()
    D.table("Mean absolute error in kt CO2 per day: weighted over system economy series, and the World residence total",
            src, t.sort_values("weighted MAE, main"), digits=2)
    o, src = rd("supplementary/OLS_all_bases_summary.csv")
    o = o[o.window == "main"].assign(base=lambda d: d.base.map(MODEL))
    D.table("OLS reconciliation as a benchmark: MASE over system economies, main window, all horizons", src,
            o[["base", "weighted_base", "weighted_OLS", "weighted_WLS", "weighted_MinT", "median_base", "median_OLS",
               "median_WLS", "median_MinT"]])
    fr, src = rd("supplementary/friedman_nemenyi.csv")
    fr = fr[fr.treatments == "base, WLS, MinT"].assign(base=lambda d: d.base.map(MODEL))
    fr = fr[fr.horizon.isin(H4)].assign(k=lambda d: d.horizon.map(H4.index)).sort_values(["base", "k"]).drop(columns="k")
    D.table("Friedman test and Nemenyi critical difference over base / WLS / MinT, per base, main window", src,
            fr[["base", "horizon", "n_series", "friedman_chi2", "friedman_p", "nemenyi_cd", "mean_rank_base",
                "mean_rank_WLS", "mean_rank_MinT"]], cols={"friedman_chi2": 1, "friedman_p": 4},
            note="Blocks = system economy series; loss = MASE. The four-treatment version with OLS is in the source file.")
    dm, src = rd("supplementary/dm_counts.csv")
    dm = dm[dm.unit_type == "economy"]
    for loss, lab in [("absolute error", "absolute error"), ("scaled CRPS", "scaled CRPS")]:
        x = dm[dm.loss.str.startswith(loss)].copy()
        x["better / worse"] = x.n_better.astype(str) + " / " + x.n_worse.astype(str)
        x["after BH"] = x.n_better_bh.astype(str) + " / " + x.n_worse_bh.astype(str)
        x["lag-0 fallback"] = x.n_lag0_fallback.map(lambda v: "not recorded" if pd.isna(v) else str(int(v)))
        D.table(f"Diebold-Mariano tests per system economy series, {lab} loss, main window: counts of significant "
                f"differences (5%)", src,
                x[["comparison", "horizon", "n_tests", "n_untestable", "better / worse", "after BH", "lag-0 fallback",
                   "mean_n_origins"]], cols={"mean_n_origins": 0},
                note="HLN-corrected DM, rectangular kernel with lag h-1, t(n-1), two-sided. 'better' = first forecast "
                     "better. BH = Benjamini-Hochberg at 5% within comparison x horizon (1,100 tests). Lag-0 fallback = "
                     "tests where the long-run variance was not positive and gamma_0 was used"
                     + ("; not recorded for the stored CRPS tests." if loss == "scaled CRPS" else "."))
    dc, src = rd("supplementary/dm_base_models_vs_ETS.csv")
    dc = dc[dc.unit_type == "economy"].copy()
    dc["v"] = dc.n_better_bh.astype(str) + " / " + dc.n_worse_bh.astype(str)
    t = dc.pivot(index="model", columns="horizon", values="v").reset_index()
    t.columns = ["model"] + [f"h {h}" for h in t.columns[1:]]
    t["model"] = t.model.map(MODEL)
    D.table("Base models against ETS, Diebold-Mariano on absolute error, system economy series: better / worse than "
            "ETS after BH", src, t)
    mc, src = rd("supplementary/mcs_pvalues.csv")
    x = mc[(mc.source == "eval_fm_w")].pivot(index="candidate", columns="horizon_group", values="mcs_pvalue")[["all", "1", "3", "6", "12"]]
    x.columns = [f"p, h {c}" for c in x.columns]
    loss = mc[(mc.source == "eval_fm_w") & (mc.horizon_group == "all")].set_index("candidate").mean_loss
    t = x.assign(mean_loss_all=loss).sort_values("mean_loss_all").reset_index()
    D.table("Model Confidence Set p-values, base x reconciliation method, system economies, main window", src, t,
            cols={c: 4 for c in t.columns if c.startswith("p,")},
            note="arch 8.0.0, 90% MCS, range statistic, stationary bootstrap with mean block int(sqrt(T)) (5 for 31 "
                 "origins, 4 for 24), 10,000 replications, seed 42. Loss per origin = mean scaled absolute error. "
                 "[ETS-W] = W of AutoETS. In the 90% set: p > 0.10.")
    x = mc[mc.source == "eval_fm_v2"].pivot_table(index="candidate", columns=["scope", "horizon_group"], values="mcs_pvalue")
    t = pd.DataFrame({"model": [MODEL[m] for m in x.index]})
    for sc in ["system_economies", "all_series"]:
        for h in ["all", "1", "12"]:
            t[f"p, {sc.replace('_', ' ')}, h {h}"] = x[(sc, h)].to_numpy()
    D.table("Model Confidence Set p-values of the nine base models (system economies and all 1,170 series)", src, t, digits=4)

    # ================================================================== B. probabilistic forecasts and calibration
    D.section("Probabilistic forecasts and calibration")
    d, src = rd("supplementary/prob_tracks_by_window.csv")
    d = d[(d.level == "economies") & (d.grouping == "all") & (d.horizon == "all")]
    rows = []
    for tr in TRACKS:
        for w in WIN:
            x = d[(d.track == tr) & (d.window == w)]
            g = lambda m, it, col: x[(x.metric == m) & (x.interval == it)][col].iloc[0]
            rows.append({"track": tr, "window": WIN[w], "sCRPS base, weighted": g("sCRPS", "base", "weighted"),
                         "sCRPS rec, weighted": g("sCRPS", "rec", "weighted"), "sCRPS rec, median": g("sCRPS", "rec", "median"),
                         "cov80 ACI": g("cov80", "aci_g0.05", "mean"), "cov95 ACI": g("cov95", "aci_g0.05", "mean"),
                         "cov80 conformal": g("cov80", "conformal", "mean"), "cov95 conformal": g("cov95", "conformal", "mean")})
    D.table("Probabilistic scores of the two tracks by window, system economies, horizons 1-12 pooled", src,
            pd.DataFrame(rows), note="sCRPS over all 31 origins; coverage = mean over series of the per-series coverage "
                                     "(as in T4), calibration origins 2023-12 ... 2025-06; intervals nested.")
    e, src = rd("supplementary/prob_base_models_by_window.csv")
    e = e[(e.source == "eval_fm_v2") & (e.level == "economies") & (e.grouping == "all") & (e.horizon == "all")]
    p = e.pivot_table(index="model", columns=["metric", "window"], values=["weighted", "median", "mean"])
    t = pd.DataFrame({"model": [MODEL[m] for m in p.index], "sCRPS3 weighted, main": p[("weighted", "sCRPS3", "main")].to_numpy(),
                      "sCRPS3 median, main": p[("median", "sCRPS3", "main")].to_numpy(),
                      "sCRPS3 weighted, 2026": p[("weighted", "sCRPS3", "robB")].to_numpy(),
                      "cov80, main": p[("mean", "cov80", "main")].to_numpy(), "cov80, 2026": p[("mean", "cov80", "robB")].to_numpy()})
    D.table("Base models' own probabilistic scores (no calibration), system economies, all horizons", src,
            t.sort_values("sCRPS3 weighted, main"),
            note="sCRPS3 = mean pinball loss over the quantiles 0.1, 0.5, 0.9 shared by all models, scaled. LightGBM gives "
                 "point forecasts only.")
    ca, src = rd("supplementary/calibration_variants.csv")
    rows = []
    for (tr, it), x in ca.groupby(["track", "interval"]):
        x = x.set_index("horizon")
        rows.append({"track": tr, "interval": INTERVAL[it], "cov80 all h": x.at["all", "cov80_mean"],
                     "cov95 all h": x.at["all", "cov95_mean"], "cov80 h 1": x.at["1", "cov80_mean"],
                     "cov80 h 12": x.at["12", "cov80_mean"], "cov95 h 12": x.at["12", "cov95_mean"],
                     "sWinkler80 weighted": x.at["all", "sWinkler80_weighted"], "sWinkler95 weighted": x.at["all", "sWinkler95_weighted"]})
    D.table("Calibration variants: coverage and scaled Winkler score of every interval type, system economies, main window",
            src, pd.DataFrame(rows), note="Calibration origins 2023-12 ... 2025-06; 80% nested inside 95%. Every "
                                          "horizon is in the source file.")
    j, src = rd("supplementary/joint_scores.csv")
    p = j.pivot_table(index=["track", "kind"], columns="horizon", values=["ES_median", "ES", "VS_median"])
    t = p.reset_index()
    t.columns = ["track", "samples"] + [f"{a} h {h}" for a, h in p.columns]
    t = t[["track", "samples"] + [f"ES_median h {h}" for h in H4] + ["ES h all"] + [f"VS_median h {h}" for h in H4]]
    t.columns = [c.replace("ES_median", "ES median").replace("VS_median", "VS median").replace("ES h all", "ES mean, h all")
                 for c in t.columns]
    D.table("Energy and variogram scores, base and reconciled samples, median over origin x horizon cells", src, t,
            digits=1, note="Reconciled samples: exact non-negative QP; VS on scaled values. Means for every horizon are in "
                           "the source file.")
    s7, src = rd("supplementary/T4_origins.csv")
    s7 = s7[s7.track == "FM"].pivot(index="horizon", columns="metric", values="n_origins").reset_index()
    s7.columns.name = None
    D.table("Number of origins behind each T4 column, by horizon (both tracks)", src, s7,
            note="sCRPS, ES and VS: origins 2022-12 ... 2025-06; coverage and Winkler: calibration origins 2023-12 ... "
                 "2025-06. Targets after 2025-12 and 2025-09 excluded.")
    al, src = rd("supplementary/aci_alpha_by_origin.csv")
    al = al[(al.gamma == 0.05) & al.calibration_origin]
    t = al.pivot(index="origin", columns=["track", "level"], values="share_at_floor").mul(100)
    t.columns = [f"{a} {b}%, % at floor" for a, b in t.columns]
    t = t.reset_index()
    D.table("ACI state at each calibration origin: share of system economy cells with alpha at its floor 0.001 (%), "
            "gamma 0.05", src, t, digits=1, note="No cell reaches the ceiling 0.999. Percentiles of alpha and gamma 0.01 "
                                                  "are in the source file.")
    wr, src = rd("supplementary/aci_width_ratio.csv")
    wr = wr[wr.horizon == "all"]
    D.table("Ratio of the ACI interval width to the uncalibrated width, per cell, calibration origins", src,
            wr[["track", "level", "group", "n_cells", "ratio_median", "ratio_p25", "ratio_p75", "ratio_p95",
                "share_ratio_gt_2", "share_ratio_lt_1"]])
    bb, src = rd("supplementary/coverage_block_bootstrap.csv")
    bb = bb[bb.horizon.isna()].copy()
    bb["group"] = bb.bgroup.fillna("all system economy series")
    bb = bb.assign(k=bb.bgroup.notna()).sort_values(["k", "track", "interval", "level"], kind="stable").drop(columns="k")
    bb["interval"] = bb.interval.map(INTERVAL)
    D.table("Pooled coverage with 95% moving-block bootstrap intervals over origins (block 6), by measure group", src,
            bb[["track", "interval", "level", "group", "coverage", "boot_lo95", "boot_hi95"]],
            note="Calibration origins 2023-12 ... 2025-06, main window. By horizon in the source file.")
    wl, src = rd("supplementary/coverage_wilson.csv")
    wl = wl[(wl.grouping == "basis x tercile") & (wl.interval == "aci_g0.05")]
    t = wl.pivot_table(index=["track", "level", "bgroup"], columns="tercile", values="coverage")[["small", "medium", "large"]]
    t = t.reset_index()
    t.columns = ["track", "level", "group", "coverage, small", "coverage, medium", "coverage, large"]
    D.table("ACI coverage by measure group and size tercile (terciles of the 1,100 series by 2022 weight)", src, t,
            note="Wilson intervals in the source file assume independent cells and are too narrow.")
    cr, src = rd("conformal_residence/coverage_residence_totals.csv")
    cr["interval"] = cr.interval.map(INTERVAL)
    D.table("Coverage of the residence totals (RES_TOTAL_SYS of the 110 system economies): pooled split conformal against "
            "ACI and uncalibrated intervals", src,
            cr[["track", "interval", "level", "n_cells", "coverage_pooled", "boot_lo95", "boot_hi95",
                "median_width_ratio_to_aci", "world_coverage"]],
            note="Calibration origins 2023-12 ... 2025-06, horizons 1-12, main window. Bootstrap: circular moving blocks "
                 "of 6 origins, 2,000 resamples. Width ratio = median per-cell width relative to the ACI interval. World "
                 "= W|RES_TOTAL_SYS alone.")
    ch, src = rd("conformal_residence/coverage_by_horizon.csv")
    ch = ch[ch.horizon.isin([1, 3, 6, 12])]
    t = ch.pivot_table(index=["track", "interval", "level"], columns="horizon", values="coverage_pooled").reset_index()
    t.columns = ["track", "interval", "level"] + [f"h {h}" for h in t.columns[3:]]
    t["interval"] = t.interval.map(INTERVAL)
    D.table("Coverage of the residence totals by horizon", src, t)

    # ================================================================== C. outside system, the Fig. 4b spike, settings, data
    D.section("Outside-system series and the STAT CRPS spike")
    os_, src = rd("supplementary/outside_system_summary.csv")
    os_ = os_[(os_.window == "main") & (os_.horizon == "all") & (os_.measure == "all")].assign(model=lambda d: d.model.map(MODEL))
    D.table("Outside-system series (49 series of 25 economies, not reconciled), main window, all horizons", src,
            os_[["model", "MASE_median", "MASE_weighted", "RMSSE_median", "RMSSE_weighted", "sCRPS3_median", "cov80_mean"]]
            .sort_values("MASE_median"), note="By measure, window and horizon in the source file; per series in outside_system_series.csv.")
    s10, src = rd("supplementary/stat_crps_by_horizon.csv")
    t = s10.pivot(index="horizon", columns="kind", values="weighted").reset_index()
    t.columns = ["horizon", "weighted sCRPS, base samples", "weighted sCRPS, reconciled samples"]
    # the Fig. 4b diagnostic line (no origin is excluded from the results)
    dg, src2 = rd("paper/fig4b_stat_crps_without_2024-09.csv")
    t = t.merge(dg[["horizon", "without_2024-09"]].rename(
        columns={"without_2024-09": "reconciled samples without origin 2024-09 (diagnostic)"}), on="horizon")
    D.table("STAT track: weighted scaled CRPS over system economies by horizon (Fig. 4b)", [src, src2], t,
            note="All origins are kept in the results. The last column is the dashed diagnostic line of Fig. 4b; "
                 "at h = 12 origin 2024-09 has no target in the main window (2025-09 is excluded). Fig. 4b limits its "
                 "axis at 1.4 and marks the off-scale value at h = 11 with an arrow.")
    s10d, src = rd("supplementary/stat_crps_by_origin.csv")
    t = s10d[s10d.kind == "rec"].sort_values("weighted_cell_sCRPS", ascending=False).head(8)
    D.table("STAT, reconciled samples, h = 10-12: the origins with the largest weighted cell sCRPS, and the horizon's "
            "weighted sCRPS without that origin", src,
            t[["horizon", "origin_left_out", "weighted_cell_sCRPS", "weighted_without_origin"]])
    s10e, src = rd("supplementary/hkg_res_int_to_2024-09.csv")
    D.table("Scaled CRPS of HKG RES_INT_TO at origin 2024-09, AutoETS base and reconciled samples", src,
            s10e.rename(columns={"base": "base samples", "rec": "reconciled samples"}), digits=1)
    st, src = rd("supplementary/fm_settings.csv")
    t = st.drop(columns=["source", "environment"]).set_index("model").T.reset_index().rename(columns={"index": "setting"})
    t.columns.name = None
    D.table("Inference settings of the foundation models (zero-shot)", src, t,
            note="Foundation-model environment: " + st.environment.iloc[0].replace("dl-gpu: ", "") + ".")
    sh, src = rd("supplementary/intl_share_by_month.csv")
    D.table("World international share of the territory total (%), by year and month", src, sh, digits=1,
            note="Last column: January minus the mean of the preceding December and the following February (2019: "
                 "February only).")
    sw, src = rd("supplementary/switzerland_summary.csv")
    sw["v"] = sw.published.astype(str) + " / " + sw.structural_zero.astype(str) + " / " + sw.missing.astype(str)
    t = sw.pivot(index="year", columns="letter", values="v").reset_index()
    t.columns.name = None
    D.table("Switzerland: months per component published / set to zero by the structural-zero rule / still missing", src, t,
            note="A, B, C = domestic components; D, E, F, G = international. Month-by-month detail in switzerland_components.csv.")

    # ================================================================== D. skill and H*
    D.section("Skill and H*")
    hb, src = rd("paper/T6b_hstar_variants.csv")
    for col in [c for c in hb.columns if c.startswith(("FM", "STAT"))]:
        hb[col] = hb[col].map(lambda v: "" if pd.isna(v) else (">36" if v > 36 else str(int(v))))
    D.table("H* for every variant of the skill test", src, hb,
            note="H* = first horizon at which the lower bound of s(h) (80%, 90%, 95% columns) is at or below zero, or at "
                 "which the one-sided HAC test (Newey-West, lag h-1) no longer rejects at 5% ('test'). >36: the bound "
                 "stays above zero up to h = 36. Main row: block 6, 95%.")
    # ================================================================== E. reconciliation diagnostics
    D.section("Post-release scores and the coherence of the evaluation samples")
    wc, src = rd("w_diagnostics/w_condition_by_origin.csv")
    zv, src2 = rd("w_diagnostics/zero_variance_series.csv")
    m = wc.merge(zv[["track", "origin", "n_var_below_floor", "n_dead"]], on=["track", "origin"])
    t = m.pivot(index="origin", columns="track", values=["lambda", "cond_W_unit_diag", "cond_AWAt_unit_diag",
                                                          "n_var_below_floor", "n_dead"])
    t.columns = [f"{tr} {a}".replace("cond_W_unit_diag", "cond(W), unit diag").replace(
        "cond_AWAt_unit_diag", "cond(A W A'), unit diag").replace("n_var_below_floor", "floored").replace("n_dead", "dead")
        for a, tr in t.columns]
    t = t[[c for tr in ["FM", "STAT"] for c in t.columns if c.startswith(tr)]].reset_index()
    D.table("W by origin: shrinkage intensity, condition numbers on a unit diagonal, floored and dead series", [src, src2],
            t, cols={c: (3 if "lambda" in c else (1 if "cond" in c else 0)) for c in t.columns},
            note="Raw condition numbers (about 1e14, reflecting scale) are in the source file. Floored = residual variance "
                 "below 1e-4 (t/day)^2; dead = zero in all 24 months up to the origin, fixed at zero.")
    cs_, src = rd("w_diagnostics/correlation_similarity.csv")
    cs_ = cs_[cs_.h_step_moments == "uncentered"]
    D.table("Similarity of the one-step error correlations (W) with the rolling-origin h-step error correlations", src,
            cs_[["track", "h", "reference", "n_origins", "corr_offdiag", "rel_frobenius_offdiag", "mean_abs_offdiag_ref",
                 "mean_abs_offdiag_h"]], note="1,120 series, 626,640 off-diagonal pairs, origin 2025-06 W; centered "
                                              "versions in the source file.")
    dr, src = rd("w_diagnostics/dead_series_revival_summary.csv")
    dv, src2 = rd("w_diagnostics/dead_series_revival.csv")
    D.table("Dead series (fixed at zero at an origin) that later become non-zero", [src, src2], dr,
            note=f"Largest value reached after revival: {dv.max_per_day_t.max():.1f} t/day; per-cell detail in the second "
                 f"source file.")
    pc, src = rd("fm_postrelease/coverage.csv")
    D.table("Post-release evaluation: target months after each model's release available in the stored forecasts",
            src, pc.drop(columns=["target_months"]))
    ps, src = rd("fm_postrelease/postrelease_scores.csv")
    ps["v"] = ps["median"].map(lambda v: f"{v:.3f}") + " / " + ps.weighted.map(lambda v: f"{v:.3f}")
    for metric in ["MASE", "scaled CRPS"]:
        t = ps[ps.metric == metric].pivot_table(index=["model", "set"], columns="forecast", values="v", aggfunc="first")
        t = t.reset_index()
        t.columns.name = None
        D.table(f"Post-release evaluation: {metric}, median / weighted over system economies, on the same "
                f"post-release target months", src, t,
                note="Foundation model = base forecasts of the model in the row; FM and STAT = reconciled tracks.")
    ab, src = rd("ablation/mase_summary.csv")
    D.table("Constraint ablation: MASE of MinT exact QP under three constraint sets, system economies, main window",
            src, ab, note="Set 1 = full dual allocation; 2 = without the two world-balance rows; 3 = aggregation only.")
    aw, src = rd("ablation/mase_world_series.csv")
    t = aw.pivot(index="unique_id", columns=["track", "set"], values="MASE")
    t.columns = [f"{tr} set {s}" for tr, s in t.columns]
    D.table("Constraint ablation: MASE of the World series by constraint set", src, t.reset_index())
    av, src = rd("ablation/violations_within_unit.csv")
    av = av[av.set > 1]
    D.table("Constraint ablation: violations of the within-unit identities left by each set (%)", src,
            av[["track", "set", "total", "n_cells", "median_rel_pct", "p95_rel_pct", "max_rel_pct",
                "share_cells_rel_gt_1pct", "world_mean_rel_pct"]], cols={"max_rel_pct": 1},
            note="Relative violation = |identity residual| / total, %. Set 1 (not shown) and set 2 meet these "
                 "identities to rounding (below 1e-9%).")
    ag, src = rd("ablation/violations_world_gap_by_horizon.csv")
    ag = ag[(ag.set > 1) & ag.horizon.isin([1, 3, 6, 12])]
    D.table("Constraint ablation: World residence-territory gap and B(W) - C(W), % of the World total", src,
            ag, note="Gap = TER_DOM + TER_INT + r G - RES_TOTAL_SYS. Every horizon in the source file.")
    vs, src = rd("sample_coherence/violations_summary.csv")
    D.table("Evaluation samples with negatives set to zero (superseded by the exact samples): violations of C y = 0, % "
            "of the sample's World residence total, all 372 cells per track", src, vs.drop(columns=["max_crps_check"]), digits=4,
            cols={"max_of_cell_medians_pct": 3, "max_pct": 2, "mean_share_samples_with_zeroing": 2},
            note="The regenerated samples reproduce the stored per-cell CRPS to " +
                 f"{vs.max_crps_check.max():.1e} t/day. inf = one sample whose World residence total was set to zero.")
    vh, src = rd("sample_coherence/violations_by_horizon.csv")
    t = vh.pivot(index="horizon", columns="track", values=["median_pct", "share_gt_01pct"])
    t.columns = [f"{tr} {a}" for a, tr in t.columns]
    D.table("Evaluation samples with negatives set to zero: violation by horizon (median %, share of samples above 0.1%)",
            src, t.reset_index(), digits=4)
    qs, src = rd("sample_coherence/qp_vs_zeroed_summary.csv")
    D.table("The same 500 samples, set to zero against reconciled with the exact non-negative QP, 16 cells", src,
            qs.drop(columns=["n_series"]), cols={"es_median": 1, "es_mean": 1, "vs_median": 0, "vs_mean": 0,
                                                 "max_violation_pct": 3},
            note="Cells: origins 2023-12, 2024-06, 2024-12, 2025-06 x h = 1, 3, 6, 12; system economies.")
    qd, src = rd("sample_coherence/qp_vs_zeroed_fm_minus_stat.csv")
    D.table("FM minus STAT under the two sample treatments (negative = FM better for scores)", src, qd,
            cols={"FM_minus_STAT_es_median": 2, "FM_minus_STAT_vs_median": 1})

    # ================================================================== F. vintage and the paired test
    D.section("Data vintage and the paired test at origin 2026-06")
    vc, src = rd("vintage/coverage.csv")
    D.table("October 2025 against October 2026 vintage: coverage of the comparison", src, vc)
    rv, src = rd("vintage/revision_stats.csv")
    D.table("Revisions from the October 2025 to the October 2026 vintage (%)", src, rv,
            note="Absolute revision |2026 / 2025 - 1|; mean_pct is signed.")
    mr, src = rd("vintage/most_revised_economy_months.csv")
    D.table("The ten most revised economy-months", src,
            mr.head(10)[["area_name", "area_code", "period", "co2_t_2025", "co2_t_2026", "revision_pct"]],
            cols={"co2_t_2025": 0, "co2_t_2026": 0, "revision_pct": 1}, note="30 rows in the source file.")
    pr, src = rd("vintage/paired_2026-06.csv")
    D.table("Paired comparison of FM and STAT at the real-time origin 2026-06 (published residence totals, 110 economies, "
            "2026-07 to 2026-09)", src,
            pr[["d_variant", "forecasts", "median_MASE_FM", "median_MASE_STAT", "median_diff_FM_minus_STAT", "boot95_lo",
                "boot95_hi", "share_FM_better", "wilcoxon_stat", "wilcoxon_p"]], cols={"wilcoxon_stat": 0},
            note="Per-economy MASE = mean over the three horizons. Bootstrap: 2,000 resamples of economies, seed 42.")

    # ================================================================== G. exactly reconciled samples
    D.section("Exactly reconciled samples: solver, and zeroed against exact")
    q_eval, src = rd("exact_eval/negative_counts_summary.csv")
    qs4, src4 = rd("exact_samples/qp_summary.csv")
    rows = []
    for tr in ["FM", "STAT"]:
        x = q_eval[q_eval.track == tr]
        rows.append({"samples": "evaluation origins 2022-12 ... 2025-06", "track": tr,
                     "QPs": int(x.samples_needing_qp.sum()), "solved by a fallback stage": int(tr == "STAT")})
        for so, lab in [("calib", "calibration origins 2025-07 ... 2026-05"),
                        ("realtime", "real-time origins 2025-06 and 2026-06"), ("final", "final origin 2026-06 (STAT; the FM final samples were solved exactly when drawn)")]:
            y = qs4[(qs4.track == tr) & (qs4.source == so)]
            if len(y):
                rows.append({"samples": lab, "track": tr, "QPs": int(y.qps.sum()),
                             "solved by a fallback stage": int(y.solved_by_fallback.sum())})
    D.table("Exact non-negative QP of the samples: number of QPs and of samples that needed a fallback stage", [src, src4],
            pd.DataFrame(rows), note="Solver: quadprog (Goldfarb-Idnani) in the null space of the equality constraints, "
                                     "exact active set, multipliers checked; fallback stages: quadprog on the rescaled "
                                     "sample, then OSQP (qp_pool.py). Every solution passed the coherence, non-negativity "
                                     "and KKT checks. Per-sample detail of the evaluation origins in results/exact_eval.")
    sb, src = rd("exact_eval/T4_side_by_side.csv")
    sb = sb[sb.horizon.astype(str) == "all"]
    t = sb.pivot_table(index="quantity", columns="track", values=["zeroed", "exact"])
    t.columns = [f"{tr} {v}" for v, tr in t.columns]
    t = t[["FM zeroed", "FM exact", "STAT zeroed", "STAT exact"]].reset_index()
    D.table("Table 4 quantities, samples with negatives set to zero against exactly reconciled samples, all horizons", src,
            t, note="By horizon in the source file. Coverage = mean over series; ACI gamma 0.05, nested.")
    bt, src = rd("exact_eval/fm_minus_stat_crps_bootstrap.csv")
    D.table("FM minus STAT weighted scaled CRPS, origin-block bootstrap, zeroed and exact samples", src,
            bt[["samples", "horizon", "FM", "STAT", "difference", "lo95", "hi95", "share_below_zero"]])
    wo, src = rd("exact_eval/crps_without_2024-09.csv")
    D.table("Weighted scaled CRPS with and without origin 2024-09 (diagnostic only, no origin is excluded)", src, wo)
    ti = []
    for tr in ["FM", "STAT"]:
        x, src = rd(f"prob_main_exact/track={tr}/aci_tie_events.csv")
        ti.append(x.assign(track=tr))
    ti = pd.concat(ti, ignore_index=True)
    ti["tie share %"] = 100 * ti.tie_events / ti.events
    D.table("Coverage events with the outcome on an interval bound within 1e-9 t/day",
            [f"results/prob_main_exact/track={tr}/aci_tie_events.csv" for tr in ["FM", "STAT"]],
            ti[["track", "version", "level", "gamma", "events", "tie_events", "y_zero_events", "tie share %"]],
            cols={"tie share %": 2}, note="Coverage definition unchanged; the tie share bounds the effect of ties "
                                          "on coverage.")
    t5, src = rd("compare/T5a_zeroed_vs_exact.csv")
    t5 = t5[t5.quantity.isin(["sCRPS_weighted", "cov80_aci_mean", "cov95_aci_mean", "cov80_uncal_mean",
                              "cov95_uncal_mean"])]
    t5 = t5.rename(columns={"exact_ext_state": "exact (T5a)", "exact": "exact, evaluation-origin state"})
    D.table("Tests on later-published data (published residence totals, 110 economies): zeroed against exact samples", src,
            t5[["origin", "targets", "track", "quantity", "zeroed", "exact (T5a)", "exact, evaluation-origin state"]],
            note="exact (T5a) = the paper version: at origin 2026-06 the ACI state extended with the calibration origins "
                 "2025-07 to 2026-05, the state of the final intervals (data file S1), so the test scores the intervals delivered "
                 "with the final forecasts. "
                 "exact, evaluation-origin state = the state of the evaluation origins only (robustness; "
                 "results/paper/T5a_realtime_eval_state.csv). At origin 2025-06 both use the state at 2025-06. Zeroed = "
                 "samples with negatives set to zero, with the evaluation-origin state. MASE does not depend on the samples.")
    s1c, src = rd("compare/S1_intervals_zeroed_vs_exact.csv")
    D.table("Final intervals at origin 2026-06 (data files S1, S1a): ACI recursion on zeroed against exact samples", src,
            s1c.drop(columns=["share_point_equal"]), note="Width ratio = median over cells of the exact interval width "
                                                          "over the zeroed one. Floor = alpha at 0.001 at either level.")

    # ------------------------------------------------------------------ write
    missing = sorted(set(IN_SUPPLEMENT.values()) - set(D.skipped), key=lambda x: int(x[1:]))
    if missing:                                                          # a caption changed: fail loudly
        raise SystemExit(f"tables of the supplement not found among the rendered tables: {missing}")
    head = ["# Additional tables", "",
            "Tables computed for the paper that are not among the Supplementary Tables S1-S45 "
            "(`SUPPLEMENTARY_TABLES.md`). They are not numbered. "
            "Generated by `src/supplementary_tables.py` from stored results; nothing is refitted. Each table lists its "
            "source file; the source files hold the full tables. Abbreviations: FM = TimesFM 2.5 + MinT (W from the "
            "model's own one-step errors); STAT = ETS + MinT; main window = target months to 2025-12 without 2025-09; "
            "weighted = 2022 per-day emission weights within one level; scaled metrics divide by the in-sample "
            "seasonal-naive MAE at the origin; MASE, sCRPS and Winkler scores are scaled. Every probabilistic table uses "
            "exactly reconciled non-negative samples. The test at origin 2026-06 scores the intervals delivered with the "
            "final forecasts (data file S1; ACI state extended with the calibration origins).", "",
            "## Contents", ""] + D.index + [""]
    out = out_path(args.out, args.overwrite)
    out.write_text("\n".join(head + D.parts), encoding="utf-8", newline="\n")
    log.finish(f"{D.n} additional tables ({len(D.skipped)} in the supplement skipped), "
               f"{out.stat().st_size / 1024:.0f} KiB")


if __name__ == "__main__":
    main()
