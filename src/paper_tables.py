"""
paper_tables.py: the tables and supplementary files of the paper, from stored results.

Nothing is refitted: the script reads the stored evaluation results and recomputes only the incoherence of the base
forecasts (T3c) from the stored base forecasts. Output (results/paper/): T1a ... T6b, S1, S1a, S3, the robustness
versions of T5a and T5b, tables.md (every table with its note) and MANIFEST.csv (sources of every table).

  T1   data and constraint system          results/system_economies_v2.csv, C_matrix_*_v2, coverage_2025_v2.csv,
                                             structural_zeros.csv, episode_2022/, data/clean/panel_outside_v2.parquet
  T2   base models, main window            results/eval_fm_v2 (summary_weighted.csv, mcs_90.csv)
  T3   reconciliation and incoherence      results/eval_fm_w/summary.csv; stored base forecasts (incoherence)
  T4   probabilistic forecasts             results/eval_prob_exact_nested (exactly reconciled samples, nested intervals)
  T5   tests on later-published data       results/eval_realtime_exact_ext_nested (published residence totals scored
                                             with the forecast divided by (1 - d); nested ACI intervals; at origin
                                             2026-06 the ACI state extended with the calibration origins, the state of
                                             the S1 intervals). Robustness: results/eval_realtime_exact_nested (ACI state
                                             of the evaluation origins only): T5a_realtime_eval_state.csv,
                                             T5b_realtime_units_eval_state.csv
  T6   skill and H*                         results/skill/skill.csv, hstar.csv (skill.py)
  S1   FM forecasts 2026-07 ... 2027-06     results/final_exact/track=FM/forecasts.parquet; split-conformal bounds from
                                             results/conformal_s1/s1_conformal.csv
  S1a  ACI alpha at its floor               results/final_exact/track=<FM|STAT>/forecasts.parquet
  S3   data dictionary of S1

Incoherence (T3c): for every base model, main origins 2022-12 ... 2025-06, horizons 1-12, the World constraint
TER_DOM + TER_INT = RES_TOTAL_SYS - r G (B = C and E + F = (1 + r) G, with r the mean ratio DELTA_INT / G(World) over
the 12 months up to the origin) is checked on the base point forecasts: deviation = (TER_DOM + TER_INT + r G -
RES_TOTAL_SYS) / RES_TOTAL_SYS x 100. Reconciled forecasts have deviation 0.

Usage:
  python src/paper_tables.py [--overwrite]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from reconcile import load_system
from runlog import RunLog, out_path

MODELS = ["SeasonalNaive", "RandomWalkWithDrift", "AutoETS", "AutoTheta", "AutoARIMA", "LightGBM", "ChronosBolt",
          "TimesFM", "TimesFM25"]
LABEL = {"SeasonalNaive": "Seasonal naive", "RandomWalkWithDrift": "Drift", "AutoETS": "ETS (AutoETS)",
         "AutoTheta": "Theta (AutoTheta)", "AutoARIMA": "ARIMA (AutoARIMA)", "LightGBM": "LightGBM (global)",
         "ChronosBolt": "Chronos-Bolt", "TimesFM": "TimesFM 3.0", "TimesFM25": "TimesFM 2.5"}
FM_MODELS = {"ChronosBolt", "TimesFM", "TimesFM25"}
MAIN_ORIGINS = pd.period_range("2022-12", "2025-06", freq="M").astype(str).tolist()
# sources of the probabilistic tables (exactly reconciled non-negative samples)
EVAL_EXACT = "eval_prob_exact_nested"
RT_EXACT = "eval_realtime_exact_ext_nested"           # origin 2026-06: ACI state of the S1 intervals
RT_EVAL_STATE = "eval_realtime_exact_nested"          # robustness: ACI state of the evaluation origins only
FINAL_EXACT = "final_exact"


def md(df, digits=3):
    """Markdown table; floats rounded to `digits` (ints and strings as they are)."""
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else f"{v:,.{digits}f}"
        return str(v)
    out = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    out += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(out)


def incoherence(RES, sysd):
    """Deviation (%) of the base point forecasts from the World residence/territory constraint, all base models."""
    rows = []
    ratio = sysd["ratio"]
    r_o = {o: float(ratio.loc[:o].tail(12).mean()) for o in MAIN_ORIGINS}
    cols = ["W|RES_TOTAL_SYS", "W|TER_DOM", "W|TER_INT", "W|NRES_INT_FROM"]
    for m in MODELS:
        d = RES / ("base_forecasts_fm" if m in FM_MODELS else "base_forecasts_v2") / f"model={m}" / "forecasts.parquet"
        f = pd.read_parquet(d, columns=["unique_id", "origin", "horizon", "mean"], filters=[("unique_id", "in", cols)])
        f = f[f.origin.isin(MAIN_ORIGINS) & (f.horizon <= 12)]
        w = f.pivot_table(index=["origin", "horizon"], columns="unique_id", values="mean").reset_index()
        r = w.origin.map(r_o)
        dev = (w["W|TER_DOM"] + w["W|TER_INT"] + r * w["W|NRES_INT_FROM"] - w["W|RES_TOTAL_SYS"]) / w["W|RES_TOTAL_SYS"] * 100
        rows.append(pd.DataFrame({"model": m, "origin": w.origin, "horizon": w.horizon, "deviation_pct": dev}))
    return pd.concat(rows, ignore_index=True)


def t4_table(RES, src):
    """Table 4 quantities from the evaluate_prob.py summaries in RES / src (also used by exact_eval.py)."""
    c = pd.read_csv(RES / src / "summary_crps.csv")
    c = c[(c.window == "main") & (c.level == "economies") & (c.grouping == "all")]
    iv = pd.read_csv(RES / src / "summary_intervals.csv")
    iv = iv[(iv.window == "main") & (iv.level == "economies") & (iv.grouping == "all")]
    j = pd.read_csv(RES / src / "summary_joint.csv")
    j = j[j.window == "main"]
    rows = []
    for tr in ["FM", "STAT"]:
        for h in ["1", "3", "6", "12", "all"]:
            row = {"track": tr, "horizon": h}
            for kind in ["base", "rec"]:
                x = c[(c.track == tr) & (c.kind == kind) & (c.horizon.astype(str) == h)].iloc[0]
                row[f"sCRPS_{kind}_median"], row[f"sCRPS_{kind}_weighted"] = x["median"], x["weighted"]
            for it, lab in [("uncal_rec", "uncal"), ("aci_g0.05", "aci")]:
                for met in ["cov80", "cov95", "sWinkler80", "sWinkler95"]:
                    x = iv[(iv.track == tr) & (iv.interval == it) & (iv.metric == met) & (iv.horizon.astype(str) == h)]
                    col = "mean" if met.startswith("cov") else "weighted"
                    row[f"{met}_{lab}"] = x[col].iloc[0]
            for kind in ["base", "rec"]:
                x = j[(j.track == tr) & (j.kind == kind) & (j.horizon.astype(str) == h)]
                row[f"ES_median_{kind}"] = x.ES_median.iloc[0] if len(x) else np.nan
                row[f"VS_median_{kind}"] = x.VS_median.iloc[0] if len(x) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean", default="data/clean")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/paper")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    log = RunLog("paper_tables.py", vars(args))
    RES, OUT = Path(args.results), Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    sysd = load_system(RES, args.clean)
    doc = ["# Paper tables", "",
           "Generated by `src/paper_tables.py` from stored results; the CSV files hold the full tables. "
           "MASE and CRPS are scaled by the seasonal-naive scale at each origin. Weighted means use the 2022 mean "
           "per-day emissions of each series and are computed within one level at a time (system economies; World "
           "series one by one). Main window: target months to 2025-12 without 2025-09, origins 2022-12 to 2025-06.", ""]

    def save(df, name, title, note="", digits=3, show=None):
        df.to_csv(out_path(OUT / f"{name}.csv", args.overwrite), index=False, lineterminator="\n", float_format="%.6g")
        doc.extend([f"## {title}", "", f"File: `results/paper/{name}.csv`.", ""])
        if note:
            doc.extend([note, ""])
        doc.extend([md(show if show is not None else df, digits), ""])

    # ------------------------------------------------------------------ T1
    se = pd.read_csv(RES / "system_economies_v2.csv")
    cols = pd.read_csv(RES / "C_matrix_cols_v2.csv")
    crow = pd.read_csv(RES / "C_matrix_rows_v2.csv")
    outside = pd.read_parquet(Path(args.clean) / "panel_outside_v2.parquet")
    z = pd.read_csv(RES / "structural_zeros.csv")
    zs = z[z.in_system].drop_duplicates(["unit", "measure", "period"])
    t1a = pd.DataFrame([
        ("System economies (all seven components 2019-01 to 2026-06 after the structural-zero rule)", len(se)),
        ("  of which joined through structural zeros", int(se.joined_through_structural_zeros.sum())),
        ("Units in the constraint system (economies + Residual + World)", cols.unit.nunique()),
        ("Series in the constraint system (10 measures per unit)", len(cols) - 1),
        ("  plus the World discrepancy node DELTA_INT", 1),
        ("Outside-system economies (published totals only, forecast separately)", outside.unit.nunique()),
        ("Outside-system series", outside.unit.str.cat(outside.measure, sep="|").nunique()),
        ("Constraint rows (within-unit, aggregation, world balance)", len(crow)),
        ("  within-unit rows", int(crow.type.str.startswith("within").sum())),
        ("  aggregation rows", int((crow.type == "aggregation").sum())),
        ("  world-balance rows", int(crow.type.str.startswith("world_balance").sum())),
        ("Rank of the constraint matrix (redundant rows removed)", int((~crow.redundant).sum())),
        ("Structural zeros filled (system unit-component-months)", len(zs)),
        ("Months", "2019-01 to 2026-06 (90); residence totals published to 2026-09"),
    ], columns=["Item", "Value"])
    save(t1a, "T1a_system", "T1a. Data and constraint system")
    cov = pd.read_csv(RES / "coverage_2025_v2.csv").rename(columns={"share_2025": "share_of_World_2025"})
    save(cov, "T1b_coverage", "T1b. Coverage of World by the system economies, 2025 (share of World by measure)")
    EP = RES / "episode_2022"
    e22 = pd.read_csv(EP / "episode_summary.csv", dtype={"episode": str}).set_index("episode").loc["2022"]
    wc = pd.read_csv(EP / "world_components.csv", dtype={"episode": str})
    wc = wc[wc.episode == "2022"]
    c22 = {m[0]: v for m, v in zip(wc.measure, wc.change_kt_day)}
    n_long_in = int(pd.read_csv(EP / "long_origins.csv").origin_in_episode.sum())
    t1c = pd.DataFrame([
        ("Missing component values where the published totals identify them as zero",
         "Structural zeros (every published total containing the value adds up without it); logged; nothing else filled"),
        ("Published residence total differs from the component sum",
         "System total = component sum A+B+D+E+F; the published total is scored with the forecast divided by (1 - d)"),
        ("E(World) + F(World) differs from G(World) by about 0.1%",
         "Discrepancy node DELTA_INT; in reconciliation the ratio r = DELTA_INT / G(World) enters the balance row"),
        ("OECD aggregate is not the sum of its members (territory)", "Group totals built from economies"),
        ("September 2025 spike (suspected processing artefact)", "flag_sep2025: every evaluation with and without it"),
        (f"2022 domestic/international episode, {e22.start} to {e22.end}: World international share "
         f"{e22.share_mean_reference_pct:.1f}% (same months a year earlier) to {e22.share_mean_episode_pct:.1f}% "
         f"(minimum {e22.share_min_pct:.1f}% in {e22.share_min_month}); B = C {c22['B']:+.0f} kt/day, F {c22['F']:+.0f}, "
         f"G {c22['G']:+.0f}; World totals {e22.dRES_TOTAL_SYS_pct:+.1f}%",
         f"Documented (DATA_RULES.md, Fig. 1); {n_long_in} of the 13 long-horizon origins (2021-12 to 2022-12) lie "
         "inside it, which affects the reading of the skill estimates for components and territory measures; "
         "diagnostic skill of the totals in results/episode_2022/skill_totals.csv"),
        ("2026 domestic/international reclassification (about 300 kt/day by June 2026)",
         "Main evaluation window ends 2025-12; 2026-01 to 2026-06 is a separate stress test"),
        ("Series that are zero for two years", "Fixed at zero at that origin, still scored"),
        ("Switzerland: A, B, F unpublished for 2025", "Outside the system; published residence total forecast as an outside-system series"),
        ("Territory assignment of ports (e.g. Malaysia high, Hong Kong low)", "Not handled; reported as a limitation (DATA_RULES.md)"),
        ("Components published to 2026-06, residence totals to 2026-09",
         "The test at origin 2026-06 scores the published residence totals of 2026-07 to 2026-09"),
    ], columns=["Data issue", "How it is handled"])
    save(t1c, "T1c_issues", "T1c. Main data issues and how they are handled")

    # ------------------------------------------------------------------ T2
    s = pd.read_csv(RES / "eval_fm_v2" / "summary_weighted.csv")
    s = s[(s.window == "main") & (s.level == "economies") & (s.grouping == "all") & (s.horizon == "all")]
    mcs = pd.read_csv(RES / "eval_fm_v2" / "mcs_90.csv")
    rows = []
    for m in MODELS:
        x = s[s.model == m].set_index("metric")
        row = {"model": LABEL[m], "MASE_median": x.at["MASE", "median"], "MASE_weighted": x.at["MASE", "weighted"],
               "RMSSE_weighted": x.at["RMSSE", "weighted"],
               "cov80_mean": x.at["cov80", "mean"] if "cov80" in x.index else np.nan,
               "cov80_weighted": x.at["cov80", "weighted"] if "cov80" in x.index else np.nan}
        for scope in ["system_economies", "all_series"]:
            mm = mcs[(mcs.scope == scope) & (mcs.model == m)]
            row[f"MCS90_{scope}_all_h"] = bool(mm[mm.horizon_group.astype(str) == "all"].in_mcs_90.iloc[0])
            row[f"MCS90_{scope}_h_groups"] = " ".join(str(g) for g in mm[mm.in_mcs_90 & (mm.horizon_group.astype(str) != "all")].horizon_group)
        rows.append(row)
    t2 = pd.DataFrame(rows)
    save(t2, "T2_base_models", "T2. Base models, main window, system economies, all horizons",
         "Coverage = empirical coverage of the uncalibrated 80% interval (LightGBM gives point forecasts only). "
         "MCS: 90% Model Confidence Set, loss = mean MASE across series per origin; h groups 1, 3, 6, 12.")

    # ------------------------------------------------------------------ T3
    w = pd.read_csv(RES / "eval_fm_w" / "summary.csv")
    w = w[(w.window == "main") & (w.horizon.astype(str) == "all") & (w.metric == "MASE")]
    rows, wrows = [], []
    for tr, base in [("FM (TimesFM 2.5, own W)", "TimesFM25"), ("STAT (AutoETS)", "AutoETS")]:
        for meth, var in [("base", "base"), ("WLS", "nn_qp"), ("MinT", "nn_qp")]:
            x = w[(w.base == base) & (w.method == meth) & (w.variant == var)]
            e = x[(x.level == "economies") & (x.grouping == "all")].iloc[0]
            rows.append({"track": tr, "method": meth, "MASE_median_economies": e["median"], "MASE_weighted_economies": e["weighted"]})
            ww = x[(x.level == "world") & (x.grouping == "series")]
            wrows.append({"track": tr, "method": meth, **dict(zip("World " + ww.grp, ww["median"]))})
    save(pd.DataFrame(rows), "T3a_reconciliation_economies",
         "T3a. Reconciliation, main window, all horizons: base, WLS and MinT (exact non-negative QP)")
    t3w = pd.DataFrame(wrows)
    save(t3w, "T3b_reconciliation_world", "T3b. Reconciliation, World series one by one (MASE, all horizons)")
    inc = incoherence(RES, sysd)
    inc.to_csv(out_path(OUT / "T3c_incoherence_cells.csv", args.overwrite), index=False, lineterminator="\n", float_format="%.6g")
    t3c = inc.groupby(["model", "horizon"]).deviation_pct.apply(lambda x: x.abs().mean()).unstack("model")[MODELS]
    t3c.columns = [LABEL[m] for m in MODELS]
    save(t3c.reset_index(), "T3c_incoherence", "T3c. Incoherence of the base forecasts by horizon",
         "Mean absolute deviation (% of World RES_TOTAL_SYS) from the World constraint TER_DOM + TER_INT = "
         "RES_TOTAL_SYS - r G, over the 31 main origins. Reconciled forecasts have deviation 0.")

    # ------------------------------------------------------------------ T4
    t4_cols = ["track", "horizon", "sCRPS_base_weighted", "sCRPS_rec_weighted", "cov80_uncal", "cov80_aci", "cov95_uncal",
               "cov95_aci", "sWinkler80_aci", "sWinkler95_aci", "ES_median_rec", "VS_median_rec"]
    t4 = t4_table(RES, EVAL_EXACT)
    save(t4, "T4_probabilistic", "T4. Probabilistic forecasts, main window, system economies",
         "sCRPS: scaled CRPS (weighted mean shown; medians in the CSV). Coverage: mean over series of the empirical "
         "coverage; uncal = reconciled sample quantiles, aci = ACI with gamma 0.05, with the 80% interval clipped inside "
         "the 95% interval (nesting). Winkler: scaled, weighted. ES, VS: medians across origin-horizon cells, reconciled "
         "samples (base in the CSV); VS on scaled values. Horizon 'all' pools horizons 1-12. Reconciled samples: exact "
         "non-negative QP.", show=t4[t4_cols])

    # ------------------------------------------------------------------ T5
    def t5_tables(src):
        rt = pd.read_csv(RES / src / "summary_res_total_pub.csv")
        rt = rt[(rt.d_variant == "d") & (rt.level == "economies") & (rt.period != "all incl. 2025-09")]
        rows = []
        for (tr, o, per), x in rt.groupby(["track", "origin", "period"]):
            x = x.set_index("metric")
            rows.append({"track": tr, "origin": o, "targets": per, "MASE_median": x.at["MASE", "median"],
                         "MASE_weighted": x.at["MASE", "weighted"], "sCRPS_weighted": x.at["sCRPS", "weighted"],
                         "cov80_aci": x.at["cov80_aci", "mean"], "cov95_aci": x.at["cov95_aci", "mean"],
                         "cov80_conformal": x.at["cov80_conformal", "mean"], "cov95_conformal": x.at["cov95_conformal", "mean"],
                         "n_economies": int(x.at["MASE", "n"])})
        t5 = pd.DataFrame(rows).sort_values(["origin", "targets", "track"])
        u = pd.read_csv(RES / src / "units_res_total_pub.csv")
        u = u[(u.origin == "2026-06") & (u.d_variant == "d") & u.unit.isin(["W", "CHN", "GRC", "SGP", "JPN", "USA"])]
        t5b = u.pivot_table(index="unit", columns="track", values=["MASE", "cov80_aci", "cov95_aci"]).reindex(
            ["W", "CHN", "GRC", "SGP", "JPN", "USA"])
        t5b.columns = [f"{a}_{b}" for a, b in t5b.columns]
        return t5, t5b.reset_index()

    t5, t5b = t5_tables(RT_EXACT)
    save(t5, "T5a_realtime", "T5a. Tests on later-published data: published residence totals of the system economies",
         "Forecasts are made at the origin and scored against everything published since (OECD release of "
         "14 October 2026). The published residence total is scored with the forecast divided by (1 - d), d = mean "
         "relative discrepancy (published minus component sum, over published) in the 6 months up to the origin. "
         "Origin 2026-06 is leakage-free: its target months are later than every model release (T5c). ACI intervals "
         "nested (80% clipped inside 95%). Samples reconciled with the exact non-negative QP. ACI state: at origin "
         "2025-06 the state at 2025-06; at origin 2026-06 the state extended with the calibration origins 2025-07 to "
         "2026-05, the state of the S1 intervals, so the test scores the intervals delivered in S1. Robustness, state "
         "of the evaluation origins only: T5a_realtime_eval_state.csv.")
    save(t5b, "T5b_realtime_units", "T5b. Origin 2026-06 (targets 2026-07 to 2026-09): World and the five largest economies",
         "ACI intervals nested; exact samples; ACI state extended with the calibration origins, as S1. Robustness, state "
         "of the evaluation origins only: T5b_realtime_units_eval_state.csv.")
    t5r, t5br = t5_tables(RT_EVAL_STATE)
    for df, name in [(t5r, "T5a_realtime_eval_state"), (t5br, "T5b_realtime_units_eval_state")]:
        df.to_csv(out_path(OUT / f"{name}.csv", args.overwrite), index=False, lineterminator="\n", float_format="%.6g")
    doc.extend(["Robustness versions of T5a and T5b with the ACI state of the evaluation origins only at origin 2026-06: "
                "`results/paper/T5a_realtime_eval_state.csv`, `results/paper/T5b_realtime_units_eval_state.csv`.", ""])
    t5c = pd.DataFrame([
        ("TimesFM 2.5", "google/timesfm-2.5-200m-pytorch", "repository created 2025-09-02; weights last modified 2025-10-02"),
        ("TimesFM 3.0", "google/timesfm-3.0-pytorch", "created 2026-08-24; revision last modified 2026-09-02"),
        ("Chronos-Bolt", "amazon/chronos-bolt-base", "created 2024-11-25; revision last modified 2025-11-21"),
        ("OECD data (targets 2026-07 to 2026-09)", "OECD.SDD.NAD.SEEA:DSD_MARITIME_TRANSPORT@DF_MARITIME_TRANSPORT(2.0)",
         "published 14 October 2026"),
    ], columns=["Item", "Identifier", "Date (Hugging Face API)"])
    save(t5c, "T5c_release_dates", "T5c. Model release dates and data publication",
         "Dates of the model repositories from the Hugging Face API. Overlap between the evaluation period and the "
         "models' pretraining data cannot be ruled out; target months published after every model release are free "
         "of it.")

    # ------------------------------------------------------------------ T6: skill and H*
    sk = pd.read_csv(RES / "skill" / "skill.csv")
    sk = sk[(sk.group == "all") & (sk.reference == "persistence") & (sk.variant == "weighted") &
            (sk.origins == "all origins") & (sk.bootstrap == "block6")]
    hx = pd.read_csv(RES / "skill" / "hstar.csv")
    main_h = hx[(hx.group == "all") & (hx.reference == "persistence") & (hx.variant == "weighted") &
                (hx.origins == "all origins") & (hx.bootstrap == "block6") & (hx.level == 95)].set_index("track").Hstar
    rows = []
    for tr in ["FM", "STAT"]:
        x = sk[sk.track == tr].set_index("horizon")
        hs = int(x.index[x.lo95 <= 0].min())                       # first horizon with the lower bound at or below 0
        assert hs == int(main_h[tr]), (tr, hs, main_h[tr])
        for h in range(1, 37):
            rows.append({"track": tr, "horizon": h, "n_origins": int(x.at[h, "n_origins"]), "s": x.at[h, "s"],
                         "lo95": x.at[h, "lo95"], "hi95": x.at[h, "hi95"], "Hstar": hs})
    t6 = pd.DataFrame(rows)
    hs48 = t6.groupby("track").Hstar.first()
    show = t6[t6.horizon.isin(list(range(1, 15)) + [18, 20, 21, 22, 23, 24, 30, 36])]
    show = show.pivot(index=["horizon", "n_origins"], columns="track", values=["s", "lo95", "hi95"])
    show.columns = [f"{b} {a}" for a, b in show.columns]
    show = show[[f"{tr} {c}" for tr in ["FM", "STAT"] for c in ["s", "lo95", "hi95"]]].reset_index()
    save(t6, "T6_skill", "T6. Skill against the persistence reference and H*",
         "Skill s(h) = 1 - weighted MASE of the track / weighted MASE of the persistence reference (trailing 12-month mean "
         "times a seasonal profile, both from data up to the origin), system economies, 2022 weights. Origins: at every "
         "horizon, every one of the 43 origins 2021-12 ... 2025-06 whose target month lies in the main window (to "
         "2025-12, without 2025-09); n_origins is the same for both tracks. lo95, hi95: 95% moving-block bootstrap "
         "interval over consecutive origins (block 6, 1,000 resamples). "
         f"H* = first horizon whose lower bound is at or below zero: FM {hs48['FM']}, STAT {hs48['STAT']}. H* varies "
         "strongly across tests and origin sets (T6b).", show=show)

    # ------------------------------------------------------------------ T6b: H* for every variant
    hac = hx.bootstrap.str.startswith("HAC")
    hx["test"] = np.where(hac, "HAC test, 5%", hx.bootstrap.map(
        {"block6": "block 6 (main)", "block3": "block 3", "block12": "block 12", "iid": "i.i.d."}))
    hx["col"] = hx.track + " " + np.where(hac, "test", hx.level.astype(str) + "%")
    keys = ["group", "reference", "variant", "origins", "test"]
    t6b = hx.pivot_table(index=keys, columns="col", values="Hstar", aggfunc="first").reset_index()
    t6b.columns.name = None
    k_ord = [("group", {"all": 0, "residence totals": 1, "territory totals": 2}),
             ("reference", {"persistence": 0, "seasonal naive": 1}), ("variant", {"weighted": 0, "median": 1}),
             ("origins", {"all origins": 0, "(i) 13 origins 2021-12..2022-12": 1, "(ii) without 2021-12..2022-10": 2}),
             ("test", {"block 6 (main)": 0, "block 3": 1, "block 12": 2, "i.i.d.": 3, "HAC test, 5%": 4})]
    t6b = t6b.sort_values([c for c, _ in k_ord], key=lambda col: col.map(dict(k_ord)[col.name]))
    cols = [c for c in [f"{tr} {x}" for tr in ["FM", "STAT"] for x in ["80%", "90%", "95%", "test"]] if c in t6b.columns]
    t6b = t6b[keys + cols]
    t6b.insert(0, "main", (t6b.group == "all") & (t6b.reference == "persistence") & (t6b.variant == "weighted") &
               (t6b.origins == "all origins") & (t6b.test == "block 6 (main)"))
    assert int(t6b[t6b.main]["FM 95%"].iloc[0]) == hs48["FM"] and int(t6b[t6b.main]["STAT 95%"].iloc[0]) == hs48["STAT"]
    show = t6b.copy()
    for c in cols:
        show[c] = show[c].map(lambda v: "" if pd.isna(v) else (">36" if int(v) > 36 else str(int(v))))
    show["main"] = show.main.map({True: "yes", False: ""})
    fm_all = t6b[t6b.group == "all"][[c for c in cols if c.startswith("FM")]].stack()
    fm95 = t6b[(t6b.group == "all") & (t6b.variant == "weighted")][["FM 95%", "FM test"]].stack()
    save(t6b, "T6b_hstar_variants", "T6b. H* for every variant of the skill test",
         "H* = first horizon at which the lower bound of s(h) is at or below zero (bootstrap rows: 80%, 90% and 95% "
         "levels of the bound) or at which the one-sided test of H0: s(h) <= 0 (per-origin difference in weighted MASE, "
         "persistence minus track, Newey-West variance with lag h-1) no longer rejects at 5% (HAC rows, 'test' columns). "
         ">36: the bound stays above zero up to h = 36, the longest horizon tested. group: all system economy series, or "
         "only their residence totals (RES_TOTAL_SYS) or territory totals (TER_DOM + TER_INT); reference: persistence "
         "or seasonal naive; variant: weighted skill or median-based skill; origins: all 43, (i) the 13 origins "
         "2021-12 ... 2022-12, (ii) without the 2022 episode 2021-12 ... 2022-10. Main row: "
         f"H* = {hs48['FM']} (FM) and {hs48['STAT']} (STAT). FM, weighted skill of all series, 95% bound or 5% test "
         f"(every origin set, block length and reference): H* from {int(fm95.min())} to "
         f"{int(fm95.max()) if fm95.max() <= 36 else '>36'}; with the median-based skill and the 80% and 90% levels: "
         f"{int(fm_all.min())} to {int(fm_all.max()) if fm_all.max() <= 36 else '>36'}.", show=show)

    # ------------------------------------------------------------------ S1, S1a, S3
    meta = pd.read_parquet(RES / "base_forecasts_v2" / "series_meta.parquet").set_index("unique_id")
    cf = pd.read_csv(RES / "conformal_s1" / "s1_conformal.csv", dtype={"target": str})   # split-conformal bounds

    fc = pd.read_parquet(RES / FINAL_EXACT / "track=FM" / "forecasts.parquet")
    s1 = fc[["unique_id", "target", "horizon", "point", "lo95", "lo80", "hi80", "hi95", "aci_at_floor"]].copy()
    s1[["unit", "measure"]] = s1.unique_id.str.split("|", expand=True)
    s1["unit_type"] = s1.unique_id.map(meta.unit_type)
    s1 = s1.merge(cf, on=["unique_id", "horizon", "target"], how="left", validate="one_to_one")
    assert s1.hi95_conformal.notna().all()
    s1 = s1.rename(columns={"target": "month"})[["unit", "unit_type", "measure", "month", "horizon", "point", "lo95",
                                                  "lo80", "hi80", "hi95", "aci_at_floor", "lo95_conformal",
                                                  "lo80_conformal", "hi80_conformal", "hi95_conformal"]]
    s1.to_csv(out_path(OUT / "S1_forecasts_FM_2026-07_2027-06.csv", args.overwrite), index=False, lineterminator="\n", float_format="%.6g")
    # S1a: share of forecast cells (series x horizon) whose ACI alpha is at its floor, by measure group
    grp = {"RES_TOTAL_SYS": "total (RES_TOTAL_SYS)", "TER_DOM": "territory measures (TER_DOM, TER_INT)",
           "TER_INT": "territory measures (TER_DOM, TER_INT)"}
    order = ["total (RES_TOTAL_SYS)", "territory measures (TER_DOM, TER_INT)", "components (A-G)"]
    fl = []
    for tr in ["FM", "STAT"]:
        f = pd.read_parquet(RES / FINAL_EXACT / f"track={tr}" / "forecasts.parquet", columns=["unique_id", "aci_at_floor"])
        f["group"] = f.unique_id.str.split("|").str[1].map(grp).fillna("components (A-G)")
        fl.append(f.assign(track=tr))
    fl = pd.concat(fl, ignore_index=True)
    s1a = pd.crosstab([fl.track, fl.group], fl.aci_at_floor, normalize="index").mul(100)
    s1a = s1a.reindex(columns=["none", "95", "80_and_95", "80"], fill_value=0.0)
    s1a.columns = [f"pct_{c}" for c in s1a.columns]
    s1a["n_cells"] = fl.groupby(["track", "group"]).size()
    s1a = s1a.reset_index()
    s1a["k"] = s1a.group.map(order.index)
    s1a = s1a.sort_values(["track", "k"]).drop(columns="k")
    save(s1a, "S1a_aci_floor_flags", "S1a. Final intervals at origin 2026-06: share of cells with the ACI alpha at its floor",
         "Cells = series x horizon (FM h 1-12, STAT h 1-7), system economies, Residual and World. aci_at_floor: 95 = the "
         "95% level is at its floor (alpha = 0.001; the interval is almost the range of the reconciled samples); "
         "80_and_95 = both levels; 80 = the 80% level only (after nesting the 80% interval equals the 95% interval); "
         "values in % of the cells of the group. Headline intervals are those of the totals, RES_TOTAL_SYS and "
         "TER_DOM + TER_INT, which the 2026 reclassification does not affect. ACI state from the recursion on exactly "
         "reconciled samples over the evaluation and calibration origins; final samples exact.", digits=1)

    s3 = pd.DataFrame([
        ("S1", "unit", "OECD area code of a system economy; RESIDUAL = everything not modelled individually; W = World"),
        ("S1", "unit_type", "economy, residual or world"),
        ("S1", "measure", "RES_DOM_IN (A), RES_DOM_OUT (B), NRES_DOM_IN (C), RES_INT_FROM (D), RES_INT_TO (E), RES_INT_OUT (F), "
                          "NRES_INT_FROM (G), RES_TOTAL_SYS = A+B+D+E+F, TER_DOM = A+C, TER_INT = D+G"),
        ("S1", "month", "target month (YYYY-MM), 2026-07 to 2027-06 (horizons 1-12 from origin 2026-06)"),
        ("S1", "horizon", "months after the origin 2026-06"),
        ("S1", "point", "FM track point forecast: TimesFM 2.5 base, MinT reconciliation with its own W, exact non-negative QP; t CO2 per day"),
        ("S1", "lo95, hi95", "calibrated 95% interval (ACI, gamma 0.05, state extended to 2026-06; the ACI recursion and the final samples use samples reconciled with the exact non-negative QP); t CO2 per day"),
        ("S1", "lo80, hi80", "calibrated 80% ACI interval, clipped to lie inside the 95% interval; t CO2 per day"),
        ("S1", "lo95_conformal, hi95_conformal", "pooled split-conformal 95% interval: point -/+ q(h, g) x scale, where q is the "
                                                 "split-conformal quantile (level ceil((N + 1)(1 - alpha)) / N) of the scaled absolute errors of the FM "
                                                 "track's rolling-origin point forecasts (origins 2022-12 to 2025-06) of all system series of the same "
                                                 "measure group g (residence total; territory measures; components) at horizon h, with targets known "
                                                 "at 2026-06, and scale = the series' seasonal-naive scale up to 2026-06; lower bound clipped at zero. "
                                                 "As in the real-time test (T5a). Recommended for the residence totals; t CO2 per day"),
        ("S1", "lo80_conformal, hi80_conformal", "pooled split-conformal 80% interval, as above (nested inside the 95% interval by "
                                                 "construction); t CO2 per day"),
        ("S1", "aci_at_floor", "flag: which ACI levels have alpha at its floor (0.001) after the 2026 outcomes: none; 95 (the 95% "
                               "interval is almost the range of the 500 reconciled samples); 80_and_95 (both levels); 80 (the 80% "
                               "level only; after nesting the 80% interval equals the 95% interval). Flagged intervals carry a caveat"),
        ("S1", "headline intervals", "headline intervals are those of the totals, RES_TOTAL_SYS and TER_DOM + TER_INT, which the 2026 "
                                     "reclassification does not affect. Intervals of the components and of the domestic/international "
                                     "split are given with the flag and a caveat. S1 has TER_DOM and TER_INT as separate series; their "
                                     "interval bounds do not add up to an interval of TER_DOM + TER_INT"),
        ("all", "units", "t CO2 per day. Point forecasts are coherent across measures and units; interval bounds are per series and do not add up"),
    ], columns=["file", "column", "description"])
    s3.to_csv(out_path(OUT / "S3_data_dictionary.csv", args.overwrite), index=False, lineterminator="\n")
    doc.extend(["## Supplementary data", "", "- `S1_forecasts_FM_2026-07_2027-06.csv`: " + f"{len(s1):,} rows "
                "(the share of flagged cells is in S1a above). S1 holds the FM track forecasts at origin 2026-06 "
                "(h = 1-12) with their calibrated ACI intervals and the pooled split-conformal 80% and 95% bounds "
                "(recommended for the residence totals).", "- `S3_data_dictionary.csv`:", "", md(s3), ""])
    (OUT / "tables.md").write_text("\n".join(doc), encoding="utf-8", newline="\n")

    # ------------------------------------------------------------------ manifest
    man = [
        ("T1a_system.csv, T1b_coverage.csv, T1c_issues.csv", "table", "src/paper_tables.py",
         "results/system_economies_v2.csv; results/C_matrix_cols_v2.csv; results/C_matrix_rows_v2.csv; results/coverage_2025_v2.csv; results/structural_zeros.csv; data/clean/panel_outside_v2.parquet; results/episode_2022 (src/episode_2022.py)"),
        ("T2_base_models.csv", "table", "src/paper_tables.py", "results/eval_fm_v2/summary_weighted.csv; results/eval_fm_v2/mcs_90.csv (src/evaluate_fm.py)"),
        ("T3a_reconciliation_economies.csv, T3b_reconciliation_world.csv", "table", "src/paper_tables.py", "results/eval_fm_w/summary.csv (src/evaluate_fm_w.py)"),
        ("T3c_incoherence.csv, T3c_incoherence_cells.csv", "table", "src/paper_tables.py", "results/base_forecasts_v2, results/base_forecasts_fm (forecasts.parquet); data/clean/panel_system_wide_v2.parquet"),
        ("T4_probabilistic.csv", "table", "src/paper_tables.py", "results/eval_prob_exact_nested/summary_crps.csv, summary_intervals.csv, summary_joint.csv (src/exact_eval.py; src/evaluate_prob.py --nest --prob-dir results/prob_main_exact)"),
        ("T5a_realtime.csv, T5b_realtime_units.csv, T5c_release_dates.csv", "table", "src/paper_tables.py",
         "results/eval_realtime_exact_ext_nested (src/exact_samples.py; src/evaluate_realtime.py --d-form exact --nest --prob-dir results/prob_realtime_exact --aci-name aci_intervals_ext.parquet); release dates from the Hugging Face API"),
        ("T5a_realtime_eval_state.csv, T5b_realtime_units_eval_state.csv", "table (robustness)", "src/paper_tables.py",
         "results/eval_realtime_exact_nested (as T5a, T5b; origin 2026-06 with the ACI state of the evaluation origins only)"),
        ("T6_skill.csv, T6b_hstar_variants.csv", "table", "src/paper_tables.py", "results/skill/skill.csv, hstar.csv (src/skill.py)"),
        ("S1_forecasts_FM_2026-07_2027-06.csv", "supplementary file", "src/paper_tables.py", "results/final_exact/track=FM/forecasts.parquet (src/final_samples.py, src/exact_samples.py --part aci, src/final_bands.py --out results/final_exact); results/conformal_s1/s1_conformal.csv (src/conformal_s1.py)"),
        ("S1a_aci_floor_flags.csv", "supplementary file", "src/paper_tables.py", "results/final_exact/track=FM/forecasts.parquet, track=STAT/forecasts.parquet (src/final_bands.py)"),
        ("S3_data_dictionary.csv", "supplementary file", "src/paper_tables.py", "-"),
        ("tables.md", "tables (Markdown)", "src/paper_tables.py", "all of the above"),
    ]
    pd.DataFrame(man, columns=["output", "type", "script", "inputs"]).to_csv(out_path(OUT / "MANIFEST.csv", args.overwrite), index=False, lineterminator="\n")
    log.finish(f"tables written; S1 {len(s1)} rows")


if __name__ == "__main__":
    main()
