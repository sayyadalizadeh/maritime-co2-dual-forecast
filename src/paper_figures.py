"""
paper_figures.py: the figures of the paper, as vector PDF and 600 dpi TIFF in figures/, named by the manuscript
numbering, at their printed size (190 mm full width; Fig3 at 90 mm single column), so every font size below is the
printed size; no text is smaller than 7 pt. No overall figure titles (the manuscript captions carry them); panels are
labelled (a), (b), ...

  Fig1  World totals on the two accounting bases and the international share (2022 episode, 2026 reclassification)
  Fig3  Incoherence of the base forecasts by horizon
  Fig4  Accuracy by horizon: (a) MASE, (b) scaled CRPS of the base and reconciled samples
  Fig5  Calibration: coverage of the uncalibrated and ACI intervals (nested) by horizon
  Fig6  Test on later-published data from origin 2026-06: World and the five largest economies
  Fig7  Skill against persistence with the block-bootstrap band, the number of origins and H*

Fig. 2 (the framework diagram) is not drawn by code.

Probabilistic inputs are those of exactly reconciled non-negative samples (results/eval_prob_exact_nested,
results/eval_realtime_exact_ext_nested, results/prob_realtime_exact). At origin 2026-06, Fig6 shows the ACI intervals
with the state extended with the calibration origins 2025-07 to 2026-05, the state of the S1 intervals.

Fig4 panel (b): every origin is kept; the axis is limited at FIG4B_YMAX, the off-scale STAT value (h = 11) is marked with
an arrow and its value, and a dashed line shows STAT without origin 2024-09 as a diagnostic (stat_crps_without_origin;
data in results/paper/fig4b_stat_crps_without_2024-09.csv).

Usage:
  python src/paper_figures.py [--overwrite] [--preview <dir>]
"""
import argparse
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from PIL import Image

from runlog import RunLog, out_path

# colours, model labels and the time helper
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e5e4e0"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]   # reference order
C_TRACK = {"FM": "#e34948", "STAT": "#1baf7a"}
STAT_MODELS = ["SeasonalNaive", "RandomWalkWithDrift", "AutoETS", "AutoTheta", "AutoARIMA", "LightGBM"]
FM_MODELS = ["ChronosBolt", "TimesFM", "TimesFM25"]
LABEL = {"SeasonalNaive": "Seasonal naive", "RandomWalkWithDrift": "Drift", "AutoETS": "ETS", "AutoTheta": "Theta",
         "AutoARIMA": "ARIMA", "LightGBM": "LightGBM", "ChronosBolt": "Chronos-Bolt", "TimesFM": "TimesFM 3.0",
         "TimesFM25": "TimesFM 2.5"}


def ts(periods):
    return pd.PeriodIndex(periods, freq="M").to_timestamp() + pd.Timedelta(days=14)


# base style (the second update below overrides most of it; savefig.dpi applies to the PDF output)
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": GRID, "grid.linewidth": 0.8, "lines.linewidth": 1.5, "legend.frameon": False,
    "savefig.dpi": 200, "savefig.bbox": "tight", "pdf.fonttype": 42})

MM = 1 / 25.4
FULL, SINGLE = 190 * MM, 90 * MM
TRACK_LABEL = {"FM": "FM (TimesFM 2.5 + MinT)", "STAT": "STAT (ETS + MinT)"}   # legends
CO2 = "CO₂"
plt.rcParams.update({
    "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 1.2,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False,
    "savefig.bbox": None, "pdf.fonttype": 42, "hatch.linewidth": 0.6})
ANN = 7   # annotation font size (pt)
# inputs of the sample-based figures (exactly reconciled samples); at origin 2026-06 the ACI intervals with the state
# extended with the calibration origins (the state of the S1 intervals)
SRC = {"crps": "eval_prob_exact_nested", "intervals": "eval_prob_exact_nested",
       "rt_cells": "eval_realtime_exact_ext_nested", "rt_prob": "prob_realtime_exact", "rt_aci": "aci_intervals_ext.parquet"}
# Fig4b: upper limit of panel (b), and the STAT diagnostic without origin 2024-09 (no origin is excluded from the results)
FIG4B_YMAX, DIAG_ORIGIN, DIAG_LS = 1.4, "2024-09", (0, (5, 2))




def save(fig, figs, name, overwrite, preview=None):
    """Vector PDF and 600 dpi RGB TIFF (LZW) at the figure's printed size; optional PNG preview."""
    fig.savefig(out_path(Path(figs) / f"{name}.pdf", overwrite), metadata={"CreationDate": None})
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=600)
    buf.seek(0)
    Image.open(buf).convert("RGB").save(out_path(Path(figs) / f"{name}.tif", overwrite),
                                        compression="tiff_lzw", dpi=(600, 600))
    if preview:
        Path(preview).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(preview) / f"{name}.png", dpi=150)
    plt.close(fig)
    print("saved", name)


def style_legend(handles_tracks, styles):
    """Legend handles: one coloured line per track, one grey line per line style."""
    h = [Line2D([], [], color=C_TRACK[tr], marker="o", ms=3) for tr in handles_tracks]
    lab = [TRACK_LABEL[tr] for tr in handles_tracks]
    h += [Line2D([], [], color=INK2, ls=ls) for ls, _ in styles]
    lab += [l_ for _, l_ in styles]
    return h, lab


def stat_crps_without_origin(RES, clean, prob_dir, eval_dir, origin=DIAG_ORIGIN):
    """Fig4b diagnostic (no origin is excluded from the results; this line is a diagnostic only): STAT weighted
    scaled CRPS of the reconciled samples by horizon, main window, system economies, with and without one origin.
    Same aggregation as T4 (mean over origins per series, then weighted over series with the 2022 weights).
    Checks: with all origins it reproduces summary_crps.csv of eval_dir, and at h = 10-12 without the origin it
    reproduces the leave-one-origin-out values of results/supplementary/stat_crps_by_origin.csv."""
    from evalcore import WINDOWS, load_history, origin_scales
    sc = origin_scales(load_history(clean))[["unique_id", "origin", "scale1"]]
    m = pd.read_parquet(RES / "base_forecasts_v2" / "series_meta.parquet")
    c = pd.read_parquet(RES / prob_dir / "track=STAT" / "crps.parquet", filters=[("kind", "==", "rec")])
    c = c.merge(sc, on=["unique_id", "origin"])
    c["sCRPS"] = c.crps / c.scale1
    c = c[WINDOWS["main"](c.target) & np.isfinite(c.sCRPS)].merge(m, on="unique_id")
    c = c[c.unit_type == "economy"]

    def weighted(d):
        ps = d.groupby(["horizon", "unique_id", "weight_2022"]).sCRPS.mean().reset_index()
        return ps.groupby("horizon").apply(lambda x: np.average(x.sCRPS, weights=x.weight_2022))

    out = pd.DataFrame({"all_origins": weighted(c), f"without_{origin}": weighted(c[c.origin != origin])}).reset_index()
    out.insert(0, "track", "STAT")
    out.insert(1, "kind", "rec")
    cr = pd.read_csv(RES / eval_dir / "summary_crps.csv")
    cr = cr[(cr.window == "main") & (cr.level == "economies") & (cr.grouping == "all") & (cr.horizon != "all")
            & (cr.track == "STAT") & (cr.kind == "rec")]
    ref = cr.set_index(cr.horizon.astype(int)).weighted
    assert np.allclose(out.set_index("horizon").all_origins, ref.loc[out.horizon], rtol=1e-5), "Fig4b: T4 not reproduced"
    lo = pd.read_csv(RES / "supplementary" / "stat_crps_by_origin.csv", dtype={"origin_left_out": str})
    lo = lo[(lo.kind == "rec") & (lo.origin_left_out == origin)].set_index("horizon").weighted_without_origin
    assert np.allclose(out.set_index("horizon")[f"without_{origin}"].loc[lo.index], lo, rtol=1e-5), "Fig4b: leave-one-origin-out values not reproduced"
    return out


def fig4(RES, sh, src, save_fn, sfx, ymax=None, diag=None):
    """Fig4: (a) MASE by horizon; (b) scaled CRPS of the base and reconciled samples (src["crps"]).
    ymax: upper limit of panel (b); every value above it is marked with an arrow and its value.
    diag: STAT reconciled samples without one origin (stat_crps_without_origin), drawn as a diagnostic line in (b)."""
    cr = pd.read_csv(RES / src["crps"] / "summary_crps.csv")
    cr = cr[(cr.window == "main") & (cr.level == "economies") & (cr.grouping == "all") & (cr.horizon != "all")]
    cr["h"] = cr.horizon.astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 80 * MM), layout="constrained")
    off = []                                                  # (horizon, value, track) above ymax in panel (b)
    for tr, base in [("FM", "TimesFM25"), ("STAT", "AutoETS")]:
        for meth, var, ls in [("base", "base", "--"), ("MinT", "nn_qp", "-")]:
            d = sh[(sh.base == base) & (sh.method == meth) & (sh.variant == var)].sort_values("horizon")
            axes[0].plot(d.horizon, d.weighted, color=C_TRACK[tr], ls=ls, marker="o", ms=3)
        for kind, ls in [("base", "--"), ("rec", "-")]:
            d = cr[(cr.track == tr) & (cr.kind == kind)].sort_values("h")
            axes[1].plot(d.h, d.weighted, color=C_TRACK[tr], ls=ls, marker="o", ms=3)
            if ymax is not None:
                off += [(h_, v, tr) for h_, v in zip(d.h, d.weighted) if v > ymax]
    if diag is not None:
        col = [c for c in diag.columns if c.startswith("without_")][0]
        axes[1].plot(diag.horizon, diag[col], color=C_TRACK["STAT"], ls=DIAG_LS, lw=0.9, marker="o", ms=3,
                     mfc="white", mew=0.8)
    if ymax is not None:
        axes[1].set_ylim(top=ymax)
        for h_, v, tr in off:
            axes[1].annotate(f"{v:.2f}", xy=(h_, ymax), xytext=(h_, ymax * 0.86), fontsize=ANN, color=C_TRACK[tr],
                             ha="center", va="center", annotation_clip=False,
                             arrowprops={"arrowstyle": "->", "color": C_TRACK[tr], "lw": 0.8})
    axes[0].set_title("(a) MASE (weighted, system economies)", loc="left")
    axes[1].set_title("(b) Scaled CRPS (weighted, system economies)", loc="left")
    for ax in axes:
        ax.set_xlabel("Horizon (months)")
        ax.set_xticks(range(1, 13))
    h, lab = style_legend(["FM", "STAT"], [("--", "Base forecasts"),
                                           ("-", "Reconciled: (a) MinT, exact QP; (b) reconciled samples")])
    if diag is not None:
        h.append(Line2D([], [], color=C_TRACK["STAT"], ls=DIAG_LS, lw=0.9, marker="o", ms=3, mfc="white", mew=0.8))
        lab.append(f"(b) STAT reconciled samples without origin {DIAG_ORIGIN} (diagnostic)")
    fig.legend(h, lab, loc="outside lower center", ncol=4 if diag is None else 2)
    save_fn(fig, "Fig4" + sfx)


def fig5(RES, src, save_fn, sfx):
    """Fig5: coverage by horizon of the uncalibrated and ACI intervals (nested; src["intervals"])."""
    iv = pd.read_csv(RES / src["intervals"] / "summary_intervals.csv")
    iv = iv[(iv.window == "main") & (iv.level == "economies") & (iv.grouping == "all") & (iv.horizon != "all")]
    iv["h"] = iv.horizon.astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 80 * MM), sharex=True, layout="constrained")
    for ax, lv, pl in zip(axes, [80, 95], "ab"):
        for tr in ["FM", "STAT"]:
            for it, ls in [("uncal_rec", "--"), ("aci_g0.05", "-")]:
                d = iv[(iv.track == tr) & (iv.interval == it) & (iv.metric == f"cov{lv}")].sort_values("h")
                ax.plot(d.h, d["mean"] * 100, color=C_TRACK[tr], ls=ls, marker="o", ms=3)
        ax.axhline(lv, color=INK, lw=0.8, ls=":")
        ax.set_title(f"({pl}) {lv}% intervals, mean over system economies", loc="left")
        ax.set_xlabel("Horizon (months)")
        ax.set_xticks(range(1, 13))
        ax.set_ylabel("Empirical coverage (%)")
    h, lab = style_legend(["FM", "STAT"], [("--", "Uncalibrated"), ("-", "ACI (γ = 0.05)")])
    fig.legend(h, lab, loc="outside lower center", ncol=4)
    save_fn(fig, "Fig5" + sfx)


def fig6(RES, args, src, save_fn, sfx):
    """Fig6: real-time test from origin 2026-06 (src["rt_cells"] for points and factors, src["rt_prob"] for the ACI
    intervals, nested here as in evaluate_realtime.py --nest)."""
    cells = pd.read_csv(RES / src["rt_cells"] / "res_total_pub_cells.csv")
    cells = cells[(cells.origin == "2026-06") & (cells.d_variant == "d")]
    units = ["W", "CHN", "GRC", "SGP", "JPN", "USA"]
    p = pd.read_parquet(Path(args.clean) / "panel_system_v2.parquet")
    pub = p[p.measure == "RES_TOTAL_PUB"].pivot(index="period", columns="unit", values="co2_t_per_day")
    rt = pd.read_parquet(Path(args.clean) / "realtime_res_total_pub_2026-07_2026-09_v2.parquet")
    rt = rt[rt.unit_type != "outside"].pivot(index="period", columns="unit", values="co2_t_per_day")
    fig, axes = plt.subplots(2, 3, figsize=(FULL, 115 * MM), layout="constrained")
    for k, u in enumerate(units):
        ax = axes.flat[k]
        div = 1e6 if u == "W" else 1e3
        hst = pub.loc["2025-01":"2026-06", u] / div
        ax.plot(ts(hst.index), hst, color=INK, lw=1.1, label="Published residence total, to 2026-06")
        ax.plot(ts(rt.index), rt[u] / div, "o", color=INK, ms=4, label="Published residence total, July to September 2026")
        for off, tr in [(-4, "FM"), (4, "STAT")]:
            c = cells[(cells.track == tr) & (cells.unit == u)].sort_values("horizon")
            aci = pd.read_parquet(RES / src["rt_prob"] / f"track={tr}" / src["rt_aci"],
                                  filters=[("unique_id", "==", f"{u}|RES_TOTAL_SYS"), ("origin", "==", "2026-06"),
                                           ("gamma", "==", 0.05)])
            xx = ts(c.target) + pd.Timedelta(days=off)
            kf = c.factor.to_numpy()
            b95 = aci[aci.level == 95].sort_values("horizon")
            b80 = aci[aci.level == 80].sort_values("horizon")
            lo95, hi95 = b95.lo.clip(lower=0).to_numpy(), b95.hi.to_numpy()
            lo80, hi80 = np.maximum(b80.lo.clip(lower=0).to_numpy(), lo95), np.minimum(b80.hi.to_numpy(), hi95)
            ax.vlines(xx, lo95 * kf / div, hi95 * kf / div, color=C_TRACK[tr], lw=0.9, alpha=0.75)
            ax.vlines(xx, lo80 * kf / div, hi80 * kf / div, color=C_TRACK[tr], lw=2.6, alpha=0.75)
            ax.plot(xx, c.point / div, "s", color=C_TRACK[tr], ms=3.5, mec="white", mew=0.5,
                    label=f"{TRACK_LABEL[tr]}: forecast, 80% and 95% intervals")
        ax.set_title(f"({'abcdef'[k]}) " + ("World (Mt per day)" if u == "W" else f"{u} (kt per day)"), loc="left")
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.tick_params(axis="x", rotation=45)
    h, lab = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside lower center", ncol=2)
    save_fn(fig, "Fig6" + sfx)



def build(args, save_fn):
    RES, PAP = Path(args.results), Path(args.results) / "paper"
    hist = pd.read_parquet(Path(args.clean) / "panel_system_wide_v2.parquet")

    # ---------------------------------------------------------------- Fig1
    fig, axes = plt.subplots(2, 1, figsize=(FULL, 120 * MM), sharex=True, layout="constrained",
                             gridspec_kw={"height_ratios": [1.5, 1]})
    x = ts(hist.index)
    ax = axes[0]
    ax.plot(x, hist["W|RES_TOTAL_SYS"] / 1e3, color=CAT[0], label="Residence total")
    ax.plot(x, (hist["W|TER_DOM"] + hist["W|TER_INT"]) / 1e3, color=CAT[1], ls="--", label="Territory total")
    sep = ts(["2025-09"])[0]
    ax.annotate("September 2025 anomaly", (sep, hist.at["2025-09", "W|RES_TOTAL_SYS"] / 1e3),
                xytext=(pd.Timestamp("2022-06-01"), 2850), fontsize=ANN, color=INK2,
                arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.6})
    ax.set_ylabel(f"kt {CO2} per day")
    ax.set_title("(a) World totals: the two accounting bases differ only by the world balance discrepancy (about 0.1%)", loc="left")
    ax.legend(loc="lower right")
    ax = axes[1]
    share = hist["W|TER_INT"] / (hist["W|TER_DOM"] + hist["W|TER_INT"]) * 100
    ax.plot(x, share, color=INK)
    ax.axvspan(pd.Timestamp("2026-01-01"), pd.Timestamp("2026-07-01"), color=GRID, alpha=0.7, lw=0)
    ax.text(pd.Timestamp("2026-01-10"), share.min() + 0.5, "2026\nreclassification", fontsize=ANN, color=INK2, ha="right")
    ep = pd.read_csv(RES / "episode_2022" / "episode_summary.csv", dtype={"episode": str}).set_index("episode").loc["2022"]
    e0, e1 = pd.Period(ep.start, "M").to_timestamp(), pd.Period(ep.end, "M").to_timestamp(how="end")
    for a_ in axes:
        a_.axvspan(e0, e1, color=GRID, alpha=0.7, lw=0, zorder=0)
    ax.text(e0 - pd.Timedelta(days=20), share.min() + 0.5,
            f"2022 reclassification episode\n{ep.start} to {ep.end}:\n"
            f"{ep.share_mean_reference_pct:.1f}% to {ep.share_mean_episode_pct:.1f}%",
            fontsize=ANN, color=INK2, ha="right")
    ax.set_ylabel("International share (%)")
    ax.set_title("(b) International share of the territory total", loc="left")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    save_fn(fig, "Fig1")

    # ---------------------------------------------------------------- Fig3: single column, panels stacked
    inc = pd.read_csv(PAP / "T3c_incoherence_cells.csv")
    m = inc.groupby(["model", "horizon"]).deviation_pct.apply(lambda v: v.abs().mean()).reset_index()
    fig, axes = plt.subplots(2, 1, figsize=(SINGLE, 120 * MM), sharey=True, sharex=True, layout="constrained")
    for ax, models, title in [(axes[0], STAT_MODELS, "(a) Statistical and machine-learning models"),
                              (axes[1], FM_MODELS, "(b) Foundation models (zero-shot)")]:
        for k, mod in enumerate(models):
            d = m[m.model == mod]
            ax.plot(d.horizon, d.deviation_pct, color=CAT[k], marker="o", ms=3, label=LABEL[mod])
        ax.set_title(title, loc="left")
        ax.set_xticks(range(1, 13))
        ax.set_ylabel("Mean |residence − territory gap|\n(% of World)")
        ax.legend(loc="upper left", ncol=2)
    axes[1].set_xlabel("Horizon (months)")
    save_fn(fig, "Fig3")

    # ---------------------------------------------------------------- Fig4 and Fig5
    sh = pd.read_csv(RES / "eval_fm_w" / "summary_by_horizon.csv")
    sh = sh[(sh.window == "main") & (sh.level == "economies") & (sh.grouping == "all") & (sh.metric == "MASE")]
    diag = stat_crps_without_origin(RES, args.clean, "prob_main_exact", SRC["crps"])
    fig4(RES, sh, SRC, save_fn, "", ymax=FIG4B_YMAX, diag=diag)
    fig5(RES, SRC, save_fn, "")

    # ---------------------------------------------------------------- Fig7: skill
    def fig7_skill(t6):
        """Skill of the two tracks: s(h) with the block-6 bootstrap band, origins per horizon, H*."""
        fig, axes = plt.subplots(1, 2, figsize=(FULL, 80 * MM), sharey=True, layout="constrained")
        for j, tr in enumerate(["FM", "STAT"]):
            d = t6[t6.track == tr]
            hs = int(d.Hstar.iloc[0])
            ax = axes[j]
            ax.fill_between(d.horizon, d.lo95, d.hi95, color=C_TRACK[tr], alpha=0.18, lw=0,
                            label="95% bootstrap band (blocks of 6 origins)")
            ax.plot(d.horizon, d.s, "-o", color=C_TRACK[tr], ms=2.5, lw=0.8, label="s(h)")
            ax.axhline(0, color=INK, lw=0.7)
            ax.axvline(hs, color=INK, lw=0.7, ls="--")
            ax.set_title(f"({'ab'[j]}) {TRACK_LABEL[tr]}", loc="left")
            ax.set_ylabel("Skill against persistence" if j == 0 else "")
            ax.set_xlabel("Horizon (months)")
            ax.set_xticks([1, 6, 12, 18, 24, 30, 36])
        lo, hi = axes[0].get_ylim()
        axes[0].set_ylim(lo, hi + 0.14 * (hi - lo))
        for j, tr in enumerate(["FM", "STAT"]):
            d = t6[t6.track == tr]
            hs = int(d.Hstar.iloc[0])
            ax = axes[j]
            tr_ = ax.get_xaxis_transform()
            for h in (1, 12, 24, 36):
                n = int(d.loc[d.horizon == h, "n_origins"].iloc[0])
                ax.text(h, 0.97, f"{n} origins", transform=tr_, fontsize=ANN, color=INK2, zorder=4,
                        ha={1: "left", 36: "right"}.get(h, "center"), va="top",
                        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6})
            ax.annotate(f"H* = {hs}", (hs, 0.80), xycoords=("data", "axes fraction"), xytext=(3, 0),
                        textcoords="offset points", fontsize=ANN)
        h1, l1 = axes[0].get_legend_handles_labels()
        fig.legend(h1, l1, loc="outside lower center", ncol=2)
        return fig

    save_fn(fig7_skill(pd.read_csv(PAP / "T6_skill.csv")), "Fig7")

    # ---------------------------------------------------------------- Fig6
    fig6(RES, args, SRC, save_fn, "")
    return {"fig4b_diagnostic": diag}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean", default="data/clean")
    ap.add_argument("--results", default="results")
    ap.add_argument("--figures", default="figures")
    ap.add_argument("--preview", default=None, help="optional folder for 150 dpi PNG previews (not part of the outputs)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    log = RunLog("paper_figures.py", vars(args))
    Path(args.figures).mkdir(parents=True, exist_ok=True)
    built = build(args, lambda fig, name: save(fig, args.figures, name, args.overwrite, args.preview))
    # the data of the Fig4b diagnostic line
    built["fig4b_diagnostic"].to_csv(out_path(Path(args.results) / "paper" / f"fig4b_stat_crps_without_{DIAG_ORIGIN}.csv",
                                              args.overwrite), index=False, lineterminator="\n", float_format="%.6g")
    # manifest of the figures: manuscript number, files and inputs
    man = [("Fig1", "data/clean/panel_system_wide_v2.parquet; results/episode_2022/episode_summary.csv"),
           ("Fig3", "results/paper/T3c_incoherence_cells.csv"),
           ("Fig4", "results/eval_fm_w/summary_by_horizon.csv; results/eval_prob_exact_nested/summary_crps.csv; "
                    "results/paper/fig4b_stat_crps_without_2024-09.csv (from results/prob_main_exact/track=STAT/crps.parquet)"),
           ("Fig5", "results/eval_prob_exact_nested/summary_intervals.csv"),
           ("Fig6", "results/eval_realtime_exact_ext_nested/res_total_pub_cells.csv; "
                    "results/prob_realtime_exact/track=*/aci_intervals_ext.parquet; "
                    "data/clean/realtime_res_total_pub_2026-07_2026-09_v2.parquet; data/clean/panel_system_v2.parquet"),
           ("Fig7", "results/paper/T6_skill.csv")]
    pd.DataFrame([(f"{n}.pdf; {n}.tif", n, "src/paper_figures.py", i) for n, i in man],
                 columns=["files", "manuscript", "script", "inputs"]) \
        .to_csv(out_path(Path(args.figures) / "MANIFEST.csv", args.overwrite), index=False, lineterminator="\n")
    log.finish("Fig1, Fig3-Fig7")


if __name__ == "__main__":
    main()
