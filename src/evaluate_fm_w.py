"""
evaluate_fm_w.py: reconciliation of the foundation-model bases with their own W (from their one-step forecast
errors), robustness against the AutoETS W, and the selection of the foundation-model track.

Scored series: the 1,120 reconciled system series. Combinations (base x method):
  statistical bases  AutoETS, AutoTheta, AutoARIMA, SeasonalNaive: base, WLS, MinT
                     (reconciled_v2_burnin12, own in-sample W with a 12-month burn-in)
  foundation models  ChronosBolt, TimesFM (3.0), TimesFM25: base;
                     WLS, MinT            own one-step W (reconciled_v2_fmownw)
                     WLS[ETS-W], MinT[ETS-W]  AutoETS W (reconciled_v2_burnin12)
  OLS (every base, reconciled_v2_burnin12) is scored as a benchmark only and left out of all tests.
Primary variant of every reconciliation method: nn_qp (exact non-negative QP); free and nn_iter are
scored too.

Outputs (results/eval_fm_w/):
  metrics_per_series.parquet (not committed), summary.csv (weighted within one level at a time,
  evalcore.summarise_levels),
  summary_by_horizon.csv (system economies, MASE, horizons 1..12),
  effect.csv            % change of summary MASE against the same base, share of series improved
  friedman_nemenyi.csv  per base, treatments base / WLS / MinT (foundation models: own W);
                        blocks: system economy series; MASE, main window, all horizons
  mcs_90.csv            Model Confidence Set (90%) over every combination except OLS; loss per origin =
                        mean scaled absolute error across system economy series (main window)
  robustness_fm_w.csv  foundation models: own W against AutoETS W
  track_selection.csv   TimesFM + WLS and TimesFM + MinT (own W), MCS loss over all horizons
  tracks_economies.csv, tracks_world.csv   the two tracks side by side
  dm_tracks.csv, dm_tracks_summary.csv     Diebold-Mariano (HLN, absolute error, h = 1, 3, 6, 12, main
                        window), per reconciled series: FM track vs its base; FM track vs AutoETS + MinT

Usage:
  python src/evaluate_fm_w.py [--overwrite]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from arch.bootstrap import MCS
from scipy import stats

from evalcore import (HORIZON_GROUPS, WINDOWS, add_scaled_errors, dm_hln, load_history, origin_scales,
                      per_series_metrics, summarise_levels)
from runlog import RunLog, out_path

STAT = ["AutoETS", "AutoTheta", "AutoARIMA", "SeasonalNaive"]
FM = ["ChronosBolt", "TimesFM", "TimesFM25"]
BASES = STAT + FM
TAG_OWN, TAG_ETS_W = "v2_fmownw", "v2_burnin12"
METHODS = ["base", "WLS", "MinT"]                  # treatments of the Friedman test (FM: own W)
PRIMARY = "nn_qp"
FM_TRACK_BASE, STAT_TRACK = "TimesFM", "AutoETS+MinT"     # the two tracks
Q05_NEMENYI = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850}     # Demsar (2006), alpha = 0.05
SEED = 42


def load_base(res, base, uids=None):
    fdir = "base_forecasts_fm" if base in FM else "base_forecasts_v2"
    b = pd.read_parquet(res / fdir / f"model={base}" / "forecasts.parquet",
                        columns=["unique_id", "origin", "target", "horizon", "y", "mean"])
    return b if uids is None else b[b.unique_id.isin(uids)]


def load_reconciled(res, base):
    """Reconciled forecasts of one base with method labels of this script."""
    ets_w = pd.read_parquet(res / f"reconciled_{TAG_ETS_W}" / f"base={base}" / "reconciled.parquet")
    if base not in FM:
        return ets_w
    ets_w["method"] = ets_w.method.where(ets_w.method == "OLS", ets_w.method + "[ETS-W]")
    own = pd.read_parquet(res / f"reconciled_{TAG_OWN}" / f"base={base}" / "reconciled.parquet")
    own = own[own.method != "OLS"]                 # OLS does not depend on W: kept once, from v2_burnin12
    return pd.concat([ets_w, own], ignore_index=True)


def dm_table(L, a, b, label):
    """DM (HLN) per series and horizon: combination a against combination b (absolute error)."""
    P = L[L.combo.isin([a, b]) & L.horizon.isin([1, 3, 6, 12])].pivot_table(
        index=["unique_id", "horizon", "origin"], columns="combo", values="ae")
    rows = []
    for (uid, h), g in P.groupby(level=["unique_id", "horizon"]):
        stat, p = dm_hln(g[a] - g[b], int(h))
        rows.append({"comparison": label, "unique_id": uid, "horizon": int(h), "n": len(g), "dm_stat": stat,
                     "p_value": p, "mean_loss_diff": (g[a] - g[b]).mean()})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean", default="data/clean")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/eval_fm_w")
    ap.add_argument("--mcs-reps", type=int, default=10000)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    log = RunLog("evaluate_fm_w.py", vars(args))
    RES, OUT = Path(args.results), Path(args.out)

    def save(df, name):
        df.to_csv(out_path(OUT / name, args.overwrite), index=False, float_format="%.6g")

    meta = pd.read_parquet(RES / "base_forecasts_v2" / "series_meta.parquet")
    sc = origin_scales(load_history(args.clean))

    ps_all, loss_all = [], []
    for base in BASES:
        rec = load_reconciled(RES, base)
        uids = sorted(set(rec.unique_id) - {"W|DELTA_INT"})
        b = load_base(RES, base, uids)
        b = b[b.origin.isin(rec.origin.unique())]
        rec = rec[rec.unique_id.isin(uids)].merge(b[["unique_id", "origin", "target", "y"]],
                                                  on=["unique_id", "origin", "target"])
        f = pd.concat([b.assign(method="base", variant="base"), rec.rename(columns={"value": "mean"})],
                      ignore_index=True)
        assert f.groupby(["method", "variant"]).size().nunique() == 1        # complete grid for every method
        f = add_scaled_errors(f, sc).assign(base=base)
        ps_all.append(per_series_metrics(f, ["base", "method", "variant"]))
        keep = f.variant.isin(["base", PRIMARY]) & (f.method != "OLS") & WINDOWS["main"](f.target)
        loss_all.append(f.loc[keep, ["base", "method", "unique_id", "origin", "horizon", "ae", "sae"]])
        print(f"{base}: scored {len(f)} forecasts; methods {sorted(f.method.unique())}")

    ps = pd.concat(ps_all, ignore_index=True).merge(meta, on="unique_id")
    ps.to_parquet(out_path(OUT / "metrics_per_series.parquet", args.overwrite), index=False)
    keys = ["base", "method", "variant"]
    summ = summarise_levels(ps, keys)
    save(summ, "summary.csv")
    pp = ps[ps.variant.isin(["base", PRIMARY])]
    byh = summarise_levels(pp, keys, metrics=("MASE",), horizons=[str(h) for h in range(1, 13)])
    save(byh, "summary_by_horizon.csv")

    # ---- effect of reconciliation against the same base
    sm = summ[summ.metric == "MASE"]
    ref = sm[sm.method == "base"].drop(columns=["method", "variant"])
    on = ["window", "level", "grouping", "grp", "base", "horizon", "metric"]
    eff = sm[sm.method != "base"].merge(ref, on=on, suffixes=("", "_base"))
    for st in ["median", "mean", "weighted"]:
        eff[f"pct_change_{st}"] = (eff[st] / eff[f"{st}_base"] - 1) * 100
    pm = ps[ps.horizon.isin(HORIZON_GROUPS)]
    pb = pm[pm.method == "base"][["unique_id", "base", "window", "horizon", "MASE"]]
    imp = pm[pm.method != "base"].merge(pb, on=["unique_id", "base", "window", "horizon"], suffixes=("", "_base"))
    imp["improved"] = imp.MASE < imp.MASE_base
    imp["level"] = imp.unit_type.map({"economy": "economies", "world": "world", "residual": "residual"})
    share = imp.groupby(["window", "level", "base", "method", "variant", "horizon"]).improved.mean() \
        .rename("share_series_improved").reset_index()
    eff = eff.merge(share, on=["window", "level", "base", "method", "variant", "horizon"], how="left")
    eff.loc[eff.grouping != "all", "share_series_improved"] = np.nan
    save(eff, "effect.csv")

    # ---- Friedman test with Nemenyi comparisons, per base
    fr = []
    for base in BASES:
        d = pp[(pp.base == base) & (pp.window == "main") & (pp.horizon == "all") & (pp.unit_type == "economy")
               & pp.method.isin(METHODS)].pivot(index="unique_id", columns="method", values="MASE")[METHODS].dropna()
        chi2, p = stats.friedmanchisquare(*[d[m] for m in METHODS])
        ranks = d.rank(axis=1).mean()
        k, N = len(METHODS), len(d)
        cd = Q05_NEMENYI[k] * np.sqrt(k * (k + 1) / (6 * N))
        for i, m1 in enumerate(METHODS):
            for m2 in METHODS[i + 1:]:
                fr.append({"base": base, "w": "own_onestep" if base in FM else "own_insample", "n_series": N,
                           "friedman_chi2": chi2, "friedman_p": p, "method_1": m1, "mean_rank_1": ranks[m1],
                           "method_2": m2, "mean_rank_2": ranks[m2], "nemenyi_cd": cd,
                           "rank_diff": ranks[m1] - ranks[m2], "significant_5pct": abs(ranks[m1] - ranks[m2]) > cd})
    save(pd.DataFrame(fr), "friedman_nemenyi.csv")

    # ---- Model Confidence Set over every combination except OLS (system economies)
    L = pd.concat(loss_all, ignore_index=True).merge(meta[["unique_id", "unit_type"]], on="unique_id")
    L["combo"] = L.base + "+" + L.method
    Le = L[L.unit_type == "economy"]
    combos = sorted(Le.combo.unique(), key=lambda c: (BASES.index(c.split("+")[0]), c))
    mcs_rows = []
    for hg in HORIZON_GROUPS:
        d = Le if hg == "all" else Le[Le.horizon == int(hg)]
        loss = d.pivot_table(index="origin", columns="combo", values="sae", aggfunc="mean")[combos]
        mcs = MCS(loss.to_numpy(), size=0.10, reps=args.mcs_reps, method="R", bootstrap="stationary", seed=SEED)
        mcs.compute()
        pv = mcs.pvalues.reindex(range(len(combos)))
        for i, c in enumerate(combos):
            mcs_rows.append({"horizon_group": hg, "combo": c, "n_origins": len(loss), "mean_loss": loss[c].mean(),
                             "mcs_pvalue": float(pv.iloc[i, 0]), "in_mcs_90": i in mcs.included})
    mcs_df = pd.DataFrame(mcs_rows)
    save(mcs_df, "mcs_90.csv")
    print(mcs_df[mcs_df.horizon_group == "all"].sort_values("mean_loss").to_string(index=False))

    # ---- foundation-model track selection: TimesFM 3.0 with its own W, WLS or MinT, lower MCS loss over all horizons
    sel = mcs_df[(mcs_df.horizon_group == "all") & mcs_df.combo.isin([f"{FM_TRACK_BASE}+WLS", f"{FM_TRACK_BASE}+MinT"])]
    sel = sel.sort_values("mean_loss").assign(chosen=lambda d: [True] + [False] * (len(d) - 1))
    save(sel, "track_selection.csv")
    fm_track = sel.combo.iloc[0]
    print("foundation-model track:", fm_track)

    # ---- robustness: own W against AutoETS W, foundation-model bases
    rows = []
    s_all = summ[(summ.metric == "MASE") & (summ.variant == PRIMARY) & (summ.horizon == "all")]
    for base in FM:
        for m in ["WLS", "MinT"]:
            for w, meth in [("own", m), ("AutoETS", f"{m}[ETS-W]")]:
                r = {"base": base, "method": m, "W": w}
                for win in ["main", "robB"]:
                    e = s_all[(s_all.window == win) & (s_all.level == "economies") & (s_all.grouping == "all")
                              & (s_all.base == base) & (s_all.method == meth)]
                    r[f"{win}_median"], r[f"{win}_weighted"] = e["median"].iloc[0], e["weighted"].iloc[0]
                for lvl, g in [("world", "RES_TOTAL_SYS"), ("world", "TER_INT"), ("residual", "RES_TOTAL_SYS")]:
                    e = s_all[(s_all.window == "main") & (s_all.level == lvl) & (s_all.grp == g)
                              & (s_all.base == base) & (s_all.method == meth)]
                    r[f"main_{lvl}_{g}"] = e["median"].iloc[0]
                mc = mcs_df[(mcs_df.combo == f"{base}+{meth}")].set_index("horizon_group")
                r["mcs_loss_all"], r["in_mcs_all"] = mc.at["all", "mean_loss"], mc.at["all", "in_mcs_90"]
                r["mcs_loss_h1"], r["mcs_loss_h12"] = mc.at["1", "mean_loss"], mc.at["12", "mean_loss"]
                rows.append(r)
    save(pd.DataFrame(rows), "robustness_fm_w.csv")

    # ---- the two tracks side by side
    tb, tm = fm_track.split("+")
    sb, sm_ = STAT_TRACK.split("+")
    pick = lambda d: d[((d.base == tb) & (d.method == tm)) | ((d.base == sb) & (d.method == sm_))
                       | (d.base.isin([tb, sb]) & (d.method == "base"))]
    te = pick(byh[(byh.level == "economies") & (byh.grouping == "all")])
    te_all = pick(summ[(summ.metric == "MASE") & (summ.level == "economies") & (summ.grouping == "all")
                       & summ.variant.isin(["base", PRIMARY]) & (summ.horizon == "all")])
    save(pd.concat([te, te_all], ignore_index=True), "tracks_economies.csv")
    tw = pick(summ[(summ.level == "world") & summ.variant.isin(["base", PRIMARY])])
    save(tw, "tracks_world.csv")

    # ---- Diebold-Mariano for the tracks
    dm = pd.concat([dm_table(L, fm_track, f"{tb}+base", f"{fm_track} vs {tb}+base"),
                    dm_table(L, fm_track, STAT_TRACK, f"{fm_track} vs {STAT_TRACK}")], ignore_index=True)
    dm = dm.merge(meta[["unique_id", "unit_type"]], on="unique_id")
    save(dm, "dm_tracks.csv")
    dm["better"] = (dm.p_value < 0.05) & (dm.mean_loss_diff < 0)
    dm["worse"] = (dm.p_value < 0.05) & (dm.mean_loss_diff > 0)
    dms = dm.groupby(["comparison", "unit_type", "horizon"]).agg(
        n=("unique_id", "size"), share_better=("better", "mean"), share_worse=("worse", "mean")).reset_index()
    save(dms, "dm_tracks_summary.csv")
    print(dms.to_string(index=False))
    log.finish(f"FM track {fm_track}")


if __name__ == "__main__":
    main()
