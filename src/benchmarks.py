"""
benchmarks.py: reconciliation benchmarks, origin-block inference and two checks. Nothing is refitted and no QP is run: every number comes from stored forecasts, samples and data.

Bases: TimesFM 2.5 (results/base_forecasts_fm/model=TimesFM25) and ETS (results/base_forecasts_v2/model=AutoETS).
Series: the 1,120 reconciled system series (W|DELTA_INT excluded). Origins 2022-12 ... 2025-06, h 1-12.

1. Benchmarks
  BU    bottom-up from each base: the seven component forecasts of every unit (110 system economies and the Residual)
        are kept; each unit's totals are RES_TOTAL_SYS = A+B+D+E+F, TER_DOM = A+C, TER_INT = D+G; every World component
        is the sum over the economies and the Residual, and the World totals are sums of the World components. The
        world-balance conditions are not imposed. Base forecasts are used as stored (no clipping, no dead-series zeros).
  COMB  the simple average of the TimesFM 2.5 and ETS base point forecasts, for every series.
  MASE  per series = mean scaled absolute error over the main window (targets <= 2025-12 without 2025-09), scaled by the
        seasonal-naive scale at the origin (evalcore); system economies: median and weighted mean (2022 weights);
        the ten World series one by one. Base, WLS and MinT (exact QP; FM with its own one-step W; ETS with the 12-month burn-in) are scored
        with the same code and checked against results/paper/T3a, T3b.
  Violations, per origin and horizon, in % of the forecast's own World RES_TOTAL_SYS:
        world balance  B(W) - C(W) and E(W) + F(W) - (1 + r) G(W) (r = mean DELTA_INT / G over the 12 months
                       up to the origin), and the residence-territory gap TER_DOM + TER_INT + r G - RES_TOTAL_SYS (T3c);
        all identities every row of the full constraint matrix (results/C_matrix_v2.npz, 348 rows: within-unit,
                       aggregation, world balance; DELTA_INT column replaced by r G): max |C y| over the rows of
                       each type and over all rows.
        Mean over the 31 origins by horizon (and the maximum). BU, COMB and, for reference, the two bases.
2. Inference
  Moving-block bootstrap over the 31 consecutive origins (blocks of 6, start uniform on 1 ... 26, concatenated and cut
  to 31; skill.mbb_counts), all series jointly, 2,000 resamples, seed 42. Each resample gives origin counts; the
  per-series score is the count-weighted mean of its main-window cells, then the weighted mean (2022 weights) over the system
  economies. Differences: weighted MASE FM - STAT (reconciled, MinT exact QP); reconciled - base for each track;
  weighted sCRPS FM - STAT (reconciled samples of results/prob_main, as in T4). Percentile 95% interval and the share
  of resamples below zero. All horizons pooled (main), and h = 1, 3, 6, 12.
3. Checks
  World MASE coincidence  FM base MASE of World TER_INT and of World G (NRES_INT_FROM), six decimals.
  Published totals        RES_TOTAL_PUB in the main window: forecast = track RES_TOTAL_SYS forecast / (1 - d), d =
                          mean (published - component sum) / published over the 6 months up to the origin (as in
                          evaluate_realtime.py); scale = seasonal-naive scale of the published series up to the origin;
                          weights = 2022 weight of RES_TOTAL_SYS. Also d = 0, the base forecasts, and RES_TOTAL_SYS.

Outputs: results/benchmarks/, results/inference/, results/checks/
Usage:
  python src/benchmarks.py [--reps 2000] [--overwrite]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from evalcore import WINDOWS, load_history, origin_scales
from reconcile import load_system
from skill import mbb_counts
from runlog import RunLog, out_path

ORIGINS = pd.period_range("2022-12", "2025-06", freq="M").astype(str).tolist()          # 31
COMPONENTS = ["RES_DOM_IN", "RES_DOM_OUT", "NRES_DOM_IN", "RES_INT_FROM", "RES_INT_TO", "RES_INT_OUT", "NRES_INT_FROM"]
TOTAL_PARTS = {"RES_TOTAL_SYS": ["RES_DOM_IN", "RES_DOM_OUT", "RES_INT_FROM", "RES_INT_TO", "RES_INT_OUT"],
               "TER_DOM": ["RES_DOM_IN", "NRES_DOM_IN"], "TER_INT": ["RES_INT_FROM", "NRES_INT_FROM"]}
TRACKS = {"FM": ("TimesFM25", "base_forecasts_fm", "v2_fmownw"), "STAT": ("AutoETS", "base_forecasts_v2", "v2_burnin12")}
SEED, BLOCK = 42, 6
H_SEL = [1, 3, 6, 12]


def bottom_up_matrix(uids):
    """S (n x n): y_BU = S b, where b holds the base forecasts of all n series and only the unit components are used."""
    pos = {u: i for i, u in enumerate(uids)}
    units = sorted({u.split("|")[0] for u in uids} - {"W"})
    S = np.zeros((len(uids), len(uids)))
    for i, uid in enumerate(uids):
        unit, k = uid.split("|")
        parts = TOTAL_PARTS.get(k, [k])
        for p in parts:
            for v in (units if unit == "W" else [unit]):
                S[i, pos[f"{v}|{p}"]] = 1.0
    return S


def per_series_mase(cells):
    """evalcore semantics: mean over the cells (NaN skipped, inf kept), then finite series only."""
    return cells.groupby("unique_id").sae.mean()


def summary(ms, meta, econ):
    e = ms.reindex(econ)
    e = e[np.isfinite(e)]
    w = meta.loc[e.index, "weight_2022"]
    out = {"MASE_median_economies": e.median(), "MASE_weighted_economies": np.average(e, weights=w), "n_economy_series": len(e)}
    out.update({f"World {u.split('|')[1]}": ms.get(u, np.nan) for u in sorted(ms.index) if u.startswith("W|")})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean", default="data/clean")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results")
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    log = RunLog("benchmarks.py", vars(args))
    RES, OUT = Path(args.results), Path(args.out)

    def save(df, name):
        df.to_csv(out_path(OUT / name, args.overwrite), index=False, float_format="%.6g")

    meta = pd.read_parquet(RES / "base_forecasts_v2" / "series_meta.parquet").set_index("unique_id")
    Y = load_history(args.clean)
    sc = origin_scales(Y)[["unique_id", "origin", "scale1"]]
    y_long = Y.stack().rename("y").reset_index()
    y_long.columns = ["target", "unique_id", "y"]
    sysd = load_system(RES, args.clean)
    uids = sysd["uids"]                                                     # 1,120 system series, DELTA_INT dropped
    n = len(uids)
    econ = [u for u in uids if meta.at[u, "unit_type"] == "economy"]
    r_o = {o: float(sysd["ratio"].loc[:o].tail(12).mean()) for o in ORIGINS}     # DELTA_INT ratio r (12 months up to the origin)

    # ---------------------------------------------------------------- forecasts: wide arrays [origin, horizon, series]
    def wide(f, col):
        p = f.pivot_table(index=["origin", "horizon"], columns="unique_id", values=col)
        p = p.reindex(index=pd.MultiIndex.from_product([ORIGINS, range(1, 13)]), columns=uids)
        assert np.isfinite(p.to_numpy()).all(), col
        return p.to_numpy().reshape(len(ORIGINS), 12, n)

    def to_long(X, name):
        o, h = np.meshgrid(np.arange(len(ORIGINS)), np.arange(12), indexing="ij")
        d = pd.DataFrame(X.reshape(-1, n), columns=uids)
        d["origin"] = np.array(ORIGINS)[o.ravel()]
        d["horizon"] = h.ravel() + 1
        d = d.melt(id_vars=["origin", "horizon"], var_name="unique_id", value_name="value")
        d["target"] = (pd.PeriodIndex(d.origin, freq="M") + (d.horizon.to_numpy())).astype(str)
        return d.assign(method=name)

    base, rec = {}, {}
    flt = [("method", "in", ["WLS", "MinT"]), ("variant", "==", "nn_qp")]
    for tr, (b, fdir, tag) in TRACKS.items():
        f = pd.read_parquet(RES / fdir / f"model={b}" / "forecasts.parquet",
                            columns=["unique_id", "origin", "horizon", "mean"])
        f = f[f.unique_id.isin(uids) & f.origin.isin(ORIGINS) & (f.horizon <= 12)]
        base[tr] = wide(f, "mean")
        rr = pd.read_parquet(RES / f"reconciled_{tag}" / f"base={b}" / "reconciled.parquet", filters=flt)
        rr = rr[rr.unique_id.isin(uids) & rr.origin.isin(ORIGINS)]
        rec[tr] = {m: wide(g, "value") for m, g in rr.groupby("method")}
    S = bottom_up_matrix(uids)
    fc = {}
    for tr in TRACKS:
        fc[(tr, "base")] = base[tr]
        fc[(tr, "WLS")] = rec[tr]["WLS"]
        fc[(tr, "MinT")] = rec[tr]["MinT"]
        fc[(tr, "BU")] = base[tr] @ S.T
    fc[("COMB", "COMB")] = 0.5 * (base["FM"] + base["STAT"])

    # ---------------------------------------------------------------- benchmarks: MASE
    cells = {}
    for key, X in fc.items():
        d = to_long(X, "/".join(key)).merge(y_long, on=["unique_id", "target"]).merge(sc, on=["unique_id", "origin"])
        d = d[WINDOWS["main"](d.target)]
        d["sae"] = (d.y - d.value).abs() / d.scale1
        cells[key] = d[["unique_id", "origin", "horizon", "sae"]]
    rows, byh = [], []
    for key, d in cells.items():
        ms = per_series_mase(d)
        rows.append({"base": key[0], "method": key[1], **summary(ms, meta, econ)})
        e = d[d.unique_id.isin(econ)]
        for h, g in e.groupby("horizon"):
            m = g.groupby("unique_id").sae.mean()
            m = m[np.isfinite(m)]
            byh.append({"base": key[0], "method": key[1], "horizon": int(h), "MASE_median_economies": m.median(),
                        "MASE_weighted_economies": np.average(m, weights=meta.loc[m.index, "weight_2022"])})
    mase = pd.DataFrame(rows)
    # check against T3a / T3b (base, WLS, MinT)
    t3a = pd.read_csv(RES / "paper" / "T3a_reconciliation_economies.csv")
    t3b = pd.read_csv(RES / "paper" / "T3b_reconciliation_world.csv")
    lab = {"FM": "FM (TimesFM 2.5, own W)", "STAT": "STAT (AutoETS)"}
    chk = []
    for tr in TRACKS:
        for m in ["base", "WLS", "MinT"]:
            a = t3a[(t3a.track == lab[tr]) & (t3a.method == m)].iloc[0]
            b = t3b[(t3b.track == lab[tr]) & (t3b.method == m)].iloc[0]
            x = mase[(mase.base == tr) & (mase.method == m)].iloc[0]
            cols = ["MASE_median_economies", "MASE_weighted_economies"] + [c for c in t3b.columns if c.startswith("World")]
            dmax = max(abs(x[c] - (a[c] if c in a else b[c])) for c in cols)
            chk.append({"base": tr, "method": m, "max_abs_diff_vs_T3a_T3b": dmax})
    chk = pd.DataFrame(chk)
    assert (chk.max_abs_diff_vs_T3a_T3b < 1e-5).all(), chk          # T3 is stored to 6 significant digits
    mase["base"] = mase.base.map({"FM": "TimesFM 2.5", "STAT": "ETS", "COMB": "TimesFM 2.5 + ETS"})
    save(mase, "benchmarks/mase_summary.csv")
    save(pd.DataFrame(byh).assign(base=lambda x: x.base.map({"FM": "TimesFM 2.5", "STAT": "ETS", "COMB": "TimesFM 2.5 + ETS"})),
         "benchmarks/mase_by_horizon.csv")
    save(chk, "benchmarks/check_T3.csv")

    # ---------------------------------------------------------------- benchmarks: violations
    Cf = sp.load_npz(RES / "C_matrix_v2.npz").toarray()
    cols = pd.read_csv(RES / "C_matrix_cols_v2.csv")
    rtype = pd.read_csv(RES / "C_matrix_rows_v2.csv").type.to_numpy()
    uids_all = (cols.unit + "|" + cols.measure).tolist()
    Cf = Cf[:, [uids_all.index(u) for u in uids]]
    ef = int(np.flatnonzero(rtype == "world_balance_EF_G")[0])
    bc = int(np.flatnonzero(rtype == "world_balance_B_C")[0])
    g_col = uids.index("W|NRES_INT_FROM")
    iw = {k: uids.index(f"W|{k}") for k in ["RES_TOTAL_SYS", "TER_DOM", "TER_INT", "NRES_INT_FROM"]}
    groups = {"within-unit": np.char.startswith(rtype.astype(str), "within"), "aggregation": rtype == "aggregation",
              "world balance": np.char.startswith(rtype.astype(str), "world_balance")}
    viol, viol_cells = [], []
    for key, X in fc.items():
        if key[1] in ("WLS", "MinT"):
            continue
        for i, o in enumerate(ORIGINS):
            r = r_o[o]
            Co = Cf.copy()
            Co[ef, g_col] = -(1.0 + r)
            R = X[i] @ Co.T                                                  # [h, rows]
            wt = X[i][:, iw["RES_TOTAL_SYS"]]
            gap = X[i][:, iw["TER_DOM"]] + X[i][:, iw["TER_INT"]] + r * X[i][:, iw["NRES_INT_FROM"]] - wt
            for h in range(12):
                rec_ = {"base": key[0], "method": key[1], "origin": o, "horizon": h + 1, "world_res_total": wt[h],
                        "B_minus_C_pct": 100 * R[h, bc] / wt[h], "EF_minus_1r_G_pct": 100 * R[h, ef] / wt[h],
                        "res_ter_gap_pct": 100 * gap[h] / wt[h], "max_all_rows_pct": 100 * np.abs(R[h]).max() / wt[h]}
                for gname, msk in groups.items():
                    rec_[f"max_{gname.replace(' ', '_').replace('-', '_')}_pct"] = 100 * np.abs(R[h, msk]).max() / wt[h]
                viol_cells.append(rec_)
    vc = pd.DataFrame(viol_cells)
    vc["base"] = vc.base.map({"FM": "TimesFM 2.5", "STAT": "ETS", "COMB": "TimesFM 2.5 + ETS"})
    save(vc, "benchmarks/violations_cells.csv")
    num = [c for c in vc.columns if c.endswith("_pct")]
    va = vc.assign(**{c: vc[c].abs() for c in num})
    k = ["base", "method", "horizon"]
    v1 = va.groupby(k)[num].mean().add_prefix("mean_abs_").join(va.groupby(k)[num].max().add_prefix("max_abs_")).reset_index()
    va["horizon"] = "all"
    v2 = va.groupby(k)[num].mean().add_prefix("mean_abs_").join(va.groupby(k)[num].max().add_prefix("max_abs_")).reset_index()
    save(pd.concat([v1, v2], ignore_index=True), "benchmarks/violations_by_horizon.csv")
    neg = [{"base": k[0], "method": k[1], "min_value": float(X.min()), "share_negative": float((X < 0).mean())}
           for k, X in fc.items()]
    save(pd.DataFrame(neg), "benchmarks/negative_values.csv")

    # ---------------------------------------------------------------- inference
    rng = np.random.default_rng(SEED)
    cnt = mbb_counts(len(ORIGINS), BLOCK, args.reps, rng)                  # [reps, 31]
    oi = {o: i for i, o in enumerate(ORIGINS)}
    ei = {u: i for i, u in enumerate(econ)}
    w_e = meta.loc[econ, "weight_2022"].to_numpy()

    def sums(d, col, hsel=None):
        """[series, origin] sums and counts of col over the main-window cells (NaN cells skipped, inf kept: the
        series then drops out, as in evalcore)."""
        d = d[d.unique_id.isin(econ)]
        if hsel is not None:
            d = d[d.horizon == hsel]
        d = d[d[col].notna()]
        Ssum, N = np.zeros((len(econ), len(ORIGINS))), np.zeros((len(econ), len(ORIGINS)))
        np.add.at(Ssum, (d.unique_id.map(ei).to_numpy(), d.origin.map(oi).to_numpy()), d[col].to_numpy())
        np.add.at(N, (d.unique_id.map(ei).to_numpy(), d.origin.map(oi).to_numpy()), 1.0)
        return Ssum, N

    def wmean(Ssum, N, c, keep):
        with np.errstate(invalid="ignore", divide="ignore"):
            m = (Ssum[keep] @ c.T) / (N[keep] @ c.T)                         # [series, reps]
        return (w_e[keep][:, None] * m).sum(axis=0) / w_e[keep].sum()

    crps = {}
    for tr in TRACKS:
        c = pd.read_parquet(RES / "prob_main" / f"track={tr}" / "crps.parquet", filters=[("kind", "==", "rec")])
        c = c.merge(sc, on=["unique_id", "origin"])
        c = c[WINDOWS["main"](c.target)]
        c["sae"] = c.crps / c.scale1                                         # sCRPS, in the column used by sums()
        crps[tr] = c[["unique_id", "origin", "horizon", "sae"]]
    comps = [("weighted MASE, FM - STAT (reconciled, MinT)", cells[("FM", "MinT")], cells[("STAT", "MinT")]),
             ("weighted MASE, FM reconciled - FM base", cells[("FM", "MinT")], cells[("FM", "base")]),
             ("weighted MASE, STAT reconciled - STAT base", cells[("STAT", "MinT")], cells[("STAT", "base")]),
             ("weighted sCRPS, FM - STAT (reconciled samples)", crps["FM"], crps["STAT"])]
    inf_rows, dist = [], {}
    ones = np.ones((1, len(ORIGINS)))
    for label, a, b in comps:
        for hsel in [None] + H_SEL:
            Sa, Na = sums(a, "sae", hsel)
            Sb, Nb = sums(b, "sae", hsel)
            with np.errstate(invalid="ignore", divide="ignore"):
                keep = np.isfinite(Sa.sum(1) / Na.sum(1)) & np.isfinite(Sb.sum(1) / Nb.sum(1))
            pa, pb = wmean(Sa, Na, ones, keep)[0], wmean(Sb, Nb, ones, keep)[0]
            ba, bb = wmean(Sa, Na, cnt, keep), wmean(Sb, Nb, cnt, keep)
            dd = ba - bb
            ok = np.isfinite(dd)
            inf_rows.append({"comparison": label, "horizon": "all" if hsel is None else hsel, "a": pa, "b": pb,
                             "difference": pa - pb, "rel_difference_pct": 100 * (pa / pb - 1),
                             "lo95": np.percentile(dd[ok], 2.5), "hi95": np.percentile(dd[ok], 97.5),
                             "share_below_zero": float((dd[ok] < 0).mean()), "n_series": int(keep.sum()),
                             "n_resamples": int(ok.sum()), "boot_mean": float(dd[ok].mean())})
            if hsel is None:
                dist[label] = dd
    inf = pd.DataFrame(inf_rows)
    save(inf, "inference/block_bootstrap.csv")
    save(pd.DataFrame(dist), "inference/bootstrap_draws.csv")
    save(pd.DataFrame(cnt.astype(int), columns=ORIGINS), "inference/origin_counts.csv")

    # ---------------------------------------------------------------- check: World MASE coincidence
    fmb = cells[("FM", "base")]
    wrows = []
    raw = to_long(fc[("FM", "base")], "FM/base").merge(y_long, on=["unique_id", "target"]).merge(sc, on=["unique_id", "origin"])
    raw = raw[WINDOWS["main"](raw.target)]
    for u in ["W|TER_INT", "W|NRES_INT_FROM", "W|RES_INT_FROM"]:
        g = raw[raw.unique_id == u]
        wrows.append({"series": u, "MASE": fmb[fmb.unique_id == u].sae.mean(), "n_cells": len(g),
                      "mean_abs_error_t_per_day": (g.y - g.value).abs().mean(), "mean_scale_t_per_day": g.scale1.mean(),
                      "mean_actual_t_per_day": g.y.mean()})
    wm = pd.DataFrame(wrows)
    a = raw[raw.unique_id == "W|TER_INT"].set_index(["origin", "horizon"])
    g = raw[raw.unique_id == "W|NRES_INT_FROM"].set_index(["origin", "horizon"])
    dser = raw[raw.unique_id == "W|RES_INT_FROM"].set_index(["origin", "horizon"])
    extra = {"share_D_in_TER_INT_actual": float((dser.y / a.y).mean()),
             "corr_errors_TER_INT_G": float(np.corrcoef(a.y - a.value, g.y - g.value)[0, 1]),
             "mean_abs_diff_sae": float(((a.y - a.value).abs() / a.scale1 - (g.y - g.value).abs() / g.scale1).abs().mean()),
             "mean_ratio_scale_TER_INT_to_G": float((a.scale1 / g.scale1).mean()),
             "mean_ratio_abs_error_TER_INT_to_G": float(((a.y - a.value).abs() / (g.y - g.value).abs()).mean()),
             "T3b_TER_INT": float(t3b.loc[t3b.method.eq("base") & t3b.track.eq(lab["FM"]), "World TER_INT"].iloc[0]),
             "T3b_G": float(t3b.loc[t3b.method.eq("base") & t3b.track.eq(lab["FM"]), "World NRES_INT_FROM"].iloc[0])}
    save(wm.assign(**extra), "checks/world_mase_coincidence.csv")

    # ---------------------------------------------------------------- check: published residence totals
    p = pd.read_parquet(Path(args.clean) / "panel_system_v2.parquet")
    pub = p[p.measure == "RES_TOTAL_PUB"].pivot(index="period", columns="unit", values="co2_t_per_day")
    sysT = p[p.measure == "RES_TOTAL_SYS"].pivot(index="period", columns="unit", values="co2_t_per_day")
    units = list(pub.columns)
    prow = []
    for o in ORIGINS:
        win = pub.loc[:o].tail(6)
        dvec = ((win - sysT.loc[win.index, units]) / win.where(win != 0)).mean().fillna(0.0)
        hp = pub.loc[:o]
        scp = (hp - hp.shift(12)).loc["2020-01":].abs().mean()
        prow.append(pd.DataFrame({"unit": units, "origin": o, "d": dvec[units].to_numpy(), "scale_pub": scp[units].to_numpy()}))
    dd_ = pd.concat(prow, ignore_index=True)
    ypub = pub.stack().rename("y_pub").reset_index()
    ypub.columns = ["target", "unit", "y_pub"]
    res_ids = [f"{u}|RES_TOTAL_SYS" for u in units]
    pubc = []
    for tr in TRACKS:
        for m in ["MinT", "base"]:
            d = to_long(fc[(tr, m)], m)
            d = d[d.unique_id.isin(res_ids)]
            d["unit"] = d.unique_id.str.split("|").str[0]
            d = d.merge(dd_, on=["unit", "origin"]).merge(ypub, on=["unit", "target"], how="left")
            d = d.merge(y_long.rename(columns={"y": "y_sys"}), on=["unique_id", "target"]).merge(sc, on=["unique_id", "origin"])
            d = d[WINDOWS["main"](d.target)]
            for dv in ["(1 - d)", "d = 0", "RES_TOTAL_SYS"]:
                if dv == "RES_TOTAL_SYS":
                    e = (d.y_sys - d.value).abs() / d.scale1
                else:
                    k = 1.0 / (1.0 - d.d) if dv == "(1 - d)" else 1.0
                    e = (d.y_pub - d.value * k).abs() / d.scale_pub
                pubc.append(pd.DataFrame({"track": tr, "forecast": "reconciled (MinT)" if m == "MinT" else "base",
                                          "target_series": "RES_TOTAL_PUB, " + dv if dv != "RES_TOTAL_SYS" else dv,
                                          "unit": d.unit, "origin": d.origin, "horizon": d.horizon, "sae": e,
                                          "y_missing": d.y_pub.isna() if dv != "RES_TOTAL_SYS" else False}))
    pc = pd.concat(pubc, ignore_index=True)
    utype = {u: ("world" if u == "W" else "residual" if u == "RESIDUAL" else "economy") for u in units}
    srows = []
    for k, gk in pc.groupby(["track", "forecast", "target_series"]):
        ps = gk.groupby("unit").sae.mean()
        e = ps[[u for u in ps.index if utype[u] == "economy"]]
        e = e[np.isfinite(e)]
        srows.append({**dict(zip(["track", "forecast", "target_series"], k)), "MASE_median_economies": e.median(),
                      "MASE_weighted_economies": np.average(e, weights=meta.loc[[f"{u}|RES_TOTAL_SYS" for u in e.index], "weight_2022"]),
                      "MASE_World": ps.get("W"), "MASE_Residual": ps.get("RESIDUAL"), "n_economies": len(e),
                      "n_cells": int(gk.sae.notna().sum()), "n_cells_unpublished": int(gk.y_missing.sum())})
    save(pd.DataFrame(srows), "checks/published_totals_mase.csv")
    save(dd_, "checks/published_totals_d.csv")

    pd.set_option("display.width", 250)
    print(mase.to_string(index=False))
    print(inf[inf.horizon == "all"].to_string(index=False))
    print(wm.to_string(index=False), extra)
    print(pd.DataFrame(srows).to_string(index=False))
    log.finish("benchmarks, inference, checks")


if __name__ == "__main__":
    main()
