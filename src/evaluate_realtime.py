"""
evaluate_realtime.py: real-time tests of the two tracks.

Origins
  2025-06  horizons 1-15. System series scored to 2026-06 (h1-12): targets 2025-07 ... 2025-12 and,
           separately, 2026-01 ... 2026-06 (after the 2026 reclassification). RES_TOTAL_PUB scored to 2026-09
           (h1-15).
  2026-06  horizons 1-3 (2026-07 ... 2026-09; only RES_TOTAL_PUB is published).

Forecasts per track (FM: TimesFM25 + MinT with own W; STAT: AutoETS + MinT):
  point        exact non-negative MinT reconciliation (results/reconciled_rt, nn_qp); base point for reference
  samples      results/prob_realtime/track=<T>/ (prob_samples.py --mode realtime)
  intervals    ACI (gamma 0.05; state of the main run) and pooled split conformal (half-widths from the
               main-run point forecasts with outcomes known at the origin; horizons > 12 use horizon 12)
RES_TOTAL_PUB: forecast = RES_TOTAL_SYS forecast x (1 + d), d = mean of (published - component sum) /
published over the 6 months up to and including the origin (months with a zero published value are
skipped; d = 0 if none is left); samples and interval bounds are scaled the same way. A second table
uses d = 0.
With --d-form exact the forecast is divided by (1 - d) instead (the exact form); the default is --d-form plus.

Metrics: MASE and sCRPS scaled by the seasonal-naive scale up to the origin (for RES_TOTAL_PUB, from its
own published history), coverage of the 80% and 95% calibrated intervals. Summaries: system economies
(median; weighted by 2022 level), World, Residual (reported, not used for choices) and the 10 largest system
economies (mean RES_TOTAL_SYS per day in 2025) one by one.

Checks: at 2025-06, horizons 1-12, the real-time point forecasts and CRPS must equal the main run.

With --nest the 80% ACI interval is clipped to lie inside the 95% ACI interval before
scoring (results/eval_realtime_zeroed_nested).

--prob-dir reads the real-time samples from another folder (exact samples:
results/prob_realtime_exact), --main-prob-dir the main-run CRPS of the 2025-06 check (results/prob_main_exact),
and --aci-name another interval file of that folder (aci_intervals_ext.parquet: origin 2026-06 with the extended
ACI state). The defaults are unchanged.

Outputs: results/eval_realtime/
Usage:
  python src/evaluate_realtime.py [--overwrite]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_prob import ALPHA, GROUP, TRACKS, conformal_halfwidths, nest, track_points
from evalcore import load_history, origin_scales
from prob_samples import crps_fair
from runlog import RunLog, out_path

RT_BASE = {"FM": "TimesFM25", "STAT": "AutoETS"}
ORIGINS = {"2025-06": 15, "2026-06": 3}
TOP = 10


def period_label(o, target):
    if o == "2026-06":
        return "2026-07..09"
    if target <= "2025-12":
        return "2025-07..12" + (" (2025-09)" if target == "2025-09" else "")
    return "2026-01..06" if target <= "2026-06" else "2026-07..09"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean", default="data/clean")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/eval_realtime")
    ap.add_argument("--d-form", default="plus", choices=["plus", "exact"],
                    help="plus: x (1 + d); exact: / (1 - d)")
    ap.add_argument("--nest", action="store_true", help="clip 80%% ACI intervals inside the 95%% ones")
    ap.add_argument("--prob-dir", default=None, help="real-time samples (default results/prob_realtime)")
    ap.add_argument("--main-prob-dir", default=None, help="main-run CRPS for the 2025-06 check (default results/prob_main)")
    ap.add_argument("--aci-name", default="aci_intervals.parquet", help="ACI interval file in --prob-dir")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    log = RunLog("evaluate_realtime.py", vars(args))
    RES, OUT = Path(args.results), Path(args.out)

    def save(df, name):
        df.to_csv(out_path(OUT / name, args.overwrite), index=False, float_format="%.6g")

    meta = pd.read_parquet(RES / "base_forecasts_v2" / "series_meta.parquet")
    Y = load_history(args.clean)
    y_long = Y.stack().rename("y").reset_index().rename(columns={"period": "target", "uid": "unique_id"})
    scales = origin_scales(Y)[["unique_id", "origin", "scale1"]]
    hist = Y.loc[:, [c for c in Y.columns if not c.startswith("OUT:")]]

    # published residence totals: history to 2026-06 and the real-time months
    p = pd.read_parquet(Path(args.clean) / "panel_system_v2.parquet")
    pub = p[p.measure == "RES_TOTAL_PUB"].pivot(index="period", columns="unit", values="co2_t_per_day")
    rt = pd.read_parquet(Path(args.clean) / "realtime_res_total_pub_2026-07_2026-09_v2.parquet")
    rt = rt[rt.unit_type != "outside"].pivot(index="period", columns="unit", values="co2_t_per_day")
    pub = pd.concat([pub, rt[pub.columns.intersection(rt.columns)]]).sort_index()
    sysT = p[p.measure == "RES_TOTAL_SYS"].pivot(index="period", columns="unit", values="co2_t_per_day")
    units = list(pub.columns)
    rank = hist.loc["2025-01":"2025-12", [f"{u}|RES_TOTAL_SYS" for u in units]].mean()
    eco_units = [u for u in units if meta.set_index("unique_id").at[f"{u}|RES_TOTAL_SYS", "unit_type"] == "economy"]
    top = [u for u in rank.sort_values(ascending=False).index.str.split("|").str[0] if u in eco_units][:TOP]
    wmap = meta.set_index("unique_id").weight_2022

    checks, sys_rows, pub_rows = [], [], []
    for tr in TRACKS:
        P = (Path(args.prob_dir) if args.prob_dir else RES / "prob_realtime") / f"track={tr}"
        rec = pd.read_parquet(RES / "reconciled_rt" / f"base={RT_BASE[tr]}" / "reconciled.parquet",
                              filters=[("method", "==", "MinT"), ("variant", "==", "nn_qp")])
        rec = rec[rec.unique_id != "W|DELTA_INT"]
        base = pd.read_parquet(RES / "realtime" / "base_forecasts" / f"model={RT_BASE[tr]}" / "forecasts.parquet",
                               columns=["unique_id", "origin", "target", "horizon", "mean"])
        base = base[base.unique_id.isin(rec.unique_id.unique())]
        aci = pd.read_parquet(P / args.aci_name)
        aci = aci[aci.gamma == 0.05]
        if args.nest:
            aci = nest(aci, ["unique_id", "origin", "horizon"])
        main_pts = track_points(RES, tr)
        hw = conformal_halfwidths(main_pts, y_long, scales, [(o, h) for o, H in ORIGINS.items() for h in range(1, H + 1)])

        # ---- checks against the main run (origin 2025-06, h <= 12)
        m = rec[rec.origin == "2025-06"].merge(main_pts[main_pts.origin == "2025-06"],
                                               on=["unique_id", "origin", "target", "horizon"])
        c_rt = pd.read_parquet(P / "crps.parquet")
        c_main = pd.read_parquet((Path(args.main_prob_dir) if args.main_prob_dir else RES / "prob_main") / f"track={tr}" /
                                 "crps.parquet", filters=[("origin", "==", "2025-06")])
        mc = c_rt.merge(c_main, on=["unique_id", "origin", "horizon", "kind"])
        checks.append({"track": tr, "point_rows": len(m), "point_max_abs_diff": float((m.value_x - m.value_y).abs().max()),
                       "crps_rows": len(mc), "crps_max_abs_diff": float((mc.crps_x - mc.crps_y).abs().max())})

        # ---- A: system series, origin 2025-06, targets to 2026-06
        d = rec[rec.origin == "2025-06"].rename(columns={"value": "point"}) \
            .merge(base.rename(columns={"mean": "base_point"}), on=["unique_id", "origin", "target", "horizon"]) \
            .merge(y_long, on=["unique_id", "target"]).merge(scales, on=["unique_id", "origin"])
        d = d.merge(c_rt.pivot_table(index=["unique_id", "origin", "horizon"], columns="kind", values="crps")
                    .reset_index().rename(columns={"base": "crps_base", "rec": "crps_rec"}),
                    on=["unique_id", "origin", "horizon"])
        d["group"] = d.unique_id.str.split("|").str[1].map(GROUP).fillna("components")
        for lv in ALPHA:
            a = aci[aci.level == lv][["unique_id", "origin", "horizon", "lo", "hi"]]
            d = d.merge(a.rename(columns={"lo": f"aci_lo{lv}", "hi": f"aci_hi{lv}"}), on=["unique_id", "origin", "horizon"])
            q = hw[hw.level == lv][["origin", "horizon", "group", "q"]].rename(columns={"q": f"cq{lv}"})
            d = d.merge(q, on=["origin", "horizon", "group"])
            d[f"cov{lv}_aci"] = ((d.y >= d[f"aci_lo{lv}"].clip(lower=0)) & (d.y <= d[f"aci_hi{lv}"])).astype(float)
            half = d[f"cq{lv}"] * d.scale1
            d[f"cov{lv}_conformal"] = ((d.y >= (d.point - half).clip(lower=0)) & (d.y <= d.point + half)).astype(float)
        d["MASE"] = (d.y - d.point).abs() / d.scale1
        d["MASE_base"] = (d.y - d.base_point).abs() / d.scale1
        d["sCRPS"], d["sCRPS_base"] = d.crps_rec / d.scale1, d.crps_base / d.scale1
        d["period"] = [period_label("2025-06", t) for t in d.target]
        sys_rows.append(d.assign(track=tr))

        # ---- B: RES_TOTAL_PUB, both origins
        smp = pd.read_parquet(P / "samples_res_total_sys.parquet")
        smp["unit"] = smp.unique_id.str.split("|").str[0]
        smp_g = {k: g for k, g in smp.groupby(["unit", "origin", "horizon"])}
        aci_i = aci.set_index(["unique_id", "origin", "horizon", "level"])[["lo", "hi"]]
        sc_i = scales.set_index(["unique_id", "origin"]).scale1
        hw_i = hw.set_index(["origin", "horizon", "group", "level"]).q
        for o, H in ORIGINS.items():
            win = pub.loc[:o].tail(6)
            ws = sysT.loc[win.index]
            rel = ((win - ws) / win.where(win != 0)).mean()
            dvec = rel.fillna(0.0)
            pts_o = rec[(rec.origin == o) & rec.unique_id.str.endswith("|RES_TOTAL_SYS")].copy()
            bs_o = base[(base.origin == o) & base.unique_id.str.endswith("|RES_TOTAL_SYS")]
            pts_o = pts_o.merge(bs_o.rename(columns={"mean": "base_point"}), on=["unique_id", "origin", "target", "horizon"])
            pts_o["unit"] = pts_o.unique_id.str.split("|").str[0]
            hist_pub = pub.loc[:o]
            sc_pub = (hist_pub - hist_pub.shift(12)).loc["2020-01":].abs().mean()
            for dname in ["d", "d0"]:
                for _, r in pts_o.iterrows():
                    u, h, t = r.unit, int(r.horizon), r.target
                    if u not in pub.columns or t not in pub.index or pd.isna(pub.at[t, u]):
                        continue
                    dd = dvec[u] if dname == "d" else 0.0
                    k = 1.0 + dd if args.d_form == "plus" else 1.0 / (1.0 - dd)
                    y = pub.at[t, u]
                    s = smp_g[(u, o, h)]
                    crps = {kd: float(crps_fair((g.value.to_numpy() * k)[:, None], np.array([y]))[0])
                            for kd, g in s.groupby("kind")}
                    rec_s = np.sort(s[s.kind == "rec"].value.to_numpy() * k)
                    row = {"track": tr, "origin": o, "horizon": h, "target": t, "unit": u, "d_variant": dname,
                           "d": dd, "factor": k, "y_pub": y, "point": r.value * k, "base_point": r.base_point * k,
                           "scale_pub": sc_pub[u], "crps_rec": crps.get("rec"), "crps_base": crps.get("base"),
                           "period": period_label(o, t)}
                    for lv in ALPHA:
                        a = aci_i.loc[(r.unique_id, o, h, lv)]
                        lo, hi = max(a.lo, 0) * k, a.hi * k
                        row[f"cov{lv}_aci"] = float(lo <= y <= hi)
                        cq = hw_i.loc[(o, h, "residence", lv)]
                        sc_sys = sc_i.loc[(r.unique_id, o)]
                        half = cq * sc_sys * k
                        row[f"cov{lv}_conformal"] = float(max(r.value * k - half, 0) <= y <= r.value * k + half)
                        al = ALPHA[lv]
                        row[f"cov{lv}_uncal"] = float(np.quantile(rec_s, al / 2) <= y <= np.quantile(rec_s, 1 - al / 2))
                    pub_rows.append(row)
        print(f"{tr}: done")

    save(pd.DataFrame(checks), "checks.csv")
    A = pd.concat(sys_rows, ignore_index=True).merge(meta[["unique_id", "unit", "unit_type", "measure", "weight_2022"]],
                                                     on="unique_id")
    A.to_parquet(out_path(OUT / "system_2025-06_cells.parquet", args.overwrite), index=False)
    B = pd.DataFrame(pub_rows)
    B["MASE"] = (B.y_pub - B.point).abs() / B.scale_pub
    B["MASE_base"] = (B.y_pub - B.base_point).abs() / B.scale_pub
    B["sCRPS"], B["sCRPS_base"] = B.crps_rec / B.scale_pub, B.crps_base / B.scale_pub
    B["unit_type"] = ["world" if u == "W" else "residual" if u == "RESIDUAL" else "economy" for u in B.unit]
    B["weight_2022"] = [wmap[f"{u}|RES_TOTAL_SYS"] for u in B.unit]
    save(B, "res_total_pub_cells.csv")

    metrics = ["MASE", "MASE_base", "sCRPS", "sCRPS_base", "cov80_aci", "cov95_aci", "cov80_conformal", "cov95_conformal"]

    def summarise(df, keys, extra=()):
        rows = []
        for k, g in df.groupby(keys):
            ps = g.groupby("unique_id" if "unique_id" in g else "unit").agg(
                {**{m: "mean" for m in metrics + list(extra)}, "weight_2022": "first", "unit_type": "first"})
            rec_ = dict(zip(keys, k))
            e = ps[ps.unit_type == "economy"]
            for m in metrics + list(extra):
                rows.append({**rec_, "level": "economies", "metric": m, "median": e[m].median(),
                             "mean": e[m].mean(), "weighted": np.average(e[m], weights=e.weight_2022), "n": len(e)})
        return pd.DataFrame(rows)

    # A: system economies by period; World one by one; top economies (median over their 10 series)
    A2 = A[A.target != "2025-09"].copy()
    sA = pd.concat([summarise(A2, ["track", "period"]), summarise(A, ["track"]).assign(period="2025-07..2026-06 incl. 2025-09")])
    save(sA, "summary_system_2025-06.csv")
    w = A2[A2.unit_type == "world"].groupby(["track", "period", "measure"])[metrics].mean().reset_index()
    save(w, "world_system_2025-06.csv")
    t10 = A2[A2.unit.isin(top)].groupby(["track", "period", "unit"])[metrics].median().reset_index()
    save(t10, "top10_system_2025-06.csv")
    # B: RES_TOTAL_PUB
    B_ = B.copy()
    B_["unique_id"] = B_.unit
    sB = pd.concat([summarise(B_[B_.target != "2025-09"], ["track", "origin", "d_variant", "period"], ("cov80_uncal", "cov95_uncal")),
                    summarise(B_, ["track", "origin", "d_variant"], ("cov80_uncal", "cov95_uncal")).assign(period="all incl. 2025-09")])
    save(sB, "summary_res_total_pub.csv")
    one = B_[B_.unit.isin(["W", "RESIDUAL"] + top)].groupby(["track", "origin", "d_variant", "period", "unit"])[
        metrics + ["cov80_uncal", "cov95_uncal", "d"]].mean().reset_index()
    save(one, "units_res_total_pub.csv")
    pd.Series({"top10": " ".join(top)}).to_csv(out_path(OUT / "top10_units.csv", args.overwrite))
    print(pd.DataFrame(checks).to_string(index=False))
    log.finish("real-time tests")


if __name__ == "__main__":
    main()
