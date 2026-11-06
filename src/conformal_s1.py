"""
conformal_s1.py: pooled split-conformal 80% and 95% intervals for the final FM forecasts of S1.
Nothing is refitted: the intervals come from stored point forecasts and errors.

Interval (as the conformal intervals of the real-time test at origin 2026-06, evaluate_realtime.py, T5a): for horizon h
and measure group g (residence: RES_TOTAL_SYS; territory: TER_DOM, TER_INT; components A-G), the half-width is
q(h, g) x scale, where q is the split-conformal quantile (level min(1, ceil((N + 1)(1 - alpha)) / N)) of the scaled
absolute errors |y - point| / scale of the FM track point forecasts (TimesFM 2.5 + MinT, exact QP; main-run origins
2022-12 ... 2025-06) of every system series, at horizon h, whose target month is known at the origin 2026-06
(evaluate_prob.conformal_halfwidths); scale = the series' seasonal-naive scale up to 2026-06. Bounds: point -/+ half-width,
the lower bound clipped at zero. Centre: the S1 point (results/final_exact/track=FM/forecasts.parquet). Nested by construction.
Sensitivity: the same pool plus the errors of the extended calibration origins 2025-07 ... 2026-05 (calibration only;
results/reconciled_calib, MinT exact QP), whose targets are known at 2026-06.
Width ratio to ACI: conformal width / ACI width (gamma 0.05, nested, as in S1), per series and horizon; median over the
residence totals (RES_TOTAL_SYS of the 110 system economies; World and Residual separately) and over all series.

Outputs: results/conformal_s1/ (s1_conformal.csv: one row per series and horizon, read by paper_tables.py for S1;
halfwidths.csv; width_ratio.csv)
--final-dir and --realtime-dir read the final forecasts and the real-time cells of the exact samples
(results/final_exact, results/eval_realtime_exact_nested); only the width ratios to ACI change, the
conformal bounds do not depend on the samples.

Usage:
  python src/conformal_s1.py
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from evalcore import load_history, origin_scales
from evaluate_prob import GROUP, conformal_halfwidths, track_points
from runlog import RunLog, out_path

ORIGIN = "2026-06"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean", default="data/clean")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/conformal_s1")
    ap.add_argument("--final-dir", default=None, help="folder with track=FM/forecasts.parquet (default results/final_exact)")
    ap.add_argument("--realtime-dir", default=None, help="real-time results (default results/eval_realtime_exact_ext_nested)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    log = RunLog("conformal_s1.py", vars(args))
    RES, OUT = Path(args.results), Path(args.out)

    def save(df, name):
        df.to_csv(out_path(OUT / name, args.overwrite), index=False, float_format="%.6g")

    meta = pd.read_parquet(RES / "base_forecasts_v2" / "series_meta.parquet").set_index("unique_id")
    Y = load_history(args.clean)
    y_long = Y.stack().rename("y").reset_index().rename(columns={"period": "target", "uid": "unique_id"})
    scales = origin_scales(Y)[["unique_id", "origin", "scale1"]]
    fc = pd.read_parquet((Path(args.final_dir) if args.final_dir else RES / "final_exact") / "track=FM" / "forecasts.parquet",
                         columns=["unique_id", "horizon", "target", "point", "lo95", "lo80", "hi80", "hi95"])
    H = int(fc.horizon.max())
    pts = track_points(RES, "FM")
    calib = pd.read_parquet(RES / "reconciled_calib" / "base=TimesFM25" / "reconciled.parquet",
                            filters=[("method", "==", "MinT"), ("variant", "==", "nn_qp")])
    calib = calib[calib.unique_id != "W|DELTA_INT"]
    at = [(ORIGIN, h) for h in range(1, H + 1)]
    hw = {"main": conformal_halfwidths(pts, y_long, scales, at),
          "with extended calibration origins": conformal_halfwidths(pd.concat([pts, calib], ignore_index=True),
                                                                     y_long, scales, at)}
    save(pd.concat([d.assign(pool=k) for k, d in hw.items()], ignore_index=True), "halfwidths.csv")

    sc =scales[scales.origin == ORIGIN].set_index("unique_id").scale1
    d = fc.assign(group=fc.unique_id.str.split("|").str[1].map(GROUP).fillna("components"))
    d["scale"] = d.unique_id.map(sc)
    assert d.scale.notna().all()
    out, ratios = None, []
    for pool, h_ in hw.items():
        x = d.copy()
        for lv in (80, 95):
            q = h_[h_.level == lv].set_index(["horizon", "group"]).q
            half = q.loc[list(zip(x.horizon, x.group))].to_numpy() * x.scale.to_numpy()
            x[f"lo{lv}_conformal"] = np.maximum(x.point - half, 0.0)
            x[f"hi{lv}_conformal"] = x.point + half
        if pool == "main":
            out = x
        for lv in (80, 95):
            w_aci = x[f"hi{lv}"] - x[f"lo{lv}"]
            w_cf = x[f"hi{lv}_conformal"] - x[f"lo{lv}_conformal"]
            x[f"ratio{lv}"] = np.where(w_aci > 0, w_cf / w_aci.where(w_aci > 0), np.nan)
        measure = x.unique_id.str.split("|").str[1]
        utype = x.unique_id.map(meta.unit_type)
        sets = {"residence totals, system economies (110)": (measure == "RES_TOTAL_SYS") & (utype == "economy"),
                "residence total, World": x.unique_id == "W|RES_TOTAL_SYS",
                "residence total, Residual": x.unique_id == "RESIDUAL|RES_TOTAL_SYS",
                "territory totals, system economies": measure.isin(["TER_DOM", "TER_INT"]) & (utype == "economy"),
                "components, system economies": ~measure.isin(["RES_TOTAL_SYS", "TER_DOM", "TER_INT"]) & (utype == "economy"),
                "all system series": pd.Series(True, index=x.index)}
        for name, msk in sets.items():
            for hg in ["all"] + [1, 3, 6, 12]:
                xm = x[msk & ((x.horizon == hg) if hg != "all" else True)]
                ratios.append({"pool": pool, "set": name, "horizon": hg, "n_cells": len(xm),
                               "median_ratio80": xm.ratio80.median(), "median_ratio95": xm.ratio95.median(),
                               "share_conformal_wider80": float((xm.ratio80 > 1).mean()),
                               "share_conformal_wider95": float((xm.ratio95 > 1).mean()),
                               "n_aci_width_zero": int(xm.ratio95.isna().sum())})
    # check: with the stored factor 1 / (1 - d), the main-pool intervals reproduce the stored conformal coverage of every
    # FM real-time cell at origin 2026-06 (RES_TOTAL_PUB, h 1-3; evaluate_realtime.py --d-form exact --nest)
    rt = pd.read_csv((Path(args.realtime_dir) if args.realtime_dir else RES / "eval_realtime_exact_ext_nested") / "res_total_pub_cells.csv")
    rt = rt[(rt.track == "FM") & (rt.origin == ORIGIN)]
    m = rt.merge(out.assign(unit=out.unique_id.str.split("|").str[0])[out.unique_id.str.endswith("|RES_TOTAL_SYS")],
                 on=["unit", "horizon"])
    assert len(m) == len(rt) > 0 and np.allclose(m.point_x, m.point_y * m.factor, rtol=1e-5)  # stored to 6 digits
    for lv in (80, 95):
        hit = ((m.y_pub >= m[f"lo{lv}_conformal"] * m.factor) & (m.y_pub <= m[f"hi{lv}_conformal"] * m.factor)).astype(float)
        assert (hit == m[f"cov{lv}_conformal"]).all(), lv
    print(f"check: stored real-time conformal coverage reproduced ({len(m)} cells)")
    cols =["unique_id", "horizon", "target", "lo95_conformal", "lo80_conformal", "hi80_conformal", "hi95_conformal"]
    save(out[cols], "s1_conformal.csv")
    rat = pd.DataFrame(ratios)
    save(rat, "width_ratio.csv")
    pd.set_option("display.width", 220)
    print(hw["main"].pivot_table(index="horizon", columns=["group", "level"], values="q").round(3).to_string())
    print(rat[rat.horizon.astype(str) == "all"].to_string(index=False))
    log.finish(f"S1 conformal bounds: {len(out)} rows")


if __name__ == "__main__":
    main()
