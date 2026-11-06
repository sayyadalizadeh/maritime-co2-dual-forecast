"""
fm_postrelease.py: post-release evaluation of the foundation models.

Only target months after a model's release are scored: Chronos-Bolt from 2024-12, TimesFM 2.5 from 2025-10,
TimesFM 3.0 from 2026-09. Stored forecasts are rescored; nothing is refitted.

Sets
  rolling-main    origins 2022-12 ... 2025-06, h 1-12, main window (targets <= 2025-12 without 2025-09), targets >= release
  rolling-stress  the same origins, targets 2026-01 ... 2026-06 (stress window after the 2026 reclassification), targets >= release
  realtime-sys    origin 2025-06, system series, targets 2025-07 ... 2026-06 >= release (results/eval_realtime_exact_ext_nested/
                  system_2025-06_cells.parquet; base = the track's base model, rec = the track)
  realtime-pub    origins 2025-06 (h <= 15) and 2026-06 (h 1-3), published residence totals RES_TOTAL_PUB of the system
                  economies, scored with the RES_TOTAL_SYS forecast divided by (1 - d) and with d = 0
                  (results/eval_realtime_exact_ext_nested/res_total_pub_cells.csv), targets >= release
Compared on exactly the same series, origins and target months (inner join of the cells): the foundation model's
base forecast, ETS (AutoETS base), the STAT track (AutoETS + MinT, exact QP) and, for TimesFM 2.5, the FM track
(TimesFM 2.5 + MinT, exact QP). Real-time forecasts are stored only for TimesFM 2.5 and AutoETS, so Chronos-Bolt and
TimesFM 3.0 have no real-time rows; TimesFM 3.0 has no stored forecast for any target from 2026-09.
Metrics: MASE per series (mean scaled absolute error, seasonal scale m = 12 up to the origin), median and weighted
(2022 weights) over system economies; scaled CRPS (CRPS / scale1) where samples exist (base and reconciled
samples of the two tracks; the base samples of the FM track are TimesFM 2.5's, of the STAT track AutoETS's);
the World residence total (W|RES_TOTAL_SYS) MASE one by one; numbers of target months, origins, series and cells.

Output: results/fm_postrelease/postrelease_scores.csv, coverage.csv
--prob-dir and --realtime-dir read the CRPS of the exact evaluation samples and the real-time cells of the
exact samples (results/prob_main_exact, results/eval_realtime_exact_nested); MASE does not change.

Usage:
  python src/fm_postrelease.py
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from evalcore import WINDOWS, load_history, origin_scales
from runlog import RunLog, out_path

RELEASE = {"ChronosBolt": "2024-12", "TimesFM25": "2025-10", "TimesFM": "2026-09"}
LABEL = {"ChronosBolt": "Chronos-Bolt", "TimesFM25": "TimesFM 2.5", "TimesFM": "TimesFM 3.0"}
ORIGINS = pd.period_range("2022-12", "2025-06", freq="M").astype(str).tolist()
CELL = ["unique_id", "origin", "target"]


def summarise(d, col, meta):
    """Per-series mean of `col`, then median and weighted mean over system economies; World residence total."""
    s = d.groupby("unique_id")[col].mean()
    e = s[s.index.isin(meta.index[meta.unit_type == "economy"])].dropna()
    w = meta.loc[e.index, "weight_2022"]
    wr = s.get("W|RES_TOTAL_SYS", np.nan)
    return {"median": float(e.median()), "weighted": float(np.average(e, weights=w)), "world_res_total": float(wr),
            "n_series": int(len(e))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean", default="data/clean")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/fm_postrelease")
    ap.add_argument("--prob-dir", default=None, help="evaluation samples' CRPS (default results/prob_main_exact)")
    ap.add_argument("--realtime-dir", default=None, help="real-time cells (default results/eval_realtime_exact_ext_nested)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    log = RunLog("fm_postrelease.py", vars(args))
    RES, OUT = Path(args.results), Path(args.out)
    meta = pd.read_parquet(RES / "base_forecasts_v2" / "series_meta.parquet").set_index("unique_id")
    Y = load_history(args.clean)
    sc = origin_scales(Y)[["unique_id", "origin", "scale1"]]
    y_long = Y.stack().rename("y_obs").reset_index()
    y_long.columns = ["target", "unique_id", "y_obs"]
    uids = set(meta.index[meta.unit_type != "outside"])

    def point(path, filters=None, col="mean"):
        f = pd.read_parquet(path, columns=CELL + [col], filters=filters).rename(columns={col: "pred"})
        return f[f.unique_id.isin(uids) & f.origin.isin(ORIGINS)]

    rec_flt = [("method", "==", "MinT"), ("variant", "==", "nn_qp")]
    rows, cov = [], []
    stat_rec = point(RES / "reconciled_v2_burnin12" / "base=AutoETS" / "reconciled.parquet", rec_flt, "value")
    ets = point(RES / "base_forecasts_v2" / "model=AutoETS" / "forecasts.parquet")
    PD = Path(args.prob_dir) if args.prob_dir else RES / "prob_main_exact"
    RT = Path(args.realtime_dir) if args.realtime_dir else RES / "eval_realtime_exact_ext_nested"
    crps = {tr: pd.read_parquet(PD / f"track={tr}" / "crps.parquet") for tr in ("FM", "STAT")}

    # ---------------------------------------------------------------- rolling origins
    for m, rel in RELEASE.items():
        cand = {LABEL[m]: point(RES / "base_forecasts_fm" / f"model={m}" / "forecasts.parquet"), "ETS": ets,
                "STAT (ETS + MinT)": stat_rec}
        if m == "TimesFM25":
            cand["FM (TimesFM 2.5 + MinT)"] = point(RES / "reconciled_v2_fmownw" / "base=TimesFM25" / "reconciled.parquet",
                                                     rec_flt, "value")
        for sname, win in [("rolling-main", WINDOWS["main"]), ("rolling-stress", WINDOWS["robB"])]:
            keys = None
            for f in cand.values():
                k = f.loc[win(f.target) & (f.target >= rel), CELL]
                keys = k if keys is None else keys.merge(k, on=CELL)
            keys = keys.merge(y_long, on=["unique_id", "target"]).dropna(subset=["y_obs"])
            cov.append({"model": LABEL[m], "release_from": rel, "set": sname, "n_cells": len(keys),
                        "target_months": " ".join(sorted(keys.target.unique())) if len(keys) else "",
                        "n_target_months": keys.target.nunique(), "n_origins": keys.origin.nunique(),
                        "n_series": keys.unique_id.nunique()})
            if keys.empty:
                continue
            for name, f in cand.items():
                d = keys.merge(f, on=CELL).merge(sc, on=["unique_id", "origin"])
                d["sae"] = (d.y_obs - d.pred).abs() / d.scale1
                d.loc[~np.isfinite(d.sae), "sae"] = np.nan
                r = summarise(d, "sae", meta)
                rows.append({"model": LABEL[m], "set": sname, "forecast": name, "metric": "MASE", **r,
                             "n_cells": len(d), "n_target_months": d.target.nunique()})
            for tr, kind, name in [("FM", "base", "TimesFM 2.5 (base samples)"), ("FM", "rec", "FM (TimesFM 2.5 + MinT)"),
                                   ("STAT", "base", "ETS (base samples)"), ("STAT", "rec", "STAT (ETS + MinT)")]:
                if tr == "FM" and m != "TimesFM25":
                    continue
                c = crps[tr][crps[tr].kind == kind][CELL + ["crps"]]
                d = keys.merge(c, on=CELL).merge(sc, on=["unique_id", "origin"])
                d["scrps"] = d.crps / d.scale1
                d.loc[~np.isfinite(d.scrps), "scrps"] = np.nan
                r = summarise(d, "scrps", meta)
                rows.append({"model": LABEL[m], "set": sname, "forecast": name, "metric": "scaled CRPS", **r,
                             "n_cells": len(d), "n_target_months": d.target.nunique()})

    # ---------------------------------------------------------------- real-time origins (stored for TimesFM 2.5 only)
    rel = RELEASE["TimesFM25"]
    sysc = pd.read_parquet(RT / "system_2025-06_cells.parquet")
    sysc = sysc[(sysc.target >= rel) & (sysc.method == "MinT") & (sysc.variant == "nn_qp")]
    pub = pd.read_csv(RT / "res_total_pub_cells.csv")
    pub = pub[(pub.target >= rel) & (pub.unit_type == "economy")]
    for sname, d0, lab in [("realtime-sys (origin 2025-06)", sysc, None)] + \
            [(f"realtime-pub (origins 2025-06, 2026-06; d = {'(1-d) correction' if dv == 'd' else '0'})",
              pub[pub.d_variant == dv], dv) for dv in ("d", "d0")]:
        keys = None
        for tr in ("FM", "STAT"):
            k = d0[d0.track == tr][["unit" if lab else "unique_id", "origin", "target"]]
            keys = k if keys is None else keys.merge(k)
        idc = "unit" if lab else "unique_id"
        cov.append({"model": "TimesFM 2.5", "release_from": rel, "set": sname, "n_cells": len(keys),
                    "target_months": " ".join(sorted(keys.target.unique())), "n_target_months": keys.target.nunique(),
                    "n_origins": keys.origin.nunique(), "n_series": keys[idc].nunique()})
        for tr, col, name in [("FM", "MASE_base", "TimesFM 2.5"), ("STAT", "MASE_base", "ETS"),
                              ("STAT", "MASE", "STAT (ETS + MinT)"), ("FM", "MASE", "FM (TimesFM 2.5 + MinT)"),
                              ("FM", "sCRPS_base", "TimesFM 2.5 (base samples)"), ("STAT", "sCRPS_base", "ETS (base samples)"),
                              ("STAT", "sCRPS", "STAT (ETS + MinT)"), ("FM", "sCRPS", "FM (TimesFM 2.5 + MinT)")]:
            d = keys.merge(d0[d0.track == tr], on=[idc, "origin", "target"])
            if lab:                                    # one series per economy: RES_TOTAL_PUB
                d = d.assign(unique_id=d.unit + "|RES_TOTAL_SYS")
            r = summarise(d, col, meta)
            if lab:
                r["world_res_total"] = np.nan
            rows.append({"model": "TimesFM 2.5", "set": sname, "forecast": name,
                         "metric": "MASE" if col.startswith("MASE") else "scaled CRPS", **r, "n_cells": len(d),
                         "n_target_months": d.target.nunique()})
    for m in ("ChronosBolt", "TimesFM"):
        cov.append({"model": LABEL[m], "release_from": RELEASE[m], "set": "realtime", "n_cells": 0,
                    "target_months": "no stored real-time forecasts", "n_target_months": 0, "n_origins": 0, "n_series": 0})
    out = pd.DataFrame(rows)
    out.to_csv(out_path(OUT / "postrelease_scores.csv", args.overwrite), index=False, float_format="%.6g")
    cv = pd.DataFrame(cov)
    cv.to_csv(out_path(OUT / "coverage.csv", args.overwrite), index=False)
    pd.set_option("display.width", 220)
    print(cv.drop(columns="target_months").to_string(index=False))
    print(out[["model", "set", "forecast", "metric", "median", "weighted", "world_res_total", "n_cells"]].to_string(index=False))
    log.finish("post-release evaluation")


if __name__ == "__main__":
    main()
