"""
realtime_base.py: AutoETS base forecasts and in-sample residuals for the real-time origins.

Origins:
  2025-06  horizons 1-15 (targets 2025-07 ... 2026-09)
  2026-06  horizons 1-3  (targets 2026-07 ... 2026-09, months never used in any estimation)
Series: the 1,120 reconciled system series (DELTA_INT excluded). AutoETS (season length 12) is fitted
on 2019-01 ... origin with prediction intervals at 50, 80 and 95%, and its in-sample one-step residuals
are kept for the MinT W (12-month burn-in applied in reconcile.py).

statsforecast (2.1.1) simulates the intervals of ETS classes 4 and 5 (5,000 paths, fixed seed), so they
depend on the requested horizon length. As in the rolling design (base_forecasts.py), horizons 1-12 are taken
from a call with h = 12 (for 2026-06 the first 3 of them); horizons 13-15 of origin 2025-06 come from a call
with h = 15. Point forecasts do not depend on h.

Checks: at 2025-06, horizons 1-12 must equal the rolling-origin forecasts (base_forecasts.py) and the
residuals must equal results/insample_residuals_v2 (same data, same model).

Outputs (results/realtime/):
  base_forecasts/model=AutoETS/forecasts.parquet   layout of base_forecasts_v2 (y empty after 2026-06)
  insample_residuals/model=AutoETS/residuals.parquet  unique_id, origin, period, resid

Usage:
  python src/realtime_base.py
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoETS

from runlog import RunLog, out_path

ORIGINS = {"2025-06": 15, "2026-06": 3}
LEVELS = [50, 80, 95]
REL_TOL = 1e-8


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean", default="data/clean")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/realtime")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    log = RunLog("realtime_base.py", vars(args))
    RES, OUT = Path(args.results), Path(args.out)

    p = pd.read_parquet(Path(args.clean) / "panel_system_v2.parquet")
    p = p[p.in_constraints & (p.measure != "DELTA_INT")]
    df = pd.DataFrame({"unique_id": p.unit + "|" + p.measure, "ds": pd.to_datetime(p.period, format="%Y-%m"),
                       "y": p.co2_t_per_day}).sort_values(["unique_id", "ds"]).reset_index(drop=True)
    meta = p.assign(unique_id=p.unit + "|" + p.measure).drop_duplicates("unique_id")[
        ["unique_id", "unit", "measure", "unit_type"]]
    actual = df.assign(target=df.ds.dt.strftime("%Y-%m"))[["unique_id", "target", "y"]]

    fc, rs = [], []
    for o, h in ORIGINS.items():
        t0 = time.perf_counter()
        train = df[df.ds <= pd.Timestamp(o + "-01")]
        parts = []
        for hh in sorted({12, max(h, 12)}):
            sf = StatsForecast(models=[AutoETS(season_length=12)], freq="MS", n_jobs=args.n_jobs)
            f = sf.forecast(df=train, h=hh, level=LEVELS, fitted=True)
            f = f.reset_index() if "unique_id" not in f.columns else f
            if hh == 12:
                fv = sf.forecast_fitted_values()
                fv = fv.reset_index() if "unique_id" not in fv.columns else fv
            d = pd.DataFrame({"unique_id": f.unique_id, "origin": o, "target": f.ds.dt.strftime("%Y-%m"),
                              "mean": f["AutoETS"]})
            d["horizon"] = d.groupby("unique_id").cumcount() + 1
            for lv in LEVELS:
                d[f"lo_{lv}"], d[f"hi_{lv}"] = f[f"AutoETS-lo-{lv}"].to_numpy(), f[f"AutoETS-hi-{lv}"].to_numpy()
            parts.append(d[(d.horizon <= 12) & (d.horizon <= h)] if hh == 12 else d[d.horizon > 12])
        out = pd.concat(parts, ignore_index=True)
        out = out.merge(meta, on="unique_id").merge(actual, on=["unique_id", "target"], how="left")
        fc.append(out)
        rs.append(pd.DataFrame({"unique_id": fv.unique_id, "origin": o, "period": fv.ds.dt.strftime("%Y-%m"),
                                "resid": (fv["y"] - fv["AutoETS"]).to_numpy(float)}))
        print(f"origin {o}: {len(out)} forecasts, {train.ds.nunique()} months of data, {time.perf_counter() - t0:.0f} s")
    fc = pd.concat(fc, ignore_index=True)[["unique_id", "unit", "measure", "unit_type", "origin", "target", "horizon",
                                           "y", "mean"] + [f"{s}_{lv}" for lv in LEVELS for s in ("lo", "hi")]]
    rs = pd.concat(rs, ignore_index=True)

    # ---- checks against the rolling-origin forecasts and residuals at origin 2025-06
    old = pd.read_parquet(RES / "base_forecasts_v2" / "model=AutoETS" / "forecasts.parquet",
                          filters=[("origin", "==", "2025-06")])
    m = fc[fc.origin == "2025-06"].merge(old, on=["unique_id", "origin", "target"], suffixes=("", "_old"))
    rel_f = max(((m[c] - m[c + "_old"]).abs() / m[c + "_old"].abs().clip(lower=1.0)).max()
                for c in ["mean", "lo_95", "hi_95", "lo_50"])
    oldr = pd.read_parquet(RES / "insample_residuals_v2" / "model=AutoETS" / "residuals.parquet",
                           filters=[("origin", "==", "2025-06")])
    mr = rs[rs.origin == "2025-06"].merge(oldr, on=["unique_id", "origin", "period"], suffixes=("", "_old"))
    abs_r = float((mr.resid - mr.resid_old).abs().max())
    print(f"check 2025-06: {len(m)} forecasts, max relative difference to base_forecasts.py {rel_f:.2e}; "
          f"{len(mr)} residuals, max abs difference to insample_residuals.py {abs_r:.2e}")
    if rel_f > REL_TOL:
        raise SystemExit("real-time refit differs from the rolling-origin forecasts: stop")

    fc.to_parquet(out_path(OUT / "base_forecasts" / "model=AutoETS" / "forecasts.parquet", args.overwrite), index=False)
    rs.to_parquet(out_path(OUT / "insample_residuals" / "model=AutoETS" / "residuals.parquet", args.overwrite), index=False)
    log.finish(f"check: forecasts rel {rel_f:.1e}, residuals abs {abs_r:.1e}")


if __name__ == "__main__":
    main()
