"""
calib_base.py: AutoETS base forecasts for the extended calibration origins and the final origin.

Origins: monthly 2025-07 ... 2026-05 (calibration only: used to update the ACI state, never for evaluation)
and 2026-06 (final probabilistic forecasts). Series: the 1,120 reconciled system series (DELTA_INT excluded; it is
forecast as a ratio to the reconciled G(World)). AutoETS (season length 12) is fitted on 2019-01 ... origin (data
after 2026-06 are never used) with a call with h = 12 and intervals at 50, 80 and 95% (the convention of
base_forecasts.py and realtime_base.py: the simulated ETS intervals depend on the requested horizon length).
In-sample one-step residuals are kept for the MinT W (12-month burn-in in reconcile.py).

Checks: at 2026-06, the point forecasts and intervals of h 1-3 equal the real-time forecasts of realtime_base.py
(also an h = 12 call) and the residuals equal results/realtime/insample_residuals (stop otherwise).

Outputs (results/calib/):
  base_forecasts/model=AutoETS/forecasts.parquet      layout of results/realtime/base_forecasts
  insample_residuals/model=AutoETS/residuals.parquet  unique_id, origin, period, resid

Usage:
  python src/calib_base.py
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoETS

from runlog import RunLog, out_path

ORIGINS = pd.period_range("2025-07", "2026-06", freq="M").astype(str).tolist()
H = 12
LEVELS = [50, 80, 95]
REL_TOL = 1e-8


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean", default="data/clean")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/calib")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    log = RunLog("calib_base.py", vars(args))
    RES, OUT = Path(args.results), Path(args.out)

    p = pd.read_parquet(Path(args.clean) / "panel_system_v2.parquet")
    p = p[p.in_constraints & (p.measure != "DELTA_INT")]
    df = pd.DataFrame({"unique_id": p.unit + "|" + p.measure, "ds": pd.to_datetime(p.period, format="%Y-%m"),
                       "y": p.co2_t_per_day}).sort_values(["unique_id", "ds"]).reset_index(drop=True)
    meta = p.assign(unique_id=p.unit + "|" + p.measure).drop_duplicates("unique_id")[
        ["unique_id", "unit", "measure", "unit_type"]]
    actual = df.assign(target=df.ds.dt.strftime("%Y-%m"))[["unique_id", "target", "y"]]
    actual = actual[actual.target <= "2026-06"]

    fc, rs, t_all = [], [], time.perf_counter()
    for o in ORIGINS:
        t0 = time.perf_counter()
        train = df[df.ds <= pd.Timestamp(o + "-01")]
        assert train.ds.max() == pd.Timestamp(o + "-01")
        sf = StatsForecast(models=[AutoETS(season_length=12)], freq="MS", n_jobs=args.n_jobs)
        f = sf.forecast(df=train, h=H, level=LEVELS, fitted=True)
        fv = sf.forecast_fitted_values()
        f = f.reset_index() if "unique_id" not in f.columns else f
        fv = fv.reset_index() if "unique_id" not in fv.columns else fv
        d = pd.DataFrame({"unique_id": f.unique_id, "origin": o, "target": f.ds.dt.strftime("%Y-%m"), "mean": f["AutoETS"]})
        d["horizon"] = d.groupby("unique_id").cumcount() + 1
        for lv in LEVELS:
            d[f"lo_{lv}"], d[f"hi_{lv}"] = f[f"AutoETS-lo-{lv}"].to_numpy(), f[f"AutoETS-hi-{lv}"].to_numpy()
        fc.append(d.merge(meta, on="unique_id").merge(actual, on=["unique_id", "target"], how="left"))
        rs.append(pd.DataFrame({"unique_id": fv.unique_id, "origin": o, "period": fv.ds.dt.strftime("%Y-%m"),
                                "resid": (fv["y"] - fv["AutoETS"]).to_numpy(float)}))
        print(f"origin {o}: {len(d)} forecasts, {time.perf_counter() - t0:.0f} s", flush=True)
    fc = pd.concat(fc, ignore_index=True)[["unique_id", "unit", "measure", "unit_type", "origin", "target", "horizon",
                                           "y", "mean"] + [f"{s}_{lv}" for lv in LEVELS for s in ("lo", "hi")]]
    rs = pd.concat(rs, ignore_index=True)

    # ---- checks against the real-time run at 2026-06 (realtime_base.py)
    old = pd.read_parquet(RES / "realtime" / "base_forecasts" / "model=AutoETS" / "forecasts.parquet",
                          filters=[("origin", "==", "2026-06")])
    m = fc[fc.origin == "2026-06"].merge(old, on=["unique_id", "origin", "target"], suffixes=("", "_old"))
    rel = max(float(((m[c] - m[c + "_old"]).abs() / m[c + "_old"].abs().clip(lower=1.0)).max())
              for c in ["mean", "lo_95", "hi_95", "lo_80", "hi_80", "lo_50", "hi_50"])
    oldr = pd.read_parquet(RES / "realtime" / "insample_residuals" / "model=AutoETS" / "residuals.parquet",
                           filters=[("origin", "==", "2026-06")])
    mr = rs[rs.origin == "2026-06"].merge(oldr, on=["unique_id", "origin", "period"], suffixes=("", "_old"))
    absr = float((mr.resid - mr.resid_old).abs().max())
    print(f"{len(fc)} forecasts in {(time.perf_counter() - t_all) / 60:.1f} min | check 2026-06 h1-3: {len(m)} rows, "
          f"max rel diff {rel:.2e} (mean and all intervals); {len(mr)} residuals, max abs diff {absr:.2e}")
    if rel > REL_TOL or absr > 1e-6:
        raise SystemExit("2026-06 differs from the real-time run: stop")
    fc.to_parquet(out_path(OUT / "base_forecasts" / "model=AutoETS" / "forecasts.parquet", args.overwrite), index=False)
    rs.to_parquet(out_path(OUT / "insample_residuals" / "model=AutoETS" / "residuals.parquet", args.overwrite), index=False)
    log.finish(f"{len(ORIGINS)} origins; check rel {rel:.1e}, residuals {absr:.1e}")


if __name__ == "__main__":
    main()
