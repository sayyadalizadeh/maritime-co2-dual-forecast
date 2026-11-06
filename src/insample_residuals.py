"""
insample_residuals.py: in-sample one-step residuals of the base models at every origin.

The rolling-origin fits of base_forecasts.py are repeated for the system series (all v2 constraint-system
series except DELTA_INT, which is not reconciled: it is forecast as a ratio to the reconciled G(World)) with
fitted values switched on: cross_validation(h=12, step_size=1, n_windows=31, refit=True, fitted=True).
For every origin, the residual y - fitted is stored for every training month. The forecasts of
the refit are compared with the stored base_forecasts.py forecasts; the script stops if they differ.

Models: AutoETS, AutoTheta, AutoARIMA, SeasonalNaive (season length 12).

Output (over 10 MB, not in git): results/insample_residuals_v2/model=<name>/residuals.parquet with
unique_id, origin, period, resid. One file per model, so an interrupted run resumes.

Usage:
  python src/insample_residuals.py --estimate      # time one origin per model, write nothing
  python src/insample_residuals.py [--models AutoETS ...] [--skip-if-over-min 60]
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, AutoTheta, SeasonalNaive

from runlog import RunLog

SEASON, H, STEP, N_WINDOWS = 12, 12, 1, 31
MODELS = {
    "AutoETS": lambda: AutoETS(season_length=SEASON),
    "AutoTheta": lambda: AutoTheta(season_length=SEASON),
    "AutoARIMA": lambda: AutoARIMA(season_length=SEASON),
    "SeasonalNaive": lambda: SeasonalNaive(season_length=SEASON),
}
REL_TOL = 1e-8          # refit forecasts must equal the base_forecasts.py forecasts to this relative tolerance


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean", default="data/clean")
    ap.add_argument("--suffix", default="v2")
    ap.add_argument("--base", default="results/base_forecasts_v2", help="rolling-origin base forecasts (for the check)")
    ap.add_argument("--out", default="results/insample_residuals_v2")
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--skip-if-over-min", type=float, default=60.0,
                    help="with --estimate: report models whose estimated time exceeds this (minutes)")
    args = ap.parse_args()
    log = RunLog("insample_residuals.py", vars(args))
    OUT = Path(args.out)

    p = pd.read_parquet(Path(args.clean) / f"panel_system_{args.suffix}.parquet")
    p = p[p.in_constraints & (p.measure != "DELTA_INT")]
    df = pd.DataFrame({"unique_id": p.unit + "|" + p.measure, "ds": pd.to_datetime(p.period, format="%Y-%m"),
                       "y": p.co2_t_per_day}).sort_values(["unique_id", "ds"]).reset_index(drop=True)
    print(f"series: {df.unique_id.nunique()} | months: {df.ds.nunique()}")

    def run(name, n_windows):
        sf = StatsForecast(models=[MODELS[name]()], freq="MS", n_jobs=args.n_jobs)
        cv = sf.cross_validation(h=H, df=df, n_windows=n_windows, step_size=STEP, refit=True, fitted=True)
        fv = sf.cross_validation_fitted_values()
        return cv, fv

    if args.estimate:
        for name in args.models:
            t0 = time.perf_counter()
            run(name, 1)
            est = (time.perf_counter() - t0) * N_WINDOWS / 60
            flag = "  > limit: skip" if est > args.skip_if_over_min else ""
            print(f"{name:14s} estimated {est:6.1f} min for {N_WINDOWS} origins{flag}")
        log.finish("estimate only")
        return

    for name in args.models:
        path = OUT / f"model={name}" / "residuals.parquet"
        if path.exists():
            print(f"{name}: exists, skipped")
            continue
        t0 = time.perf_counter()
        cv, fv = run(name, N_WINDOWS)
        secs = time.perf_counter() - t0

        # Check: refit forecasts equal the base_forecasts.py forecasts (all series, origins and horizons).
        cv = cv.assign(origin=cv.cutoff.dt.strftime("%Y-%m"), target=cv.ds.dt.strftime("%Y-%m"))
        old = pd.read_parquet(Path(args.base) / f"model={name}" / "forecasts.parquet",
                              columns=["unique_id", "origin", "target", "mean"])
        cmp = cv.merge(old, on=["unique_id", "origin", "target"], how="left")
        if cmp["mean"].isna().any():
            raise SystemExit(f"{name}: refit rows without a stored base forecast")
        rel = (cmp[name] - cmp["mean"]).abs() / cmp["mean"].abs().clip(lower=1.0)
        print(f"{name}: compared {len(cmp)} forecasts | max relative difference {rel.max():.3e}")
        if rel.max() > REL_TOL:
            raise SystemExit(f"{name}: refit forecasts differ from base_forecasts.py (max rel {rel.max():.3e}): stop")

        res = pd.DataFrame({"unique_id": fv.unique_id.values, "origin": fv.cutoff.dt.strftime("%Y-%m").values,
                            "period": fv.ds.dt.strftime("%Y-%m").values,
                            "resid": (fv["y"] - fv[name]).to_numpy(float)})
        path.parent.mkdir(parents=True, exist_ok=True)
        res.to_parquet(path, index=False)
        print(f"{name}: {len(res)} residual rows ({int(res.resid.isna().sum())} NaN) in {secs / 60:.1f} min")
        log.finish(f"{name}: {secs:.0f} s, max rel diff to the stored residuals {rel.max():.2e}")
        log = RunLog("insample_residuals.py", vars(args))    # one log row per model


if __name__ == "__main__":
    main()
