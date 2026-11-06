"""
base_forecasts.py: rolling-origin base forecasts of the statistical models and LightGBM.

Series: every series of the v2 constraint system (system economies, Residual, World and
DELTA_INT) and every outside-system series (panel_outside_v2). The modelling variable is
the value per day (co2_t_per_day).

Protocol: expanding window from 2019-01; origins monthly from 2022-12
to 2025-06 (31 origins); horizons 1 to 12; models refitted at every origin. With data to
2026-06, h=12, step_size=1 and n_windows=31 give exactly these origins.

Models
  statsforecast (season length 12): SeasonalNaive, RandomWalkWithDrift, AutoETS, AutoTheta,
      AutoARIMA, with prediction intervals at the 50, 80 and 95% levels.
  mlforecast global LightGBM (point forecasts only): target log1p(value per day), then local
      standard scaling; features lags 1, 2, 3, 6, 12, rolling means of lag 1 over 3 and 12
      months, calendar month, series identifier (categorical); fixed hyperparameters
      (500 trees, learning rate 0.05, 31 leaves, min 20 samples per leaf, seed 42);
      recursive multi-step forecasts.

Output: results/base_forecasts_v2/model=<name>/forecasts.parquet (hive partitioning by model),
one row per series, origin and target month: unique_id, unit, measure, unit_type, origin,
target, horizon, y (actual), mean, lo_50, hi_50, lo_80, hi_80, lo_95, hi_95.
A model whose output file exists is skipped, so an interrupted run resumes where it stopped.
results/base_forecasts_v2/series_meta.parquet holds one row per series.

Usage:
  python src/base_forecasts.py --estimate          # time one origin (2025-06) per model, write nothing
  python src/base_forecasts.py [--models AutoETS LightGBM ...] [--n-jobs -1]
"""
import argparse
import os
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean
from mlforecast.target_transforms import GlobalSklearnTransformer, LocalStandardScaler
from sklearn.preprocessing import FunctionTransformer
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, AutoTheta, RandomWalkWithDrift, SeasonalNaive

from runlog import RunLog

SEASON = 12
H, STEP, N_WINDOWS = 12, 1, 31                 # origins 2022-12 .. 2025-06 with data to 2026-06
FIRST_ORIGIN, LAST_ORIGIN = "2022-12", "2025-06"
LEVELS = [50, 80, 95]
SEED = 42
STAT_MODELS = {
    "SeasonalNaive": lambda: SeasonalNaive(season_length=SEASON),
    "RandomWalkWithDrift": lambda: RandomWalkWithDrift(alias="RandomWalkWithDrift"),  # default alias is RWD
    "AutoETS": lambda: AutoETS(season_length=SEASON),
    "AutoTheta": lambda: AutoTheta(season_length=SEASON),
    "AutoARIMA": lambda: AutoARIMA(season_length=SEASON),
}
ALL_MODELS = list(STAT_MODELS) + ["LightGBM"]

# On Windows, multiprocessing starts workers by re-importing this module, so all work
# must sit inside main() behind the __name__ guard.
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean", default="data/clean")
    ap.add_argument("--out", default="results/base_forecasts_v2")
    ap.add_argument("--suffix", default="v2")
    ap.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    ap.add_argument("--n-jobs", type=int, default=-1, help="parallel jobs for statsforecast and LightGBM threads")
    ap.add_argument("--estimate", action="store_true", help="time a single origin per model and exit")
    args = ap.parse_args()
    log = RunLog("base_forecasts.py", vars(args))
    CLEAN, OUT = Path(args.clean), Path(args.out)
    np.random.seed(SEED)

    # ---------------------------------------------------------------- series
    sys_long = pd.read_parquet(CLEAN / f"panel_system_{args.suffix}.parquet")
    sys_long = sys_long[sys_long.in_constraints]
    out_long = pd.read_parquet(CLEAN / f"panel_outside_{args.suffix}.parquet")
    cols = ["unit", "unit_type", "measure", "basis", "period", "co2_t_per_day"]
    long = pd.concat([sys_long[cols], out_long[cols]], ignore_index=True)
    # Outside-system units are economies too; the prefix keeps their ids distinct from system ids.
    long["unique_id"] = np.where(long.unit_type == "outside", "OUT:", "") + long.unit + "|" + long.measure
    long["ds"] = pd.to_datetime(long.period, format="%Y-%m")
    df = long[["unique_id", "ds", "co2_t_per_day"]].rename(columns={"co2_t_per_day": "y"})
    df = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    meta = long.drop_duplicates("unique_id")[["unique_id", "unit", "unit_type", "measure", "basis"]]
    w22 = long[long.period.str.startswith("2022")].groupby("unique_id").co2_t_per_day.mean().rename("weight_2022")
    meta = meta.merge(w22, on="unique_id").sort_values("unique_id").reset_index(drop=True)
    assert df.groupby("unique_id").size().nunique() == 1, "all series must cover the same months"
    print(f"series: {len(meta)} ({meta.unit_type.value_counts().to_dict()}) | months: {df.ds.nunique()}")


    def tidy(cv, model):
        """cross_validation output -> common long format."""
        o = cv.rename(columns={model: "mean"}).copy()
        for lv in LEVELS:
            for side in ("lo", "hi"):
                c = f"{model}-{side}-{lv}"
                o[f"{side}_{lv}"] = o[c] if c in o.columns else np.nan
        o["origin"] = o.cutoff.dt.strftime("%Y-%m")
        o["target"] = o.ds.dt.strftime("%Y-%m")
        o["horizon"] = ((o.ds.dt.year - o.cutoff.dt.year) * 12 + o.ds.dt.month - o.cutoff.dt.month).astype(int)
        o = o.merge(meta[["unique_id", "unit", "measure", "unit_type"]], on="unique_id")
        keep = ["unique_id", "unit", "measure", "unit_type", "origin", "target", "horizon", "y", "mean"] + \
               [f"{s}_{lv}" for lv in LEVELS for s in ("lo", "hi")]
        return o[keep].sort_values(["unique_id", "origin", "horizon"]).reset_index(drop=True)


    def run_stat(name, n_windows):
        sf = StatsForecast(models=[STAT_MODELS[name]()], freq="MS", n_jobs=args.n_jobs)
        cv = sf.cross_validation(h=H, df=df, n_windows=n_windows, step_size=STEP, level=LEVELS, refit=True)
        return tidy(cv.reset_index() if "unique_id" not in cv.columns else cv, name)


    def run_lgbm(n_windows):
        d = df.assign(series=pd.Categorical(df.unique_id))          # series identifier as categorical feature
        model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31, min_child_samples=20,
                                  random_state=SEED, deterministic=True, force_row_wise=True,
                                  n_jobs=os.cpu_count() if args.n_jobs == -1 else args.n_jobs, verbosity=-1)
        mlf = MLForecast(models={"LightGBM": model}, freq="MS", lags=[1, 2, 3, 6, 12],
                         lag_transforms={1: [RollingMean(window_size=3), RollingMean(window_size=12)]},
                         date_features=["month"],
                         target_transforms=[GlobalSklearnTransformer(FunctionTransformer(np.log1p, np.expm1)),
                                            LocalStandardScaler()])
        cv = mlf.cross_validation(d, n_windows=n_windows, h=H, step_size=STEP, refit=True, static_features=["series"])
        feats = mlf.ts.features_order_
        print("LightGBM features:", feats, "| categorical:", [c for c in feats if str(d.dtypes.get(c)) == "category"])
        return tidy(cv, "LightGBM")


    def run(name, n_windows):
        return run_lgbm(n_windows) if name == "LightGBM" else run_stat(name, n_windows)


    # ---------------------------------------------------------------- estimate mode
    if args.estimate:
        total = 0.0
        for name in args.models:
            t0 = time.perf_counter()
            f = run(name, 1)
            dt = time.perf_counter() - t0
            assert f.origin.unique().tolist() == [LAST_ORIGIN], f.origin.unique()
            # Training windows at earlier origins are shorter, so origin 2025-06 is the slowest one.
            print(f"{name:20s} one origin: {dt:7.1f} s -> {N_WINDOWS} origins approx {dt * N_WINDOWS / 60:6.1f} min "
                  f"| NaN means: {int(f['mean'].isna().sum())}")
            total += dt * N_WINDOWS
        print(f"estimated total: {total / 60:.1f} min")
        log.finish(f"estimate only: {total / 60:.1f} min")
        return

    # ---------------------------------------------------------------- full run (resumable per model)
    OUT.mkdir(parents=True, exist_ok=True)
    meta_path = OUT / "series_meta.parquet"
    if not meta_path.exists():
        meta.to_parquet(meta_path, index=False)
    times = {}
    for name in args.models:
        path = OUT / f"model={name}" / "forecasts.parquet"
        if path.exists():
            print(f"{name}: exists, skipped")
            continue
        t0 = time.perf_counter()
        f = run(name, N_WINDOWS)
        times[name] = time.perf_counter() - t0
        origins = sorted(f.origin.unique())
        assert origins[0] == FIRST_ORIGIN and origins[-1] == LAST_ORIGIN and len(origins) == N_WINDOWS, origins
        path.parent.mkdir(parents=True, exist_ok=True)
        f.to_parquet(path, index=False)
        print(f"{name}: {len(f)} rows in {times[name] / 60:.1f} min | NaN means: {int(f['mean'].isna().sum())}")
    log.finish("run times (s): " + ", ".join(f"{k} {v:.0f}" for k, v in times.items()))


if __name__ == "__main__":
    main()
