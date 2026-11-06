"""
longhorizon_base.py: AutoETS forecasts to 36 months for the skill-decay analysis.

Origins: monthly 2021-12 ... 2022-12 (13); horizons 1-36, so every target lies in 2022-01 ... 2025-12.
Series: the 1,120 reconciled system series (DELTA_INT excluded). AutoETS (season length 12) is refitted
at every origin on 2019-01 ... origin (statsforecast cross_validation on the data to 2025-12 with h = 36,
step 1, 13 windows, refit, fitted values on). Point forecasts only: the analysis uses MASE, and point
forecasts of ETS do not depend on the requested horizon length (only the simulated intervals of ETS
classes 4 and 5 do).

Outputs (results/longhorizon/):
  base_forecasts/model=AutoETS/forecasts.parquet   unique_id, unit, measure, unit_type, origin, target, horizon, y, mean
  insample_residuals/model=AutoETS/residuals.parquet  unique_id, origin, period, resid (W; 12-month burn-in in reconcile.py)
Checks at origin 2022-12: horizons 1-12 equal results/base_forecasts_v2; residuals equal results/insample_residuals_v2.

Usage:
  python src/longhorizon_base.py [--estimate]
"""
import argparse
import time
from pathlib import Path

import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoETS

from runlog import RunLog, out_path

H, N_WINDOWS, LAST_TARGET = 36, 13, "2025-12"
FIRST_ORIGIN, LAST_ORIGIN = "2021-12", "2022-12"
REL_TOL = 1e-8


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean", default="data/clean")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/longhorizon")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--estimate", action="store_true", help="time the last origin only, write nothing")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    log = RunLog("longhorizon_base.py", vars(args))
    RES, OUT = Path(args.results), Path(args.out)

    p = pd.read_parquet(Path(args.clean) / "panel_system_v2.parquet")
    p = p[p.in_constraints & (p.measure != "DELTA_INT")]
    df = pd.DataFrame({"unique_id": p.unit + "|" + p.measure, "ds": pd.to_datetime(p.period, format="%Y-%m"),
                       "y": p.co2_t_per_day}).sort_values(["unique_id", "ds"]).reset_index(drop=True)
    meta = p.assign(unique_id=p.unit + "|" + p.measure).drop_duplicates("unique_id")[
        ["unique_id", "unit", "measure", "unit_type"]]
    df = df[df.ds <= pd.Timestamp(LAST_TARGET + "-01")]

    t0 = time.perf_counter()
    sf = StatsForecast(models=[AutoETS(season_length=12)], freq="MS", n_jobs=args.n_jobs)
    nw = 1 if args.estimate else N_WINDOWS
    cv = sf.cross_validation(h=H, df=df, n_windows=nw, step_size=1, refit=True, fitted=True)
    fv = sf.cross_validation_fitted_values()
    secs = time.perf_counter() - t0
    if args.estimate:
        print(f"one origin: {secs:.0f} s -> {N_WINDOWS} origins ~ {secs * N_WINDOWS / 60:.1f} min (the last origin is the "
              f"longest fit; an upper bound)")
        log.finish(f"estimate {secs:.0f} s per origin")
        return
    cv = cv.reset_index() if "unique_id" not in cv.columns else cv
    fv = fv.reset_index() if "unique_id" not in fv.columns else fv
    f = pd.DataFrame({"unique_id": cv.unique_id, "origin": cv.cutoff.dt.strftime("%Y-%m"),
                      "target": cv.ds.dt.strftime("%Y-%m"), "y": cv.y, "mean": cv.AutoETS})
    f["horizon"] = ((cv.ds.dt.year - cv.cutoff.dt.year) * 12 + cv.ds.dt.month - cv.cutoff.dt.month).astype(int)
    f = f.merge(meta, on="unique_id")[["unique_id", "unit", "measure", "unit_type", "origin", "target", "horizon",
                                       "y", "mean"]]
    origins = sorted(f.origin.unique())
    assert origins[0] == FIRST_ORIGIN and origins[-1] == LAST_ORIGIN and len(origins) == N_WINDOWS, origins
    assert f.horizon.min() == 1 and f.horizon.max() == H and f.target.max() == LAST_TARGET
    res = pd.DataFrame({"unique_id": fv.unique_id, "origin": fv.cutoff.dt.strftime("%Y-%m"),
                        "period": fv.ds.dt.strftime("%Y-%m"), "resid": (fv["y"] - fv["AutoETS"]).to_numpy(float)})

    # ---- checks at origin 2022-12 against the stored base forecasts and in-sample residuals
    old = pd.read_parquet(RES / "base_forecasts_v2" / "model=AutoETS" / "forecasts.parquet",
                          filters=[("origin", "==", LAST_ORIGIN)])
    m = f.merge(old, on=["unique_id", "origin", "target"], suffixes=("", "_old"))
    rel = float(((m["mean"] - m["mean_old"]).abs() / m["mean_old"].abs().clip(lower=1.0)).max())
    oldr = pd.read_parquet(RES / "insample_residuals_v2" / "model=AutoETS" / "residuals.parquet",
                           filters=[("origin", "==", LAST_ORIGIN)])
    mr = res.merge(oldr, on=["unique_id", "origin", "period"], suffixes=("", "_old"))
    absr = float((mr.resid - mr.resid_old).abs().max())
    print(f"{len(f)} forecasts in {secs / 60:.1f} min | check 2022-12: {len(m)} forecasts, max rel diff {rel:.2e}; "
          f"{len(mr)} residuals, max abs diff {absr:.2e}")
    if rel > REL_TOL:
        raise SystemExit("forecasts differ from results/base_forecasts_v2 at origin 2022-12: stop")
    f.to_parquet(out_path(OUT / "base_forecasts" / "model=AutoETS" / "forecasts.parquet", args.overwrite), index=False)
    res.to_parquet(out_path(OUT / "insample_residuals" / "model=AutoETS" / "residuals.parquet", args.overwrite),
                   index=False)
    log.finish(f"{secs:.0f} s; check rel {rel:.1e}, residuals {absr:.1e}")


if __name__ == "__main__":
    main()
