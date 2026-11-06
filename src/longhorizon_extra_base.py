"""
longhorizon_extra_base.py: AutoETS forecasts to 36 months at the 31 main origins.

Origins: monthly 2022-12 ... 2025-06; horizons 1-36. Series: the 1,120 reconciled system series (DELTA_INT excluded).
At every origin AutoETS (season length 12) is fitted on 2019-01 ... origin and forecasts 36 months
(statsforecast forecast; the cross-validation of longhorizon_base.py cannot reach h = 36 from origins up to 2025-06
because the data end in 2026-06). Point forecasts only (MASE skill).
W for the reconciliation comes from the in-sample residuals of insample_residuals.py (results/insample_residuals_v2), which belong
to the same fits; they are not rewritten.

Check: horizons 1-12 must equal the stored base forecasts (results/base_forecasts_v2, relative tolerance 1e-8), and
horizons 1-36 at origin 2022-12 the long-horizon forecasts of longhorizon_base.py (results/longhorizon).

Output: results/longhorizon_extra/base_forecasts/model=AutoETS/forecasts.parquet
        unique_id, unit, measure, unit_type, origin, target, horizon, y, mean

Usage:
  python src/longhorizon_extra_base.py [--estimate]
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoETS

from runlog import RunLog, out_path

H = 36
ORIGINS = pd.period_range("2022-12", "2025-06", freq="M").astype(str).tolist()
REL_TOL = 1e-8


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean", default="data/clean")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/longhorizon_extra/base_forecasts")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--estimate", action="store_true", help="time the last origin only, write nothing")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    log = RunLog("longhorizon_extra_base.py", vars(args))
    RES = Path(args.results)

    p = pd.read_parquet(Path(args.clean) / "panel_system_v2.parquet")
    p = p[p.in_constraints & (p.measure != "DELTA_INT")]
    df = pd.DataFrame({"unique_id": p.unit + "|" + p.measure, "ds": pd.to_datetime(p.period, format="%Y-%m"),
                       "y": p.co2_t_per_day}).sort_values(["unique_id", "ds"]).reset_index(drop=True)
    meta = p.assign(unique_id=p.unit + "|" + p.measure).drop_duplicates("unique_id")[
        ["unique_id", "unit", "measure", "unit_type"]]
    actual = df.assign(target=df.ds.dt.strftime("%Y-%m"))[["unique_id", "target", "y"]]

    frames, t_start = [], time.perf_counter()
    for o in (ORIGINS[-1:] if args.estimate else ORIGINS):
        t0 = time.perf_counter()
        d = df[df.ds <= pd.Timestamp(o + "-01")]
        assert d.ds.max() == pd.Timestamp(o + "-01") and d.ds.min() == pd.Timestamp("2019-01-01")
        sf = StatsForecast(models=[AutoETS(season_length=12)], freq="MS", n_jobs=args.n_jobs)
        fc = sf.forecast(df=d, h=H)
        fc = fc.reset_index() if "unique_id" not in fc.columns else fc
        f = pd.DataFrame({"unique_id": fc.unique_id, "origin": o, "target": fc.ds.dt.strftime("%Y-%m"),
                          "mean": fc.AutoETS.to_numpy(float)})
        f["horizon"] = (pd.PeriodIndex(f.target, freq="M") - pd.Period(o, "M")).map(lambda x: x.n).astype(int)
        frames.append(f)
        print(f"origin {o}: {len(f)} forecasts in {time.perf_counter() - t0:.0f} s", flush=True)
    if args.estimate:
        per = time.perf_counter() - t_start
        print(f"one origin (the longest fit): {per:.0f} s -> {len(ORIGINS)} origins <= {per * len(ORIGINS) / 60:.1f} min")
        log.finish("estimate")
        return
    f = pd.concat(frames, ignore_index=True).merge(actual, on=["unique_id", "target"], how="left").merge(meta, on="unique_id")
    f = f[["unique_id", "unit", "measure", "unit_type", "origin", "target", "horizon", "y", "mean"]]
    assert f.horizon.min() == 1 and f.horizon.max() == H and len(f) == len(ORIGINS) * H * meta.shape[0]

    # ---- checks: horizons 1-12 against base_forecasts_v2; origin 2022-12 against longhorizon
    old = pd.read_parquet(RES / "base_forecasts_v2" / "model=AutoETS" / "forecasts.parquet",
                          columns=["unique_id", "origin", "target", "mean"])
    m = f.merge(old, on=["unique_id", "origin", "target"], suffixes=("", "_old"))
    rel = float(((m["mean"] - m["mean_old"]).abs() / m["mean_old"].abs().clip(lower=1.0)).max())
    lh = pd.read_parquet(RES / "longhorizon" / "base_forecasts" / "model=AutoETS" / "forecasts.parquet",
                         columns=["unique_id", "origin", "target", "mean"])
    m2 = f.merge(lh, on=["unique_id", "origin", "target"], suffixes=("", "_old"))
    rel2 = float(((m2["mean"] - m2["mean_old"]).abs() / m2["mean_old"].abs().clip(lower=1.0)).max())
    secs = time.perf_counter() - t_start
    print(f"{len(f)} forecasts in {secs / 60:.1f} min | h1-12 vs base_forecasts_v2: {len(m)} cells, max rel diff {rel:.2e} | "
          f"2022-12 h1-36 vs longhorizon: {len(m2)} cells, max rel diff {rel2:.2e}")
    if len(m) != len(ORIGINS) * 12 * meta.shape[0] or rel > REL_TOL or rel2 > REL_TOL:
        raise SystemExit("forecasts differ from the earlier runs: stop")
    f.to_parquet(out_path(Path(args.out) / "model=AutoETS" / "forecasts.parquet", args.overwrite), index=False)
    log.finish(f"{secs:.0f} s; check h1-12 rel {rel:.1e}; 2022-12 rel {rel2:.1e}")


if __name__ == "__main__":
    main()
