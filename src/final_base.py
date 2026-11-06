"""
final_base.py: AutoETS base forecasts of the final forecasts at origin 2026-06, horizons 1-36.

Series: the 1,120 reconciled system series (DELTA_INT excluded; it is forecast as a ratio to the reconciled
G(World)). AutoETS (season length 12) is fitted on 2019-01 ... 2026-06 (later months are never used) and forecasts
h = 1-36 (targets 2026-07 ... 2029-06), with intervals at 50, 80 and 95% from the same h = 36 call. The intervals of
ETS classes 4 and 5 are simulated and depend on the requested horizon length, so they differ from the h = 12
intervals of realtime_base.py and calib_base.py; they are stored for reference only. Point forecasts do not depend
on h.

Checks against the real-time run at origin 2026-06 (realtime_base.py): point forecasts at h 1-3 and the in-sample
residuals must be equal (stop otherwise).

Outputs (results/final_base/):
  base_forecasts/model=AutoETS/forecasts.parquet      layout of results/realtime/base_forecasts (y empty)
  insample_residuals/model=AutoETS/residuals.parquet  unique_id, origin, period, resid (12-month burn-in in reconcile.py)

Usage:
  python src/final_base.py
"""
import argparse
import time
from pathlib import Path

import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoETS

from runlog import RunLog, out_path

ORIGIN, H = "2026-06", 36
LEVELS = [50, 80, 95]
REL_TOL = 1e-8


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean", default="data/clean")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/final_base")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    log = RunLog("final_base.py", vars(args))
    RES, OUT = Path(args.results), Path(args.out)

    p = pd.read_parquet(Path(args.clean) / "panel_system_v2.parquet")
    p = p[p.in_constraints & (p.measure != "DELTA_INT")]
    df = pd.DataFrame({"unique_id": p.unit + "|" + p.measure, "ds": pd.to_datetime(p.period, format="%Y-%m"),
                       "y": p.co2_t_per_day}).sort_values(["unique_id", "ds"]).reset_index(drop=True)
    meta = p.assign(unique_id=p.unit + "|" + p.measure).drop_duplicates("unique_id")[
        ["unique_id", "unit", "measure", "unit_type"]]
    train = df[df.ds <= pd.Timestamp(ORIGIN + "-01")]
    assert train.ds.min() == pd.Timestamp("2019-01-01") and train.ds.max() == pd.Timestamp(ORIGIN + "-01")

    t0 = time.perf_counter()
    sf = StatsForecast(models=[AutoETS(season_length=12)], freq="MS", n_jobs=args.n_jobs)
    f = sf.forecast(df=train, h=H, level=LEVELS, fitted=True)
    fv = sf.forecast_fitted_values()
    f = f.reset_index() if "unique_id" not in f.columns else f
    fv = fv.reset_index() if "unique_id" not in fv.columns else fv
    secs = time.perf_counter() - t0
    out = pd.DataFrame({"unique_id": f.unique_id, "origin": ORIGIN, "target": f.ds.dt.strftime("%Y-%m"),
                        "mean": f["AutoETS"]})
    out["horizon"] = out.groupby("unique_id").cumcount() + 1
    for lv in LEVELS:
        out[f"lo_{lv}"], out[f"hi_{lv}"] = f[f"AutoETS-lo-{lv}"].to_numpy(), f[f"AutoETS-hi-{lv}"].to_numpy()
    out = out.merge(meta, on="unique_id").assign(y=float("nan"))
    out = out[["unique_id", "unit", "measure", "unit_type", "origin", "target", "horizon", "y", "mean"]
              + [f"{s}_{lv}" for lv in LEVELS for s in ("lo", "hi")]]
    assert out.horizon.max() == H and out.target.max() == "2029-06"
    rs = pd.DataFrame({"unique_id": fv.unique_id, "origin": ORIGIN, "period": fv.ds.dt.strftime("%Y-%m"),
                       "resid": (fv["y"] - fv["AutoETS"]).to_numpy(float)})

    # ---- checks against the real-time run (realtime_base.py)
    old = pd.read_parquet(RES / "realtime" / "base_forecasts" / "model=AutoETS" / "forecasts.parquet",
                          filters=[("origin", "==", ORIGIN)])
    m = out.merge(old, on=["unique_id", "origin", "target"], suffixes=("", "_old"))
    rel = float(((m["mean"] - m["mean_old"]).abs() / m["mean_old"].abs().clip(lower=1.0)).max())
    oldr = pd.read_parquet(RES / "realtime" / "insample_residuals" / "model=AutoETS" / "residuals.parquet",
                           filters=[("origin", "==", ORIGIN)])
    mr = rs.merge(oldr, on=["unique_id", "origin", "period"], suffixes=("", "_old"))
    absr = float((mr.resid - mr.resid_old).abs().max())
    print(f"{len(out)} forecasts in {secs:.0f} s | check h1-3: {len(m)} forecasts, max rel diff {rel:.2e}; "
          f"{len(mr)} residuals, max abs diff {absr:.2e}")
    if rel > REL_TOL or len(mr) != len(rs):
        raise SystemExit("forecasts or residuals differ from the real-time run at origin 2026-06: stop")
    out.to_parquet(out_path(OUT / "base_forecasts" / "model=AutoETS" / "forecasts.parquet", args.overwrite), index=False)
    rs.to_parquet(out_path(OUT / "insample_residuals" / "model=AutoETS" / "residuals.parquet", args.overwrite),
                  index=False)
    log.finish(f"{secs:.0f} s; check rel {rel:.1e}, residuals {absr:.1e}")


if __name__ == "__main__":
    main()
