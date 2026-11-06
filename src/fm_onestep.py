"""
fm_onestep.py: one-step forecasts of the foundation models, the basis of their own W.

Runs in the foundation-model environment (requirements-fm.txt), on the GPU:
  <python of the foundation-model environment> src/fm_onestep.py ...

For every system series (the 1,121 constraint series, DELTA_INT included) and every month t from
2020-01 to 2025-06, month t is forecast one step ahead from the history 2019-01 ... t-1 only
(12 to 77 months of context). All series of one month are forecast as one batch, with the models
and inference settings of fm_forecasts.py. Each model returns its whole output block in one pass,
so step 1 of the 12-step output is the one-step forecast.

Leakage checks (results/onestep_fm/leakage_checks.csv):
  1. construction: for every t the context frame is asserted to end at t-1 and to start at 2019-01;
  2. poisoned panel: for CHECK_MONTHS, every value from month t on is replaced by 1e12 (and, in a
     second run, the rows are deleted) and the batch is forecast again; the forecasts must be equal
     to the stored ones;
  3. agreement with the base forecasts: the one-step forecast of month o+1 must equal the h = 1
     forecast of origin o in results/base_forecasts_fm (same information set), for every origin
     with o+1 <= 2025-06.

For the real-time origins the months are extended to 2025-07 ... 2026-06 for TimesFM25 (W at the real-time
origin 2026-06): --start 2025-07 --end 2026-06 --check-months 2025-07 2026-01 2026-06
--out results/onestep_fm_rt. Check 3 is then possible for 2025-07 only (base origin 2025-06).

Output: results/onestep_fm/model=<name>/onestep.parquet
  unique_id, period (target month t), context_end (t-1), context_months, y, median (q0.5),
  error (y - median), q0.1 ... q0.9. Existing files are skipped (resume).

Usage:
  python src/fm_onestep.py [--models ChronosBolt TimesFM TimesFM25] [--checks-only]
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fm_forecasts import MODEL_CLASSES, QCOLS, SEED, load_series
from runlog import RunLog

START, END = "2020-01", "2025-06"
CHECK_MONTHS = ["2020-01", "2022-12", "2025-06"]
POISON = 1e12


def contexts(Y, t):
    """Context of every series for target month t: rows 2019-01 ... t-1 (asserted)."""
    ctx = Y.loc[:str(t - 1)]
    assert ctx.index[-1] == str(t - 1) and ctx.index[0] == "2019-01"
    assert len(ctx) == (t - 1 - pd.Period("2019-01", "M")).n + 1
    return ctx, [ctx[c].to_numpy(dtype=np.float32) for c in Y.columns]


def onestep(model, Y, t):
    """(deciles [n, 9], context length) of the one-step forecasts of month t."""
    ctx, c = contexts(Y, t)
    q, _ = model.predict(c)
    return q[:, 0, :], len(ctx)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean", default="data/clean")
    ap.add_argument("--out", default="results/onestep_fm")
    ap.add_argument("--base-forecasts", default="results/base_forecasts_fm")
    ap.add_argument("--models", nargs="+", default=list(MODEL_CLASSES), choices=list(MODEL_CLASSES))
    ap.add_argument("--checks-only", action="store_true", help="only rerun the leakage checks on stored output")
    ap.add_argument("--start", default=START)
    ap.add_argument("--end", default=END)
    ap.add_argument("--check-months", nargs="+", default=CHECK_MONTHS)
    args = ap.parse_args()
    MONTHS = pd.period_range(args.start, args.end, freq="M")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Y, meta = load_series(args.clean)
    Y = Y.loc[:"2026-06", meta.index[meta.unit_type != "outside"]]       # system series only
    print(f"device {device} | system series: {Y.shape[1]} | target months: {len(MONTHS)} ({args.start} to {args.end})")
    OUT = Path(args.out)
    checks = []

    for name in args.models:
        log = RunLog("fm_onestep.py", {**vars(args), "model": name})
        t0 = time.perf_counter()
        model = MODEL_CLASSES[name](device)
        path = OUT / f"model={name}" / "onestep.parquet"
        if not path.exists() and not args.checks_only:
            frames = []
            for t in MONTHS:
                q, nctx = onestep(model, Y, t)
                d = pd.DataFrame(q, columns=QCOLS)
                d.insert(0, "unique_id", Y.columns)
                d.insert(1, "period", str(t))
                d.insert(2, "context_end", str(t - 1))
                d.insert(3, "context_months", nctx)
                d.insert(4, "y", Y.loc[str(t)].to_numpy())
                d.insert(5, "median", d["q0.5"])
                d.insert(6, "error", d.y - d["median"])
                frames.append(d)
            f = pd.concat(frames, ignore_index=True)
            path.parent.mkdir(parents=True, exist_ok=True)
            f.to_parquet(path, index=False)
            print(f"{name}: {len(f)} one-step forecasts in {(time.perf_counter() - t0) / 60:.1f} min | "
                  f"NaN {int(f[QCOLS].isna().sum().sum())}")
        f = pd.read_parquet(path)
        t_run = time.perf_counter() - t0

        # ---- check 2: poisoned and truncated panels
        for tm in args.check_months:
            t = pd.Period(tm, "M")
            stored = f[f.period == tm].set_index("unique_id").loc[Y.columns, QCOLS].to_numpy()
            Yp = Y.copy()
            Yp.loc[str(t):] = POISON                       # every value from month t on poisoned
            for label, Yv in [("poisoned_from_t", Yp), ("rows_from_t_deleted", Y.loc[:str(t - 1)])]:
                q, _ = onestep(model, Yv, t)
                d = np.abs(q - stored)
                checks.append({"model": name, "check": label, "period": tm, "n_series": len(q),
                               "max_abs_diff": float(d.max()),
                               "max_rel_diff": float((d / np.maximum(np.abs(stored), 1.0)).max())})
        # ---- check 3: agreement with the h = 1 base forecasts of origin t-1
        b = pd.read_parquet(Path(args.base_forecasts) / f"model={name}" / "forecasts.parquet",
                            columns=["unique_id", "origin", "target", "horizon"] + QCOLS)
        b = b[(b.horizon == 1) & b.unique_id.isin(Y.columns)]
        m = b.merge(f, left_on=["unique_id", "target"], right_on=["unique_id", "period"], suffixes=("_base", ""))
        assert (m.origin == m.context_end).all()
        d = np.abs(m[[c + "_base" for c in QCOLS]].to_numpy() - m[QCOLS].to_numpy())
        rel = d / np.maximum(np.abs(m[QCOLS].to_numpy()), 1.0)
        checks.append({"model": name, "check": "equals_base_h1_of_origin_t-1",
                       "period": f"{m.period.min()}..{m.period.max()} ({m.period.nunique()} months)",
                       "n_series": m.unique_id.nunique(), "max_abs_diff": float(d.max()),
                       "max_rel_diff": float(rel.max())})
        print(pd.DataFrame([c for c in checks if c["model"] == name]).to_string(index=False))
        log.finish(f"{name}: {t_run:.0f} s")
        del model
        torch.cuda.empty_cache()

    c = pd.DataFrame(checks)
    cpath = OUT / "leakage_checks.csv"
    if cpath.exists():                                    # keep the checks of models run earlier
        old = pd.read_csv(cpath)
        c = pd.concat([old[~old.model.isin(args.models)], c], ignore_index=True)
    c.to_csv(cpath, index=False, float_format="%.6g")


if __name__ == "__main__":
    main()
