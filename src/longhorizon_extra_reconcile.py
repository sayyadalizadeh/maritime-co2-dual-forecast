"""
longhorizon_extra_reconcile.py: MinT exact-QP reconciliation of the long-horizon forecasts at the 31 main origins.

Input: results/longhorizon_extra/base_forecasts/model=<base>/forecasts.parquet (longhorizon_extra_base.py,
fm_longhorizon_extra.py; h 1-36). Horizons 1-12 are kept from the existing runs (results/reconciled_v2_burnin12,
reconciled_v2_fmownw), so only h 13-36 are reconciled: they are written to
results/longhorizon_extra/base_forecasts_h13_36/ and passed to reconcile.py unchanged (MinT, exact non-negative QP;
STAT: AutoETS in-sample residuals with the 12-month burn-in; FM: own one-step errors). The projection and the QP work
column by column with the same A and W, so a horizon's result does not depend on which other horizons are present.
Check: origin 2022-12, h 13-36, against the long-horizon run (results/reconciled_long), which reconciled the same
base forecasts with the same W.

Output: results/reconciled_long_extra/base=<base>/reconciled.parquet, results/reconcile_long_extra/
Usage:
  python src/longhorizon_extra_reconcile.py --base AutoETS
  python src/longhorizon_extra_reconcile.py --base TimesFM25
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import reconcile as R
from runlog import RunLog, out_path

SRC = Path("results/longhorizon_extra/base_forecasts")
SUB = Path("results/longhorizon_extra/base_forecasts_h13_36")
OUT = Path("results")
ORIGINS = pd.period_range("2022-12", "2025-06", freq="M").astype(str).tolist()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, choices=["AutoETS", "TimesFM25"])
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if not args.check_only:
        f = pd.read_parquet(SRC / f"model={args.base}" / "forecasts.parquet")
        f = f[f.horizon >= 13]
        f.to_parquet(out_path(SUB / f"model={args.base}" / "forecasts.parquet", args.overwrite), index=False)
        argv = ["reconcile.py", "--bases", args.base, "--methods", "MinT", "--tag", "long_extra", "--out", str(OUT),
                "--forecasts-dir", str(SUB), "--origins", *ORIGINS]
        if args.base == "TimesFM25":
            argv += ["--fm-w", "own"]
        if args.overwrite:
            argv.append("--overwrite")
        sys.argv = argv
        R.main()
    log = RunLog("longhorizon_extra_reconcile.py", vars(args))
    flt = [("method", "==", "MinT"), ("variant", "==", "nn_qp"), ("origin", "==", "2022-12")]
    new = pd.read_parquet(OUT / "reconciled_long_extra" / f"base={args.base}" / "reconciled.parquet", filters=flt)
    old = pd.read_parquet(Path("results/reconciled_long") / f"base={args.base}" / "reconciled.parquet", filters=flt)
    m = new.merge(old[old.horizon >= 13], on=["unique_id", "origin", "target", "horizon"], suffixes=("", "_old"))
    d = (m.value - m.value_old).abs()
    rel = float((d / m.value_old.abs().clip(lower=1.0)).max())
    print(f"{args.base}: check 2022-12 h13-36 against reconciled_long: {len(m)} cells, max abs {d.max():.3g} t/day, "
          f"max rel {rel:.2e}")
    pd.DataFrame([{"base": args.base, "cells": len(m), "max_abs_t_per_day": float(d.max()), "max_rel": rel}]).to_csv(
        out_path(OUT / "reconcile_long_extra" / f"check_2022-12_{args.base}.csv", args.overwrite), index=False)
    log.finish(f"check rel {rel:.1e}")


if __name__ == "__main__":
    main()
