"""
fm_final.py: TimesFM 2.5 base forecasts of the final forecasts at origin 2026-06, horizons 1-36.

Runs in the foundation-model environment (requirements-fm.txt):
  <python of the foundation-model environment> src/fm_final.py

Series: the system series (fm_forecasts.load_series; DELTA_INT included, it is dropped in reconciliation).
Context: 2019-01 up to and including 2026-06 (asserted). Model and settings of fm_forecasts.TimesFM25.
TimesFM 2.5 decodes its whole output block in one pass, so horizons 1-3 must equal the real-time
forecasts at 2026-06 (fm_realtime.py) (checked).

Output: results/final_base/base_forecasts/model=TimesFM25/forecasts.parquet (layout of base_forecasts_fm; y empty)

Usage:
  python src/fm_final.py
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fm_forecasts import QCOLS, SEED, TimesFM25, load_series
from runlog import RunLog, out_path

ORIGIN, H = "2026-06", 36


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean", default="data/clean")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/final_base/base_forecasts")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    log = RunLog("fm_final.py", vars(args))
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    Y, meta = load_series(args.clean)
    Y = Y.loc[:ORIGIN, meta.index[meta.unit_type != "outside"]]
    model = TimesFM25("cuda" if torch.cuda.is_available() else "cpu")

    t0 = time.perf_counter()
    assert Y.index[0] == "2019-01" and Y.index[-1] == ORIGIN
    q, pt = model.predict([Y[c].to_numpy(dtype=np.float32) for c in Y.columns], horizon=H)
    n = Y.shape[1]
    targets = pd.period_range(pd.Period(ORIGIN, "M") + 1, periods=H, freq="M").astype(str)
    f = pd.DataFrame({"unique_id": np.repeat(Y.columns.to_numpy(), H), "origin": ORIGIN,
                      "target": np.tile(targets, n), "horizon": np.tile(np.arange(1, H + 1), n)})
    f = f.join(meta, on="unique_id")
    f["y"] = np.nan
    for j, c in enumerate(QCOLS):
        f[c] = q.reshape(n * H, len(QCOLS))[:, j]
    f["mean"] = f["q0.5"]
    f["model_point"] = pt.reshape(-1)
    f["mean_head"] = model.mean_head.reshape(-1)
    secs = time.perf_counter() - t0

    old = pd.read_parquet(Path(args.results) / "realtime" / "base_forecasts" / "model=TimesFM25" / "forecasts.parquet",
                          filters=[("origin", "==", ORIGIN)])
    m = f.merge(old, on=["unique_id", "origin", "target"], suffixes=("", "_old"))
    d = np.abs(m[QCOLS].to_numpy() - m[[c + "_old" for c in QCOLS]].to_numpy())
    rel = float((d / np.maximum(np.abs(m[QCOLS].to_numpy()), 1.0)).max())
    print(f"{len(f)} forecasts in {secs:.1f} s | check h1-3 against the real-time run: {len(m)} forecasts, max abs {d.max():.3g}, "
          f"max relative {rel:.2e}")
    f.to_parquet(out_path(Path(args.out) / "model=TimesFM25" / "forecasts.parquet", args.overwrite), index=False)
    log.finish(f"{secs:.0f} s; check rel {rel:.1e}")


if __name__ == "__main__":
    main()
