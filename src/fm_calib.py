"""
fm_calib.py: TimesFM 2.5 base forecasts for the extended calibration origins and the final origin.

Runs in the foundation-model environment (requirements-fm.txt):
  <python of the foundation-model environment> src/fm_calib.py

Origins 2025-07 ... 2026-05 (calibration only) and 2026-06 (final forecasts), horizons 1-12; system series
(fm_forecasts.load_series; DELTA_INT included, dropped in reconciliation); context 2019-01 up to and including the
origin (asserted). Model and settings of fm_forecasts.TimesFM25. TimesFM 2.5 decodes its whole output block in one
pass, so 2026-06 must equal the base forecasts of the final forecasts (h 1-12 of results/final_base, fm_final.py)
(checked).

Output: results/calib/base_forecasts/model=TimesFM25/forecasts.parquet (layout of base_forecasts_fm)

Usage:
  python src/fm_calib.py
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fm_forecasts import QCOLS, SEED, TimesFM25, load_series
from runlog import RunLog, out_path

ORIGINS = pd.period_range("2025-07", "2026-06", freq="M").astype(str).tolist()
H = 12


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean", default="data/clean")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/calib/base_forecasts")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    log = RunLog("fm_calib.py", vars(args))
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    Y, meta = load_series(args.clean)
    Y = Y.loc[:"2026-06", meta.index[meta.unit_type != "outside"]]
    model = TimesFM25("cuda" if torch.cuda.is_available() else "cpu")

    frames, t_all = [], time.perf_counter()
    for o in ORIGINS:
        ctx = Y.loc[:o]
        assert ctx.index[0] == "2019-01" and ctx.index[-1] == o
        q, pt = model.predict([ctx[c].to_numpy(dtype=np.float32) for c in Y.columns], horizon=H)
        n = Y.shape[1]
        targets = pd.period_range(pd.Period(o, "M") + 1, periods=H, freq="M").astype(str)
        df = pd.DataFrame({"unique_id": np.repeat(Y.columns.to_numpy(), H), "origin": o,
                           "target": np.tile(targets, n), "horizon": np.tile(np.arange(1, H + 1), n)})
        df = df.join(meta, on="unique_id")
        df["y"] = [Y.at[t, u] if t in Y.index else np.nan for u, t in zip(df.unique_id, df.target)]
        for j, c in enumerate(QCOLS):
            df[c] = q.reshape(n * H, len(QCOLS))[:, j]
        df["mean"] = df["q0.5"]
        df["model_point"] = pt.reshape(-1)
        df["mean_head"] = model.mean_head.reshape(-1)
        frames.append(df)
    f = pd.concat(frames, ignore_index=True)
    secs = time.perf_counter() - t_all

    old = pd.read_parquet(Path(args.results) / "final_base" / "base_forecasts" / "model=TimesFM25" / "forecasts.parquet",
                          filters=[("origin", "==", "2026-06"), ("horizon", "<=", H)])
    m = f.merge(old, on=["unique_id", "origin", "target"], suffixes=("", "_old"))
    d = np.abs(m[QCOLS].to_numpy() - m[[c + "_old" for c in QCOLS]].to_numpy())
    rel = float((d / np.maximum(np.abs(m[QCOLS].to_numpy()), 1.0)).max())
    print(f"{len(f)} forecasts in {secs:.1f} s | check 2026-06 h1-12 against fm_final.py: {len(m)} rows, max abs {d.max():.3g}, "
          f"max relative {rel:.2e}")
    f.to_parquet(out_path(Path(args.out) / "model=TimesFM25" / "forecasts.parquet", args.overwrite), index=False)
    log.finish(f"{len(ORIGINS)} origins, {secs:.0f} s; check rel {rel:.1e}")


if __name__ == "__main__":
    main()
