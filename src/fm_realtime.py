"""
fm_realtime.py: TimesFM 2.5 base forecasts for the real-time origins.

Runs in the foundation-model environment (requirements-fm.txt):
  <python of the foundation-model environment> src/fm_realtime.py

Origins: 2025-06 (horizons 1-15) and 2026-06 (horizons 1-3). Series: the system series (fm_forecasts.load_series;
DELTA_INT included, it is dropped in reconciliation). Context: 2019-01 up to and including the origin
(asserted). Model and inference settings as in fm_forecasts.py (TIMESFM25_CONFIG); TimesFM 2.5 decodes its
whole output block in one pass, so horizons 1-12 at 2025-06 must equal the rolling-origin forecasts of
fm_forecasts.py (checked).

Output: results/realtime/base_forecasts/model=TimesFM25/forecasts.parquet (layout of base_forecasts_fm;
y empty after 2026-06).

Usage:
  python src/fm_realtime.py
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fm_forecasts import QCOLS, SEED, TimesFM25, load_series
from runlog import RunLog, out_path

ORIGINS = {"2025-06": 15, "2026-06": 3}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean", default="data/clean")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/realtime/base_forecasts")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    log = RunLog("fm_realtime.py", vars(args))
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    Y, meta = load_series(args.clean)
    Y = Y.loc[:"2026-06", meta.index[meta.unit_type != "outside"]]
    model = TimesFM25("cuda" if torch.cuda.is_available() else "cpu")

    frames = []
    for o, h in ORIGINS.items():
        t0 = time.perf_counter()
        ctx = Y.loc[:o]
        assert ctx.index[0] == "2019-01" and ctx.index[-1] == o
        q, pt = model.predict([ctx[c].to_numpy(dtype=np.float32) for c in Y.columns], horizon=h)
        n = Y.shape[1]
        targets = pd.period_range(pd.Period(o, "M") + 1, periods=h, freq="M").astype(str)
        df = pd.DataFrame({"unique_id": np.repeat(Y.columns.to_numpy(), h), "origin": o,
                           "target": np.tile(targets, n), "horizon": np.tile(np.arange(1, h + 1), n)})
        df = df.join(meta, on="unique_id")
        df["y"] = [Y.at[t, u] if t in Y.index else np.nan for u, t in zip(df.unique_id, df.target)]
        for j, c in enumerate(QCOLS):
            df[c] = q.reshape(n * h, len(QCOLS))[:, j]
        df["mean"] = df["q0.5"]
        df["model_point"] = pt.reshape(-1)
        df["mean_head"] = model.mean_head.reshape(-1)
        frames.append(df)
        print(f"origin {o}: {len(df)} forecasts ({len(ctx)} months of context) in {time.perf_counter() - t0:.1f} s")
    f = pd.concat(frames, ignore_index=True)

    old = pd.read_parquet(Path(args.results) / "base_forecasts_fm" / "model=TimesFM25" / "forecasts.parquet",
                          filters=[("origin", "==", "2025-06")])
    m = f[f.origin == "2025-06"].merge(old, on=["unique_id", "origin", "target"], suffixes=("", "_old"))
    d = np.abs(m[QCOLS].to_numpy() - m[[c + "_old" for c in QCOLS]].to_numpy())
    rel = float((d / np.maximum(np.abs(m[QCOLS].to_numpy()), 1.0)).max())
    print(f"check 2025-06 h1-12 against fm_forecasts.py: {len(m)} forecasts, max abs {d.max():.3g}, max relative {rel:.2e}")
    f.to_parquet(out_path(Path(args.out) / "model=TimesFM25" / "forecasts.parquet", args.overwrite), index=False)
    log.finish(f"check 2025-06 rel {rel:.1e}")


if __name__ == "__main__":
    main()
