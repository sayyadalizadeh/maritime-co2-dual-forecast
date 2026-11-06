"""
fm_forecasts.py: zero-shot forecasts of the foundation models.

Runs in the foundation-model environment (requirements-fm.txt), on the GPU:
  <python of the foundation-model environment> src/fm_forecasts.py ...

Models (no fine-tuning, no covariates, default inference settings):
  ChronosBolt  amazon/chronos-bolt-base   (chronos-forecasting; quantiles 0.1, 0.2, ..., 0.9)
  TimesFM      google/timesfm-3.0-pytorch (timesfm TimesFM3Forecaster; median and deciles 0.1 ... 0.9)
  TimesFM25    google/timesfm-2.5-200m-pytorch (timesfm TimesFM_2p5_200M_torch; base of the FM track).
               The dataclass defaults of ForecastConfig cannot run (max_context = 0), so the
               configuration of the model card is used (TIMESFM25_CONFIG); torch.compile is off
               (speed only). Outputs: mean channel and deciles 0.1 ... 0.9; point output = median.

Design as in base_forecasts.py: all 1,170 series (system and outside-system), values per day, origins 2022-12 to
2025-06, horizons 1 to 12. Context: the full history from 2019-01 up to and including the origin
month. All series of one origin are forecast as one batch.

Output (over 10 MB, not in git): results/base_forecasts_fm/model=<name>/forecasts.parquet, same layout as
results/base_forecasts_v2: unique_id, unit, measure, unit_type, origin, target, horizon, y,
mean (= the median, used as the point forecast), q0.1 ... q0.9 (every quantile the model outputs),
model_point (the model's own point output: Chronos-Bolt mean, TimesFM forecast, TimesFM 2.5 median);
TimesFM25 also has mean_head (its mean output channel).
A model whose file exists is skipped (resume).

Usage:
  python src/fm_forecasts.py --estimate                 # time one origin (2025-06) per model
  python src/fm_forecasts.py [--models ChronosBolt TimesFM]
  python src/fm_forecasts.py --spot-check               # rerun a few series one by one and compare
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from runlog import RunLog

H = 12
ORIGINS = pd.period_range("2022-12", "2025-06", freq="M").astype(str).tolist()
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
QCOLS = [f"q{q}" for q in QUANTILES]
MODEL_IDS = {"ChronosBolt": "amazon/chronos-bolt-base", "TimesFM": "google/timesfm-3.0-pytorch",
             "TimesFM25": "google/timesfm-2.5-200m-pytorch"}
# TimesFM 2.5: configuration of the model card (revision 1d952420fba87f3c6dee4f240de0f1a0fbc790e3)
TIMESFM25_CONFIG = dict(max_context=1024, max_horizon=256, normalize_inputs=True, use_continuous_quantile_head=True,
                        force_flip_invariance=True, infer_is_positive=True, fix_quantile_crossing=True,
                        per_core_batch_size=256)
SEED = 42
CHRONOS_BATCH = 256


def load_series(clean, suffix="v2"):
    """Wide frame of values per day (months x series) and series metadata, as in base_forecasts.py."""
    s = pd.read_parquet(Path(clean) / f"panel_system_{suffix}.parquet")
    s = s[s.in_constraints]
    o = pd.read_parquet(Path(clean) / f"panel_outside_{suffix}.parquet")
    cols = ["unit", "unit_type", "measure", "period", "co2_t_per_day"]
    long = pd.concat([s[cols], o[cols]], ignore_index=True)
    long["unique_id"] = np.where(long.unit_type == "outside", "OUT:", "") + long.unit + "|" + long.measure
    Y = long.pivot(index="period", columns="unique_id", values="co2_t_per_day").sort_index()
    meta = long.drop_duplicates("unique_id").set_index("unique_id")[["unit", "measure", "unit_type"]].loc[Y.columns]
    assert Y.notna().all().all()
    return Y, meta


class Chronos:
    def __init__(self, device):
        from chronos import BaseChronosPipeline
        self.p = BaseChronosPipeline.from_pretrained(MODEL_IDS["ChronosBolt"], device_map=device,
                                                     torch_dtype=torch.float32)

    def predict(self, contexts, horizon=H):
        """contexts: list of 1-D float arrays -> (quantiles [n, horizon, 9], point [n, horizon])."""
        qs, pts = [], []
        for i in range(0, len(contexts), CHRONOS_BATCH):
            batch = [torch.tensor(c, dtype=torch.float32) for c in contexts[i:i + CHRONOS_BATCH]]
            q, m = self.p.predict_quantiles(batch, prediction_length=horizon, quantile_levels=QUANTILES)
            qs.append(q.float().cpu().numpy())
            pts.append(m.float().cpu().numpy())
        return np.concatenate(qs), np.concatenate(pts)


class TimesFM:
    def __init__(self, device):
        import timesfm
        self.m = timesfm.TimesFM3Forecaster.from_pretrained(MODEL_IDS["TimesFM"], device=device)

    def predict(self, contexts, horizon=H):
        outs = list(self.m.predict_batch([np.asarray(c, dtype=np.float32) for c in contexts], horizon=horizon,
                                         return_quantiles=True))
        q = np.stack([np.asarray(o.quantiles, dtype=float) for o in outs])          # [n, H, 9] deciles
        pt = np.stack([np.asarray(o.forecast, dtype=float) for o in outs])           # [n, H] median
        return q, pt


class TimesFM25:
    def __init__(self, device):
        import timesfm
        self.m = timesfm.TimesFM_2p5_200M_torch.from_pretrained(MODEL_IDS["TimesFM25"], torch_compile=False)
        self.m.compile(timesfm.ForecastConfig(**TIMESFM25_CONFIG))

    def predict(self, contexts, horizon=H):
        """-> (deciles [n, horizon, 9], point [n, horizon]); the mean channel is kept in self.mean_head."""
        pt, q = self.m.forecast(horizon=horizon, inputs=[np.asarray(c, dtype=np.float32) for c in contexts])
        self.mean_head = np.asarray(q[..., 0], dtype=float)                       # channel 0: mean
        return np.asarray(q[..., 1:], dtype=float), np.asarray(pt, dtype=float)    # channels 1..9: deciles


MODEL_CLASSES = {"ChronosBolt": Chronos, "TimesFM": TimesFM, "TimesFM25": TimesFM25}


def forecast_origin(model, Y, origin):
    """Forecasts of all series at one origin -> (quantiles [n, H, 9], point [n, H])."""
    ctx = Y.loc[:origin]
    return model.predict([ctx[c].to_numpy(dtype=np.float32) for c in Y.columns])


def to_frame(Y, meta, origin, q, pt):
    targets = pd.period_range(pd.Period(origin, "M") + 1, periods=H, freq="M").astype(str)
    n = len(Y.columns)
    df = pd.DataFrame({"unique_id": np.repeat(Y.columns.to_numpy(), H), "origin": origin,
                       "target": np.tile(targets, n), "horizon": np.tile(np.arange(1, H + 1), n)})
    df = df.join(meta, on="unique_id")
    df["y"] = [Y.at[t, u] if t in Y.index else np.nan for u, t in zip(df.unique_id, df.target)]
    qf = q.reshape(n * H, len(QUANTILES))
    for j, c in enumerate(QCOLS):
        df[c] = qf[:, j]
    df["mean"] = df["q0.5"]                                   # point forecast = median
    df["model_point"] = pt.reshape(-1)
    return df[["unique_id", "unit", "measure", "unit_type", "origin", "target", "horizon", "y", "mean"]
              + QCOLS + ["model_point"]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean", default="data/clean")
    ap.add_argument("--out", default="results/base_forecasts_fm")
    ap.add_argument("--models", nargs="+", default=list(MODEL_IDS), choices=list(MODEL_IDS))
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--spot-check", action="store_true")
    args = ap.parse_args()
    log = RunLog("fm_forecasts.py", vars(args))
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, torch.cuda.get_device_name(0) if device == "cuda" else "")
    Y, meta = load_series(args.clean)
    Y = Y.loc[:"2026-06"]
    print(f"series: {Y.shape[1]} | months: {Y.shape[0]}")
    OUT = Path(args.out)

    for name in args.models:
        t0 = time.perf_counter()
        model = MODEL_CLASSES[name](device)
        t_load = time.perf_counter() - t0
        if args.estimate:
            t0 = time.perf_counter()
            q, pt = forecast_origin(model, Y, ORIGINS[-1])
            dt = time.perf_counter() - t0
            print(f"{name}: load {t_load:.1f} s | one origin {dt:.1f} s -> {len(ORIGINS)} origins ~ "
                  f"{dt * len(ORIGINS) / 60:.1f} min | quantiles {q.shape} | NaN {int(np.isnan(q).sum())}")
            continue
        path = OUT / f"model={name}" / "forecasts.parquet"
        if args.spot_check:
            f = pd.read_parquet(path)
            rng = np.random.default_rng(SEED)
            uids = list(rng.choice(Y.columns, 4, replace=False)) + ["W|RES_TOTAL_SYS"]
            for o in ["2022-12", "2024-06", "2025-06"]:
                for u in uids:
                    qq, pp = model.predict([Y.loc[:o, u].to_numpy(dtype=np.float32)])   # batch of one
                    st = f[(f.unique_id == u) & (f.origin == o)].sort_values("horizon")
                    d = np.abs(st[QCOLS].to_numpy() - qq[0]).max()
                    rel = d / max(np.abs(qq[0]).max(), 1.0)
                    print(f"spot-check {name} {u} {o}: max |stored - manual| = {d:.3e} (relative {rel:.2e})")
            continue
        if path.exists():
            print(f"{name}: exists, skipped")
            continue
        frames = []
        for o in ORIGINS:
            q, pt = forecast_origin(model, Y, o)
            fr = to_frame(Y, meta, o, q, pt)
            if hasattr(model, "mean_head"):
                fr["mean_head"] = model.mean_head.reshape(-1)
            frames.append(fr)
        f = pd.concat(frames, ignore_index=True)
        crossing = int((np.diff(f[QCOLS].to_numpy(), axis=1) < -1e-9).any(axis=1).sum())
        path.parent.mkdir(parents=True, exist_ok=True)
        f.to_parquet(path, index=False)
        secs = time.perf_counter() - t0
        print(f"{name}: {len(f)} rows in {secs / 60:.1f} min (load {t_load:.0f} s) | NaN {int(f[QCOLS].isna().sum().sum())}"
              f" | rows with crossing quantiles {crossing} | negative medians {int((f['mean'] < 0).sum())}")
        log.finish(f"{name}: {secs:.0f} s")
        log = RunLog("fm_forecasts.py", vars(args))
        del model
        torch.cuda.empty_cache()
    if args.estimate or args.spot_check:
        log.finish("estimate" if args.estimate else "spot-check")


if __name__ == "__main__":
    main()
