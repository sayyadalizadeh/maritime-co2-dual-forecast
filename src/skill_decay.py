"""
skill_decay.py: the persistence reference and the skill functions used by skill.py, skill_drift.py and
episode_2022.py (a library; it writes nothing).

Persistence reference, per series and origin o, from data up to o only:
  level      L_o = mean(y_{o-11}, ..., y_o)
  profile    for every month t with o-23 <= t <= o-6 (the months of the 24-month window whose centred mean
             uses no data after o): CMA_t = (y_{t-6}/2 + y_{t-5} + ... + y_{t+5} + y_{t+6}/2) / 12 and
             r_t = y_t / CMA_t (only where CMA_t > 0). S_m = mean of r_t over the months t in calendar month m
             (each calendar month appears once or twice); a calendar month without a ratio gets S_m = 1; the
             12 factors are rescaled to mean 1.
  forecast   ref(o, h) = L_o * S_{month(o + h)}
Skill: s(h) = 1 - wMASE_track(h) / wMASE_ref(h) over system economies, MASE scaled by the in-sample
seasonal-naive scale at the origin (evalcore), per-series MASE = mean over origins, weights = 2022 level (within the
system economies); main-window targets (<= 2025-12, without 2025-09); series with a zero scale are left out of both.
Median-based skill: s_med(h) = 1 - median MASE_track / median MASE_ref.

Tracks: FM = TimesFM25 + MinT (own W), STAT = AutoETS + MinT. Long-horizon origins 2021-12 ... 2022-12
(results/reconciled_long, h 1-36); main origins 2022-12 ... 2025-06.
"""
import numpy as np
import pandas as pd

TRACKS = {"FM": "TimesFM25", "STAT": "AutoETS"}
MAIN_TAGS = {"FM": "v2_fmownw", "STAT": "v2_burnin12"}
LONG_ORIGINS = pd.period_range("2021-12", "2022-12", freq="M").astype(str).tolist()
MAIN_ORIGINS = pd.period_range("2022-12", "2025-06", freq="M").astype(str).tolist()
H_LONG, H_OUT, W_MIN, SEED = 36, 60, 0.05, 42
TER_TOTAL = "W|TER_DOM+TER_INT"
GROUP = {"RES_TOTAL_SYS": "residence", "TER_DOM": "territory", "TER_INT": "territory"}


def reference(Y, origins, H):
    """Persistence reference ref(o, h) for every column of Y (months x series). Long frame."""
    idx = list(Y.index)
    V = Y.to_numpy(float)
    rows = []
    for o in origins:
        p = idx.index(o)
        level = V[p - 11: p + 1].mean(axis=0)
        ratios = {m: [] for m in range(1, 13)}
        for q in range(p - 23, p - 5):                       # t = o-23 ... o-6
            cma = (0.5 * V[q - 6] + V[q - 5: q + 6].sum(axis=0) + 0.5 * V[q + 6]) / 12
            with np.errstate(divide="ignore", invalid="ignore"):
                r = np.where(cma > 0, V[q] / cma, np.nan)
            ratios[pd.Period(idx[q], "M").month].append(r)
        S = np.vstack([np.nanmean(np.vstack(ratios[m]), axis=0) if ratios[m] else np.full(V.shape[1], np.nan)
                       for m in range(1, 13)])               # (12, n)
        S = np.where(np.isfinite(S), S, 1.0)
        S = S * 12 / S.sum(axis=0)
        for h in range(1, H + 1):
            t = pd.Period(o, "M") + h
            rows.append(pd.DataFrame({"unique_id": Y.columns, "origin": o, "target": str(t), "horizon": h,
                                      "ref": level * S[t.month - 1]}))
    return pd.concat(rows, ignore_index=True)


def add_ter_total(df, value_col):
    """Append the World TER_DOM + TER_INT series to a long forecast frame."""
    w = df[df.unique_id.isin(["W|TER_DOM", "W|TER_INT"])]
    s = w.groupby(["origin", "target", "horizon"], as_index=False)[value_col].sum().assign(unique_id=TER_TOTAL)
    return pd.concat([df, s], ignore_index=True)


def cube(f, col, origins, H, uids):
    """Array [h, origin, series] of column col (NaN where missing)."""
    piv = f.set_index(["horizon", "origin", "unique_id"])[col]
    full = pd.MultiIndex.from_product([range(1, H + 1), origins, uids], names=["horizon", "origin", "unique_id"])
    return piv.reindex(full).to_numpy().reshape(H, len(origins), len(uids))


def skill_from(At, Ar, cnt, w, med=False):
    """s(h) from scaled-error cubes At, Ar [h, o, i] with origin counts cnt [o] and series weights w [i]."""
    ok = np.isfinite(At) & np.isfinite(Ar)
    c = cnt[None, :, None] * ok
    den = c.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mt = (np.where(ok, At, 0) * c).sum(axis=1) / den          # per-series MASE [h, i]
        mr = (np.where(ok, Ar, 0) * c).sum(axis=1) / den
    out = np.full(At.shape[0], np.nan)
    for h in range(At.shape[0]):
        v = np.isfinite(mt[h]) & np.isfinite(mr[h])
        if not v.any():
            continue
        if med:
            out[h] = 1 - np.median(mt[h, v]) / np.median(mr[h, v])
        else:
            out[h] = 1 - np.average(mt[h, v], weights=w[v]) / np.average(mr[h, v], weights=w[v])
    return out, mt, mr
