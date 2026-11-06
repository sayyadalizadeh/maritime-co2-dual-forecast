"""
prob_samples.py: joint predictive samples, sample reconciliation, sample-based scores and adaptive
conformal intervals for the two tracks (main origins and real-time origins).

Tracks:
  FM    TimesFM25 base, MinT with its own one-step W; point forecasts: reconciled_v2_fmownw
  STAT  AutoETS base, MinT with the in-sample W after the 12-month burn-in; point forecasts: reconciled_v2_burnin12

For every origin and horizon, over the 1,120 reconciled system series (DELTA_INT excluded):
  1. Marginals. Base quantiles of each series: TimesFM 2.5 deciles 0.1 ... 0.9; AutoETS 0.025, 0.1,
     0.25, 0.5, 0.75, 0.9, 0.975 (bounds of its 95, 80 and 50% intervals and its mean, the median of
     its Gaussian predictive distribution). The quantile function is piecewise linear in
     z = Phi^-1(p) between these points (monotone: the values are sorted first; rows that needed
     sorting are counted) and is extended linearly in z beyond the outermost points with the
     slope of the outermost pair. For a Gaussian marginal this reproduces it exactly.
  2. Dependence. Gaussian copula with correlation matrix R = the track's MinT shrinkage covariance W
     at that origin (reconcile.mint_w, variance floor included), scaled to unit diagonal.
  3. Sampling. S = 500 joint draws: z = e Lc' with e ~ N(0, I) and R = Lc Lc'; each series is
     mapped through its quantile function (equivalent to u = Phi(z), y = Q(u)). The standard normal
     draws e depend only on the seed and the origin (drawn horizon by horizon), so both tracks use
     the same e (common random numbers) and a real-time rerun of an origin reproduces it.
  4. Reconciliation. Every sample is projected with the track's free MinT projection (the linear
     map used for the point forecasts, reconcile.project); negative values are then set to zero.
     The share of values set to zero and the largest |C y| afterwards are recorded. The track's
     point forecast remains the exact non-negative QP result of reconcile.py.
  5. Storage: quantiles 0.025, 0.05, 0.1, 0.2, ..., 0.9, 0.95, 0.975 of base and reconciled samples.

Scores computed from the samples (targets with an observed value):
  crps   fair CRPS estimator: mean|X - y| - sum_i (2i - S - 1) x_(i) / (S (S - 1)), t CO2 per day
  es     energy score of the vector scaled by each series' MASE scale at the origin (series with a
         zero scale excluded); O(S n) estimator: mean_s ||x_s - y|| - 1/2 mean_{s<S/2} ||x_s - x_{s+S/2}||
  vs     variogram score, p = 0.5, unit weights, pairs i < j, same scaling, on World, Residual and the
         20 system economies with the largest mean RES_TOTAL_SYS per day in 2025 (all 10 measures)

Adaptive conformal inference (ACI), per series and horizon, nominal 80% and 95%, gamma 0.05 and 0.01,
on the reconciled sample quantiles: the interval at origin o is [Q(alpha/2), Q(1 - alpha/2)] of the
reconciled samples (linear interpolation of the order statistics). At every month t (in order), for
each horizon h the forecast made at t - h is scored against y_t (known at t) and
alpha <- clip(alpha + gamma (alpha_nominal - err), 0.001, 0.999); alpha starts at the nominal value.
After the last origin (2025-06) the updates continue month by month to 2026-06 with the outcomes of
forecasts already made; the state at 2025-06 and at 2026-06 is saved for the real-time origins.

Modes
  main      origins 2022-12 ... 2025-06, horizons 1-12, base forecasts of base_forecasts.py and fm_forecasts.py.
  realtime  origins and forecasts given by --forecasts-dir (results/realtime/base_forecasts), W from
            --residuals-dir / --onestep-dirs; ACI from the saved main-run state (no further updates;
            horizons 13-15 use the state of horizon 12). Also stores the base and reconciled
            samples of every RES_TOTAL_SYS series (for the RES_TOTAL_PUB scoring).

Outputs: results/prob_<mode>/track=<FM|STAT>/
  quantiles.parquet (over 10 MB, not in git), crps.parquet, joint_scores.csv, aci_intervals.parquet,
  aci_state.parquet (main), sampling_diagnostics.csv, samples_res_total_sys.parquet (realtime)

Usage:
  python src/prob_samples.py --track STAT [--estimate]
  python src/prob_samples.py --track FM --mode realtime --forecasts-dir results/realtime/base_forecasts ...
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import cholesky
from scipy.stats import norm

from evalcore import load_history, origin_scales
from reconcile import (NEG_TOL, load_system, load_w_residuals, mint_w, origin_constraints, project,
                       residual_matrix)
from runlog import RunLog, out_path

S, SEED = 500, 42
H_MAIN = 12
STORE_Q = [0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975]
QNAMES = [f"q{q:g}" for q in STORE_Q]
LEVELS = {80: 0.20, 95: 0.05}                     # nominal coverage -> alpha
GAMMAS = [0.05, 0.01]                             # main, robustness
ALPHA_CLIP = (0.001, 0.999)
VS_P, VS_TOP = 0.5, 20
TRACKS = {
    "FM": {"base": "TimesFM25", "fm_w": "own", "fdir": "base_forecasts_fm", "point_tag": "v2_fmownw",
           "qcols": [f"q{q:g}" for q in np.arange(1, 10) / 10], "p": list(np.arange(1, 10) / 10)},
    "STAT": {"base": "AutoETS", "fm_w": "AutoETS", "fdir": "base_forecasts_v2", "point_tag": "v2_burnin12",
             "qcols": ["lo_95", "lo_80", "lo_50", "mean", "hi_50", "hi_80", "hi_95"],
             "p": [0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975]},
}


def quantile_map(z, zk, Q):
    """Piecewise-linear quantile function in z-space with linear tails.
    z: (S, n) normal scores; zk: (K,) knots; Q: (n, K) sorted values -> (S, n)."""
    slope = np.diff(Q, axis=1) / np.diff(zk)[None, :]                    # (n, K-1)
    j = np.clip(np.searchsorted(zk, z) - 1, 0, len(zk) - 2)              # segment; outer segments extrapolate
    cols = np.arange(Q.shape[0])[None, :]
    return Q[cols, j] + (z - zk[j]) * slope[cols, j]


def crps_fair(X, y):
    """Fair CRPS per column. X: (S, n) samples, y: (n,)."""
    Xs = np.sort(X, axis=0)
    i = np.arange(1, X.shape[0] + 1)[:, None]
    spread = ((2 * i - X.shape[0] - 1) * Xs).sum(axis=0) / (X.shape[0] * (X.shape[0] - 1))
    return np.abs(X - y[None, :]).mean(axis=0) - spread


def energy_score(X, y):
    half = X.shape[0] // 2
    return (np.linalg.norm(X - y[None, :], axis=1).mean()
            - 0.5 * np.linalg.norm(X[:half] - X[half:2 * half], axis=1).mean())


def variogram_score(X, y, p=VS_P):
    total = 0.0
    for i in range(X.shape[1] - 1):
        ey = np.abs(X[:, i:i + 1] - X[:, i + 1:]) ** p
        total += ((np.abs(y[i] - y[i + 1:]) ** p - ey.mean(axis=0)) ** 2).sum()
    return total


def sample_quantile(Xsorted, p):
    """Quantile at level p[i] of column i of the sorted samples (linear interpolation of order statistics)."""
    pos = np.clip(p, 0, 1) * (Xsorted.shape[0] - 1)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, Xsorted.shape[0] - 1)
    c = np.arange(Xsorted.shape[1])
    return Xsorted[lo, c] + (pos - lo) * (Xsorted[hi, c] - Xsorted[lo, c])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", required=True, choices=list(TRACKS))
    ap.add_argument("--mode", default="main", choices=["main", "realtime"])
    ap.add_argument("--clean", default="data/clean")
    ap.add_argument("--results", default="results")
    ap.add_argument("--origins", nargs="+", default=None)
    ap.add_argument("--forecasts-dir", default=None)
    ap.add_argument("--residuals-dir", default=None)
    ap.add_argument("--onestep-dirs", nargs="+", default=None)
    ap.add_argument("--aci-state", default=None, help="realtime: aci_state.parquet of the main run (default: derived)")
    ap.add_argument("--out", default=None, help="default: results/prob_<mode>/track=<track>")
    ap.add_argument("--estimate", action="store_true", help="time two origins, write nothing")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    log = RunLog("prob_samples.py", vars(args))
    RES = Path(args.results)
    T = TRACKS[args.track]
    OUT = Path(args.out) if args.out else RES / f"prob_{args.mode}" / f"track={args.track}"

    sysd = load_system(RES, args.clean)
    uids, n = sysd["uids"], sysd["n"]
    hist = sysd["Yh"]                                                   # actual values per day, to 2026-06
    meta = pd.read_parquet(RES / "base_forecasts_v2" / "series_meta.parquet").set_index("unique_id").loc[uids]
    sc = origin_scales(load_history(args.clean)).pivot(index="origin", columns="unique_id", values="scale1")[uids]
    res, own_fm, w_from = load_w_residuals(RES, T["base"], T["fm_w"], 12, args.residuals_dir, args.onestep_dirs)

    # variogram-score subset: World, Residual, 20 largest system economies (mean RES_TOTAL_SYS per day, 2025)
    rts = hist.loc["2025-01":"2025-12", [u for u in uids if u.endswith("|RES_TOTAL_SYS")]].mean()
    eco = [u for u in rts.index if meta.at[u, "unit_type"] == "economy"]
    top = [u.split("|")[0] for u in rts[eco].sort_values(ascending=False).index[:VS_TOP]]
    vs_units = ["W", "RESIDUAL"] + top
    vs_idx = np.array([i for i, u in enumerate(uids) if u.split("|")[0] in vs_units])

    # base forecasts
    fdir = Path(args.forecasts_dir) if args.forecasts_dir else RES / T["fdir"]
    f = pd.read_parquet(fdir / f"model={T['base']}" / "forecasts.parquet",
                        columns=["unique_id", "origin", "target", "horizon", "mean"] +
                                [c for c in T["qcols"] if c != "mean"])
    f = f[f.unique_id.isin(uids)]
    if args.mode == "main":
        origins = args.origins or pd.period_range("2022-12", "2025-06", freq="M").astype(str).tolist()
        rec_free = pd.read_parquet(RES / f"reconciled_{T['point_tag']}" / f"base={T['base']}" / "reconciled.parquet",
                                   filters=[("method", "==", "MinT"), ("variant", "==", "free")])
    else:
        origins = args.origins or sorted(f.origin.unique())
        state_path = Path(args.aci_state) if args.aci_state else \
            RES / "prob_main" / f"track={args.track}" / "aci_state.parquet"
        aci_state = pd.read_parquet(state_path)
    if args.estimate:
        origins = origins[-2:]
    f = f[f.origin.isin(origins)].set_index(["origin", "unique_id", "horizon"]).sort_index()
    zk = norm.ppf(T["p"])

    q_rows, crps_rows, joint_rows, diag_rows, aci_rows, pts_rows = [], [], [], [], [], []
    alpha = {}                                                            # (level, gamma) -> (n, H) state
    pending = {}                                                          # (origin, h) -> {(level, gamma): (lo, hi)}
    state_rows = []

    def aci_update(t):
        """Score every pending forecast whose target is month t and update alpha (main mode)."""
        if t not in hist.index:
            return
        y = hist.loc[t].to_numpy()
        for h in range(1, H_MAIN + 1):
            o_prev = str(pd.Period(t, "M") - h)
            if (o_prev, h) not in pending:
                continue
            for key, (lo, hi) in pending.pop((o_prev, h)).items():
                err = ((y < lo) | (y > hi)).astype(float)
                lv, g = key
                alpha[key][:, h - 1] = np.clip(alpha[key][:, h - 1] + g * (LEVELS[lv] - err), *ALPHA_CLIP)

    def save_state(label):
        for (lv, g), a in alpha.items():
            for h in range(a.shape[1]):
                state_rows.append(pd.DataFrame({"state_at": label, "unique_id": uids, "horizon": h + 1,
                                                "level": lv, "gamma": g, "alpha": a[:, h]}))

    if args.mode == "main":
        for key in [(lv, g) for lv in LEVELS for g in GAMMAS]:
            alpha[key] = np.full((n, H_MAIN), LEVELS[key[0]])

    t_start = time.perf_counter()
    for o in origins:
        t0 = time.perf_counter()
        A, Acheck, r, dead, _ = origin_constraints(sysd, o)
        Rm, _ = residual_matrix(res, o, own_fm, uids)
        W, lam = mint_w(Rm)
        L = cholesky(W, lower=True)
        sd = np.sqrt(np.diag(W))
        Lc = cholesky(W / np.outer(sd, sd), lower=True)                   # copula correlation R = Lc Lc'
        fo = f.loc[o]
        H = int(fo.index.get_level_values("horizon").max())
        Q = np.stack([fo[c].unstack("horizon").loc[uids].to_numpy() for c in T["qcols"]], axis=2)   # (n, H, K)
        n_sorted = int((np.diff(Q, axis=2) < 0).any(axis=2).sum())
        Q = np.sort(Q, axis=2)
        point = fo["mean"].unstack("horizon").loc[uids].to_numpy()            # base point forecast (n, H)
        targets = [str(pd.Period(o, "M") + h) for h in range(1, H + 1)]
        scale = sc.loc[o].to_numpy()
        ok_sc = np.isfinite(scale) & (scale > 0)

        # ---- check: the free MinT projection reproduces the stored free point reconciliation
        check = np.nan
        if args.mode == "main":
            ref = rec_free[rec_free.origin == o].pivot(index="unique_id", columns="horizon", values="value").loc[uids]
            check = float(np.abs(project(point, A, L, False)[0] - ref.to_numpy()).max())

        # ---- joint samples (common random numbers: e depends on seed and origin only)
        rng = np.random.default_rng([SEED, int(o.replace("-", ""))])
        Xb = np.empty((H, S, n))
        for h in range(H):
            Xb[h] = quantile_map(rng.standard_normal((S, n)) @ Lc.T, zk, Q[:, h, :])
        Xr = project(Xb.transpose(2, 0, 1).reshape(n, H * S), A, L, False)[0].reshape(n, H, S).transpose(1, 2, 0)
        neg = Xr < 0
        share_neg_rec, share_negtol_rec = float(neg.mean()), float((Xr < NEG_TOL).mean())
        Xr[neg] = 0.0
        Cy = np.einsum("rn,hsn->hsr", Acheck, Xr)
        scale_c = np.einsum("rn,hsn->hsr", np.abs(Acheck), Xr)
        diag_rows.append({"origin": o, "track": args.track, "base": T["base"], "w_from": w_from, "mint_lambda": lam,
                          "n_quantile_rows_sorted": n_sorted, "share_neg_base_samples": float((Xb < 0).mean()),
                          "share_neg_rec_samples_set_to_zero": share_neg_rec, "share_below_neg_tol": share_negtol_rec,
                          "max_abs_Cy_after_zeroing": float(np.abs(Cy).max()),
                          "max_rel_Cy_after_zeroing": float((np.abs(Cy) / np.maximum(scale_c, 1.0)).max()),
                          "free_projection_check_max_abs": check})

        # ---- quantiles, scores
        for kind, X in [("base", Xb), ("rec", Xr)]:
            qs = np.quantile(X, STORE_Q, axis=1)                          # (13, H, n)
            for h in range(H):
                d = pd.DataFrame(qs[:, h, :].T, columns=QNAMES)
                d.insert(0, "unique_id", uids)
                d.insert(1, "origin", o)
                d.insert(2, "horizon", h + 1)
                d.insert(3, "target", targets[h])
                d.insert(4, "kind", kind)
                q_rows.append(d)
                t = targets[h]
                if t not in hist.index:
                    continue
                y = hist.loc[t].to_numpy()
                crps_rows.append(pd.DataFrame({"unique_id": uids, "origin": o, "horizon": h + 1, "target": t,
                                               "kind": kind, "crps": crps_fair(X[h], y)}))
                Xs = X[h][:, ok_sc] / scale[ok_sc]
                ys = y[ok_sc] / scale[ok_sc]
                vmask = np.isin(np.flatnonzero(ok_sc), vs_idx)
                joint_rows.append({"origin": o, "horizon": h + 1, "target": t, "kind": kind,
                                   "es": energy_score(Xs, ys), "n_es": int(ok_sc.sum()),
                                   "vs": variogram_score(Xs[:, vmask], ys[vmask]), "n_vs": int(vmask.sum())})
        if args.mode == "realtime":                                       # samples of RES_TOTAL_SYS (RES_TOTAL_PUB scoring)
            ir = [i for i, u in enumerate(uids) if u.endswith("|RES_TOTAL_SYS")]
            for kind, X in [("base", Xb), ("rec", Xr)]:
                v = X[:, :, ir]                                           # (H, S, m)
                pts_rows.append(pd.DataFrame({
                    "unique_id": np.tile(np.array(uids)[ir], H * S), "origin": o,
                    "horizon": np.repeat(np.arange(1, H + 1), S * len(ir)),
                    "sample": np.tile(np.repeat(np.arange(S), len(ir)), H), "kind": kind, "value": v.reshape(-1)}))

        # ---- ACI
        Xsorted = np.sort(Xr, axis=1)
        if args.mode == "main":
            aci_update(o)
            if o == origins[-1]:
                save_state(o)
        for h in range(H):
            per_key = {}
            for lv in LEVELS:
                for g in GAMMAS:
                    if args.mode == "main":
                        a = alpha[(lv, g)][:, h]
                    else:
                        st = "2025-06" if o <= "2025-06" else "2026-06"
                        a_df = aci_state[(aci_state.state_at == st) & (aci_state.level == lv) & (aci_state.gamma == g)
                                         & (aci_state.horizon == min(h + 1, H_MAIN))]
                        a = a_df.set_index("unique_id").alpha.loc[uids].to_numpy()
                    lo, hi = sample_quantile(Xsorted[h], a / 2), sample_quantile(Xsorted[h], 1 - a / 2)
                    per_key[(lv, g)] = (lo, hi)
                    aci_rows.append(pd.DataFrame({"unique_id": uids, "origin": o, "horizon": h + 1,
                                                  "target": targets[h], "level": lv, "gamma": g, "alpha": a,
                                                  "lo": lo, "hi": hi}))
            if args.mode == "main":
                pending[(o, h + 1)] = per_key
        diag_rows[-1]["seconds"] = time.perf_counter() - t0
        print(f"{args.track} {o}: {time.perf_counter() - t0:.1f} s | neg set to 0 {share_neg_rec:.4%} | "
              f"max|Cy| {diag_rows[-1]['max_abs_Cy_after_zeroing']:.3g} | free check {check:.3g}", flush=True)

    if args.estimate:
        per = (time.perf_counter() - t_start) / len(origins)
        print(f"estimate: {per:.1f} s per origin -> 31 origins ~ {per * 31 / 60:.1f} min")
        log.finish(f"estimate {per:.1f} s per origin")
        return

    if args.mode == "main":                    # continue the ACI updates month by month to 2026-06
        for t in pd.period_range(pd.Period(origins[-1], "M") + 1, "2026-06", freq="M").astype(str):
            aci_update(t)
        save_state("2026-06")

    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat(q_rows, ignore_index=True).to_parquet(out_path(OUT / "quantiles.parquet", args.overwrite), index=False)
    if crps_rows:
        pd.concat(crps_rows, ignore_index=True).to_parquet(out_path(OUT / "crps.parquet", args.overwrite), index=False)
    pd.DataFrame(joint_rows).to_csv(out_path(OUT / "joint_scores.csv", args.overwrite), index=False, float_format="%.8g")
    pd.concat(aci_rows, ignore_index=True).to_parquet(out_path(OUT / "aci_intervals.parquet", args.overwrite), index=False)
    pd.DataFrame(diag_rows).to_csv(out_path(OUT / "sampling_diagnostics.csv", args.overwrite), index=False,
                                   float_format="%.6g")
    if state_rows:
        pd.concat(state_rows, ignore_index=True).to_parquet(out_path(OUT / "aci_state.parquet", args.overwrite), index=False)
    if pts_rows:
        pd.concat(pts_rows, ignore_index=True).to_parquet(out_path(OUT / "samples_res_total_sys.parquet", args.overwrite),
                                                          index=False)
    pd.Series({"vs_units": " ".join(vs_units)}).to_csv(out_path(OUT / "vs_subset.csv", args.overwrite))
    log.finish(f"{args.track} {args.mode}: {len(origins)} origins in {time.perf_counter() - t_start:.0f} s")


if __name__ == "__main__":
    main()
