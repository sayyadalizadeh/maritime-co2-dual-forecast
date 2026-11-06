"""
final_samples.py: joint predictive samples at the final origin 2026-06, reconciled.

Horizons 1 ... FINAL_H of the track: FM 12 (the calibrated range of S1), STAT 7 (set from an earlier skill
estimate as H* - 1; the stored STAT samples have 7 horizons). Samples are drawn exactly as
in prob_samples.py: marginals from the base quantiles (TimesFM 2.5 deciles; AutoETS mean and 50/80/95% bounds from
an h = 12 call), piecewise linear in z = Phi^-1(p) with linear tails; Gaussian copula with R = the track's MinT
shrinkage W at 2026-06 scaled to unit diagonal; S joint draws; standard normals from the seed (42, 202606), drawn
horizon by horizon, so horizons 1-3 reproduce the base samples of the real-time run (prob_samples.py, results/prob_realtime; checked).

Reconciliation of every sample:
  FM    exact non-negative QP: min (y - x)' W^-1 (y - x) s.t. A y = 0, y >= 0, with x the base sample,
        solved with Goldfarb-Idnani (reconcile.nonneg_gi: quadprog plus an exact projection on the active set;
        the smallest bound multiplier is recorded as the certificate). Samples whose free MinT projection has no
        value below NEG_TOL need no QP (the projection is the optimum; values in [NEG_TOL, 0) are set to 0).
  STAT  free MinT projection, negatives set to zero (robustness track only).
Checks for every sample: max |C y| (all constraint rows of origin 2026-06) and min y.

Outputs (results/final/track=<FM|STAT>/):
  samples_rec.npy     reconciled samples, array (H, S, n), series order of reconcile.load_system (over 10 MB, not in git)
  samples_base.npy    base samples (H, S, n) (over 10 MB, not in git)
  sample_diagnostics.csv  per (horizon, sample): negatives in the free projection, QP used, zeros, multiplier,
                      max |C y|, min y, seconds
  run_info.csv        settings, run time
Usage:
  python src/final_samples.py --track FM [--S 500] [--n-jobs 16] [--estimate]
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.linalg import cholesky
from scipy.stats import norm
from threadpoolctl import threadpool_limits

from prob_samples import SEED, TRACKS, quantile_map
from reconcile import NEG_TOL, load_system, load_w_residuals, mint_w, nonneg_gi, origin_constraints, project, residual_matrix
from runlog import RunLog, out_path

FINAL_H = {"FM": 12, "STAT": 7}     # horizons of the final samples

ORIGIN = "2026-06"


def qp_chunk(X, Xf, A, L, dead, Acheck):
    """Exact non-negative QP for the columns of X (base samples) whose free projection Xf has a negative value."""
    rows, Y = [], Xf.copy()
    with threadpool_limits(1):
        for j in range(X.shape[1]):
            t0 = time.perf_counter()
            nneg = int((Xf[:, j] < NEG_TOL).sum())
            if nneg:
                y, fixed, lagr, iters, flag = nonneg_gi(X[:, j], A, L, False, dead)
                Y[:, j] = y
                info = {"qp": True, "n_zero": len(fixed), "min_multiplier": lagr, "iterations": iters, "flag": flag}
            else:
                Y[:, j] = np.maximum(Xf[:, j], 0.0)
                info = {"qp": False, "n_zero": 0, "min_multiplier": np.nan, "iterations": 0, "flag": ""}
            rows.append({"n_neg_free": nneg, **info, "max_abs_Cy": float(np.abs(Acheck @ Y[:, j]).max()),
                         "min_y": float(Y[:, j].min()), "seconds": time.perf_counter() - t0})
    return Y, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", required=True, choices=list(TRACKS))
    ap.add_argument("--clean", default="data/clean")
    ap.add_argument("--results", default="results")
    ap.add_argument("--forecasts-dir", default="results/calib/base_forecasts")
    ap.add_argument("--residuals-dir", default="results/calib/insample_residuals")
    ap.add_argument("--onestep-dirs", nargs="+", default=["results/onestep_fm", "results/onestep_fm_rt"])
    ap.add_argument("--S", type=int, default=500)
    ap.add_argument("--n-jobs", type=int, default=16)
    ap.add_argument("--out", default="results/final")
    ap.add_argument("--estimate", action="store_true", help="time the QPs of the first 32 samples of horizon 1")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    log = RunLog("final_samples.py", vars(args))
    RES, T, S = Path(args.results), TRACKS[args.track], args.S
    OUT = Path(args.out) / f"track={args.track}"
    H = FINAL_H[args.track]

    sysd = load_system(RES, args.clean)
    uids, n = sysd["uids"], sysd["n"]
    A, Acheck, r, dead, _ = origin_constraints(sysd, ORIGIN)
    res, own_fm, w_from = load_w_residuals(RES, T["base"], T["fm_w"], 12, args.residuals_dir, args.onestep_dirs)
    Rm, n_months = residual_matrix(res, ORIGIN, own_fm, uids)
    W, lam = mint_w(Rm)
    L = cholesky(W, lower=True)
    sd = np.sqrt(np.diag(W))
    Lc = cholesky(W / np.outer(sd, sd), lower=True)

    f = pd.read_parquet(Path(args.forecasts_dir) / f"model={T['base']}" / "forecasts.parquet",
                        filters=[("origin", "==", ORIGIN)])
    f = f[f.unique_id.isin(uids)].set_index(["unique_id", "horizon"]).sort_index()
    Q = np.stack([f[c].unstack("horizon").loc[uids].to_numpy() for c in T["qcols"]], axis=2)[:, :H, :]
    n_sorted = int((np.diff(Q, axis=2) < 0).any(axis=2).sum())
    Q = np.sort(Q, axis=2)
    zk = norm.ppf(T["p"])
    rng = np.random.default_rng([SEED, int(ORIGIN.replace("-", ""))])
    Xb = np.empty((H, S, n))
    for h in range(H):
        Xb[h] = quantile_map(rng.standard_normal((S, n)) @ Lc.T, zk, Q[:, h, :])
    Xf = project(Xb.transpose(2, 0, 1).reshape(n, H * S), A, L, False)[0].reshape(n, H, S).transpose(1, 2, 0)
    print(f"{args.track}: H = {H}, S = {S}, W from {w_from} ({n_months} months, lambda {lam:.3f}); "
          f"samples with a negative free value: {(Xf < NEG_TOL).any(axis=2).mean():.1%}", flush=True)

    t_start = time.perf_counter()
    if args.track == "STAT":                                   # projection, negatives set to zero
        Xr = np.maximum(Xf, 0.0)
        diag = [{"horizon": h + 1, "sample": s, "n_neg_free": int((Xf[h, s] < NEG_TOL).sum()), "qp": False,
                 "max_abs_Cy": float(np.abs(Acheck @ Xr[h, s]).max()), "min_y": float(Xr[h, s].min())}
                for h in range(H) for s in range(S)]
    else:
        if args.estimate:
            m = 32
            t0 = time.perf_counter()
            out = Parallel(n_jobs=args.n_jobs)(delayed(qp_chunk)(Xb[0, j:j + 2].T, Xf[0, j:j + 2].T, A, L, dead, Acheck)
                                               for j in range(0, m, 2))
            secs = time.perf_counter() - t0
            per = np.mean([r_["seconds"] for _, rows in out for r_ in rows])
            n_qp = sum(r_["qp"] for _, rows in out for r_ in rows)
            print(f"estimate: {m} samples ({n_qp} needed the QP) in {secs:.1f} s wall with {args.n_jobs} jobs; "
                  f"{per:.2f} s per sample on one core -> {H * S} samples ~ {secs / m * H * S / 3600:.2f} h")
            log.finish(f"estimate {secs / m * H * S / 3600:.2f} h")
            return
        cols = [(h, s) for h in range(H) for s in range(S)]
        chunks = [cols[i:i + 25] for i in range(0, len(cols), 25)]
        out = Parallel(n_jobs=args.n_jobs, verbose=5)(
            delayed(qp_chunk)(np.stack([Xb[h, s] for h, s in c], axis=1), np.stack([Xf[h, s] for h, s in c], axis=1),
                              A, L, dead, Acheck) for c in chunks)
        Xr = np.empty_like(Xb)
        diag = []
        for c, (Y, rows) in zip(chunks, out):
            for k, (h, s) in enumerate(c):
                Xr[h, s] = Y[:, k]
                diag.append({"horizon": h + 1, "sample": s, **rows[k]})
    secs = time.perf_counter() - t_start
    diag = pd.DataFrame(diag)

    # ---- check: base samples of horizons 1-3 against the real-time run (RES_TOTAL_SYS series)
    old = pd.read_parquet(RES / "prob_realtime" / f"track={args.track}" / "samples_res_total_sys.parquet",
                          filters=[("origin", "==", ORIGIN), ("kind", "==", "base")])
    ir = {u: i for i, u in enumerate(uids)}
    old_v = old.value.to_numpy()
    new_v = Xb[old.horizon.to_numpy() - 1, old["sample"].to_numpy(), old.unique_id.map(ir).to_numpy()] \
        if S == 500 else np.full(len(old), np.nan)
    chk = float(np.abs(new_v - old_v).max())
    print(f"check base samples h1-3 vs the real-time run ({len(old)} values): max abs diff {chk:.3g}")
    print(diag.describe().T.to_string())

    OUT.mkdir(parents=True, exist_ok=True)
    np.save(out_path(OUT / "samples_rec.npy", args.overwrite), Xr)
    np.save(out_path(OUT / "samples_base.npy", args.overwrite), Xb)
    diag.to_csv(out_path(OUT / "sample_diagnostics.csv", args.overwrite), index=False, float_format="%.6g")
    pd.DataFrame([{"track": args.track, "origin": ORIGIN, "H": H, "S": S, "n_series": n, "w_from": w_from,
                   "w_months": n_months, "mint_lambda": lam, "r": r, "n_dead": len(dead), "n_quantile_rows_sorted": n_sorted,
                   "n_jobs": args.n_jobs, "seconds": secs, "check_base_h1_3_max_abs": chk,
                   "share_samples_qp": float(diag.qp.mean()), "max_abs_Cy": float(diag.max_abs_Cy.max()),
                   "min_y": float(diag.min_y.min())}]).to_csv(out_path(OUT / "run_info.csv", args.overwrite), index=False)
    log.finish(f"{args.track}: {H * S} samples in {secs / 60:.1f} min; max |Cy| {diag.max_abs_Cy.max():.2e}")


if __name__ == "__main__":
    main()
