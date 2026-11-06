"""
w_diagnostics.py: diagnostics of the MinT covariance W.

For both tracks (FM = TimesFM 2.5 with its own one-step errors; STAT = AutoETS in-sample residuals with the
12-month burn-in) W is rebuilt at every main origin 2022-12 ... 2025-06 with the functions of reconcile.py (residual
matrix, Schafer-Strimmer shrinkage towards the diagonal, variance floor VAR_FLOOR on the diagonal), exactly as in the
main reconciliation. The rebuilt lambda is checked against results/reconcile_<tag>/w_diagnostics_<base>.csv.

(1)   lambda; condition numbers (2-norm, ratio of extreme eigenvalues) of W and of A W A', where A is the origin's
      constraint matrix with independent rows (full-rank C with the DELTA_INT ratio and the dead-series rows); also of
      both matrices rescaled to unit diagonal (scale-free), and the ratio of the largest to the smallest variance.
(2)   Zero-variance series: per origin the number of series whose mean squared residual is below VAR_FLOOR (their W
      diagonal is raised to the floor; their correlations are zero, so W keeps them as near-fixed free variables), the
      number of dead series (fixed by an equality row y_i = 0 and given no bound in the QP), and their overlap.
(3)   Correlation structure: the one-step shrinkage correlation at origin 2025-06 (W scaled to unit diagonal) against
      the correlation of the rolling-origin h-step base forecast errors (h = 3, 6, 12) over the main window (targets
      up to 2025-12 without 2025-09, origins 2022-12 ... 2025-06; second moments uncentered, as in W). Series with
      zero error variance are excluded. Reported: correlation between the off-diagonal elements and the relative
      Frobenius distance ||R_h - R_1||_F / ||R_1||_F over the off-diagonal elements; also against the unshrunk
      one-step correlation, and with centered h-step correlations.
(4)   Dead series at any origin that become non-zero within the following 12 months (observed values).

Outputs: results/w_diagnostics/
Usage:
  python src/w_diagnostics.py
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from evalcore import WINDOWS
from reconcile import (VAR_FLOOR, load_system, load_w_residuals, mint_w, origin_constraints, residual_matrix,
                       shrink_cov)
from runlog import RunLog, out_path

TRACKS = {"FM": ("TimesFM25", "own", "v2_fmownw", "base_forecasts_fm"),
          "STAT": ("AutoETS", "AutoETS", "v2_burnin12", "base_forecasts_v2")}
ORIGINS = pd.period_range("2022-12", "2025-06", freq="M").astype(str).tolist()
CORR_ORIGIN = "2025-06"


def cond_sym(M):
    ev = np.linalg.eigvalsh(M)
    return float(ev.max() / ev.min()) if ev.min() > 0 else np.inf, float(ev.min()), float(ev.max())


def to_corr(M):
    d = np.sqrt(np.diag(M))
    ok = d > 0
    R = np.zeros_like(M)
    R[np.ix_(ok, ok)] = M[np.ix_(ok, ok)] / np.outer(d[ok], d[ok])
    return R


def offdiag_compare(R1, R2, keep):
    iu = np.triu_indices(int(keep.sum()), k=1)
    a, b = R1[np.ix_(keep, keep)][iu], R2[np.ix_(keep, keep)][iu]
    return float(np.corrcoef(a, b)[0, 1]), float(np.linalg.norm(b - a) / np.linalg.norm(a)), len(a)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean", default="data/clean")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/w_diagnostics")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    log = RunLog("w_diagnostics.py", vars(args))
    RES, OUT = Path(args.results), Path(args.out)

    def save(df, name):
        df.to_csv(out_path(OUT / name, args.overwrite), index=False, float_format="%.6g")

    sysd = load_system(RES, args.clean)
    uids, hist = sysd["uids"], sysd["hist"]
    cons = {o: origin_constraints(sysd, o) for o in ORIGINS}

    rows, zrows, crows = [], [], []
    for tr, (base, fm_w, tag, fdir) in TRACKS.items():
        res, own, _ = load_w_residuals(RES, base, fm_w, 12)
        wd = pd.read_csv(RES / f"reconcile_{tag}" / f"w_diagnostics_{base}.csv")
        wd = wd[wd.method == "MinT"].set_index("origin")
        W_corr = None
        for o in ORIGINS:
            A, _, _, dead, _ = cons[o]
            Rm, _ = residual_matrix(res, o, own, uids)
            var = (Rm ** 2).mean(axis=0)
            W, lam = mint_w(Rm)
            assert abs(lam - wd.at[o, "mint_lambda"]) <= 1e-5 * max(1.0, abs(lam)), (tr, o, lam)
            cW, wmin, wmax = cond_sym(W)
            cA, amin, amax = cond_sym(A @ W @ A.T)
            cWc = cond_sym(to_corr(W))[0]                              # scale-free: W rescaled to unit diagonal
            AWA = A @ W @ A.T
            cAc = cond_sym(to_corr(AWA))[0]
            rows.append({"track": tr, "base": base, "origin": o, "resid_months": Rm.shape[0], "lambda": lam,
                         "cond_W": cW, "eig_min_W": wmin, "eig_max_W": wmax, "cond_AWAt": cA, "eig_min_AWAt": amin,
                         "eig_max_AWAt": amax, "cond_W_unit_diag": cWc, "cond_AWAt_unit_diag": cAc,
                         "rows_A": A.shape[0], "max_var_over_min_var": float(np.diag(W).max() / np.diag(W).min())})
            floored = set(np.flatnonzero(var < VAR_FLOOR))
            zero = set(np.flatnonzero(var == 0))
            dset = set(int(i) for i in dead)
            zrows.append({"track": tr, "origin": o, "n_var_below_floor": len(floored), "n_var_exactly_zero": len(zero),
                          "n_dead": len(dset), "n_floored_and_dead": len(floored & dset),
                          "n_floored_not_dead": len(floored - dset), "n_dead_not_floored": len(dset - floored),
                          "floored_not_dead": " ".join(uids[i] for i in sorted(floored - dset))})
            if o == CORR_ORIGIN:
                S, _ = shrink_cov(Rm)
                W_corr = (to_corr(W), to_corr(Rm.T @ Rm / Rm.shape[0]), var)
        # ---- rolling-origin h-step errors of the base forecasts, main window
        f = pd.read_parquet(RES / fdir / f"model={base}" / "forecasts.parquet",
                            columns=["unique_id", "origin", "target", "horizon", "y", "mean"])
        f = f[f.unique_id.isin(uids) & f.origin.isin(ORIGINS) & WINDOWS["main"](f.target)]
        R1s, R1raw, var1 = W_corr
        for h in (3, 6, 12):
            e = f[f.horizon == h].assign(err=lambda d: d.y - d["mean"]).pivot(index="origin", columns="unique_id",
                                                                              values="err")[uids]
            E = e.dropna().to_numpy()
            for centered in (False, True):
                Ec = E - E.mean(axis=0) if centered else E
                M = Ec.T @ Ec / Ec.shape[0]
                Rh = to_corr(M)
                keep = (np.diag(M) > 0) & (var1 > 0)
                for ref, R1 in [("one-step shrinkage correlation (W)", R1s), ("one-step raw correlation", R1raw)]:
                    c, frob, npair = offdiag_compare(R1, Rh, keep)
                    crows.append({"track": tr, "h": h, "h_step_moments": "centered" if centered else "uncentered",
                                  "reference": ref, "n_origins": E.shape[0], "n_series": int(keep.sum()),
                                  "n_pairs": npair, "corr_offdiag": c, "rel_frobenius_offdiag": frob,
                                  "mean_abs_offdiag_ref": float(np.abs(R1[np.ix_(keep, keep)][np.triu_indices(int(keep.sum()), 1)]).mean()),
                                  "mean_abs_offdiag_h": float(np.abs(Rh[np.ix_(keep, keep)][np.triu_indices(int(keep.sum()), 1)]).mean())})
        print(f"{tr}: done", flush=True)
    diag = pd.DataFrame(rows)
    save(diag, "w_condition_by_origin.csv")
    save(pd.DataFrame(zrows), "zero_variance_series.csv")
    corr = pd.DataFrame(crows)
    save(corr, "correlation_similarity.csv")

    # ---- (4) dead series that revive within 12 months
    drows = []
    for o in ORIGINS:
        _, _, _, dead, _ = cons[o]
        pos = hist.index.get_loc(o)
        nxt = hist.iloc[pos + 1: pos + 13]
        for i in dead:
            u = uids[i]
            nz = nxt[u][nxt[u] != 0]
            drows.append({"origin": o, "unique_id": u, "nonzero_within_12m": len(nz) > 0,
                          "first_nonzero": nz.index[0] if len(nz) else "", "n_nonzero_months": len(nz),
                          "max_per_day_t": float(nxt[u].max())})
    dd = pd.DataFrame(drows)
    save(dd, "dead_series_revival.csv")
    summ = pd.DataFrame([{"origins": len(ORIGINS), "dead_cells": len(dd), "distinct_dead_series": dd.unique_id.nunique(),
                          "cells_nonzero_within_12m": int(dd.nonzero_within_12m.sum()),
                          "distinct_series_nonzero_within_12m": dd[dd.nonzero_within_12m].unique_id.nunique(),
                          "series_nonzero": " ".join(sorted(dd[dd.nonzero_within_12m].unique_id.unique()))}])
    save(summ, "dead_series_revival_summary.csv")
    print(diag.groupby("track")[["lambda", "cond_W", "cond_AWAt", "cond_W_unit_diag", "cond_AWAt_unit_diag", "max_var_over_min_var"]].describe().T.to_string())
    print(corr[corr.h_step_moments == "uncentered"].to_string(index=False))
    print(summ.to_string(index=False))
    log.finish("W diagnostics")


if __name__ == "__main__":
    main()
