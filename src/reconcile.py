"""
reconcile.py: dual-accounting reconciliation of the base point forecasts.

For every base model, origin and horizon the base forecast vector y_hat of the system series
(112 units x 10 measures; DELTA_INT excluded, see below) is projected onto the constraints:

    y_tilde = y_hat - W A' (A W A')^-1 A y_hat

computed through a QR factorisation of B' = L' A' (W = L L'), never with an explicit inverse:
y_tilde = y_hat - L Q R'^-1 A y_hat. A is built per origin from the full-rank v2 matrix C:
  - the DELTA_INT column is dropped and the world-balance row becomes
        E(W) + F(W) - (1 + r) G(W) = 0, with r = mean of DELTA_INT / G(W) over the 12 months
        ending at the origin (the last 12 observed months);
  - one row y_i = 0 for each dead series (zero in all 24 months ending at the origin).
Rows made linearly dependent by these additions are removed with a pivoted QR of A' (A holds
only 0, +-1 and 1 + r, so the rank decision does not depend on W).

W (estimated from the in-sample one-step residuals of the base model at that origin; months
in which any series lacks a residual are dropped; second moments are uncentered):
  - burn-in (option --burn-in, default 12): the first 12 months of every fit
    (2019-01 to 2019-12, the expanding window starts in 2019-01) are excluded. --burn-in 0 gives
    the version without burn-in (tag v2).
  - foundation-model bases (Chronos-Bolt, TimesFM, TimesFM25) have no in-sample residuals;
    with --fm-w AutoETS their W is the AutoETS W of the same origin (shared-W design).
  - --fm-w own: a foundation model's W is estimated from its own one-step errors
    (results/onestep_fm, made by fm_onestep.py), months 2020-01 up to and including the origin.
    The one-step errors start in 2020-01, so the burn-in does not apply to them.
  OLS   W = I
  WLS   W = diag(mean resid^2), each variance raised to at least VAR_FLOOR
  MinT  shrinkage covariance (Schafer-Strimmer intensity, target diag), diagonal floored as WLS

Non-negativity (every method):
  nn_qp    primary. Exact QP  min (y - y_hat)' W^-1 (y - y_hat)  s.t.  A y = 0, y >= 0 (unique optimum,
           W positive definite). Solved in whitened variables z = L^-1 (y - y_hat), i.e.
           min 1/2 z'z  s.t.  A L z = -A y_hat,  L z >= -y_hat, with the Goldfarb-Idnani dual
           active-set method (quadprog), which starts from the free projection and adds violated
           bounds; its Lagrange multipliers (all >= 0 at the optimum) are recorded as a certificate.
           The series in its active set are then fixed exactly (y_i = 0 rows) and the projection
           recomputed, so that coherence holds to machine precision (if that leaves a negative
           value, it is added and the projection repeated; flagged).
           Cross-check: OSQP solves the same QP on a sample of cells (OSQP_CHECK, every
           method); the largest difference and both objectives are recorded. OSQP is not the
           primary solver because its solution polishing failed on dense MinT problems and it
           needed 20,000+ iterations (one to several minutes) per MinT problem.
  nn_iter  check. Fix negative values at zero, re-project y_hat, repeat until no new negative.
  Both are computed only for horizons where the free projection has a negative value; otherwise
  the free projection is already the QP solution.

DELTA_INT is reported as r x reconciled G(W). Outside-system series are not reconciled.

Outputs
  (TAG = --tag: "v2" without burn-in, "v2_burnin12" with it)
  results/reconciled_TAG/base=<model>/reconciled.parquet  (over 10 MB: not in git; in the archive)
      unique_id, origin, target, horizon, method (OLS/WLS/MinT), variant (free/nn_qp/nn_iter), value
  results/reconcile_TAG/origin_info.csv    r, number of dead series, constraint counts per origin
                                           (written by the run that includes AutoETS)
  results/reconcile_TAG/w_diagnostics_<base>.csv  per origin/method: residual months, floored
                                           variances, MinT shrinkage intensity, condition of R
  results/reconcile_TAG/coherence_<base>.csv  max |A y|, max relative |A y|, min y per method/variant/origin/horizon
  results/reconcile_TAG/nonneg_<base>.csv  negatives before the step, QP vs iterative differences, run times

Base models are independent; they can be run as parallel processes, one base each.

Usage:
  python src/reconcile.py [--bases AutoETS AutoTheta AutoARIMA SeasonalNaive] [--overwrite]
  python src/reconcile.py --bases TimesFM --fm-w own --tag v2_fmownw          # own one-step W
"""
import argparse
import time
from pathlib import Path

import numpy as np
import osqp
import pandas as pd
import scipy.sparse as sp
import quadprog
from scipy.linalg import cholesky, qr, solve_triangular

from runlog import RunLog, out_path

WORLD, DELTA = "W", "W|DELTA_INT"
R_WINDOW, DEAD_WINDOW = 12, 24          # months ending at the origin (DELTA_INT ratio, dead series)
VAR_FLOOR = 1e-4                        # (t CO2 per day)^2: residual variance floor (WLS, MinT diagonal)
NEG_TOL = -1e-6                         # values below this (t/day) count as negative
DUAL_TOL = 1e-8                         # OSQP dual below -DUAL_TOL: y >= 0 constraint active
RANK_TOL = 1e-10                        # relative tolerance for dependent constraint rows
METHODS = ["OLS", "WLS", "MinT"]
OSQP_CHECK = {("2024-06", 1), ("2024-06", 12)}     # cells also solved with OSQP (cross-check)
FM_BASES = {"ChronosBolt", "TimesFM", "TimesFM25"}   # foundation models: forecasts in base_forecasts_fm
W_FROM_FM = "AutoETS"                               # --fm-w AutoETS: W from AutoETS
ONESTEP_START = "2020-01"                           # --fm-w own: first month of the one-step errors


# ---------------------------------------------------------------- linear algebra
def independent_rows(A):
    """Indices of a maximal set of linearly independent rows of the dense matrix A."""
    _, r, piv = qr(A.T, mode="economic", pivoting=True)
    d = np.abs(np.diag(r))
    rank = int((d > RANK_TOL * d[0]).sum())
    return np.sort(piv[:rank])


def project(Yhat, A, L, diag):
    """W-projection of the columns of Yhat onto {y: A y = 0}; W = L L' (L a vector if diag).
    Returns (Ytilde, condition number of R)."""
    Bt = L[:, None] * A.T if diag else L.T @ A.T
    Q, R = qr(Bt, mode="economic")
    z = solve_triangular(R, A @ Yhat, trans="T")
    corr = Q @ z
    corr = L[:, None] * corr if diag else L @ corr
    dR = np.abs(np.diag(R))
    return Yhat - corr, dR.max() / dR.min()


def with_zero_rows(A, idx, n):
    """Append rows y_i = 0 for i in idx and drop dependent rows."""
    if len(idx) == 0:
        return A
    E = np.zeros((len(idx), n))
    E[np.arange(len(idx)), list(idx)] = 1.0
    Aa = np.vstack([A, E])
    return Aa[independent_rows(Aa)]


def nonneg_iter(yhat, A, L, diag, first, max_rounds=1000):
    """Iterative set-negative-to-zero; `first` is the free projection of yhat. Only negatives
    outside the fixed set count: a fixed series can come out of the projection at a tiny
    negative value (rounding), and it is set to exactly zero at the end."""
    n, y, fixed, rounds = len(yhat), first, set(), 0
    while rounds < max_rounds:
        neg = set(np.flatnonzero(y < NEG_TOL)) - fixed
        if not neg:
            break
        fixed |= neg
        rounds += 1
        y = project(yhat[:, None], with_zero_rows(A, sorted(fixed), n), L, diag)[0][:, 0]
    y = np.where(np.isin(np.arange(n), list(fixed)), 0.0, y)     # exact zeros for fixed series
    return y, rounds, len(fixed)


def nonneg_qp(yhat, A, L, diag):
    """Cross-check: the same QP with OSQP in whitened variables, then exact re-projection on the
    active set taken from the OSQP duals. Returns (y, status, zero set, shift, objective of y,
    objective of the raw OSQP solution)."""
    n = len(yhat)
    Lm = sp.diags(L) if diag else sp.csc_matrix(L)
    AL = sp.csc_matrix(A) @ Lm
    rhs = -(A @ yhat)
    Cmat = sp.vstack([AL, Lm]).tocsc()
    lo = np.concatenate([rhs, -yhat])
    hi = np.concatenate([rhs, np.full(n, np.inf)])
    prob = osqp.OSQP()
    prob.setup(P=sp.identity(n, format="csc"), q=np.zeros(n), A=Cmat, l=lo, u=hi, verbose=False,
               eps_abs=1e-9, eps_rel=1e-9, max_iter=200000, polish=True)
    res = prob.solve()
    status = res.info.status
    y_qp = yhat + (L * res.x if diag else L @ res.x)
    fixed = set(np.flatnonzero(res.y[A.shape[0]:] < -DUAL_TOL))     # active set from the duals
    while True:                                   # exact projection on the active set
        y = project(yhat[:, None], with_zero_rows(A, sorted(fixed), n), L, diag)[0][:, 0]
        neg = set(np.flatnonzero(y < NEG_TOL)) - fixed
        if not neg:
            break
        fixed |= neg
    y = np.where(np.isin(np.arange(n), list(fixed)), 0.0, y)
    return (y, status, fixed, float(np.abs(y - y_qp).max()), objective(y, yhat, L, diag),
            objective(y_qp, yhat, L, diag))


def objective(y, yhat, L, diag):
    """(y - yhat)' W^-1 (y - yhat) with W = L L'."""
    z = (y - yhat) / L if diag else solve_triangular(L, y - yhat, lower=True)
    return float(z @ z)


def nonneg_gi(yhat, A, L, diag, dead):
    """Exact non-negative W-projection with the Goldfarb-Idnani dual active-set method (quadprog)
    in whitened variables, followed by an exact projection on the active set. Dead series are
    fixed by equality rows of A, so they get no bound (quadprog needs independent active rows).
    Returns (y, zero set, min multiplier of the bounds, quadprog iterations, flag)."""
    n, k = len(yhat), A.shape[0]
    bnd = np.setdiff1d(np.arange(n), list(dead))          # series with a y >= 0 bound
    AL = A * L[None, :] if diag else A @ L
    Lb = (np.diag(L) if diag else L)[bnd]
    Cm = np.vstack([AL, Lb]).T                            # quadprog: constraints are columns
    b = np.concatenate([-(A @ yhat), -yhat[bnd]])
    z, _, _, iters, lagr, iact = quadprog.solve_qp(np.eye(n), np.zeros(n), Cm, b, meq=k)
    fixed = set(int(bnd[i - k - 1]) for i in iact if i > k)    # iact: 1-based constraint indices
    flag = ""
    while True:                                          # exact projection on the active set
        y = project(yhat[:, None], with_zero_rows(A, sorted(fixed), n), L, diag)[0][:, 0]
        neg = set(np.flatnonzero(y < NEG_TOL)) - fixed
        if not neg:
            break
        fixed |= neg
        flag = "negatives added after quadprog"
    y = np.where(np.isin(np.arange(n), list(fixed)), 0.0, y)
    return y, fixed, float(lagr[k:].min()), int(iters[0]), flag


def nullspace_setup(A, L, dead):
    """Set-up of nonneg_nullspace for one constraint set (A, W = L L', dead series): (M, bounded series, quadprog C).
    In whitened variables u = L^-1 y the feasible set {A y = 0} is {u: A L u = 0}; with Z an orthonormal basis of the
    null space of A L (full QR of (A L)'), every coherent y is y_f + M z with M = L Z, where y_f is the free projection,
    and (y - y_hat)' W^-1 (y - y_hat) = |z|^2 + const."""
    n, k = A.shape[1], A.shape[0]
    Q, _ = qr((A @ L).T, mode="full")
    M = L @ Q[:, k:]
    bnd = np.setdiff1d(np.arange(n), list(dead))
    return M, bnd, np.ascontiguousarray(M[bnd].T)


def nonneg_nullspace(yf, M, bnd, Cb):
    """The exact non-negative QP of nonneg_gi, solved in the null space of the equality constraints.
    yf: the free projection of y_hat; (M, bnd, Cb) from nullspace_setup. The QP min 1/2 z'z s.t. (M z)_i >= -yf_i
    (i bounded) has only inequality constraints and starts at z = 0 (the free projection), so the Goldfarb-Idnani
    method (quadprog) needs about as many iterations as there are active bounds, instead of one per equality row as
    well. The optimum is the same unique point. As in nonneg_gi, the active set is then fixed exactly: z = B' lam with
    B = M[active], B B' lam = -yf[active] (lam: the multipliers of the active bounds, all >= 0 at the optimum); a
    negative value left elsewhere is added to the set and the step repeated (flagged).
    Returns (y, zero set, min multiplier of the active bounds, quadprog iterations, flag); nonneg_gi returns the same
    except that its multiplier is the minimum over all bounds (0 when a bound is inactive)."""
    m = M.shape[1]
    _, _, _, iters, _, iact = quadprog.solve_qp(np.eye(m), np.zeros(m), Cb, -yf[bnd], 0, True)
    fixed = set(int(bnd[i - 1]) for i in iact if i > 0)
    y, fixed, lam_min, flag = nullspace_fix(yf, M, fixed, "negatives added after quadprog")
    return y, fixed, lam_min, int(iters[0]), flag


def nullspace_fix(yf, M, fixed, flag_text):
    """The exact step of nonneg_nullspace for a given active set: z = B' lam with B = M[fixed], B B' lam = -yf[fixed];
    a negative value left elsewhere is added to the set and the step repeated (flag_text). Returns (y, zero set,
    min multiplier of the active bounds, flag)."""
    flag = ""
    while True:
        F = np.array(sorted(fixed), dtype=int)
        B = M[F]
        lam = np.linalg.solve(B @ B.T, -yf[F]) if len(F) else np.zeros(0)
        y = yf + M @ (B.T @ lam) if len(F) else yf.copy()
        neg = set(np.flatnonzero(y < NEG_TOL)) - fixed
        if not neg:
            break
        fixed |= neg
        flag = flag_text
    y[F] = 0.0
    return y, fixed, float(lam.min()) if len(F) else np.nan, flag


def nonneg_nullspace_osqp(yf, M, bnd, eps=1e-9, lam_tol=1e-9, max_steps=200):
    """Fallback for nonneg_nullspace: the same null-space QP, min 1/2 z'z s.t. (M z)_i >= -yf_i (i bounded),
    solved with OSQP at tight tolerances (eps_abs = eps_rel = eps, polishing on). The OSQP solution is then made exact
    where possible: an active set read from the duals is fixed exactly (z = B' lam, as in nullspace_fix) and refined
    (a value below NEG_TOL outside the set is added, otherwise the bound with the most negative multiplier dropped; at
    most max_steps steps) until it is a KKT point (no negative value, every multiplier >= -lam_tol), i.e. the unique
    optimum. Degenerate bounds (y = 0 with a multiplier near 0) can make the dual-based set too large, so three dual
    thresholds are tried in turn (DUAL_TOL, then 1e-6 and 1e-4 of the largest dual). If none gives a KKT point, the
    polished OSQP solution itself is returned, with its values below zero (round-off within eps) set to zero (flagged).
    Returns (y, zero set, min multiplier of the active bounds, OSQP iterations, flag, OSQP status)."""
    m = M.shape[1]
    prob = osqp.OSQP()
    prob.setup(P=sp.identity(m, format="csc"), q=np.zeros(m), A=sp.csc_matrix(M[bnd]), l=-yf[bnd],
               u=np.full(len(bnd), np.inf), verbose=False, eps_abs=eps, eps_rel=eps, max_iter=400000, polish=True)
    res = prob.solve()
    dmax = float(np.abs(res.y).max()) if len(res.y) else 0.0
    for k, thr in enumerate([DUAL_TOL, 1e-6 * dmax, 1e-4 * dmax]):
        fixed = set(int(i) for i in bnd[res.y < -thr])
        added = dropped = 0
        for _ in range(max_steps):
            F = np.array(sorted(fixed), dtype=int)
            B = M[F]
            lam = np.linalg.solve(B @ B.T, -yf[F]) if len(F) else np.zeros(0)
            y = yf + M @ (B.T @ lam) if len(F) else yf.copy()
            neg = set(np.flatnonzero(y < NEG_TOL)) - fixed
            if neg:
                fixed |= neg
                added += len(neg)
            elif len(F) and lam.min() < -lam_tol:
                fixed.discard(int(F[np.argmin(lam)]))
                dropped += 1
            else:
                y[F] = 0.0
                flag = f"OSQP active set (dual threshold {k + 1}) refined: {added} added, {dropped} dropped"                     if k or added or dropped else ""
                return y, fixed, float(lam.min()) if len(F) else np.nan, int(res.info.iter), flag, res.info.status
    y = np.maximum(yf + M @ res.x, 0.0)
    fixed = set(int(i) for i in np.flatnonzero(y == 0.0))
    return (y, fixed, np.nan, int(res.info.iter), "no exact active set found: polished OSQP solution, values < 0 set to 0",
            res.info.status)


# ---------------------------------------------------------------- covariance estimates
def shrink_cov(R):
    """Schafer-Strimmer shrinkage of the (uncentered) residual covariance towards its diagonal,
    as in the MinT-shrink estimator. Returns (W, lambda)."""
    n = R.shape[0]
    S = R.T @ R / n
    sd = np.sqrt(np.diag(S))
    ok = sd > 0
    Xs = np.zeros_like(R)
    Xs[:, ok] = R[:, ok] / sd[ok]
    corm = Xs.T @ Xs / n
    v = (1.0 / (n * (n - 1))) * ((Xs ** 2).T @ (Xs ** 2) - (1.0 / n) * (Xs.T @ Xs) ** 2)
    np.fill_diagonal(v, 0.0)
    off = corm.copy()
    np.fill_diagonal(off, 0.0)
    lam = float(np.clip(v.sum() / (off ** 2).sum(), 0.0, 1.0))
    W = (1 - lam) * S
    np.fill_diagonal(W, np.diag(S))
    return W, lam


# ---------------------------------------------------------------- shared set-up (also used by prob_samples.py)
def load_system(results, clean):
    """Full-rank constraint matrix without the DELTA_INT column and the wide history."""
    RES = Path(results)
    C = sp.load_npz(RES / "C_matrix_fullrank_v2.npz").toarray()
    cols = pd.read_csv(RES / "C_matrix_cols_v2.csv")
    rows = pd.read_csv(RES / "C_matrix_rows_v2.csv")
    rows = rows[~rows.redundant].reset_index(drop=True)          # rows of the full-rank matrix
    assert len(rows) == C.shape[0]
    uids_all = (cols.unit + "|" + cols.measure).tolist()
    keep = [i for i, u in enumerate(uids_all) if u != DELTA]
    uids = [uids_all[i] for i in keep]
    ef_row = int(rows.index[rows.type == "world_balance_EF_G"][0])
    assert np.allclose(C[ef_row, uids_all.index(DELTA)], -1.0)
    hist = pd.read_parquet(Path(clean) / "panel_system_wide_v2.parquet")
    return {"uids": uids, "n": len(uids), "C0": C[:, keep], "ef_row": ef_row,
            "g_col": uids.index(f"{WORLD}|NRES_INT_FROM"), "Yh": hist[uids], "hist": hist,
            "ratio": hist[DELTA] / hist[f"{WORLD}|NRES_INT_FROM"]}


def origin_constraints(sysd, o):
    """Constraints at origin o: (A with independent rows, all rows for checks, r (DELTA_INT ratio), dead series, info)."""
    n, hist = sysd["n"], sysd["hist"]
    pos = hist.index.get_loc(o)
    r = float(sysd["ratio"].iloc[pos - R_WINDOW + 1: pos + 1].mean())
    dead = np.flatnonzero((sysd["Yh"].iloc[pos - DEAD_WINDOW + 1: pos + 1] == 0).all().to_numpy())
    Co = sysd["C0"].copy()
    Co[sysd["ef_row"], sysd["g_col"]] = -(1.0 + r)
    A = with_zero_rows(Co, dead, n)
    E = np.zeros((len(dead), n))
    E[np.arange(len(dead)), dead] = 1.0
    info = {"origin": o, "r": r, "n_dead": len(dead), "dead_series": " ".join(sysd["uids"][i] for i in dead),
            "rows_C": Co.shape[0], "rows_after_dead": Co.shape[0] + len(dead), "rows_independent": A.shape[0]}
    return A, np.vstack([Co, E]), r, dead, info


def load_w_residuals(results, base, fm_w="AutoETS", burn_in=12, residuals_dir=None, onestep_dirs=None):
    """Residuals that W is estimated from: (frame, own_fm, w_from label). See the module docstring."""
    RES = Path(results)
    if base in FM_BASES and fm_w == "own":                               # own one-step errors
        dirs = onestep_dirs or [RES / "onestep_fm"]
        res = pd.concat([pd.read_parquet(Path(d) / f"model={base}" / "onestep.parquet",
                                         columns=["unique_id", "period", "context_end", "error"]) for d in dirs],
                        ignore_index=True)
        assert not res.duplicated(["unique_id", "period"]).any()
        assert (pd.PeriodIndex(res.context_end, freq="M") + 1 == pd.PeriodIndex(res.period, freq="M")).all()
        return res[res.period >= ONESTEP_START].rename(columns={"error": "resid"}), True, "own_onestep"
    w_base = W_FROM_FM if base in FM_BASES else base                    # AutoETS W for FM bases
    res = pd.read_parquet(Path(residuals_dir or RES / "insample_residuals_v2") / f"model={w_base}" / "residuals.parquet")
    first = pd.Period("2019-01", "M") + burn_in                          # burn-in: first month kept
    return res[pd.PeriodIndex(res.period, freq="M") >= first], False, w_base


def residual_matrix(res, o, own_fm, uids):
    """Months x series residual matrix used for W at origin o (months with any missing value dropped)."""
    ro = res[res.period <= o] if own_fm else res[res.origin == o]        # own one-step errors of 2020-01 ... o
    Rm = ro.pivot(index="period", columns="unique_id", values="resid")[uids]
    return Rm.dropna().to_numpy(), len(Rm)


def mint_w(Rm):
    """MinT-shrink W with the variance floor on the diagonal. Returns (W, lambda)."""
    W, lam = shrink_cov(Rm)
    np.fill_diagonal(W, np.maximum(np.diag(W), VAR_FLOOR))
    return W, lam


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean", default="data/clean")
    ap.add_argument("--results", default="results")
    ap.add_argument("--bases", nargs="+", default=["AutoETS", "AutoTheta", "AutoARIMA", "SeasonalNaive"])
    ap.add_argument("--origins", nargs="+", default=None, help="subset of origins (testing only)")
    ap.add_argument("--out", default=None, help="output root (default: --results)")
    ap.add_argument("--burn-in", type=int, default=12, help="months of in-sample residuals excluded at the start (burn-in)")
    ap.add_argument("--tag", default="v2_burnin12", help="output version tag (v2: no burn-in; v2_fmownw: FM bases with own one-step W)")
    ap.add_argument("--fm-w", default="AutoETS", choices=["AutoETS", "own"],
                    help="W of foundation-model bases: AutoETS or their own one-step errors")
    ap.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    ap.add_argument("--forecasts-dir", default=None,
                    help="folder with model=<base>/forecasts.parquet for every base (default: base_forecasts_v2 "
                         "or base_forecasts_fm); used for the real-time origins")
    ap.add_argument("--residuals-dir", default=None, help="in-sample residuals (default: results/insample_residuals_v2)")
    ap.add_argument("--onestep-dirs", nargs="+", default=None, help="one-step error folders (default: results/onestep_fm)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    log = RunLog("reconcile.py", vars(args))
    RES = Path(args.results)
    OUTR = Path(args.out) if args.out else RES
    DIAG = OUTR / f"reconcile_{args.tag}"

    # ------------------------------------------------------------ constraint matrix (full rank v2), per origin
    sysd = load_system(RES, args.clean)
    uids, n, g_col = sysd["uids"], sysd["n"], sysd["g_col"]
    origins = args.origins or pd.period_range("2022-12", "2025-06", freq="M").astype(str).tolist()
    A_by_origin, info = {}, []
    for o in origins:
        A, Acheck, r, dead, inf = origin_constraints(sysd, o)
        A_by_origin[o] = (A, Acheck, r, dead)                     # A: independent rows; Acheck: all rows (checks)
        info.append(inf)
    info = pd.DataFrame(info)
    if "AutoETS" in args.bases:
        info.to_csv(out_path(DIAG / "origin_info.csv", args.overwrite), index=False, float_format="%.6g")
    print(info[["origin", "r", "n_dead", "rows_independent"]].to_string(index=False))

    for base in args.bases:
        coh, nn, wdiag = [], [], []
        t_base = time.perf_counter()
        fdir = Path(args.forecasts_dir) if args.forecasts_dir else \
            RES / ("base_forecasts_fm" if base in FM_BASES else "base_forecasts_v2")
        f = pd.read_parquet(fdir / f"model={base}" / "forecasts.parquet",
                            columns=["unique_id", "origin", "target", "horizon", "mean"])
        f = f[f.unique_id.isin(uids) & f.origin.isin(origins)]
        res, own_fm, w_base = load_w_residuals(RES, base, args.fm_w, args.burn_in, args.residuals_dir, args.onestep_dirs)
        out = []
        for o in origins:
            print(f"{base} {o} ({time.perf_counter() - t_base:.0f} s)", flush=True)
            A, Acheck, r, dead = A_by_origin[o]
            fo = f[f.origin == o].pivot(index="unique_id", columns="horizon", values="mean").loc[uids]
            Yhat = fo.to_numpy()
            targets = f[f.origin == o].drop_duplicates("horizon").set_index("horizon").target.sort_index()
            Rm, n_months = residual_matrix(res, o, own_fm, uids)
            var = (Rm ** 2).mean(axis=0)
            for method in args.methods:
                diag, lam, mint_ok = True, np.nan, np.nan
                if method == "OLS":
                    L = np.ones(n)
                elif method == "WLS":
                    L = np.sqrt(np.maximum(var, VAR_FLOOR))
                else:
                    W, lam = mint_w(Rm)
                    try:
                        L = cholesky(W, lower=True)
                        diag, mint_ok = False, True
                    except np.linalg.LinAlgError:
                        print(f"MinT: W not positive definite ({base}, {o}); kept WLS for this origin")
                        mint_ok = False
                        L = np.sqrt(np.maximum(var, VAR_FLOOR))
                t0 = time.perf_counter()
                Yfree, condR = project(Yhat, A, L, diag)
                t_free = time.perf_counter() - t0
                wdiag.append({"base": base, "w_from": w_base, "burn_in": 0 if own_fm else args.burn_in, "origin": o, "method": method, "resid_months_total": n_months,
                              "resid_months_used": Rm.shape[0], "n_var_floored": int((var < VAR_FLOOR).sum()),
                              "mint_lambda": lam, "mint_cholesky_ok": mint_ok,
                              "cond_R": condR, "seconds_free": t_free})
                variants = {"free": Yfree, "nn_qp": Yfree.copy(), "nn_iter": Yfree.copy()}
                for j, h in enumerate(fo.columns):
                    yf = Yfree[:, j]
                    nneg = int((yf < NEG_TOL).sum())
                    rec = {"base": base, "method": method, "origin": o, "horizon": int(h),
                           "n_negative_base": int((Yhat[:, j] < NEG_TOL).sum()), "n_negative_free": nneg,
                           "neg_mass_free": float(-yf[yf < NEG_TOL].sum())}
                    if nneg:
                        t0 = time.perf_counter()
                        y_it, rounds, nfix_it = nonneg_iter(Yhat[:, j], A, L, diag, yf)
                        t_it = time.perf_counter() - t0
                        t0 = time.perf_counter()
                        y_qp, Zq, min_lagr, gi_iters, flag = nonneg_gi(Yhat[:, j], A, L, diag, dead)
                        t_qp = time.perf_counter() - t0
                        obj_qp = objective(y_qp, Yhat[:, j], L, diag)
                        osqp_check = obj_osqp = polish_shift = t_osqp = np.nan
                        status_os = ""
                        if (o, int(h)) in OSQP_CHECK:                         # cross-check with OSQP
                            t1 = time.perf_counter()
                            y_os, status_os, _, polish_shift, obj_osqp, _ = nonneg_qp(Yhat[:, j], A, L, diag)
                            t_osqp = time.perf_counter() - t1
                            osqp_check = float(np.abs(y_os - y_qp).max())
                        variants["nn_iter"][:, j], variants["nn_qp"][:, j] = y_it, y_qp
                        d = np.abs(y_qp - y_it)
                        rec.update(iter_rounds=rounds, n_zero_iter=nfix_it, n_zero_qp=len(Zq), qp_flag=flag,
                                   qp_min_multiplier=min_lagr, qp_iterations=gi_iters, osqp_status=status_os,
                                   osqp_max_abs_diff=osqp_check, osqp_polish_shift=polish_shift,
                                   obj_qp=obj_qp, obj_osqp=obj_osqp, seconds_osqp=t_osqp,
                                   obj_iter=objective(y_it, Yhat[:, j], L, diag), max_abs_diff_qp_iter=float(d.max()),
                                   max_rel_diff_qp_iter=float((d / np.maximum(np.abs(y_qp), 1.0)).max()),
                                   seconds_iter=t_it, seconds_qp=t_qp)
                    nn.append(rec)
                for vname, Y in variants.items():
                    # coherence against all rows of the origin's constraints (C with the DELTA_INT ratio, dead-series rows)
                    resid = Acheck @ Y
                    scale = np.abs(Acheck) @ np.abs(Y)
                    for j, h in enumerate(fo.columns):
                        coh.append({"base": base, "method": method, "variant": vname, "origin": o, "horizon": int(h),
                                    "max_abs_Cy": float(np.abs(resid[:, j]).max()),
                                    "max_rel_Cy": float((np.abs(resid[:, j]) / np.maximum(scale[:, j], 1.0)).max()),  # floor 1 t/day
                                    "min_y": float(Y[:, j].min())})
                    g = Y[g_col] * r                                  # DELTA_INT forecast = r x reconciled G(W)
                    long = pd.DataFrame(Y, index=uids, columns=targets.values).stack().rename("value").reset_index()
                    long.columns = ["unique_id", "target", "value"]
                    dl = pd.DataFrame({"unique_id": DELTA, "target": targets.values, "value": g})
                    long = pd.concat([long, dl], ignore_index=True)
                    long["horizon"] = long.target.map({t: h for h, t in targets.items()})
                    out.append(long.assign(origin=o, method=method, variant=vname))
        out = pd.concat(out, ignore_index=True)[["unique_id", "origin", "target", "horizon", "method", "variant", "value"]]
        path = out_path(OUTR / f"reconciled_{args.tag}" / f"base={base}" / "reconciled.parquet", args.overwrite)
        out.to_parquet(path, index=False)
        print(f"{base}: {len(out)} rows in {time.perf_counter() - t_base:.0f} s")
        for name, rows_ in [("coherence", coh), ("nonneg", nn), ("w_diagnostics", wdiag)]:
            pd.DataFrame(rows_).to_csv(out_path(DIAG / f"{name}_{base}.csv", args.overwrite), index=False,
                                       float_format="%.6g")
        c = pd.DataFrame(coh)
        print(c.groupby(["method", "variant"]).agg(max_abs_Cy=("max_abs_Cy", "max"),
                                                   max_rel_Cy=("max_rel_Cy", "max"), min_y=("min_y", "min")).to_string())
    log.finish(f"bases {args.bases}")


if __name__ == "__main__":
    main()
