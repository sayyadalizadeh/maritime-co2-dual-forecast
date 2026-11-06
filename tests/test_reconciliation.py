"""Stored-output tests of the reconciliation (the synthetic versions are in test_synthetic.py).

Stored cell: AutoETS base, MinT, origin 2024-06 (W rebuilt with the functions of src/reconcile.py): the projection
satisfies every constraint row; the exact non-negative QP solution satisfies the KKT conditions and equals the stored
forecast; the null-space form, the OSQP fallback and the worker pool give the same solution."""
import numpy as np
import pandas as pd
import pytest
from scipy.linalg import cholesky

from conftest import CLEAN, RES, need
from dual_allocation import kkt_check
from reconcile import (load_system, load_w_residuals, mint_w, nonneg_gi, nonneg_nullspace, nonneg_nullspace_osqp,
                       nullspace_setup,
                       origin_constraints, project, residual_matrix, VAR_FLOOR)

ORIGIN, BASE = "2024-06", "AutoETS"
REL_COH = 1e-9           # max |C y| relative to max |y|
KKT_TOL = 1e-8


@pytest.fixture(scope="module")
def cell():
    need(RES / "C_matrix_fullrank_v2.npz", CLEAN / "panel_system_wide_v2.parquet",
         RES / "base_forecasts_v2" / f"model={BASE}" / "forecasts.parquet",
         RES / "insample_residuals_v2" / f"model={BASE}" / "residuals.parquet")
    sysd = load_system(RES, CLEAN)
    A, Acheck, r, dead, _ = origin_constraints(sysd, ORIGIN)
    f = pd.read_parquet(RES / "base_forecasts_v2" / f"model={BASE}" / "forecasts.parquet",
                        filters=[("origin", "==", ORIGIN)], columns=["unique_id", "horizon", "mean"])
    Yhat = f.pivot(index="unique_id", columns="horizon", values="mean").loc[sysd["uids"]].to_numpy()
    res, own, _ = load_w_residuals(RES, BASE)
    Rm, _ = residual_matrix(res, ORIGIN, own, sysd["uids"])
    W, _ = mint_w(Rm)
    var = np.maximum((Rm ** 2).mean(axis=0), VAR_FLOOR)
    return {"sysd": sysd, "A": A, "Acheck": Acheck, "dead": dead, "Yhat": Yhat, "W": W, "var": var}


def coherence(Acheck, Y):
    return np.abs(Acheck @ Y).max(axis=0) / np.maximum(np.abs(Y).max(axis=0), 1.0)


def test_projection_is_coherent(cell):
    """WLS and MinT free projections of all 12 horizons satisfy every constraint row (C with the
    discrepancy ratio r in the balance row, and the rows fixing dead series at zero)."""
    Acheck, Yhat = cell["Acheck"], cell["Yhat"]
    assert coherence(Acheck, Yhat).max() > 1e-3                      # the base forecasts are incoherent
    for L, diag in [(np.sqrt(cell["var"]), True), (cholesky(cell["W"], lower=True), False)]:
        Y, _ = project(Yhat, cell["A"], L, diag)
        assert coherence(Acheck, Y).max() < REL_COH
        # a projection: projecting again changes nothing
        Y2, _ = project(Y, cell["A"], L, diag)
        assert np.abs(Y2 - Y).max() < 1e-6 * max(np.abs(Y).max(), 1.0)


@pytest.mark.parametrize("h", [1, 12])
def test_qp_kkt_stored_cell(cell, h):
    """Exact non-negative MinT QP at origin 2024-06: KKT conditions, and equality with the stored nn_qp forecast."""
    L = cholesky(cell["W"], lower=True)
    yhat = cell["Yhat"][:, h - 1]
    y, zero, min_mult, _, flag = nonneg_gi(yhat, cell["A"], L, False, cell["dead"])
    assert flag == "" and min_mult >= 0 and len(zero) > 0
    n = cell["sysd"]["n"]
    bounds = np.setdiff1d(np.arange(n), cell["dead"])
    k = kkt_check(y, yhat, cell["W"], cell["A"], bounds, active=zero)
    assert k["stationarity"] < KKT_TOL and k["primal_eq"] < KKT_TOL and k["primal_bound"] < KKT_TOL
    assert k["dual_min_mu"] > -KKT_TOL and k["complementarity"] < KKT_TOL
    assert coherence(cell["Acheck"], y[:, None]).max() < REL_COH
    # the stored reconciled forecast is this solution
    p = RES / "reconciled_v2_burnin12" / f"base={BASE}" / "reconciled.parquet"
    need(p)
    st = pd.read_parquet(p, filters=[("origin", "==", ORIGIN), ("horizon", "==", h), ("method", "==", "MinT"),
                                     ("variant", "==", "nn_qp")]).set_index("unique_id").value
    assert np.abs(st.loc[cell["sysd"]["uids"]].to_numpy() - y).max() < 1e-6 * max(np.abs(y).max(), 1.0)


@pytest.mark.parametrize("h", [1, 12])
def test_nullspace_qp_equals_gi(cell, h):
    """The null-space form of the exact QP gives the nonneg_gi solution and zero set, with non-negative
    multipliers of the active bounds (nonneg_gi reports the minimum over all bounds, 0 for the inactive ones)."""
    L = cholesky(cell["W"], lower=True)
    yhat = cell["Yhat"][:, h - 1]
    y, zero, min_mult, _, _ = nonneg_gi(yhat, cell["A"], L, False, cell["dead"])
    yf = project(yhat[:, None], cell["A"], L, False)[0][:, 0]
    y2, zero2, min_mult2, _, flag2 = nonneg_nullspace(yf, *nullspace_setup(cell["A"], L, cell["dead"]))
    assert flag2 == "" and zero2 == zero
    assert np.abs(y2 - y).max() < 1e-9 * max(np.abs(y).max(), 1.0)
    assert min_mult >= 0 and min_mult2 > 0


@pytest.mark.parametrize("h", [1, 12])
def test_nullspace_osqp_fallback_equals_quadprog(cell, h):
    """The OSQP fallback (eps 1e-9, active set refined to the KKT point) gives the quadprog solution."""
    L = cholesky(cell["W"], lower=True)
    yf = project(cell["Yhat"][:, [h - 1]], cell["A"], L, False)[0][:, 0]
    M, bnd, Cb = nullspace_setup(cell["A"], L, cell["dead"])
    y, _, _, _, _ = nonneg_nullspace(yf, M, bnd, Cb)
    y2, _, min_mult2, _, _, status = nonneg_nullspace_osqp(yf, M, bnd)
    assert status == "solved" and (np.isnan(min_mult2) or min_mult2 >= -1e-9)
    assert np.abs(y2 - y).max() < 1e-9 * max(np.abs(y).max(), 1.0) and y2.min() >= -1e-12 * np.abs(y).max()


def test_qp_pool_stages_equal_quadprog(cell):
    """qp_pool solves in worker processes; the rescaled-quadprog and OSQP stages (forced here) give the
    quadprog solution, and every solution passes the coherence, non-negativity and KKT checks."""
    from qp_pool import solve_pool
    L = cholesky(cell["W"], lower=True)
    cols = project(cell["Yhat"][:, [0, 11]], cell["A"], L, False)[0]
    M, bnd, Cb = nullspace_setup(cell["A"], L, cell["dead"])
    Y, rows, att = solve_pool(cols, M, bnd, Cb, lambda y: {"max_abs_Cy": 0.0, "max_violation_pct": 0.0}, 2, "test",
                              force_stage={0: "quadprog_rescaled", 1: "osqp"})
    assert [r["solver"] for r in rows] == ["quadprog_rescaled", "osqp"] and all(a["ok"] for a in att)
    for j in range(2):
        y = nonneg_nullspace(cols[:, j], M, bnd, Cb)[0]
        assert np.abs(Y[:, j] - y).max() < 1e-9 * np.abs(y).max()
