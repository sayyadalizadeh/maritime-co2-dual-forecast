"""Core tests on the synthetic three-economy example (examples/toy_dual_allocation.py); no stored outputs needed.

1. The generic dual-allocation builder: the synthetic voyage flows satisfy every identity (C y = 0), the full-rank
   matrix has the rank of C, the three within-unit rows of World are the redundant ones, and the rows do not depend on
   the order in which the totals are given (up to a permutation).
2. The W-projection gives coherent forecasts (C y = 0) and is idempotent, for a diagonal and a dense W.
3. The exact non-negative QP: the Goldfarb-Idnani solution (nonneg_gi) satisfies the KKT conditions; the null-space
   form (nonneg_nullspace), the OSQP fallback (nonneg_nullspace_osqp) and the worker pool with its fallback stages
   (qp_pool.solve_pool) give the same solution.
"""
import numpy as np
import pytest
from scipy.linalg import cholesky

from conftest import toy_spec
from dual_allocation import build, kkt_check
from reconcile import nonneg_gi, nonneg_nullspace, nonneg_nullspace_osqp, nullspace_setup, project

KKT_TOL = 1e-8


def noisy(toy_system, seed=2):
    """Base forecasts: the coherent flows plus noise large enough to push small series below zero."""
    y = toy_system["y"]
    rng = np.random.default_rng(seed)
    return y + rng.normal(0, 0.3 * y + 10)


def dense_w(toy_system, seed=3):
    """A dense, positive definite W with the scale of the noise."""
    y = toy_system["y"]
    n = len(y)
    rng = np.random.default_rng(seed)
    B = rng.normal(size=(n, n))
    R = 0.5 * np.eye(n) + 0.5 * (B @ B.T) / n
    d = 1 / np.sqrt(np.diag(R))
    sd = 0.3 * y + 10
    return (sd[:, None] * (d[:, None] * R * d[None, :]) * sd[None, :])


# ------------------------------------------------------------------ 1. builder
def test_flows_satisfy_every_identity(toy_system):
    C, y = toy_system["C"], toy_system["y"]
    assert np.abs(C @ y).max() < 1e-10 * y.max()


def test_full_rank_and_redundant_rows(toy_system):
    s = toy_system["spec"]
    C, A = toy_system["C"], toy_system["A"]
    assert np.linalg.matrix_rank(C) == A.shape[0] == np.linalg.matrix_rank(A) == (~s.rows.redundant).sum()
    red = s.rows[s.rows.redundant]
    assert len(red) == 3 and set(red.unit) == {"W"} and red.type.str.startswith("within").all()
    # 4 units + World, 3 within rows each; 10 aggregation rows; 2 world-balance rows
    assert C.shape[0] == 5 * 3 + 10 + 2 and C.shape[1] == 5 * 10 + 1


def test_builder_is_order_free_up_to_row_permutation(toy_system):
    toy = toy_system["toy"]
    s2 = build(toy_spec(toy, totals={"TER_INT": "D+G", "TER_DOM": "A+C", "RES_TOTAL_SYS": "A+B+D+E+F"}))
    s = toy_system["spec"]
    pos = {l_: i for i, l_ in enumerate(s2.labels)}
    perm = [pos[l_] for l_ in s.labels]
    a = {tuple(r) for r in s2.C.toarray()[:, perm]}
    b = {tuple(r) for r in toy_system["C"]}
    assert a == b and s2.C.shape == toy_system["C"].shape


# ------------------------------------------------------------------ 2. projection
@pytest.mark.parametrize("diag", [True, False])
def test_projection_is_coherent_and_idempotent(toy_system, diag):
    A, C = toy_system["A"], toy_system["C"]
    yhat = noisy(toy_system)
    W = np.diag((0.3 * toy_system["y"] + 10) ** 2) if diag else dense_w(toy_system)
    L = np.sqrt(np.diag(W)) if diag else cholesky(W, lower=True)
    yf = project(yhat[:, None], A, L, diag)[0][:, 0]
    assert np.abs(C @ yf).max() < 1e-9 * np.abs(yf).max()
    yff = project(yf[:, None], A, L, diag)[0][:, 0]
    assert np.abs(yff - yf).max() < 1e-9 * np.abs(yf).max()
    assert np.abs(C @ yhat).max() > 1.0                              # the base forecasts were incoherent


# ------------------------------------------------------------------ 3. exact non-negative QP
@pytest.mark.parametrize("diag", [True, False])
def test_qp_kkt(toy_system, diag):
    A, C = toy_system["A"], toy_system["C"]
    yhat = noisy(toy_system)
    W = np.diag((0.3 * toy_system["y"] + 10) ** 2) if diag else dense_w(toy_system)
    L = np.sqrt(np.diag(W)) if diag else cholesky(W, lower=True)
    yq, zero, min_mult, _, flag = nonneg_gi(yhat, A, L, diag, dead=[])
    k = kkt_check(yq, yhat, W, A, np.arange(len(yhat)), active=zero)
    assert flag == "" and len(zero) > 0 and min_mult >= 0
    assert max(k["stationarity"], k["primal_eq"], k["primal_bound"], k["complementarity"]) < KKT_TOL
    assert k["dual_min_mu"] > -KKT_TOL
    assert np.abs(C @ yq).max() < 1e-9 * np.abs(yq).max() and yq.min() >= -1e-12


def test_solvers_give_the_same_solution(toy_system):
    """nonneg_gi, the null-space form, the OSQP fallback and the worker pool (stages forced) agree."""
    from qp_pool import solve_pool
    A = toy_system["A"]
    W = dense_w(toy_system)
    L = cholesky(W, lower=True)
    yhats = np.column_stack([noisy(toy_system, seed) for seed in (2, 4)])
    yfs = project(yhats, A, L, False)[0]
    assert (yfs < 0).any(axis=0).all()                               # every column needs the QP
    M, bnd, Cb = nullspace_setup(A, L, [])
    sols = []
    for j in range(2):
        y1, zero1, _, _, _ = nonneg_gi(yhats[:, j], A, L, False, dead=[])
        y2, zero2, min_mult2, _, flag2 = nonneg_nullspace(yfs[:, j], M, bnd, Cb)
        y3, _, min_mult3, _, _, status = nonneg_nullspace_osqp(yfs[:, j], M, bnd)
        scale = max(np.abs(y1).max(), 1.0)
        assert flag2 == "" and zero2 == zero1 and min_mult2 > 0
        assert np.abs(y2 - y1).max() < 1e-9 * scale
        assert status == "solved" and (np.isnan(min_mult3) or min_mult3 >= -1e-9)
        assert np.abs(y3 - y1).max() < 1e-9 * scale
        sols.append(y1)
    Y, rows, att = solve_pool(yfs, M, bnd, Cb, lambda y: {"max_abs_Cy": 0.0, "max_violation_pct": 0.0}, 2, "test",
                              force_stage={0: "quadprog_rescaled", 1: "osqp"})
    assert [r["solver"] for r in rows] == ["quadprog_rescaled", "osqp"] and all(a["ok"] for a in att)
    for j in range(2):
        assert np.abs(Y[:, j] - sols[j]).max() < 1e-9 * max(np.abs(sols[j]).max(), 1.0)
