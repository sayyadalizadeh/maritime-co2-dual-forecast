"""C holds on the historical data, and the generic builder reproduces the paper's C."""
import numpy as np
import pandas as pd
import scipy.sparse as sp

from conftest import CLEAN, RES, need
from dual_allocation import build, paper_economies, paper_spec

# rounding tolerance of src/constraints.py (as in the structural-zero rule): 1 t per month, i.e. 1/28 t per day, or 0.01% of the
# row's scale series (the total, or the World value for aggregation and balance rows)
ABS_TOL, REL_TOL = 1 / 28, 1e-4


def stored():
    need(RES / "C_matrix_v2.npz", RES / "C_matrix_rows_v2.csv", RES / "C_matrix_cols_v2.csv")
    return (sp.load_npz(RES / "C_matrix_v2.npz"), sp.load_npz(RES / "C_matrix_fullrank_v2.npz"),
            pd.read_csv(RES / "C_matrix_rows_v2.csv"), pd.read_csv(RES / "C_matrix_cols_v2.csv"))


def test_C_holds_on_all_historical_months():
    C, _, rows, cols = stored()
    need(CLEAN / "panel_system_wide_v2.parquet")
    wide = pd.read_parquet(CLEAN / "panel_system_wide_v2.parquet")
    assert list(wide.columns) == (cols.unit + "|" + cols.measure).tolist()
    Y = wide.to_numpy()
    assert np.isfinite(Y).all()
    R = Y @ C.T.toarray()                                            # months x rows
    scale = np.abs(wide[rows.scale_series].to_numpy())
    rel = np.where(scale > 0, np.abs(R) / np.maximum(scale, 1e-300), np.where(R == 0, 0.0, np.inf))
    beyond = (np.abs(R) > ABS_TOL) & (rel > REL_TOL)
    assert len(wide) == 90                                           # 2019-01 ... 2026-06
    assert not beyond.any(), f"{beyond.sum()} cells beyond rounding"


def test_builder_reproduces_paper_C_exactly():
    C, Cf, rows, cols = stored()
    s = build(paper_spec(paper_economies(RES)))
    assert s.labels == (cols.unit + "|" + cols.measure).tolist()       # same columns, same order
    assert s.C.shape == C.shape and abs(s.C - C).max() == 0            # same rows in the same order, same coefficients
    for c in ["type", "unit", "measure", "redundant"]:
        assert (s.rows[c].to_numpy() == rows[c].to_numpy()).all(), c
    assert s.C_fullrank.shape == Cf.shape and abs(s.C_fullrank - Cf).max() == 0
    assert np.linalg.matrix_rank(Cf.toarray()) == Cf.shape[0] == (~rows.redundant).sum()


def test_builder_is_order_free_up_to_row_permutation():
    """A different order of the totals gives the same rows, permuted (columns mapped by their labels)."""
    C, _, _, cols = stored()
    spec = paper_spec(paper_economies(RES))
    spec.totals = dict(reversed(list(spec.totals.items())))
    s = build(spec)
    pos = {l_: i for i, l_ in enumerate(s.labels)}
    perm = [pos[l_] for l_ in (cols.unit + "|" + cols.measure)]
    a = {tuple(r) for r in s.C.toarray()[:, perm]}
    b = {tuple(r) for r in C.toarray()}
    assert a == b and s.C.shape[0] == C.shape[0]
