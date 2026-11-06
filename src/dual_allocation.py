"""
dual_allocation.py: generic constraint builder for an inventory published under two allocation principles.

An inventory of this kind publishes, for every unit u (economy, a Residual node for everything not modelled, World),
a set of components and a set of totals. Each total is a signed sum of components of the same unit (for example
residence and territory totals of the same flows). The builder takes the structure as data and returns the linear
constraint matrix C with C @ y = 0 for every coherent vector y:

  within       for every unit u and every total k:   k(u) - sum_j c_kj * component_j(u) = 0
  aggregation  for every measure m (components and totals):   m(World) - sum_u m(u) - m(Residual) = 0
  balance      optional rows at World level, each a signed sum of World measures (and of World-only columns,
               such as a discrepancy node) that must be zero

Columns: for each unit in the given order (economies, then the Residual, then World), the components followed by the
totals; World-only columns are appended at the end. Rows: within rows unit by unit (totals in the given order), then
one aggregation row per measure, then the balance rows. A full-row-rank version is obtained by a greedy pass that
keeps a row only if it increases the rank of the rows kept so far, testing the World within rows last (they follow
from the other rows), as src/constraints.py does.

With paper_spec() the builder reproduces the paper's C (results/C_matrix_v2.npz) and its full-rank version
(results/C_matrix_fullrank_v2.npz) exactly; tests/test_constraints.py checks this.

The module also has kkt_check(), which verifies the optimality conditions of the non-negative reconciliation QP
  min (y - yhat)' W^-1 (y - yhat)   s.t.   A y = 0,  y_i >= 0 (i in bounds).

Usage (example):
  from dual_allocation import Spec, build
  spec = Spec(economies=["X", "Y"], residual="REST", world="W",
              components=["A", "B", "C"], totals={"T1": "A+B", "T2": "A+C"},
              balances={"bal_B_C": "B-C"})
  out = build(spec)          # out.C (scipy.sparse.csr_matrix), out.rows, out.cols, out.C_fullrank
"""
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

RANK_TOL = 1e-9          # relative tolerance of the greedy rank pass (C holds only small integers)


@dataclass
class Spec:
    """Structure of a dual-allocation inventory.

    economies     units modelled individually, in column order
    residual      name of the Residual node (or None if every unit is modelled)
    world         name of the World node
    components    component measures, in column order within a unit
    totals        {total: expression}, expression a signed sum of components, e.g. "A+B+D+E+F" or "A-B";
                  components may be named by their letters through `letters`
    letters       optional {letter: component name} used in the expressions (and in the balance rows)
    balances      optional {row type: expression} of World-level rows; terms may be any measure or World-only column
    world_only    optional World-only columns appended after the World measures (e.g. a discrepancy node)
    balance_measure_labels  optional {row type: label of the `measure` field}; default: the terms joined by commas
    """
    economies: list
    residual: str
    world: str
    components: list
    totals: dict
    letters: dict = field(default_factory=dict)
    balances: dict = field(default_factory=dict)
    world_only: list = field(default_factory=list)


@dataclass
class System:
    C: sp.csr_matrix
    rows: pd.DataFrame          # row, type, unit, measure, equation, redundant
    cols: pd.DataFrame          # col, unit, measure, unit_type
    C_fullrank: sp.csr_matrix   # rows with redundant == False, in row order

    @property
    def labels(self):
        return (self.cols.unit + "|" + self.cols.measure).tolist()


def parse(expr, letters):
    """'A+B-C' -> [(name, +1), (name, +1), (name, -1)]; names may be letters (mapped) or measure names."""
    terms = re.findall(r"([+-]?)\s*([A-Za-z_][A-Za-z0-9_]*)", expr.replace(" ", ""))
    if not terms:
        raise ValueError(f"empty expression: {expr!r}")
    return [(letters.get(name, name), -1.0 if sign == "-" else 1.0) for sign, name in terms]


def build(spec):
    """Constraint matrix of the spec (see the module docstring)."""
    units = list(spec.economies) + ([spec.residual] if spec.residual else []) + [spec.world]
    measures = list(spec.components) + list(spec.totals)
    cols = [(u, m) for u in units for m in measures] + [(spec.world, m) for m in spec.world_only]
    idx = {c: i for i, c in enumerate(cols)}
    if len(idx) != len(cols):
        raise ValueError("duplicate columns: a World-only column repeats a measure")
    rows, entries = [], []

    def add_row(kind, unit, measure, terms, text):
        r = len(rows)
        rows.append({"row": r, "type": kind, "unit": unit, "measure": measure, "equation": text})
        for key, v in terms:
            if key not in idx:
                raise KeyError(f"row {kind} ({unit}): unknown series {key}")
            entries.append((r, idx[key], v))

    parts = {k: parse(e, spec.letters) for k, e in spec.totals.items()}
    for k, p in parts.items():
        bad = [n for n, _ in p if n not in spec.components]
        if bad:
            raise ValueError(f"total {k} uses non-components {bad}")
    for u in units:
        for k, p in parts.items():
            add_row(f"within_{k}", u, k, [((u, k), 1.0)] + [((u, n), -v) for n, v in p],
                    f"{k} - ({' + '.join(n for n, _ in p)}) = 0")
    others = list(spec.economies) + ([spec.residual] if spec.residual else [])
    for m in measures:
        add_row("aggregation", spec.world, m, [((spec.world, m), 1.0)] + [((u, m), -1.0) for u in others],
                f"{m}(W) - sum_c {m}(c) - {m}(Residual) = 0")
    for kind, e in spec.balances.items():
        p = parse(e, spec.letters)
        add_row(kind, spec.world, ",".join(n for n, _ in p), [((spec.world, n), v) for n, v in p], f"{e} = 0")

    rows = pd.DataFrame(rows)
    r_, c_, v_ = zip(*entries)
    C = sp.csr_matrix((np.asarray(v_, float), (r_, c_)), shape=(len(rows), len(cols)))
    world_within = rows.type.str.startswith("within") & (rows.unit == spec.world)
    order = list(rows.index[~world_within]) + list(rows.index[world_within])
    rows["redundant"] = ~rows.row.isin(independent_rows_greedy(C.toarray(), order))
    cols = pd.DataFrame(cols, columns=["unit", "measure"])
    cols.insert(0, "col", range(len(cols)))
    cols["unit_type"] = np.where(cols.unit == spec.world, "world",
                                 np.where(cols.unit == spec.residual, "residual", "economy"))
    return System(C=C, rows=rows, cols=cols, C_fullrank=C[np.flatnonzero(~rows.redundant.to_numpy())])


def independent_rows_greedy(M, order):
    """Rows kept by a greedy pass in `order`: a row is kept if it is not in the span of the rows kept so far
    (Gram-Schmidt with re-orthogonalisation)."""
    basis, kept = [], []
    for r in order:
        v = M[r].astype(float)
        nv = np.linalg.norm(v)
        if nv == 0:
            continue
        w = v.copy()
        for _ in range(2):
            for q in basis:
                w -= (q @ w) * q
        nw = np.linalg.norm(w)
        if nw > RANK_TOL * nv:
            basis.append(w / nw)
            kept.append(r)
    return kept


# ---------------------------------------------------------------- the paper's system (OECD maritime CO2)
PAPER_LETTERS = {"A": "RES_DOM_IN", "B": "RES_DOM_OUT", "C": "NRES_DOM_IN", "D": "RES_INT_FROM",
                 "E": "RES_INT_TO", "F": "RES_INT_OUT", "G": "NRES_INT_FROM"}


def paper_spec(economies):
    """Spec of the paper's system (residence total RES_TOTAL_SYS = A+B+D+E+F, World discrepancy node DELTA_INT)
    for the given system economies."""
    return Spec(economies=list(economies), residual="RESIDUAL", world="W",
                components=list(PAPER_LETTERS.values()),
                totals={"RES_TOTAL_SYS": "A+B+D+E+F", "TER_DOM": "A+C", "TER_INT": "D+G"},
                letters=PAPER_LETTERS,
                balances={"world_balance_B_C": "B-C", "world_balance_EF_G": "E+F-G-DELTA_INT"},
                world_only=["DELTA_INT"])


def paper_economies(results="results"):
    """System economies in the column order of the stored C (results/C_matrix_cols_v2.csv)."""
    cols = pd.read_csv(Path(results) / "C_matrix_cols_v2.csv")
    return list(dict.fromkeys(cols.loc[cols.unit_type == "economy", "unit"]))


# ---------------------------------------------------------------- verification
def kkt_check(y, yhat, W, A, bounds, active=None, tol_zero=1e-6):
    """Optimality conditions of  min (y - yhat)' W^-1 (y - yhat)  s.t.  A y = 0,  y_i >= 0 for i in bounds.

    With the Lagrangian 1/2 (y - yhat)' W^-1 (y - yhat) - nu' A y - mu' y, the conditions are
      stationarity       W^-1 (y - yhat) = A' nu + mu      (mu_i = 0 outside the bounds)
      primal feasibility A y = 0,  y_i >= 0
      dual feasibility   mu_i >= 0
      complementarity    mu_i y_i = 0
    The active set is the solver's (`active`, e.g. the zero set returned by reconcile.nonneg_gi) or, if not given,
    {i in bounds: y_i <= tol_zero}; nu and mu (on the active set) are the least-squares solution of the stationarity
    equations. W is a matrix or the vector of its diagonal. Returns a dict of residuals, relative to the scale of the
    gradient or of y."""
    W = np.atleast_2d(W) if np.ndim(W) == 2 else np.diag(W)
    g = np.linalg.solve(W, y - yhat)
    bounds = np.asarray(bounds, int)
    act = bounds[y[bounds] <= tol_zero] if active is None else np.array(sorted(active), int)
    E = np.zeros((len(act), len(y)))
    E[np.arange(len(act)), act] = 1.0
    M = np.vstack([A, E]).T
    sol, *_ = np.linalg.lstsq(M, g, rcond=None)
    nu, mu = sol[:A.shape[0]], sol[A.shape[0]:]
    gs = max(np.abs(g).max(), 1e-12)
    ys = max(np.abs(y).max(), 1.0)
    return {"stationarity": float(np.abs(M @ sol - g).max() / gs),
            "primal_eq": float(np.abs(A @ y).max() / ys),
            "primal_bound": float(max(0.0, -y[bounds].min()) / ys) if len(bounds) else 0.0,
            "dual_min_mu": float(mu.min() / gs) if len(mu) else 0.0,
            "complementarity": float(np.abs(mu * y[act]).max() / (gs * ys)) if len(mu) else 0.0,
            "n_active": int(len(act))}
