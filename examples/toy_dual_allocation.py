"""
toy_dual_allocation.py: a synthetic three-economy example of dual-allocation reconciliation.

Three economies X, Y, Z and a Residual node REST. Emissions are generated voyage by voyage as two flow arrays:
  Dom[o, c]       domestic voyages inside country c by operators resident in o
  Int[o, d, a]    international voyages departing d and arriving in a (d != a) by operators resident in o
The seven published components of a unit u then are:
  A = Dom[u, u]                      B = sum_{c != u} Dom[u, c]          C = sum_{o != u} Dom[o, u]
  D = sum_a Int[u, u, a]             E = sum_{d != u} Int[u, d, u]       F = sum_{d != u, a != u} Int[u, d, a]
  G = sum_{o != u} sum_a Int[o, u, a]
with the residence total A+B+D+E+F (everything operated by u) and the territory totals A+C and D+G. World values are
sums over all units, so B(W) = C(W) and E(W) + F(W) = G(W) hold exactly (the discrepancy node is zero here).

Steps: build C with src/dual_allocation.py, check the identities on the synthetic flows, add noise to get "base
forecasts" (the free projection then gives negative values in small series), reconcile with OLS and WLS projections and with the
exact non-negative QP (src/reconcile.py functions), and print max |C y| and the KKT residuals before and after.

Usage (from the repository root, project environment):
  .venv/Scripts/python.exe examples/toy_dual_allocation.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dual_allocation import PAPER_LETTERS, Spec, build, kkt_check      # noqa: E402
from reconcile import nonneg_gi, project                                # noqa: E402

SEED = 20261103
UNITS = ["X", "Y", "Z", "REST"]                     # REST plays the role of the Residual node


def components(Dom, Int):
    """Seven components per unit from the voyage flows: {unit: {letter: value}}."""
    n = len(UNITS)
    out = {}
    for u in range(n):
        oth = [c for c in range(n) if c != u]
        out[UNITS[u]] = {
            "A": Dom[u, u], "B": Dom[u, oth].sum(), "C": Dom[oth, u].sum(),
            "D": Int[u, u, :].sum(),
            "E": sum(Int[u, d, u] for d in oth),
            "F": sum(Int[u, d, a] for d in oth for a in oth),
            "G": Int[oth, u, :].sum()}
    return out


def vector(sysm, comp):
    """Stack the components, totals and World values in the column order of the built system."""
    world = {k: sum(comp[u][k] for u in UNITS) for k in "ABCDEFG"}
    full = dict(comp, W=world)
    vals = []
    for u, m in zip(sysm.cols.unit, sysm.cols.measure):
        if m == "DELTA_INT":
            vals.append(0.0)
            continue
        c = full[u]
        letter = {v: k for k, v in PAPER_LETTERS.items()}.get(m)
        if letter:
            vals.append(c[letter])
        elif m == "RES_TOTAL_SYS":
            vals.append(c["A"] + c["B"] + c["D"] + c["E"] + c["F"])
        elif m == "TER_DOM":
            vals.append(c["A"] + c["C"])
        elif m == "TER_INT":
            vals.append(c["D"] + c["G"])
    return np.array(vals)


def main():
    rng = np.random.default_rng(SEED)
    n = len(UNITS)
    spec = Spec(economies=UNITS[:3], residual="REST", world="W", components=list(PAPER_LETTERS.values()),
                totals={"RES_TOTAL_SYS": "A+B+D+E+F", "TER_DOM": "A+C", "TER_INT": "D+G"}, letters=PAPER_LETTERS,
                balances={"world_balance_B_C": "B-C", "world_balance_EF_G": "E+F-G-DELTA_INT"},
                world_only=["DELTA_INT"])
    sysm = build(spec)
    C = sysm.C.toarray()
    A = sysm.C_fullrank.toarray()
    print(f"system: {C.shape[1]} series, {C.shape[0]} constraints, rank {A.shape[0]} "
          f"(redundant: {', '.join(sysm.rows.loc[sysm.rows.redundant, 'type'] + ' ' + sysm.rows.loc[sysm.rows.redundant, 'unit'])})")

    # ---- synthetic flows (t per day) and the identities on them
    size = np.array([5.0, 3.0, 1.0, 0.3])                               # operator size
    Dom = rng.gamma(2.0, 1.0, (n, n)) * np.outer(size, size) * np.where(np.eye(n, dtype=bool), 20, 0.5)
    Int = rng.gamma(2.0, 1.0, (n, n, n)) * size[:, None, None] * 4.0
    Int[:, np.arange(n), np.arange(n)] = 0.0                            # departure != arrival
    y = vector(sysm, components(Dom, Int))
    print(f"identities on the flows: max |C y| = {np.abs(C @ y).max():.2e} (values up to {y.max():.1f})")

    # ---- noisy base forecasts: 5% multiplicative noise, plus additive noise that pushes small series below zero
    sd = 0.05 * y + 0.4
    yhat = y + rng.normal(0, sd)
    print(f"base forecasts: max |C yhat| = {np.abs(C @ yhat).max():.2f}, negatives: {(yhat < 0).sum()}")

    # ---- reconciliation (DELTA_INT is a free column here; in the paper it is r times the reconciled G(World))
    for name, L in [("OLS", np.ones(len(y))), ("WLS", sd)]:
        yf, _ = project(yhat[:, None], A, L, True)
        yf = yf[:, 0]
        print(f"{name} projection: max |C y| = {np.abs(C @ yf).max():.2e}, min y = {yf.min():.3f}, "
              f"RMSE base {np.sqrt(((yhat - y) ** 2).mean()):.3f} -> {np.sqrt(((yf - y) ** 2).mean()):.3f}")
        yq, zero, min_mult, iters, flag = nonneg_gi(yhat, A, L, True, dead=[])
        k = kkt_check(yq, yhat, L ** 2, A, bounds=np.arange(len(y)), active=zero)
        print(f"{name} exact non-negative QP: max |C y| = {np.abs(C @ yq).max():.2e}, min y = {yq.min():.3f}, "
              f"{len(zero)} series at zero, min multiplier {min_mult:.3g}, iterations {iters} {flag}")
        print("   KKT: " + ", ".join(f"{a} {b:.1e}" if isinstance(b, float) else f"{a} {b}" for a, b in k.items()))


if __name__ == "__main__":
    main()
