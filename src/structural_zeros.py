"""
structural_zeros.py: the structural-zero rule, as a function (used by build_panel.py).

The rule: a missing component value is set to zero only when every published total that contains it (RES_TOTAL for A, B,
D, E, F; TER_DOM for A, C; TER_INT for D, G) is published in that month and equals the sum of the other published
components within 0.01% or 1 t. Nothing else is filled.

The code is kept separate from build_panel.py so that it can be unit-tested (tests/test_structural_zeros.py).
"""
import numpy as np
import pandas as pd

COMPONENTS = ["RES_DOM_IN", "RES_DOM_OUT", "NRES_DOM_IN", "RES_INT_FROM",
              "RES_INT_TO", "RES_INT_OUT", "NRES_INT_FROM"]                 # A to G
A, B, C, D, E, F, G = COMPONENTS
PUB_TOTALS = {"RES_TOTAL": [A, B, D, E, F], "TER_DOM": [A, C], "TER_INT": [D, G]}   # as published
ZERO_REL_TOL, ZERO_ABS_TOL = 1e-4, 1.0           # within 0.01% or 1 t


def structural_zero_mask(X):
    """The structural-zero rule on a frame X with the seven components and the three published totals as columns (co2_t, tonnes per month;
    NaN = not published), one row per economy-month.
    Returns (zero_mask, present_sum, contained_in):
      zero_mask     boolean frame (rows of X, COMPONENTS): True where a missing component is a structural zero;
      present_sum   {total: sum of its published parts} (missing parts count as zero);
      contained_in  {component: the published totals that contain it}."""
    missing = X[COMPONENTS].isna()
    # confirm[T]: total T is published and equals the sum of its published parts within tolerance.
    present_sum = {T: X[parts].sum(axis=1, min_count=0) for T, parts in PUB_TOTALS.items()}
    confirm = {T: X[T].notna() & ((X[T] - present_sum[T]).abs()
                                  <= np.maximum(ZERO_ABS_TOL, ZERO_REL_TOL * X[T].abs()))
               for T in PUB_TOTALS}
    contained_in = {k: [T for T, parts in PUB_TOTALS.items() if k in parts] for k in COMPONENTS}
    zero_mask = pd.DataFrame(False, index=X.index, columns=COMPONENTS)
    for k in COMPONENTS:
        ok = pd.concat([confirm[T] for T in contained_in[k]], axis=1).all(axis=1)
        zero_mask[k] = missing[k] & ok
    return zero_mask, present_sum, contained_in
