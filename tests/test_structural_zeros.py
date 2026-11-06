"""The structural-zero rule: synthetic cases, and the stored log and panel."""
import numpy as np
import pandas as pd

from conftest import CLEAN, RES, need
from structural_zeros import COMPONENTS, PUB_TOTALS, structural_zero_mask

A, B, C, D, E, F, G = COMPONENTS


def frame(rows):
    """Economy-months with components A-G and published totals; values in t per month, None = not published.
    Unspecified totals are the sums of the given components (all of them published)."""
    out = []
    for r in rows:
        v = {k: r.get(k, 100.0) for k in COMPONENTS}
        for T, parts in PUB_TOTALS.items():
            v[T] = r[T] if T in r else sum(v[p] or 0.0 for p in parts)
        out.append(v)
    return pd.DataFrame(out, dtype=float)


def test_zero_when_every_containing_total_confirms():
    # A is in RES_TOTAL and TER_DOM; both are published and equal the sum of the other parts
    m, _, _ = structural_zero_mask(frame([{A: None}]))
    assert m.loc[0, A] and m.to_numpy().sum() == 1


def test_no_zero_when_a_containing_total_is_unpublished():
    m, _, _ = structural_zero_mask(frame([{A: None, "TER_DOM": None}]))      # RES_TOTAL confirms, TER_DOM missing
    assert not m.to_numpy().any()


def test_no_zero_when_a_containing_total_does_not_add_up():
    # D is in RES_TOTAL and TER_INT; TER_INT exceeds the other part by 50 t (> 1 t and > 0.01%)
    m, _, _ = structural_zero_mask(frame([{D: None, "TER_INT": 150.0}]))
    assert not m.to_numpy().any()


def test_tolerance_is_one_tonne_or_one_basis_point():
    big = 1e7
    rows = [{C: None, A: 100.0, "TER_DOM": 100.9},              # 0.9 t off: within 1 t
            {C: None, A: 100.0, "TER_DOM": 101.1},              # 1.1 t off, 1.1%: outside
            {C: None, A: big, "TER_DOM": big + 900.0},          # 900 t off, 0.009%: within 0.01%
            {C: None, A: big, "TER_DOM": big + 1100.0}]         # 1,100 t off, 0.011%: outside
    m, _, _ = structural_zero_mask(frame(rows))
    assert m[C].tolist() == [True, False, True, False]


def test_two_missing_parts_of_one_total():
    # A and C missing: TER_DOM = 0 confirms both (with RES_TOTAL for A); TER_DOM = 50 confirms neither
    m, _, _ = structural_zero_mask(frame([{A: None, C: None, "TER_DOM": 0.0},
                                          {A: None, C: None, "TER_DOM": 50.0}]))
    assert m.loc[0, A] and m.loc[0, C] and not m.loc[1, [A, C]].any()


def test_published_values_are_never_marked():
    X = frame([{}, {G: 0.0}, {E: None}])
    m, _, _ = structural_zero_mask(X)
    assert not (m & X[COMPONENTS].notna()).to_numpy().any()
    assert m.to_numpy().sum() == 1 and m.loc[2, E]


def test_rule_reproduces_the_stored_log_and_panel():
    need(CLEAN / "oecd_maritime_co2_monthly_v1.csv", CLEAN / "oecd_areas_v1.csv", RES / "structural_zeros.csv",
         CLEAN / "panel_system_v2.parquet")
    m = pd.read_csv(CLEAN / "oecd_maritime_co2_monthly_v1.csv", dtype={"period": str})
    areas = pd.read_csv(CLEAN / "oecd_areas_v1.csv")
    economies = sorted(areas.loc[areas.area_type == "economy", "area_code"])
    months = pd.period_range("2019-01", "2026-06", freq="M").astype(str).tolist()
    cols = COMPONENTS + list(PUB_TOTALS)
    m = m[(m.vessel == "ALL_VESSELS") & m.area_code.isin(economies) & m.period.isin(months) & m.component.isin(cols)]
    X = m.pivot(index=["area_code", "period"], columns="component", values="co2_t") \
        .reindex(index=pd.MultiIndex.from_product([economies, months], names=["area_code", "period"]), columns=cols)
    mask, _, _ = structural_zero_mask(X)
    got = {(a, k, p) for k in COMPONENTS for a, p in mask.index[mask[k]]}
    log = pd.read_csv(RES / "structural_zeros.csv", dtype={"period": str})
    assert got == set(zip(log.unit, log.measure, log.period))
    # in the system panel every flagged value is an exact zero and every imputed zero of a system economy is flagged
    p = pd.read_parquet(CLEAN / "panel_system_v2.parquet")
    flagged = p[p.flag_structural_zero]
    assert (flagged.co2_t == 0).all()
    sys_got = {x for x in got if x[0] in set(p.unit)}
    assert set(zip(flagged.unit, flagged.measure, flagged.period)) == sys_got
    assert len(sys_got) > 0 and np.isfinite(p[p.measure.isin(COMPONENTS)].co2_t).all()
