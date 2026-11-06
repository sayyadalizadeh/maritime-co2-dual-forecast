"""The ACI update (src/prob_samples.py, src/calib_aci.py):
    alpha <- clip(alpha + gamma (alpha_nominal - err), 0.001, 0.999),   err = 1 if y is outside [lo, hi], else 0,
applied when the target month of a forecast is observed; alpha starts at the nominal value (0.20 for 80%, 0.05 for
95%). The update lives inside the scripts' main(); it is checked by hand on a small case, and by replaying the
recursion from the stored intervals (results/prob_main, origins 2022-12 ... 2025-06) and comparing with the stored
alpha used at each origin and with the stored state after 2025-06."""
import numpy as np
import pandas as pd
import pytest

from conftest import CLEAN, RES, need
from prob_samples import ALPHA_CLIP, GAMMAS, LEVELS


def aci_step(alpha, gamma, nominal, y, lo, hi):
    """One ACI update, written from the rule above (independently of the scripts)."""
    err = ((y < lo) | (y > hi)).astype(float)
    return np.clip(alpha + gamma * (nominal - err), *ALPHA_CLIP)


def test_constants():
    assert LEVELS == {80: 0.20, 95: 0.05} and ALPHA_CLIP == (0.001, 0.999) and GAMMAS[0] == 0.05


def test_update_by_hand():
    a = np.array([0.20, 0.20, 0.05, 0.001, 0.999])
    y = np.array([5.0, 12.0, 5.0, 12.0, 5.0])
    lo, hi = np.full(5, 0.0), np.full(5, 10.0)
    nominal = np.array([0.20, 0.20, 0.05, 0.05, 0.20])
    out = aci_step(a, 0.05, nominal, y, lo, hi)
    # covered: alpha + 0.05 * 0.20 = 0.21; missed: 0.20 + 0.05 * (0.20 - 1) = 0.16; covered: 0.05 + 0.0025 = 0.0525;
    # missed at the floor: 0.001 + 0.05 * (0.05 - 1) < 0.001 -> 0.001; covered at the ceiling: stays 0.999
    assert np.allclose(out, [0.21, 0.16, 0.0525, 0.001, 0.999])


@pytest.mark.parametrize("track", ["STAT", "FM"])
def test_replay_reproduces_stored_alpha(track):
    p_iv = RES / "prob_main" / f"track={track}" / "aci_intervals.parquet"
    p_st = RES / "prob_main" / f"track={track}" / "aci_state.parquet"
    need(p_iv, p_st, CLEAN / "panel_system_wide_v2.parquet")
    hist = pd.read_parquet(CLEAN / "panel_system_wide_v2.parquet")
    uids = sorted(pd.read_parquet(p_st, columns=["unique_id"]).unique_id.unique())[::40]     # 28 series
    iv = pd.read_parquet(p_iv, filters=[("unique_id", "in", uids)])
    iv = iv.sort_values(["origin", "unique_id", "horizon", "level", "gamma"])
    origins = sorted(iv.origin.unique())
    keys = [(lv, g) for lv in LEVELS for g in GAMMAS]
    H = int(iv.horizon.max())
    alpha = {k: np.full((len(uids), H), LEVELS[k[0]]) for k in keys}
    pending = {}
    checked = 0
    for o in origins:
        # at origin o the last observed month is o: forecasts whose target is o are scored first
        for h in range(1, H + 1):
            for k, (lo, hi) in pending.pop((str(pd.Period(o, "M") - h), h), {}).items():
                y = hist.loc[o, uids].to_numpy()
                alpha[k][:, h - 1] = aci_step(alpha[k][:, h - 1], k[1], LEVELS[k[0]], y, lo, hi)
        g = iv[iv.origin == o]
        for (lv, gm, h), d in g.groupby(["level", "gamma", "horizon"]):
            d = d.set_index("unique_id").loc[uids]
            assert np.abs(d.alpha.to_numpy() - alpha[(lv, gm)][:, h - 1]).max() < 1e-12, (o, lv, gm, h)
            checked += len(d)
            pending.setdefault((o, h), {})[(lv, gm)] = (d.lo.to_numpy(), d.hi.to_numpy())
    # state after the last month (2025-06): every forecast with a target <= 2025-06 scored
    last = origins[-1]
    for (o, h), d in list(pending.items()):
        t = str(pd.Period(o, "M") + h)
        if t <= last:
            raise AssertionError(f"unscored pending forecast {o} h{h}")
    st = pd.read_parquet(p_st, filters=[("state_at", "==", last), ("unique_id", "in", uids)])
    for (lv, gm, h), d in st.groupby(["level", "gamma", "horizon"]):
        d = d.set_index("unique_id").loc[uids]
        assert np.abs(d.alpha.to_numpy() - alpha[(lv, gm)][:, h - 1]).max() < 1e-12
    assert checked > 10000
