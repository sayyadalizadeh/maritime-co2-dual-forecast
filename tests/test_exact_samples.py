"""The exactly reconciled samples and the ACI recursion on them (src/exact_samples.py).

1. The exact samples of one calibration origin and the real-time origin 2026-06 satisfy every constraint row of the
   origin (C y = 0 within 1e-6 % of the World residence total) and are non-negative (>= NEG_TOL).
2. Replaying the ACI rule (written independently in test_aci.py) over the stored exact intervals of all 42 origins, from
   the nominal alpha, reproduces the stored alpha used at every origin and the stored states at 2025-06 and 2026-06_ext.
3. The final FM intervals of results/final_exact use the same samples as results/final (identical quantiles and
   points); only the ACI columns differ.
4. The intervals of origin 2026-06 in the test on later-published data use the ACI state of the S1 intervals
   (2026-06_ext), and T5a reads that evaluation; the robustness table (T5a_realtime_eval_state.csv) reads the
   evaluation with the state of the evaluation origins.
"""
import numpy as np
import pandas as pd
import pytest

from conftest import CLEAN, RES, need
from prob_samples import GAMMAS, LEVELS
from test_aci import aci_step

R4 = RES


@pytest.mark.parametrize("source,origin", [("calib", "2025-12"), ("realtime", "2026-06")])
@pytest.mark.parametrize("track", ["FM", "STAT"])
def test_exact_samples_coherent_and_nonnegative(track, source, origin):
    from reconcile import NEG_TOL, load_system, origin_constraints
    p = R4 / "exact_samples" / source / f"track={track}" / f"origin={origin}.npy"
    need(p, RES / "C_matrix_fullrank_v2.npz", CLEAN / "panel_system_wide_v2.parquet")
    sysd = load_system(RES, CLEAN)
    _, Acheck, _, _, _ = origin_constraints(sysd, origin)
    X = np.load(p)                                                     # (H, S, n)
    iw = sysd["uids"].index("W|RES_TOTAL_SYS")
    viol = np.abs(np.einsum("rn,hsn->hsr", Acheck, X)).max(axis=2) / X[:, :, iw] * 100
    assert viol.max() < 1e-6
    assert X.min() >= NEG_TOL


@pytest.mark.parametrize("track", ["STAT", "FM"])
def test_replay_exact_aci_all_origins(track):
    p_iv = R4 / "aci_exact" / f"track={track}" / "aci_intervals.parquet"
    p_st = R4 / "aci_exact" / f"track={track}" / "aci_state.parquet"
    need(p_iv, p_st, CLEAN / "panel_system_wide_v2.parquet")
    hist = pd.read_parquet(CLEAN / "panel_system_wide_v2.parquet")
    uids = sorted(pd.read_parquet(p_st, columns=["unique_id"]).unique_id.unique())[::40]     # 28 series
    iv = pd.read_parquet(p_iv, filters=[("unique_id", "in", uids)])
    origins = sorted(iv.origin.unique())
    assert len(origins) == 42 and origins[0] == "2022-12" and origins[-1] == "2026-05"
    keys = [(lv, g) for lv in LEVELS for g in GAMMAS]
    H = int(iv.horizon.max())
    alpha = {k: np.full((len(uids), H), LEVELS[k[0]]) for k in keys}
    pending = {}
    st = pd.read_parquet(p_st, filters=[("unique_id", "in", uids)])

    def check_state(label):
        for (lv, gm, h), d in st[st.state_at == label].groupby(["level", "gamma", "horizon"]):
            d = d.set_index("unique_id").loc[uids]
            assert np.abs(d.alpha.to_numpy() - alpha[(lv, gm)][:, h - 1]).max() < 1e-12, (label, lv, gm, h)

    for t in pd.period_range(origins[0], "2026-06", freq="M").astype(str):
        for h in range(1, H + 1):
            for k, (lo, hi) in pending.pop((str(pd.Period(t, "M") - h), h), {}).items():
                y = hist.loc[t, uids].to_numpy()
                alpha[k][:, h - 1] = aci_step(alpha[k][:, h - 1], k[1], LEVELS[k[0]], y, lo, hi)
        if t == "2025-06":
            check_state("2025-06")
        g = iv[iv.origin == t]
        for (lv, gm, h), d in g.groupby(["level", "gamma", "horizon"]):
            d = d.set_index("unique_id").loc[uids]
            assert np.abs(d.alpha.to_numpy() - alpha[(lv, gm)][:, h - 1]).max() < 1e-12, (t, lv, gm, h)
            pending.setdefault((t, h), {})[(lv, gm)] = (d.lo.to_numpy(), d.hi.to_numpy())
    check_state("2026-06_ext")


def test_final_fm_only_aci_columns_change():
    pz, pe = RES / "final" / "track=FM" / "forecasts.parquet", R4 / "final_exact" / "track=FM" / "forecasts.parquet"
    need(pz, pe)
    z, e = pd.read_parquet(pz), pd.read_parquet(pe)
    assert (z.unique_id.to_numpy() == e.unique_id.to_numpy()).all() and (z.horizon.to_numpy() == e.horizon.to_numpy()).all()
    for c in ["point"] + [c for c in z.columns if c.startswith("q0")]:
        assert np.array_equal(z[c].to_numpy(), e[c].to_numpy()), c
    assert (e.lo80 >= e.lo95).all() and (e.hi80 <= e.hi95).all()                        # nesting


@pytest.mark.parametrize("track", ["FM", "STAT"])
def test_realtime_2026_06_uses_s1_state(track):
    """At origin 2026-06 the intervals of the test on later-published data (T5a, T5b, Fig6) use the ACI state
    2026-06_ext, the state of the S1 intervals (final_bands.py); origin 2025-06 keeps the state at 2025-06. T5a is read
    from the evaluation with these intervals; the robustness table reads the evaluation with the evaluation-origin state."""
    p_ext = R4 / "prob_realtime_exact" / f"track={track}" / "aci_intervals_ext.parquet"
    p_st = R4 / "aci_exact" / f"track={track}" / "aci_state.parquet"
    p_t5, p_t5r = RES / "paper" / "T5a_realtime.csv", RES / "paper" / "T5a_realtime_eval_state.csv"
    need(p_ext, p_st, p_t5, p_t5r)
    iv = pd.read_parquet(p_ext)
    assert set(iv[iv.origin == "2026-06"].state_at) == {"2026-06_ext"}
    assert set(iv[iv.origin == "2025-06"].state_at) == {"2025-06"}
    st = pd.read_parquet(p_st, filters=[("state_at", "==", "2026-06_ext")])
    m = iv[iv.origin == "2026-06"].merge(st, on=["unique_id", "level", "gamma", "horizon"], suffixes=("", "_st"))
    assert len(m) == (iv.origin == "2026-06").sum() and np.array_equal(m.alpha.to_numpy(), m.alpha_st.to_numpy())
    for path, src in [(p_t5, "eval_realtime_exact_ext_nested"), (p_t5r, "eval_realtime_exact_nested")]:
        t5 = pd.read_csv(path).query("track == @track")
        s = pd.read_csv(R4 / src / "summary_res_total_pub.csv")
        s = s[(s.track == track) & (s.d_variant == "d") & (s.level == "economies") & (s.period != "all incl. 2025-09")]
        for r in t5.itertuples():
            x = s[(s.origin == r.origin) & (s.period == r.targets)].set_index("metric")
            assert np.isclose(r.cov80_aci, x.at["cov80_aci", "mean"], rtol=1e-5), (path.name, r.origin, r.targets)
            assert np.isclose(r.cov95_aci, x.at["cov95_aci", "mean"], rtol=1e-5), (path.name, r.origin, r.targets)
