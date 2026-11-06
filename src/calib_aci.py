"""
calib_aci.py: extended calibration origins and the continued ACI recursion.

Origins 2025-07 ... 2026-05 (11), horizons 1-12, both tracks. They are used only to update the calibration state,
never for evaluation. For every origin, samples are drawn and reconciled exactly as in the evaluation
(prob_samples.py: Gaussian copula with the track's MinT W at the origin, S = 500, seed (42, origin), free MinT
projection with negatives set to zero), from the base forecasts of calib_base.py / fm_calib.py. The point forecasts
of these origins are reconciled separately with the exact QP (reconcile.py --tag calib).

ACI recursion (as in prob_samples.py; both gammas are carried, gamma = 0.05 is the main one):
  - start: the state of the evaluation origins after month 2025-06 (results/prob_main/track=X/aci_state.parquet,
    state_at 2025-06), plus the intervals of the evaluation origins 2024-07 ... 2025-06 whose targets lie after
    2025-06 (pending);
  - for every month t = 2025-07 ... 2026-06, in order: every pending interval with target t is scored and
    alpha(series, horizon) <- clip(alpha + gamma (alpha_nominal - err), 0.001, 0.999); then, if t is an extended
    origin, its intervals are made with the current alpha (Q(alpha/2), Q(1 - alpha/2) of the reconciled samples)
    and become pending.
  - Nesting is applied to the stored intervals: the 80% interval is clipped to lie inside the 95% interval.
    The recursion scores each level's own interval, as in prob_samples.py, so that it continues the same recursion.
Check: replaying 2025-07 ... 2026-06 with the evaluation-origin intervals only reproduces the stored evaluation-origin
state at 2026-06.

Outputs (results/calib/aci/track=<FM|STAT>/):
  aci_state.parquet       state_at = "2026-06_ext": unique_id, horizon, level, gamma, alpha (the final state)
  aci_intervals.parquet   intervals of the extended origins (lo, hi raw; lo_nested, hi_nested), with y where known
  coverage.csv            for information only: empirical coverage of the extended origins' intervals with a
                          known target, by level and horizon, all targets, targets in 2025 and targets in 2026
                          (after the 2026 reclassification)
  sampling_diagnostics.csv, check_replay.csv
Usage:
  python src/calib_aci.py --track FM
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import cholesky
from scipy.stats import norm

from prob_samples import ALPHA_CLIP, GAMMAS, H_MAIN, LEVELS, S, SEED, TRACKS, quantile_map, sample_quantile
from reconcile import load_system, load_w_residuals, mint_w, origin_constraints, project, residual_matrix
from runlog import RunLog, out_path

ORIGINS = pd.period_range("2025-07", "2026-05", freq="M").astype(str).tolist()
MONTHS = pd.period_range("2025-07", "2026-06", freq="M").astype(str).tolist()
KEYS = [(lv, g) for lv in LEVELS for g in GAMMAS]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", required=True, choices=list(TRACKS))
    ap.add_argument("--clean", default="data/clean")
    ap.add_argument("--results", default="results")
    ap.add_argument("--forecasts-dir", default="results/calib/base_forecasts")
    ap.add_argument("--residuals-dir", default="results/calib/insample_residuals")
    ap.add_argument("--onestep-dirs", nargs="+", default=["results/onestep_fm", "results/onestep_fm_rt"])
    ap.add_argument("--out", default="results/calib/aci")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    log = RunLog("calib_aci.py", vars(args))
    RES, T = Path(args.results), TRACKS[args.track]
    OUT = Path(args.out) / f"track={args.track}"

    sysd = load_system(RES, args.clean)
    uids, n, hist = sysd["uids"], sysd["n"], sysd["Yh"]
    res, own_fm, w_from = load_w_residuals(RES, T["base"], T["fm_w"], 12, args.residuals_dir, args.onestep_dirs)

    # ---- evaluation-origin state at 2025-06 and the pending evaluation-origin intervals
    st = pd.read_parquet(RES / "prob_main" / f"track={args.track}" / "aci_state.parquet")

    def state(label):
        out = {}
        for lv, g in KEYS:
            s = st[(st.state_at == label) & (st.level == lv) & (st.gamma == g)]
            out[(lv, g)] = s.pivot(index="unique_id", columns="horizon", values="alpha").loc[uids].to_numpy()
        return out

    ai = pd.read_parquet(RES / "prob_main" / f"track={args.track}" / "aci_intervals.parquet",
                         filters=[("origin", ">=", "2024-07")])
    ai = ai[ai.target > "2025-06"]
    pend_eval = {}
    for (o, h, lv, g), d in ai.groupby(["origin", "horizon", "level", "gamma"]):
        d = d.set_index("unique_id").loc[uids]
        pend_eval.setdefault((o, int(h)), {})[(int(lv), float(g))] = (d.lo.to_numpy(), d.hi.to_numpy())

    def update(alpha, pending, t):
        y = hist.loc[t].to_numpy()
        for h in range(1, H_MAIN + 1):
            o_prev = str(pd.Period(t, "M") - h)
            for key, (lo, hi) in pending.pop((o_prev, h), {}).items():
                err = ((y < lo) | (y > hi)).astype(float)
                alpha[key][:, h - 1] = np.clip(alpha[key][:, h - 1] + key[1] * (LEVELS[key[0]] - err), *ALPHA_CLIP)

    # ---- check: replaying with the evaluation-origin intervals only reproduces the stored state at 2026-06
    alpha = state("2025-06")
    pending = {k: dict(v) for k, v in pend_eval.items()}
    for t in MONTHS:
        update(alpha, pending, t)
    ref = state("2026-06")
    rep = max(float(np.abs(alpha[k] - ref[k]).max()) for k in KEYS)
    pd.DataFrame([{"track": args.track, "max_abs_diff_alpha": rep, "pending_left": len(pending)}]).to_csv(
        out_path(OUT / "check_replay.csv", args.overwrite), index=False)
    print(f"replay check: max |alpha - evaluation-origin state 2026-06| = {rep:.3g}")
    if rep > 1e-12:
        raise SystemExit("replay does not reproduce the evaluation-origin state: stop")

    # ---- the extended recursion
    f = pd.read_parquet(Path(args.forecasts_dir) / f"model={T['base']}" / "forecasts.parquet")
    f = f[f.unique_id.isin(uids) & f.origin.isin(ORIGINS)].set_index(["origin", "unique_id", "horizon"]).sort_index()
    zk = norm.ppf(T["p"])
    alpha = state("2025-06")
    pending = {k: dict(v) for k, v in pend_eval.items()}
    rows, diag = [], []
    for t in MONTHS:
        update(alpha, pending, t)
        if t not in ORIGINS:
            continue
        t0 = time.perf_counter()
        o = t
        A, Acheck, r, dead, _ = origin_constraints(sysd, o)
        Rm, nm = residual_matrix(res, o, own_fm, uids)
        W, lam = mint_w(Rm)
        L = cholesky(W, lower=True)
        sd = np.sqrt(np.diag(W))
        Lc = cholesky(W / np.outer(sd, sd), lower=True)
        fo = f.loc[o]
        Q = np.sort(np.stack([fo[c].unstack("horizon").loc[uids].to_numpy() for c in T["qcols"]], axis=2), axis=2)
        rng = np.random.default_rng([SEED, int(o.replace("-", ""))])
        Xb = np.empty((H_MAIN, S, n))
        for h in range(H_MAIN):
            Xb[h] = quantile_map(rng.standard_normal((S, n)) @ Lc.T, zk, Q[:, h, :])
        Xr = project(Xb.transpose(2, 0, 1).reshape(n, H_MAIN * S), A, L, False)[0].reshape(n, H_MAIN, S).transpose(1, 2, 0)
        share_neg = float((Xr < 0).mean())
        Xr = np.maximum(Xr, 0.0)
        Xs = np.sort(Xr, axis=1)
        for h in range(H_MAIN):
            tgt = str(pd.Period(o, "M") + h + 1)
            yv = hist.loc[tgt].to_numpy() if tgt in hist.index else np.full(n, np.nan)
            per = {}
            for key in KEYS:
                a = alpha[key][:, h].copy()
                per[key] = (sample_quantile(Xs[h], a / 2), sample_quantile(Xs[h], 1 - a / 2), a)
            pending[(o, h + 1)] = {k: v[:2] for k, v in per.items()}
            for g in GAMMAS:
                lo95, hi95, a95 = per[(95, g)]
                lo80, hi80, a80 = per[(80, g)]
                for lv, lo, hi, a, lon, hin in [(95, lo95, hi95, a95, lo95, hi95),
                                                (80, lo80, hi80, a80, np.maximum(lo80, lo95), np.minimum(hi80, hi95))]:
                    rows.append(pd.DataFrame({"unique_id": uids, "origin": o, "horizon": h + 1, "target": tgt,
                                              "level": lv, "gamma": g, "alpha": a, "lo": lo, "hi": hi,
                                              "lo_nested": lon, "hi_nested": hin, "y": yv}))
        diag.append({"origin": o, "w_from": w_from, "w_months": nm, "mint_lambda": lam, "r": r, "n_dead": len(dead),
                     "share_neg_rec_set_to_zero": share_neg, "seconds": time.perf_counter() - t0})
        print(f"{args.track} {o}: {time.perf_counter() - t0:.1f} s, negatives set to 0 {share_neg:.3%}", flush=True)
    assert not any(k[0] <= "2026-05" and str(pd.Period(k[0], "M") + k[1]) <= "2026-06" for k in pending)

    srows = []
    for (lv, g), a in alpha.items():
        for h in range(H_MAIN):
            srows.append(pd.DataFrame({"state_at": "2026-06_ext", "unique_id": uids, "horizon": h + 1, "level": lv,
                                       "gamma": g, "alpha": a[:, h]}))
    iv = pd.concat(rows, ignore_index=True)
    known = iv[iv.y.notna()].copy()
    known["covered"] = ((known.y >= known.lo_nested) & (known.y <= known.hi_nested)).astype(float)
    known["covered_raw"] = ((known.y >= known.lo) & (known.y <= known.hi)).astype(float)
    cov = []
    for sub, mask in [("all targets 2025-08 to 2026-06", known.target <= "2026-06"),
                      ("targets in 2025 (before the 2026 reclassification)", known.target <= "2025-12"),
                      ("targets in 2026 (after the 2026 reclassification)", known.target >= "2026-01")]:
        k = known[mask]
        c = k.groupby(["level", "gamma", "horizon"]).agg(coverage=("covered", "mean"), coverage_raw=("covered_raw", "mean"),
                                                         cells=("covered", "size")).reset_index()
        tot = k.groupby(["level", "gamma"]).agg(coverage=("covered", "mean"), coverage_raw=("covered_raw", "mean"),
                                                cells=("covered", "size")).reset_index().assign(horizon="all")
        cov.append(pd.concat([c, tot]).assign(subset=sub))
    cov = pd.concat(cov, ignore_index=True)
    cov.to_csv(out_path(OUT / "coverage.csv", args.overwrite), index=False, float_format="%.4f")
    pd.concat(srows, ignore_index=True).to_parquet(out_path(OUT / "aci_state.parquet", args.overwrite), index=False)
    iv.to_parquet(out_path(OUT / "aci_intervals.parquet", args.overwrite), index=False)
    pd.DataFrame(diag).to_csv(out_path(OUT / "sampling_diagnostics.csv", args.overwrite), index=False, float_format="%.6g")
    print(cov[(cov.horizon == "all") & (cov.gamma == 0.05)].to_string(index=False))
    log.finish(f"{args.track}: {len(ORIGINS)} origins; replay check {rep:.1e}")


if __name__ == "__main__":
    main()
