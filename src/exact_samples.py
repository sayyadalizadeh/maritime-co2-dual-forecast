"""
exact_samples.py: exactly reconciled samples throughout.

exact_eval.py makes the samples of the 31 evaluation origins exact. Here the remaining samples of the paper are made
exact with the same solver and checks (reconcile.nonneg_nullspace through qp_pool.solve_pool: worker processes, a time
limit of 120 s per sample, then quadprog on the rescaled sample and OSQP; every solution must pass the coherence,
non-negativity and KKT checks), and the ACI recursion and everything built on it are rerun on exact samples. No model
is refitted; the samples are regenerated exactly as before (Gaussian copula from the track's MinT W at the origin,
S = 500, standard normals from default_rng([42, origin]) drawn horizon by horizon, free MinT projection), and only the
treatment of negative values changes: "zeroed" (negatives set to 0) against "exact" (exact non-negative QP).

Sample sources (SOURCES):
  calib     calibration origins 2025-07 ... 2026-05 (calibration only), horizons 1-12; inputs of calib_aci.py
            (results/calib/base_forecasts, results/calib/insample_residuals, one-step errors to 2026-06)
  realtime  real-time origins 2025-06 (h 1-15) and 2026-06 (h 1-3); inputs of prob_samples.py --mode realtime
            (results/realtime/base_forecasts, results/realtime/insample_residuals)
  final     final origin 2026-06, h 1 ... H* - 1 (FM 12, STAT 7); inputs of final_samples.py. The FM final samples
            are already exact (reconcile.nonneg_gi): they are checked and reused, not solved again. The STAT final
            samples (stored zeroed in results/final) are solved here, because S1a reports both tracks.

Parts
  qp        --source S --track T: the exact QP for every sample of every origin of the source (one .npy per origin in
            results/exact_samples/<source>/track=<T>/, 54-67 MB, not committed to git; written last, so an
            interrupted run resumes at the next origin), with per-sample diagnostics (origin=<O>_diag.csv) and every
            failed or fallback stage (origin=<O>_fallback.csv). Before solving, the regenerated samples are checked
            against the stored zeroed outputs: calib, the share of negative values set to zero
            (results/calib/aci/track=T/sampling_diagnostics.csv); realtime, the stored base and zeroed CRPS
            (results/prob_realtime); final, the stored base samples and (STAT) the stored zeroed samples (results/final).
  check     all sources, both tracks: coherence, non-negativity and multipliers of every exact sample; fallback samples;
            the real-time samples of origin 2025-06, h 1-12, against the exact evaluation samples of the same origin; the
            FM final samples (results/final/track=FM/samples_rec.npy): coherence and non-negativity, and a re-solve of
            all 6,000 with the null-space solver through qp_pool (its checks certify the multipliers, which the
            final_samples.py diagnostics record as 0 for every sample); exact_samples/final/track=FM/resolve_diag.csv.
  aci       --track T: the ACI recursion (prob_samples.py / calib_aci.py: both levels, gamma 0.05 and 0.01, alpha starts at
            the nominal value, alpha <- clip(alpha + gamma (alpha_nominal - err), 0.001, 0.999), each level scoring its
            own unnested interval) over all origins in month order: the 31 evaluation origins 2022-12 ... 2025-06 (exact
            samples of exact_eval.py) and the 11 calibration origins 2025-07 ... 2026-05, outcomes to 2026-06. Two
            recursions run side by side: "evaluation" (the evaluation origins only; its state at 2026-06 is the one the
            robustness version uses for the real-time origin 2026-06) and "all" (evaluation and calibration origins;
            its state at 2026-06, "2026-06_ext", is the state of the final intervals). Checks: the evaluation-origin
            intervals and the states at 2025-06 and 2026-06 reproduce exact_eval.py (results/prob_main_exact) exactly;
            on the regenerated zeroed calibration samples, starting from the stored zeroed-sample state and intervals,
            the same code reproduces the stored calib_aci.py intervals wherever alpha agrees and every divergence
            starts at a tie event (y within 1e-9 t/day of a bound).
            Output: results/aci_exact/track=<T>/ (aci_intervals.parquet, aci_state.parquet, coverage of the
            calibration origins for information, checks).
  realtime  --track T: the real-time origins scored from the exact samples, in the layout of results/prob_realtime
            (quantiles, base and reconciled CRPS, joint scores, RES_TOTAL_SYS samples, ACI intervals), so that
            evaluate_realtime.py --prob-dir can read it. ACI intervals with the evaluation-origin state: origin 2025-06
            with the state at 2025-06, origin 2026-06 with the evaluation recursion's state at 2026-06
            (aci_intervals.parquet); variant with the extended state 2026-06_ext for origin 2026-06
            (aci_intervals_ext.parquet). Horizons above 12 use the state of horizon 12. Output:
            results/prob_realtime_exact/track=<T>/. The paper outputs (T5a, T5b, Fig6) use aci_intervals_ext.parquet,
            the state of the S1 intervals; the evaluation-origin state (aci_intervals.parquet) is the robustness
            version. In the compare part, "exact" is the evaluation-origin state and "exact_ext_state" the paper
            version.
  compare   after evaluate_realtime.py and final_bands.py on the exact inputs (see Usage): zeroed against exact for the
            real-time tables (T5a, T5b), the final intervals (S1 ACI columns, S1a floor shares), the ACI states and the
            calibration-origin coverage. Output: results/compare/.

Usage:
  python src/exact_samples.py --part qp --source calib --track FM --n-jobs 8      (and STAT; realtime; final STAT)
  python src/exact_samples.py --part check
  python src/exact_samples.py --part aci --track FM                              (and STAT)
  python src/exact_samples.py --part realtime --track FM                         (and STAT)
  python src/evaluate_realtime.py --d-form exact --nest --prob-dir results/prob_realtime_exact
         --main-prob-dir results/prob_main_exact --out results/eval_realtime_exact_nested
  python src/final_bands.py --track FM --samples results/final/track=FM/samples_rec.npy
         --aci-state results/aci_exact/track=FM/aci_state.parquet --out results/final_exact
  python src/exact_samples.py --part compare
"""
import argparse
import faulthandler
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import cholesky
from scipy.stats import norm

from evalcore import load_history, origin_scales
from prob_samples import (ALPHA_CLIP, GAMMAS, H_MAIN, LEVELS, QNAMES, S, SEED, STORE_Q, TRACKS, VS_TOP, crps_fair,
                          energy_score, quantile_map, sample_quantile, variogram_score)
from qp_pool import solve_pool
from reconcile import (NEG_TOL, load_system, load_w_residuals, mint_w, nullspace_setup,
                       origin_constraints, project, residual_matrix)
from exact_eval import ACI, TIE_TOL
from runlog import RunLog, out_path

OUT = Path("results")
QP = OUT / "exact_samples"
ROOT_RES = Path("results")                                                 # exact evaluation samples and ACI (exact_eval.py)
EVAL_ORIGINS = pd.period_range("2022-12", "2025-06", freq="M").astype(str).tolist()
CAL_ORIGINS = pd.period_range("2025-07", "2026-05", freq="M").astype(str).tolist()
FM_ONESTEP = ["results/onestep_fm", "results/onestep_fm_rt"]
SOURCES = {
    "calib": {"origins": CAL_ORIGINS, "H": {o: H_MAIN for o in CAL_ORIGINS}, "fdir": "results/calib/base_forecasts",
              "rdir": "results/calib/insample_residuals"},
    "realtime": {"origins": ["2025-06", "2026-06"], "H": {"2025-06": 15, "2026-06": 3},
                 "fdir": "results/realtime/base_forecasts", "rdir": "results/realtime/insample_residuals"},
    "final": {"origins": ["2026-06"], "H": None, "fdir": "results/calib/base_forecasts",
              "rdir": "results/calib/insample_residuals"},
}


def final_H(track, results="results"):
    """Horizons of the final samples (final_samples.FINAL_H)."""
    from final_samples import FINAL_H
    return FINAL_H[track]


class GenSampler:
    """Regenerates the samples of one track and source exactly as prob_samples.py (realtime), calib_aci.py and
    final_samples.py draw them (same marginals, copula, seed and projection)."""

    def __init__(self, track, source, results="results", clean="data/clean"):
        RES = Path(results)
        self.track, self.source = track, source
        self.T = T = TRACKS[track]
        src = SOURCES[source]
        self.sysd = load_system(RES, clean)
        self.uids, self.n = self.sysd["uids"], self.sysd["n"]
        self.hist = self.sysd["Yh"]
        self.meta = pd.read_parquet(RES / "base_forecasts_v2" / "series_meta.parquet").set_index("unique_id").loc[self.uids]
        self.sc = origin_scales(load_history(clean)).pivot(index="origin", columns="unique_id", values="scale1")[self.uids]
        self.res, self.own, self.w_from = load_w_residuals(RES, T["base"], T["fm_w"], 12, src["rdir"], FM_ONESTEP)
        self.origins = src["origins"]
        self.H = src["H"] or {o: final_H(track, results) for o in self.origins}
        f = pd.read_parquet(Path(src["fdir"]) / f"model={T['base']}" / "forecasts.parquet",
                            columns=["unique_id", "origin", "horizon", "mean"] + [c for c in T["qcols"] if c != "mean"])
        self.f = f[f.unique_id.isin(self.uids) & f.origin.isin(self.origins)].set_index(["origin", "unique_id", "horizon"]) \
            .sort_index()
        self.zk = norm.ppf(T["p"])
        rts = self.hist.loc["2025-01":"2025-12", [u for u in self.uids if u.endswith("|RES_TOTAL_SYS")]].mean()
        eco = [u for u in rts.index if self.meta.at[u, "unit_type"] == "economy"]
        top = [u.split("|")[0] for u in rts[eco].sort_values(ascending=False).index[:VS_TOP]]
        self.vs_idx = np.array([i for i, u in enumerate(self.uids) if u.split("|")[0] in ["W", "RESIDUAL"] + top])
        self.iw = self.uids.index("W|RES_TOTAL_SYS")

    def origin(self, o):
        """(Xb base samples (H, S, n), Xf free projection (H, S, n), A, Acheck, L, dead, targets, scale)."""
        A, Acheck, r, dead, _ = origin_constraints(self.sysd, o)
        Rm, _ = residual_matrix(self.res, o, self.own, self.uids)
        W, _ = mint_w(Rm)
        L = cholesky(W, lower=True)
        sd = np.sqrt(np.diag(W))
        Lc = cholesky(W / np.outer(sd, sd), lower=True)
        fo = self.f.loc[o]
        H = self.H[o]
        Q = np.sort(np.stack([fo[c].unstack("horizon").loc[self.uids].to_numpy() for c in self.T["qcols"]], axis=2),
                    axis=2)[:, :H, :]
        rng = np.random.default_rng([SEED, int(o.replace("-", ""))])
        Xb = np.empty((H, S, self.n))
        for h in range(H):
            Xb[h] = quantile_map(rng.standard_normal((S, self.n)) @ Lc.T, self.zk, Q[:, h, :])
        Xf = project(Xb.transpose(2, 0, 1).reshape(self.n, H * S), A, L, False)[0].reshape(self.n, H, S).transpose(1, 2, 0)
        targets = [str(pd.Period(o, "M") + h) for h in range(1, H + 1)]
        return Xb, Xf, A, Acheck, L, dead, targets, self.sc.loc[o].to_numpy()


def sample_dir(source, track):
    return QP / source / f"track={track}"


# ---------------------------------------------------------------- part qp
def zeroed_check(smp, o, Xb, Xf, targets, args):
    """The regenerated samples against the stored zeroed outputs of the source (see the module docstring)."""
    RES = Path(args.results)
    Xz = np.maximum(Xf, 0.0)
    out = {"source": smp.source, "track": smp.track, "origin": o}
    if smp.source == "calib":
        d = pd.read_csv(RES / "calib" / "aci" / f"track={smp.track}" / "sampling_diagnostics.csv").set_index("origin")
        out["share_neg_regenerated"] = float((Xf < 0).mean())
        out["share_neg_stored"] = float(d.at[o, "share_neg_rec_set_to_zero"])
        out["max_abs_diff"] = abs(out["share_neg_regenerated"] - out["share_neg_stored"])
        ok = np.isclose(out["share_neg_regenerated"], out["share_neg_stored"], rtol=1e-5, atol=0)   # stored to 6 digits
    elif smp.source == "realtime":
        c = pd.read_parquet(RES / "prob_realtime" / f"track={smp.track}" / "crps.parquet", filters=[("origin", "==", o)])
        mx = 0.0
        for h in range(Xb.shape[0]):
            if targets[h] not in smp.hist.index:
                continue
            y = smp.hist.loc[targets[h]].to_numpy()
            ref = c[c.horizon == h + 1].pivot(index="unique_id", columns="kind", values="crps").loc[smp.uids]
            mx = max(mx, float(np.abs(crps_fair(Xb[h], y) - ref["base"].to_numpy()).max()),
                     float(np.abs(crps_fair(Xz[h], y) - ref["rec"].to_numpy()).max()))
        out["max_abs_diff"] = mx                                      # CRPS, t/day (no target for origin 2026-06)
        ok = mx < 1e-6
    else:
        P = RES / "final" / f"track={smp.track}"
        mx = float(np.abs(np.load(P / "samples_base.npy") - Xb).max())
        if smp.track == "STAT":
            mx = max(mx, float(np.abs(np.load(P / "samples_rec.npy") - Xz).max()))
        out["max_abs_diff"] = mx                                      # samples, t/day
        ok = mx < 1e-6
    out["ok"] = bool(ok)
    return out


def solve_origin(smp, o, Xf, A, Acheck, L, dead, n_jobs, tag, progress):
    """Exact samples (H, S, n), per-sample diagnostics and fallback attempts of one origin (as exact_eval)."""
    M, bnd, Cb = nullspace_setup(A, L, dead)
    H = Xf.shape[0]
    cols = Xf.reshape(H * S, smp.n).T

    def check(y):
        c = float(np.abs(Acheck @ y).max())
        return {"max_abs_Cy": c, "max_violation_pct": 100 * c / y[smp.iw]}

    Y, rows, attempts = solve_pool(cols, M, bnd, Cb, check, n_jobs, tag, progress=progress)
    Xq = Y.T.reshape(H, S, smp.n)
    diag = pd.DataFrame(rows)
    diag.insert(0, "horizon", np.repeat(np.arange(1, H + 1), S))
    diag.insert(1, "sample", np.tile(np.arange(S), H))
    diag["min_y"] = Xq.min(axis=2).reshape(-1)
    Cy = np.abs(np.einsum("rn,hsn->hsr", Acheck, Xq)).max(axis=2)
    diag["max_abs_Cy"] = Cy.reshape(-1)
    diag["max_violation_pct"] = (100 * Cy / Xq[:, :, smp.iw]).reshape(-1)
    att = pd.DataFrame(attempts)
    if len(att):
        att.insert(0, "horizon", att.column // S + 1)
        att.insert(1, "sample", att.column % S)
    return Xq, diag, att


def part_qp(args):
    if args.source == "final" and args.track == "FM":
        sys.exit("The FM final samples are already exact: see --part check.")
    smp = GenSampler(args.track, args.source, args.results, args.clean)
    D = sample_dir(args.source, args.track)
    D.mkdir(parents=True, exist_ok=True)
    progress = str(QP / f"qp_progress_{args.source}_{args.track}.log")
    zc = []
    for o in smp.origins:
        p = D / f"origin={o}.npy"
        if p.exists() and not args.overwrite:
            continue
        t0 = time.perf_counter()
        Xb, Xf, A, Acheck, L, dead, targets, _ = smp.origin(o)
        z = zeroed_check(smp, o, Xb, Xf, targets, args)
        zc.append(z)
        print(f"{args.source} {args.track} {o}: regenerated against the stored zeroed outputs: max diff {z['max_abs_diff']:.2e}",
              flush=True)
        if not z["ok"]:
            sys.exit(f"the regenerated samples do not reproduce the stored outputs: {z}")
        Xq, diag, att = solve_origin(smp, o, Xf, A, Acheck, L, dead, args.n_jobs, f"{args.source} {args.track} {o}", progress)
        secs = time.perf_counter() - t0
        diag.assign(source=args.source, track=args.track, origin=o, wall_seconds_origin=secs) \
            .to_csv(D / f"origin={o}_diag.csv", index=False, float_format="%.6g")
        if len(att):
            att.assign(source=args.source, track=args.track, origin=o) \
                .to_csv(D / f"origin={o}_fallback.csv", index=False, float_format="%.6g")
        tmp = D / f"origin={o}.tmp.npy"                               # the .npy marks the origin as done: write it last
        np.save(tmp, Xq)
        tmp.replace(p)
        by = diag[diag.qp].solver.value_counts().to_dict()
        print(f"{args.source} {args.track} {o}: {secs:.0f} s, {int(diag.qp.sum())} QPs {by}, "
              f"flags {int((diag.flag.fillna('') != '').sum())}, min multiplier {diag.min_multiplier.min():.2e}, "
              f"max violation {diag.max_violation_pct.max():.1e}%, min y {diag.min_y.min():.1e}", flush=True)
    if zc:
        f = QP / f"zeroed_reproduction_{args.source}_{args.track}.csv"
        old = pd.read_csv(f) if f.exists() else pd.DataFrame()
        pd.concat([old, pd.DataFrame(zc)], ignore_index=True).drop_duplicates(["source", "track", "origin"], keep="last") \
            .to_csv(f, index=False, float_format="%.6g")


# ---------------------------------------------------------------- part check
def part_check(args):
    rows, fb = [], []
    for source in ["calib", "realtime", "final"]:
        for track in TRACKS:
            D = sample_dir(source, track)
            for p in sorted(D.glob("origin=*_diag.csv")):
                d = pd.read_csv(p, keep_default_na=False)
                o = p.name.split("_diag")[0].replace("origin=", "")
                q = d[d.qp.astype(str) == "True"]
                rows.append({"source": source, "track": track, "origin": o, "samples": len(d), "qps": len(q),
                             "solved_by_quadprog": int((q.solver == "quadprog").sum()),
                             "solved_by_fallback": int((q.solver != "quadprog").sum()),
                             "flags": int((d.flag.astype(str) != "").sum()),
                             "min_multiplier": pd.to_numeric(q.min_multiplier).min(),
                             "max_abs_Cy": pd.to_numeric(d.max_abs_Cy).max(),
                             "max_violation_pct": pd.to_numeric(d.max_violation_pct).max(),
                             "min_y": pd.to_numeric(d.min_y).min(), "wall_seconds_origin": float(d.wall_seconds_origin.iloc[0]),
                             "qp_seconds_sum": pd.to_numeric(d.seconds).sum()})
            for p in sorted(D.glob("origin=*_fallback.csv")):
                fb.append(pd.read_csv(p).assign(source=source, track=track))
    c = pd.DataFrame(rows)
    c.to_csv(out_path(QP / "qp_summary.csv", args.overwrite), index=False, float_format="%.6g")
    fbd = pd.concat(fb, ignore_index=True) if fb else pd.DataFrame(columns=["source", "track", "origin", "horizon", "sample"])
    fbd.to_csv(out_path(QP / "fallback_samples.csv", args.overwrite), index=False, float_format="%.6g")
    print(c.to_string(index=False))
    print(f"fallback attempts: {len(fbd)}")

    # real-time 2025-06, h 1-12 against the exact evaluation samples
    rt = []
    for track in TRACKS:
        a = np.load(sample_dir("realtime", track) / "origin=2025-06.npy", mmap_mode="r")[:H_MAIN]
        b = np.load(ROOT_RES / "exact_eval" / "samples" / f"track={track}" / "origin=2025-06.npy", mmap_mode="r")
        d = np.abs(np.asarray(a) - np.asarray(b))
        rt.append({"track": track, "check": "realtime 2025-06 h1-12 vs the exact evaluation samples", "max_abs_diff": float(d.max()),
                   "max_rel_diff": float((d / np.maximum(np.abs(np.asarray(b)), 1.0)).max())})

    # FM final samples (final_samples.py, nonneg_gi on every sample): exactness and a re-solve of 24 samples
    smp = GenSampler("FM", "final", args.results, args.clean)
    o = smp.origins[0]
    Xb, Xf, A, Acheck, L, dead, _, _ = smp.origin(o)
    P = Path(args.results) / "final" / "track=FM"
    Xr = np.load(P / "samples_rec.npy")
    Xb0 = np.load(P / "samples_base.npy")
    dg = pd.read_csv(P / "sample_diagnostics.csv", keep_default_na=False)
    Cy = np.abs(np.einsum("rn,hsn->hsr", Acheck, Xr)).max(axis=2)
    # every FM final sample re-solved with the null-space solver (qp_pool, with its checks and multipliers): the final_samples.py
    # diagnostics record min multiplier 0 for every sample, so the multipliers are certified here
    Xq, rdiag, ratt = solve_origin(smp, o, Xf, A, Acheck, L, dead, args.n_jobs, "final FM re-solve",
                                   str(QP / "qp_progress_final_FM_resolve.log"))
    rdiag.to_csv(out_path(QP / "final" / "track=FM" / "resolve_diag.csv", args.overwrite), index=False, float_format="%.6g")
    rel = np.abs(Xq - Xr).max(axis=2) / np.maximum(np.abs(Xr).max(axis=2), 1.0)
    rs = [float(rel.max())]
    rt.append({"track": "FM", "check": "final samples: base samples regenerated vs stored", "max_abs_diff": float(np.abs(Xb - Xb0).max()),
               "max_rel_diff": np.nan})
    rt.append({"track": "FM", "check": "final samples: max |C y| (% of World residence total)",
               "max_abs_diff": float(Cy.max()), "max_rel_diff": float((100 * Cy / Xr[:, :, smp.iw]).max())})
    rt.append({"track": "FM", "check": "final samples: min y (t/day)", "max_abs_diff": float(Xr.min()), "max_rel_diff": np.nan})
    rt.append({"track": "FM", "check": "final samples: share solved by the QP; min multiplier; flags",
               "max_abs_diff": float((dg.qp.astype(str) == "True").mean()),
               "max_rel_diff": float(pd.to_numeric(dg.min_multiplier, errors="coerce").min()),
               "note": f"{int((dg.flag.astype(str) != '').sum())} flags"})
    rt.append({"track": "FM", "check": "final samples: all re-solved with nonneg_nullspace (max abs diff; max relative diff)",
               "max_abs_diff": float(np.abs(Xq - Xr).max()), "max_rel_diff": max(rs),
               "note": f"{int(rdiag.qp.sum())} QPs, {len(ratt)} fallback attempts, min multiplier "
                       f"{rdiag.min_multiplier.min():.3g}, flags {int((rdiag.flag.fillna('') != '').sum())}, "
                       f"zero sets equal (|y| <= 1e-6): {bool(((np.abs(Xq) <= 1e-6) == (np.abs(Xr) <= 1e-6)).all())}"})
    r = pd.DataFrame(rt)
    r.to_csv(out_path(QP / "checks.csv", args.overwrite), index=False, float_format="%.6g")
    print(r.to_string(index=False))


# ---------------------------------------------------------------- part aci
def load_exact(o, track):
    """Exact samples of an evaluation (exact_eval.py) or calibration (here) origin, horizons 1-12."""
    p = ROOT_RES / "exact_eval" / "samples" / f"track={track}" / f"origin={o}.npy" if o in EVAL_ORIGINS \
        else sample_dir("calib", track) / f"origin={o}.npy"
    return np.load(p)


def part_aci(args):
    track = args.track
    sysd = load_system(Path(args.results), args.clean)
    uids, n, hist = sysd["uids"], sysd["n"], sysd["Yh"]
    D = OUT / "aci_exact" / f"track={track}"
    D.mkdir(parents=True, exist_ok=True)
    months = pd.period_range(EVAL_ORIGINS[0], "2026-06", freq="M").astype(str).tolist()
    ev, al = ACI(hist, n), ACI(hist, n)                              # evaluation origins only; all origins
    rows = []
    t0 = time.perf_counter()
    for t in months:
        ev.update(t)
        al.update(t)
        if t == EVAL_ORIGINS[-1]:
            ev.save_state(t, uids)
            al.save_state(t, uids)
        if t not in EVAL_ORIGINS + CAL_ORIGINS:
            continue
        X = np.sort(load_exact(t, track), axis=1)
        targets = [str(pd.Period(t, "M") + h) for h in range(1, H_MAIN + 1)]
        if t in EVAL_ORIGINS:
            ev.intervals(t, X, uids, targets)
        rows += [r.assign(origin_set="evaluation" if t in EVAL_ORIGINS else "calibration")
                 for r in al.intervals(t, X, uids, targets)]
        print(f"aci {track} {t}: {time.perf_counter() - t0:.0f} s", flush=True)
    ev.save_state("2026-06", uids)
    al.save_state("2026-06_ext", uids)
    assert not any(str(pd.Period(o, "M") + h) <= "2026-06" for o, h in al.pending)
    iv = pd.concat(rows, ignore_index=True)
    st = pd.concat(ev.state + [s for s in al.state if (s.state_at == "2026-06_ext").all()], ignore_index=True)

    # ---- checks against exact_eval.py: evaluation-origin intervals and the states at 2025-06 and 2026-06
    P_EVAL = ROOT_RES / "prob_main_exact" / f"track={track}"
    a_eval = pd.read_parquet(P_EVAL / "aci_intervals.parquet")
    keys = ["unique_id", "origin", "horizon", "level", "gamma"]
    m = iv[iv.origin_set == "evaluation"].merge(a_eval, on=keys, suffixes=("", "_eval"), validate="one_to_one")
    chk_iv = float(max((m.alpha - m.alpha_eval).abs().max(), (m.lo - m.lo_eval).abs().max(), (m.hi - m.hi_eval).abs().max()))
    s_eval = pd.read_parquet(P_EVAL / "aci_state.parquet")
    ms = st[st.state_at.isin(["2025-06", "2026-06"])].merge(s_eval, on=["state_at", "unique_id", "horizon", "level", "gamma"])
    chk_st = float((ms.alpha_x - ms.alpha_y).abs().max())
    a_al = pd.concat([s for s in al.state if (s.state_at == "2025-06").all()], ignore_index=True) \
        .merge(s_eval[s_eval.state_at == "2025-06"], on=["state_at", "unique_id", "horizon", "level", "gamma"])
    chk_al = float((a_al.alpha_x - a_al.alpha_y).abs().max())
    print(f"check vs exact_eval.py: evaluation intervals {chk_iv:.1e} ({len(m)} rows of {len(a_eval)}), states {chk_st:.1e} "
          f"({len(ms)} rows), state 2025-06 of the full recursion {chk_al:.1e}")
    assert len(m) == len(a_eval) and chk_iv == 0 and chk_st == 0 and chk_al == 0

    # ---- check of the code on zeroed calibration samples: from the stored zeroed-sample state at 2025-06 and the stored
    # pending intervals, the recursion must reproduce the stored calib_aci.py intervals (ties aside, as in exact_eval.py)
    RES = Path(args.results)
    P_ZERO = RES / "prob_main" / f"track={track}"
    s_zero = pd.read_parquet(P_ZERO / "aci_state.parquet")
    z = ACI(hist, n)
    for (lv, g) in z.alpha:
        z.alpha[(lv, g)] = s_zero[(s_zero.state_at == "2025-06") & (s_zero.level == lv) & (s_zero.gamma == g)] \
            .pivot(index="unique_id", columns="horizon", values="alpha").loc[uids].to_numpy()
    ai = pd.read_parquet(P_ZERO / "aci_intervals.parquet", filters=[("origin", ">=", "2024-07")])
    ai = ai[ai.target > "2025-06"]
    for (o, h, lv, g), d in ai.groupby(["origin", "horizon", "level", "gamma"]):
        d = d.set_index("unique_id").loc[uids]
        z.pending.setdefault((o, int(h)), {})[(int(lv), float(g))] = (d.lo.to_numpy(), d.hi.to_numpy())
    smp = GenSampler(track, "calib", args.results, args.clean)
    zrows = []
    for t in pd.period_range("2025-07", "2026-06", freq="M").astype(str):
        z.update(t)
        if t in CAL_ORIGINS:
            _, Xf, *_ = smp.origin(t)
            zrows += z.intervals(t, np.sort(np.maximum(Xf, 0.0), axis=1), uids,
                                 [str(pd.Period(t, "M") + h) for h in range(1, H_MAIN + 1)])
    z.save_state("2026-06_ext", uids)
    zi = pd.concat(zrows, ignore_index=True)
    cst = pd.read_parquet(RES / "calib" / "aci" / f"track={track}" / "aci_intervals.parquet")
    mz = zi.merge(cst, on=keys, suffixes=("", "_st"), validate="one_to_one")
    same = (mz.alpha - mz.alpha_st).abs() < 1e-12
    chk_z = float(max((mz.lo - mz.lo_st)[same].abs().max(), (mz.hi - mz.hi_st)[same].abs().max()))
    first = mz[~same].sort_values("origin").groupby(["unique_id", "horizon", "level", "gamma"], as_index=False).first()
    n_untied = 0
    if len(first):                                                   # the event that first changed alpha
        first["o_prev"] = [str(pd.Period(o, "M") - h) for o, h in zip(first.origin, first.horizon)]
        allz = pd.concat([ai[keys + ["target", "lo", "hi"]].assign(lo_st=ai.lo, hi_st=ai.hi),
                          mz[keys + ["target", "lo", "hi", "lo_st", "hi_st"]]], ignore_index=True)
        evz = first[["unique_id", "o_prev", "horizon", "level", "gamma", "origin"]].merge(
            allz, left_on=["unique_id", "o_prev", "horizon", "level", "gamma"], right_on=keys, suffixes=("_div", ""))
        evz["y"] = [hist.loc[o, u] for o, u in zip(evz.origin_div, evz.unique_id)]
        gap = np.minimum.reduce([np.abs(evz.y - evz[c]) for c in ["lo", "hi", "lo_st", "hi_st"]])
        n_untied = int((gap >= TIE_TOL).sum())
    cs = pd.read_parquet(RES / "calib" / "aci" / f"track={track}" / "aci_state.parquet")
    zs = pd.concat(z.state, ignore_index=True).merge(cs, on=["state_at", "unique_id", "horizon", "level", "gamma"])
    chk_zs = float((zs.alpha_x - zs.alpha_y).abs().max())
    n_zs = int(((zs.alpha_x - zs.alpha_y).abs() > 1e-12).sum())
    print(f"check of the code on zeroed calibration samples: intervals where alpha agrees {chk_z:.1e}; rows with alpha "
          f"differing {int((~same).sum())} of {len(mz)} ({len(first)} series x horizon x level x gamma), not caused by a "
          f"tie: {n_untied}; final state differs in {n_zs} of {len(zs)} cells (max {chk_zs:.2e})")
    assert len(mz) == len(cst) and chk_z < 1e-6 and n_untied == 0

    # ---- calibration-origin coverage (information only), nested intervals as in calib_aci.py
    y = hist.rename_axis(index="target", columns="unique_id").stack().rename("y").reset_index()
    cal = iv[iv.origin_set == "calibration"]
    wide = cal.pivot_table(index=["unique_id", "origin", "horizon", "target", "gamma"], columns="level",
                           values=["lo", "hi"]).reset_index()
    wide.columns = [f"{a}{b}" for a, b in wide.columns]
    wide = wide.merge(y, on=["target", "unique_id"])
    cov = []
    for lv in LEVELS:
        lo = wide[f"lo{lv}"] if lv == 95 else np.maximum(wide.lo80, wide.lo95)
        hi = wide[f"hi{lv}"] if lv == 95 else np.minimum(wide.hi80, wide.hi95)
        wide[f"cov{lv}"] = ((wide.y >= lo) & (wide.y <= hi)).astype(float)
    for sub, msk in [("all targets 2025-08 to 2026-06", wide.target <= "2026-06"),
                     ("targets in 2025 (before the 2026 reclassification)", wide.target <= "2025-12"),
                     ("targets in 2026 (after the 2026 reclassification)", wide.target >= "2026-01")]:
        for g, d in wide[msk].groupby("gamma"):
            cov.append({"subset": sub, "gamma": g, "cells": len(d), "coverage80": d.cov80.mean(), "coverage95": d.cov95.mean()})
    pd.DataFrame(cov).to_csv(out_path(D / "coverage_calibration_origins.csv", args.overwrite), index=False, float_format="%.6g")

    iv.to_parquet(out_path(D / "aci_intervals.parquet", args.overwrite), index=False)
    st.to_parquet(out_path(D / "aci_state.parquet", args.overwrite), index=False)
    pd.DataFrame([{"track": track, "eval_intervals_vs_exact_eval_max_abs_diff": chk_iv, "states_vs_exact_eval_max_abs_diff": chk_st,
                   "zeroed_calib_intervals_max_abs_diff_alpha_equal": chk_z, "zeroed_calib_rows_alpha_differs": int((~same).sum()),
                   "zeroed_calib_rows": len(mz), "zeroed_first_divergences": len(first),
                   "zeroed_first_divergences_not_tie": n_untied, "zeroed_final_state_cells_differ": n_zs,
                   "zeroed_final_state_max_abs_diff": chk_zs, "seconds": time.perf_counter() - t0}]) \
        .to_csv(out_path(D / "checks.csv", args.overwrite), index=False, float_format="%.6g")
    print(pd.DataFrame(cov).to_string(index=False))


# ---------------------------------------------------------------- part realtime
def part_realtime(args):
    track = args.track
    smp = GenSampler(track, "realtime", args.results, args.clean)
    uids, hist = smp.uids, smp.hist
    D = OUT / "prob_realtime_exact" / f"track={track}"
    D.mkdir(parents=True, exist_ok=True)
    st = pd.read_parquet(OUT / "aci_exact" / f"track={track}" / "aci_state.parquet")
    ir = [i for i, u in enumerate(uids) if u.endswith("|RES_TOTAL_SYS")]
    q_rows, crps_rows, joint_rows, aci_rows, aci_ext, pts_rows, diag = [], [], [], [], [], [], []
    for o in smp.origins:
        Xb, Xf, A, Acheck, L, dead, targets, scale = smp.origin(o)
        Xq = np.load(sample_dir("realtime", track) / f"origin={o}.npy")
        H = Xq.shape[0]
        ok = np.isfinite(scale) & (scale > 0)
        vm = np.isin(np.flatnonzero(ok), smp.vs_idx)
        Cy = np.abs(np.einsum("rn,hsn->hsr", Acheck, Xq)).max(axis=2)
        diag.append({"origin": o, "track": track, "horizons": H, "share_neg_free_projection": float((Xf < 0).mean()),
                     "max_abs_Cy": float(Cy.max()), "max_violation_pct": float((100 * Cy / Xq[:, :, smp.iw]).max()),
                     "min_y": float(Xq.min()), "share_values_zero_exact": float((Xq == 0).mean()),
                     "share_values_zero_zeroed": float((np.maximum(Xf, 0) == 0).mean())})
        for kind, X in [("base", Xb), ("rec", Xq)]:
            qs = np.quantile(X, STORE_Q, axis=1)
            for h in range(H):
                d = pd.DataFrame(qs[:, h, :].T, columns=QNAMES)
                d.insert(0, "unique_id", uids)
                d.insert(1, "origin", o)
                d.insert(2, "horizon", h + 1)
                d.insert(3, "target", targets[h])
                d.insert(4, "kind", kind)
                q_rows.append(d)
                t = targets[h]
                if t not in hist.index:
                    continue
                y = hist.loc[t].to_numpy()
                crps_rows.append(pd.DataFrame({"unique_id": uids, "origin": o, "horizon": h + 1, "target": t, "kind": kind,
                                               "crps": crps_fair(X[h], y)}))
                Xs, ys = X[h][:, ok] / scale[ok], y[ok] / scale[ok]
                joint_rows.append({"origin": o, "horizon": h + 1, "target": t, "kind": kind, "es": energy_score(Xs, ys),
                                   "n_es": int(ok.sum()), "vs": variogram_score(Xs[:, vm], ys[vm]), "n_vs": int(vm.sum())})
            v = X[:, :, ir]
            pts_rows.append(pd.DataFrame({
                "unique_id": np.tile(np.array(uids)[ir], H * S), "origin": o,
                "horizon": np.repeat(np.arange(1, H + 1), S * len(ir)),
                "sample": np.tile(np.repeat(np.arange(S), len(ir)), H), "kind": kind, "value": v.reshape(-1)}))
        Xs_ = np.sort(Xq, axis=1)
        for label, out in [("main", aci_rows), ("ext", aci_ext)]:
            state_at = "2025-06" if o <= "2025-06" else ("2026-06" if label == "main" else "2026-06_ext")
            for h in range(H):
                for lv in LEVELS:
                    for g in GAMMAS:
                        a = st[(st.state_at == state_at) & (st.level == lv) & (st.gamma == g)
                               & (st.horizon == min(h + 1, H_MAIN))].set_index("unique_id").alpha.loc[uids].to_numpy()
                        out.append(pd.DataFrame({"unique_id": uids, "origin": o, "horizon": h + 1, "target": targets[h],
                                                 "level": lv, "gamma": g, "alpha": a, "state_at": state_at,
                                                 "lo": sample_quantile(Xs_[h], a / 2), "hi": sample_quantile(Xs_[h], 1 - a / 2)}))
        print(f"realtime {track} {o}: scored", flush=True)
    pd.concat(q_rows, ignore_index=True).to_parquet(out_path(D / "quantiles.parquet", args.overwrite), index=False)
    pd.concat(crps_rows, ignore_index=True).to_parquet(out_path(D / "crps.parquet", args.overwrite), index=False)
    pd.DataFrame(joint_rows).to_csv(out_path(D / "joint_scores.csv", args.overwrite), index=False, float_format="%.8g")
    pd.concat(aci_rows, ignore_index=True).to_parquet(out_path(D / "aci_intervals.parquet", args.overwrite), index=False)
    pd.concat(aci_ext, ignore_index=True).to_parquet(out_path(D / "aci_intervals_ext.parquet", args.overwrite), index=False)
    pd.concat(pts_rows, ignore_index=True).to_parquet(out_path(D / "samples_res_total_sys.parquet", args.overwrite), index=False)
    pd.DataFrame(diag).to_csv(out_path(D / "sampling_diagnostics.csv", args.overwrite), index=False, float_format="%.6g")
    # check: the base CRPS and the main-state intervals of origin 2025-06 equal the stored zeroed run where the samples
    # agree (base samples are the same; the reconciled samples differ only through the QP)
    c0 = pd.read_parquet(Path(args.results) / "prob_realtime" / f"track={track}" / "crps.parquet", filters=[("kind", "==", "base")])
    cb = pd.concat(crps_rows, ignore_index=True)
    mb = cb[cb.kind == "base"].merge(c0, on=["unique_id", "origin", "horizon", "kind"])
    print(f"check: base CRPS against the stored real-time run, max abs diff {(mb.crps_x - mb.crps_y).abs().max():.2e} "
          f"({len(mb)} rows)")
    assert (mb.crps_x - mb.crps_y).abs().max() < 1e-6


# ---------------------------------------------------------------- part compare
def part_compare(args):
    RES = Path(args.results)
    C = OUT / "compare"
    C.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)

    def save(df, name):
        df.to_csv(out_path(C / name, args.overwrite), index=False, float_format="%.6g")
        print(f"---- {name}")
        print(df.to_string(index=False))

    # ---- T5a / T5b inputs: zeroed (eval_realtime_zeroed_nested) against exact (main state) and exact (ext state)
    src = {"zeroed": RES / "eval_realtime_zeroed_nested", "exact": OUT / "eval_realtime_exact_nested",
           "exact_ext_state": OUT / "eval_realtime_exact_ext_nested"}
    rows = []
    for k, p in src.items():
        rt = pd.read_csv(p / "summary_res_total_pub.csv")
        rt = rt[(rt.d_variant == "d") & (rt.level == "economies") & (rt.period != "all incl. 2025-09")]
        for (tr, o, per), x in rt.groupby(["track", "origin", "period"]):
            x = x.set_index("metric")
            for met, col in [("MASE", "weighted"), ("sCRPS", "weighted"), ("sCRPS", "median"), ("cov80_aci", "mean"),
                             ("cov95_aci", "mean"), ("cov80_uncal", "mean"), ("cov95_uncal", "mean"),
                             ("cov80_conformal", "mean"), ("cov95_conformal", "mean")]:
                rows.append({"track": tr, "origin": o, "targets": per, "quantity": f"{met}_{col}", "version": k,
                             "value": x.at[met, col]})
    t5 = pd.DataFrame(rows).pivot_table(index=["origin", "targets", "track", "quantity"], columns="version",
                                        values="value").reset_index()
    t5["exact_minus_zeroed"] = t5.exact - t5.zeroed
    save(t5, "T5a_zeroed_vs_exact.csv")
    rows = []
    for k, p in src.items():
        u = pd.read_csv(p / "units_res_total_pub.csv")
        u = u[(u.origin == "2026-06") & (u.d_variant == "d") & u.unit.isin(["W", "CHN", "GRC", "SGP", "JPN", "USA"])]
        for r in u.itertuples():
            for met in ["MASE", "sCRPS", "cov80_aci", "cov95_aci"]:
                rows.append({"unit": r.unit, "track": r.track, "quantity": met, "version": k, "value": getattr(r, met)})
    t5b = pd.DataFrame(rows).pivot_table(index=["unit", "track", "quantity"], columns="version", values="value").reset_index()
    t5b["exact_minus_zeroed"] = t5b.exact - t5b.zeroed
    save(t5b, "T5b_zeroed_vs_exact.csv")
    # system series of origin 2025-06 (targets to 2026-06): sCRPS and ACI coverage
    rows = []
    for k, p in src.items():
        s = pd.read_csv(p / "summary_system_2025-06.csv")
        s = s[s.level == "economies"]
        for r in s.itertuples():
            rows.append({"track": r.track, "period": r.period, "metric": r.metric, "version": k,
                         "weighted": r.weighted, "median": r.median, "mean": r.mean})
    ss = pd.DataFrame(rows).pivot_table(index=["period", "track", "metric"], columns="version",
                                        values=["weighted", "mean"]).reset_index()
    ss.columns = ["_".join(c).strip("_") for c in ss.columns]
    save(ss, "realtime_system_2025-06_zeroed_vs_exact.csv")

    # ---- final intervals (S1 ACI columns) and S1a floor shares
    grp = {"RES_TOTAL_SYS": "total (RES_TOTAL_SYS)", "TER_DOM": "territory measures (TER_DOM, TER_INT)",
           "TER_INT": "territory measures (TER_DOM, TER_INT)"}
    rows, fl = [], []
    for tr in TRACKS:
        z = pd.read_parquet(RES / "final" / f"track={tr}" / "forecasts.parquet")
        e = pd.read_parquet(OUT / "final_exact" / f"track={tr}" / "forecasts.parquet")
        m = z.merge(e, on=["unique_id", "origin", "horizon", "target"], suffixes=("_z", "_e"), validate="one_to_one")
        m["group"] = m.unique_id.str.split("|").str[1].map(grp).fillna("components (A-G)")
        for lv in (80, 95):
            for v in ("z", "e"):
                m[f"w{lv}_{v}"] = m[f"hi{lv}_{v}"] - m[f"lo{lv}_{v}"]
        for gname, d in [("all", m)] + list(m.groupby("group")):
            rows.append({"track": tr, "group": gname, "cells": len(d),
                         "share_point_equal": float(np.isclose(d.point_z, d.point_e, rtol=0, atol=1e-9).mean()),
                         "share_median_equal": float(np.isclose(d["q0.5_z"], d["q0.5_e"], rtol=1e-12, atol=1e-9).mean()),
                         **{f"median_width_ratio_{lv}_exact_over_zeroed": float(np.nanmedian(d[f"w{lv}_e"] / d[f"w{lv}_z"].where(d[f"w{lv}_z"] > 0)))
                            for lv in (80, 95)},
                         "median_alpha80_zeroed": d.alpha80_z.median(), "median_alpha80_exact": d.alpha80_e.median(),
                         "median_alpha95_zeroed": d.alpha95_z.median(), "median_alpha95_exact": d.alpha95_e.median(),
                         "share_floor_any_zeroed": float((d.aci_at_floor_z != "none").mean()),
                         "share_floor_any_exact": float((d.aci_at_floor_e != "none").mean())})
        for v, f in [("zeroed", z), ("exact", e)]:
            f = f[["unique_id", "aci_at_floor"]].copy()
            f["group"] = f.unique_id.str.split("|").str[1].map(grp).fillna("components (A-G)")
            fl.append(f.assign(track=tr, version=v))
        w = m[m.unique_id == "W|RES_TOTAL_SYS"][["horizon", "target", "point_z", "lo80_z", "hi80_z", "lo95_z", "hi95_z",
                                                   "lo80_e", "hi80_e", "lo95_e", "hi95_e", "alpha80_z", "alpha80_e",
                                                   "alpha95_z", "alpha95_e"]]
        save(w.assign(track=tr), f"S1_world_res_total_{tr}_zeroed_vs_exact.csv")
    save(pd.DataFrame(rows), "S1_intervals_zeroed_vs_exact.csv")
    fl = pd.concat(fl, ignore_index=True)
    s1a = pd.crosstab([fl.version, fl.track, fl.group], fl.aci_at_floor, normalize="index").mul(100) \
        .reindex(columns=["none", "95", "80_and_95", "80"], fill_value=0.0)
    s1a.columns = [f"pct_{c}" for c in s1a.columns]
    save(s1a.reset_index(), "S1a_zeroed_vs_exact.csv")

    # ---- ACI states: zeroed (prob_samples.py / calib_aci.py) against exact, system series, by state and gamma
    rows = []
    for tr in TRACKS:
        z = pd.concat([pd.read_parquet(RES / "prob_main" / f"track={tr}" / "aci_state.parquet"),
                       pd.read_parquet(RES / "calib" / "aci" / f"track={tr}" / "aci_state.parquet")], ignore_index=True)
        e = pd.read_parquet(OUT / "aci_exact" / f"track={tr}" / "aci_state.parquet")
        m = z.merge(e, on=["state_at", "unique_id", "horizon", "level", "gamma"], suffixes=("_z", "_e"))
        for (sa, lv, g), d in m.groupby(["state_at", "level", "gamma"]):
            rows.append({"track": tr, "state_at": sa, "level": lv, "gamma": g, "cells": len(d),
                         "median_alpha_zeroed": d.alpha_z.median(), "median_alpha_exact": d.alpha_e.median(),
                         "share_floor_zeroed": float((d.alpha_z <= ALPHA_CLIP[0] + 1e-12).mean()),
                         "share_floor_exact": float((d.alpha_e <= ALPHA_CLIP[0] + 1e-12).mean()),
                         "share_cells_alpha_differs": float(((d.alpha_z - d.alpha_e).abs() > 1e-12).mean()),
                         "max_abs_alpha_diff": float((d.alpha_z - d.alpha_e).abs().max())})
    save(pd.DataFrame(rows), "aci_state_zeroed_vs_exact.csv")
    # calibration-origin coverage (information only)
    rows = []
    for tr in TRACKS:
        z = pd.read_csv(RES / "calib" / "aci" / f"track={tr}" / "coverage.csv")
        z = z[z.horizon.astype(str) == "all"]
        e = pd.read_csv(OUT / "aci_exact" / f"track={tr}" / "coverage_calibration_origins.csv")
        for r in e.itertuples():
            for lv in (80, 95):
                zz = z[(z.subset == r.subset) & (z.gamma == r.gamma) & (z.level == lv)]
                rows.append({"track": tr, "subset": r.subset, "gamma": r.gamma, "level": lv, "cells": r.cells,
                             "zeroed": float(zz.coverage.iloc[0]), "exact": getattr(r, f"coverage{lv}")})
    cc = pd.DataFrame(rows)
    cc["exact_minus_zeroed"] = cc.exact - cc.zeroed
    save(cc, "calibration_origins_coverage_zeroed_vs_exact.csv")

    # ---- supplementary results: zeroed (RES / supplementary_zeroed) against exact (RES / supplementary)
    SZ, SE = RES / "supplementary_zeroed", RES / "supplementary"
    # point-based files of the rerun items must be unchanged
    ident = []
    for f in ["point_by_window.csv", "method_ranks.csv", "rank_correlation.csv",
              "prob_base_models_by_window.csv", "weighted_MAE_kt.csv", "dm_base_models_vs_ETS.csv",
              "T4_origins.csv"]:
        a, b = pd.read_csv(SZ / f), pd.read_csv(SE / f)
        same = a.shape == b.shape and all(np.allclose(a[c], b[c], equal_nan=True, rtol=1e-9) if np.issubdtype(a[c].dtype, np.number)
                                          else (a[c].astype(str) == b[c].astype(str)).all() for c in a.columns)
        ident.append({"file": f, "rows": len(a), "identical": bool(same)})
    dz, de = pd.read_csv(SZ / "dm_counts.csv"), pd.read_csv(SE / "dm_counts.csv")
    keys = ["loss", "comparison", "unit_type", "horizon"]
    m = dz.merge(de, on=keys, suffixes=("_zeroed", "_exact"))
    pt = m[m.loss.str.startswith("absolute")]
    ident.append({"file": "dm_counts.csv (absolute-error rows)", "rows": len(pt), "identical": bool(all(
        (pt[f"{c}_zeroed"] == pt[f"{c}_exact"]).all() for c in ["n_better", "n_worse", "n_better_bh", "n_worse_bh"]))})
    save(pd.DataFrame(ident), "supplementary_point_files_identical.csv")
    cr = m[m.loss.str.startswith("scaled CRPS") & (m.unit_type == "economy")]
    save(cr[keys + [f"{c}_{v}" for c in ["n_tests", "n_better", "n_worse", "n_better_bh", "n_worse_bh"] for v in ("zeroed", "exact")]],
         "dm_crps_counts_zeroed_vs_exact.csv")
    bz, be = pd.read_csv(SZ / "coverage_block_bootstrap.csv"), pd.read_csv(SE / "coverage_block_bootstrap.csv")
    k8 = ["track", "interval", "level", "horizon", "bgroup"]
    b = bz.merge(be, on=k8, how="outer", suffixes=("_zeroed", "_exact"))
    b = b[["track", "interval", "level", "bgroup", "horizon"] + [f"{c}_{v}" for v in ("zeroed", "exact")
                                                                 for c in ("coverage", "boot_lo95", "boot_hi95")]]
    b["exact_minus_zeroed"] = b.coverage_exact - b.coverage_zeroed
    save(b.sort_values(["track", "interval", "level", "bgroup", "horizon"]), "coverage_zeroed_vs_exact.csv")
    az, ae = pd.read_csv(SZ / "calibration_variants.csv"), pd.read_csv(SE / "calibration_variants.csv")
    a6 = az.merge(ae, on=["track", "interval", "horizon"], suffixes=("_zeroed", "_exact"))
    a6 = a6[a6.horizon.astype(str).isin(["1", "3", "6", "12", "all"])]
    save(a6[["track", "interval", "horizon"] + [f"{c}_{v}" for c in ["cov80_mean", "cov95_mean", "sWinkler80_weighted",
                                                                        "sWinkler95_weighted"] for v in ("zeroed", "exact")]],
         "calibration_variants_zeroed_vs_exact.csv")
    rz = pd.read_csv(RES / "conformal_residence_zeroed" / "coverage_residence_totals.csv")
    re_ = pd.read_csv(RES / "conformal_residence" / "coverage_residence_totals.csv")
    cres = rz.merge(re_, on=["track", "interval", "level"], suffixes=("_zeroed", "_exact"))
    save(cres[["track", "interval", "level"] + [f"{c}_{v}" for c in ["coverage_pooled", "boot_lo95", "boot_hi95",
                                                                     "median_width_ratio_to_aci", "world_coverage"]
                                               for v in ("zeroed", "exact")]], "conformal_residence_coverage_zeroed_vs_exact.csv")
    hz = pd.read_csv(RES / "conformal_residence_zeroed" / "coverage_by_horizon.csv")
    he = pd.read_csv(RES / "conformal_residence" / "coverage_by_horizon.csv")
    save(hz.merge(he, on=["track", "interval", "level", "horizon"], suffixes=("_zeroed", "_exact")),
         "conformal_residence_coverage_by_horizon_zeroed_vs_exact.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", required=True, choices=["qp", "check", "aci", "realtime", "compare"])
    ap.add_argument("--source", choices=list(SOURCES))
    ap.add_argument("--track", choices=list(TRACKS))
    ap.add_argument("--clean", default="data/clean")
    ap.add_argument("--results", default="results")
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    faulthandler.enable(sys.stderr)
    log = RunLog("exact_samples.py", vars(args))
    OUT.mkdir(parents=True, exist_ok=True)
    QP.mkdir(parents=True, exist_ok=True)
    {"qp": part_qp, "check": part_check, "aci": part_aci, "realtime": part_realtime, "compare": part_compare}[args.part](args)
    log.finish(f"{args.part} {args.source or ''} {args.track or ''}")


if __name__ == "__main__":
    main()
