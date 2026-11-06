"""
exact_eval.py: the probabilistic evaluation with exactly reconciled samples.

Samples: the evaluation samples (prob_samples.py, main mode), regenerated exactly by sample_coherence.Sampler
(same marginals, Gaussian copula from the track's MinT W, S = 500, draws from default_rng([42, origin]), free MinT
projection). prob_samples.py then sets negative values to zero ("zeroed"); here every sample is reconciled with the
exact non-negative QP instead ("exact"). All 31 main origins (2022-12 ... 2025-06) x horizons 1-12 are run for both tracks
(the main-window cells and the others, so that the ACI recursion runs exactly as in prob_samples.py).

Parts
  count     samples per cell with a negative value after the free projection (any value < 0, and < NEG_TOL = -1e-6
            t/day, the threshold of the QP step), and the share of negative values; main-window flag per cell.
  estimate  run-time estimate for the scope rule (all cells if at most 12 hours): the wall time of one full origin
            (12 x 500 samples, the samples with a value < NEG_TOL solved, --n-jobs processes) with the solver used
            here, and the single-thread time per QP of reconcile.nonneg_gi (the solver of reconcile.py) on a few
            samples for comparison. Writes estimate_<T>.csv.
  qp        the exact QP for every sample with a value < NEG_TOL; the other samples are already the QP solution
            (their tiny negatives in [NEG_TOL, 0) are set to zero, as in final_samples.qp_chunk). Solver:
            reconcile.nonneg_nullspace, the same QP (unique optimum) solved by quadprog (Goldfarb-Idnani) in the
            null space of the equality constraints, with the active set then fixed exactly and the multipliers of the
            active bounds recorded (all must be >= 0). The samples are solved by qp_pool.solve_pool: worker processes
            with a time limit of 120 s per sample; a sample that fails, hangs or crashes its worker goes to quadprog on
            the rescaled sample and then to OSQP (eps 1e-9), and every solution must pass the coherence, non-negativity
            and KKT checks. Every failed or fallback stage: samples/track=<T>/origin=<O>_fallback.csv; progress and
            fallback events: qp_progress_<T>.log (written by the parent only). One file per origin
            (results/exact_eval/samples/, 54 MB each, not committed to git; written last, so an interrupted run
            resumes at the next origin); per-sample diagnostics in origin=<O>_diag.csv.
  check     the exact samples against the cells of sample_coherence.py (results/sample_coherence/qp_cells: 16 cells
            per track, every sample solved with reconcile.nonneg_gi): the largest absolute and relative difference per cell.
  score     per track, origin by origin, from the exact samples: fair CRPS per series and cell, energy and variogram
            scores (scaled, as in prob_samples.py), stored quantiles, the largest constraint violation, and the ACI
            recursion (both gammas, both levels; exactly as prob_samples.py main mode, continued month by month to
            2026-06). Checks: the zeroed CRPS must reproduce the stored crps.parquet; the ACI code replayed on the
            stored zeroed-sample intervals must reproduce the stored alpha and state exactly; on the regenerated zeroed samples it must
            reproduce the stored intervals wherever alpha agrees, and every divergence of alpha must start at a tie
            event (y on an interval bound within TIE_TOL, e.g. y = 0 against a bound of 1e-21 left by the projection;
            aci_zeroed_divergences.csv). Tie events of both versions: aci_tie_events.csv.
            Output: results/prob_main_exact/track=<T>/,
            in the layout of results/prob_main (kind "rec" = exact samples; the "base" rows are copied from
            results/prob_main, the base samples being the same), so that evaluate_prob.py --prob-dir can read it.
  compare   after `python src/evaluate_prob.py --nest --prob-dir results/prob_main_exact --out
            results/eval_prob_exact_nested`:
              - Table 4 quantities (paper_tables.t4_table), zeroed (results/eval_prob_nested) against exact;
              - calibrated (ACI, gamma 0.05, nested) and uncalibrated coverage overall and for the residence totals
                (RES_TOTAL_SYS of the system economies): mean over series (T4 definition) and pooled hits / cells (the
                definition of coverage_wilson.csv), with 95% intervals from the circular block bootstrap
                over calibration origins of coverage_block_bootstrap.csv (block 6, 2,000 resamples, seed 42; the same resampled origins for both versions);
              - FM - STAT weighted sCRPS (reconciled samples) with the origin-block bootstrap (benchmarks.py:
                moving blocks of 6 of the 31 origins, 2,000 resamples, seed 42, all series jointly), zeroed and exact;
                the zeroed result must reproduce results/inference/block_bootstrap.csv.

Usage:
  python src/exact_eval.py --part count
  python src/exact_eval.py --part estimate --track FM --n-jobs 8
  python src/exact_eval.py --part qp --track FM --n-jobs 8
  python src/exact_eval.py --part check
  python src/exact_eval.py --part score --track FM
  python src/evaluate_prob.py --nest --prob-dir results/prob_main_exact --out results/eval_prob_exact_nested
  python src/exact_eval.py --part compare
"""
import argparse
import faulthandler
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from evalcore import WINDOWS, load_history, origin_scales
from evaluate_prob import CAL_START, nest
from paper_tables import t4_table
from prob_samples import (ALPHA_CLIP, GAMMAS, H_MAIN, LEVELS, QNAMES, S, STORE_Q, TRACKS, crps_fair, energy_score,
                          sample_quantile, variogram_score)
from qp_pool import solve_pool
from reconcile import NEG_TOL, nonneg_gi, nonneg_nullspace_osqp, nullspace_setup
from sample_coherence import ORIGINS, OUT as COH, Sampler
from skill import mbb_counts
from supplementary_analyses import BASIS_GROUP, mbb_indices
from runlog import RunLog, out_path

OUT = Path("results/exact_eval")
PROB = Path("results/prob_main_exact")
SEED, BLOCK, REPS = 42, 6, 2000                                       # origin-block (benchmarks.py) and coverage bootstraps
TIE_TOL = 1e-9                                                        # t/day; |y - bound| below this is a tie


def solve_origin(smp, o, n_jobs, track, force_stage=None):
    """Exact samples Xq (H, S, n) of origin o, the per-sample diagnostics, and the fallback attempts (qp_pool)."""
    Xb, Xf, A, Acheck, L, dead, targets, _ = smp.origin(o)
    M, bnd, Cb = nullspace_setup(A, L, dead)
    H = Xf.shape[0]
    cols = Xf.reshape(H * S, smp.n).T                                 # column h * S + s

    def check(y):
        c = float(np.abs(Acheck @ y).max())
        return {"max_abs_Cy": c, "max_violation_pct": 100 * c / y[smp.iw]}

    Y, rows, attempts = solve_pool(cols, M, bnd, Cb, check, n_jobs, f"{track} {o}",
                                   progress=str(OUT / f"qp_progress_{track}.log"), force_stage=force_stage)
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
        att.insert(0, "track", track)
        att.insert(1, "origin", o)
        att.insert(2, "horizon", att.column // S + 1)
        att.insert(3, "sample", att.column % S)
    return Xq, diag, targets, att


# ---------------------------------------------------------------- parts
def part_count(args):
    rows = []
    for track in TRACKS:
        smp = Sampler(track, args.results, args.clean)
        for o in ORIGINS:
            _, Xf, *_ , targets, _ = smp.origin(o)
            for h in range(Xf.shape[0]):
                X = Xf[h]
                rows.append({"track": track, "origin": o, "horizon": h + 1, "target": targets[h],
                             "main_window": bool(WINDOWS["main"](pd.Series([targets[h]])).iloc[0]), "samples": S,
                             "samples_with_value_lt_0": int((X < 0).any(axis=1).sum()),
                             "samples_with_value_lt_neg_tol": int((X < NEG_TOL).any(axis=1).sum()),
                             "mean_negative_values_per_sample": float((X < NEG_TOL).sum(axis=1).mean()),
                             "max_negative_values_per_sample": int((X < NEG_TOL).sum(axis=1).max()),
                             "share_values_lt_0": float((X < 0).mean())})
            print(f"count {track} {o}", flush=True)
    c = pd.DataFrame(rows)
    c.to_csv(out_path(OUT / "negative_counts.csv", args.overwrite), index=False, float_format="%.6g")
    s = c.groupby(["track", "main_window"]).agg(cells=("origin", "size"), samples=("samples", "sum"),
                                                samples_needing_qp=("samples_with_value_lt_neg_tol", "sum"),
                                                samples_with_value_lt_0=("samples_with_value_lt_0", "sum"),
                                                min_per_cell=("samples_with_value_lt_neg_tol", "min"),
                                                mean_negative_values_per_sample=("mean_negative_values_per_sample", "mean"),
                                                max_negative_values_per_sample=("max_negative_values_per_sample", "max"),
                                                mean_share_values_lt_0=("share_values_lt_0", "mean")).reset_index()
    s.to_csv(out_path(OUT / "negative_counts_summary.csv", args.overwrite), index=False, float_format="%.6g")
    print(s.to_string(index=False))


def part_estimate(args):
    smp = Sampler(args.track, args.results, args.clean)
    o = "2024-06"
    t0 = time.perf_counter()
    _, diag, _, _ = solve_origin(smp, o, args.n_jobs, args.track)
    wall = time.perf_counter() - t0
    # the solver of reconcile.py (nonneg_gi), single thread, on the first 5 samples of horizons 1 and 12
    Xb, Xf, A, Acheck, L, dead, _, _ = smp.origin(o)
    gi = []
    with threadpool_limits(1):
        for h in [1, 12]:
            for s in range(5):
                t1 = time.perf_counter()
                nonneg_gi(Xb[h - 1, s], A, L, False, dead)
                gi.append(time.perf_counter() - t1)
    n_qp = int(diag.qp.sum())
    counts = pd.read_csv(OUT / "negative_counts.csv")
    ct = counts[counts.track == args.track]
    need_main, need_all = int(ct[ct.main_window].samples_with_value_lt_neg_tol.sum()), int(ct.samples_with_value_lt_neg_tol.sum())
    est = {"track": args.track, "origin_timed": o, "n_jobs": args.n_jobs, "qps_in_origin": n_qp,
           "wall_seconds_origin": wall, "wall_seconds_per_qp": wall / n_qp,
           "single_thread_seconds_per_qp_nullspace": diag.seconds[diag.qp].mean(),
           "single_thread_seconds_per_qp_nonneg_gi": float(np.mean(gi)),
           "qps_main_window": need_main, "qps_all_cells": need_all,
           "est_hours_main_window_nullspace": wall / n_qp * need_main / 3600,
           "est_hours_all_cells_nullspace": wall / n_qp * need_all / 3600,
           "est_hours_main_window_nonneg_gi_single_thread": np.mean(gi) * need_main / 3600,
           "est_hours_main_window_nonneg_gi_ideal_parallel": np.mean(gi) * need_main / 3600 / args.n_jobs}
    pd.DataFrame([est]).to_csv(out_path(OUT / f"estimate_{args.track}.csv", args.overwrite), index=False, float_format="%.6g")
    print(pd.Series(est).to_string())


def part_qp(args):
    smp = Sampler(args.track, args.results, args.clean)
    D = OUT / "samples" / f"track={args.track}"
    D.mkdir(parents=True, exist_ok=True)
    for o in ORIGINS:
        p = D / f"origin={o}.npy"
        if p.exists() and not args.overwrite:
            continue
        t0 = time.perf_counter()
        Xq, diag, _, att = solve_origin(smp, o, args.n_jobs, args.track)
        secs = time.perf_counter() - t0
        diag.assign(track=args.track, origin=o, wall_seconds_origin=secs).to_csv(D / f"origin={o}_diag.csv", index=False,
                                                                                  float_format="%.6g")
        if len(att):                                                  # every failed or fallback stage
            att.to_csv(D / f"origin={o}_fallback.csv", index=False, float_format="%.6g")
        tmp = D / f"origin={o}.tmp.npy"                               # the .npy marks the origin as done: write it last
        np.save(tmp, Xq)
        tmp.replace(p)
        by = diag[diag.qp].solver.value_counts().to_dict()
        print(f"{args.track} {o}: {secs:.0f} s, {int(diag.qp.sum())} QPs {by}, flags {int((diag.flag.fillna('') != '').sum())}, "
              f"min multiplier {diag.min_multiplier.min():.2e}, max violation {diag.max_violation_pct.max():.1e}%, "
              f"min y {diag.min_y.min():.1e}", flush=True)


def part_check(args):
    rows = []
    for track in TRACKS:
        for p in sorted((COH / "qp_cells" / f"track={track}").glob("origin=*_h=*.npz")):
            o, h = p.stem.split("_h=")
            o, h = o.replace("origin=", ""), int(h)
            Xt = np.load(p)["Xq"]                                             # (S, n), nonneg_gi on every sample
            Xq = np.load(OUT / "samples" / f"track={track}" / f"origin={o}.npy", mmap_mode="r")[h - 1]
            d = np.abs(Xq - Xt)
            rows.append({"track": track, "origin": o, "horizon": h, "samples": len(Xt),
                         "max_abs_diff_t_per_day": float(d.max()),
                         "max_rel_diff": float((d / np.maximum(np.abs(Xt), 1.0)).max()),
                         # zero sets at |y| <= 1e-6 t/day (NEG_TOL); exact zeros differ by round-off (|y| ~1e-13 to
                         # 1e-9 in one solution where the other has 0, e.g. dead series)
                         "same_zero_sets_tol_1e-6": bool(((np.abs(Xq) <= 1e-6) == (np.abs(Xt) <= 1e-6)).all()),
                         "cells_exact_zero_in_one_only": int(((Xq == 0) != (Xt == 0)).sum()),
                         "max_abs_where_exact_zero_in_one_only": float(np.abs(np.where((Xq == 0) != (Xt == 0), Xq + Xt, 0)).max())})
    c = pd.DataFrame(rows)
    c.to_csv(out_path(OUT / "check_vs_sample_coherence_qp.csv", args.overwrite), index=False, float_format="%.6g")
    print(c.to_string(index=False))
    # every sample that needed a fallback stage (qp_pool), with the residuals of the stored solution; a solution of the
    # rescaled quadprog stage is also compared with an OSQP solution of the same sample (eps 1e-9)
    fb = []
    for track in TRACKS:
        smp = None
        for f in sorted((OUT / "samples" / f"track={track}").glob("origin=*_fallback.csv")):
            a = pd.read_csv(f, keep_default_na=False)
            fin = a[a.ok.astype(str) == "True"]
            if not len(fin):
                continue
            o = str(fin.origin.iloc[0])
            smp = smp or Sampler(track, args.results, args.clean)
            _, Xf, A, Acheck, L, dead, _, _ = smp.origin(o)
            M, bnd, _ = nullspace_setup(A, L, dead)
            Xq = np.load(OUT / "samples" / f"track={track}" / f"origin={o}.npy", mmap_mode="r")
            for r in fin.itertuples():
                y = np.array(Xq[r.horizon - 1, r.sample])
                row = {"track": track, "origin": o, "horizon": r.horizon, "sample": r.sample,
                       "stages_failed": "; ".join(f"{x.stage}: {x.reason}" for x in a[(a.column == r.column) &
                                                                                     (a.ok.astype(str) != "True")].itertuples()),
                       "solved_by": r.stage, "note": r.note, "seconds": r.seconds, "n_zero": r.n_zero,
                       "min_multiplier": r.min_multiplier, "min_y": float(y.min()),
                       "max_abs_Cy": float(np.abs(Acheck @ y).max()),
                       "max_violation_pct": float(100 * np.abs(Acheck @ y).max() / y[smp.iw]),
                       "osqp_max_abs_diff": np.nan, "osqp_max_rel_diff": np.nan, "osqp_status": ""}
                if r.stage == "quadprog_rescaled":
                    with threadpool_limits(1):
                        yo, _, _, _, _, st = nonneg_nullspace_osqp(np.ascontiguousarray(Xf[r.horizon - 1, r.sample]), M, bnd)
                    row.update(osqp_max_abs_diff=float(np.abs(y - yo).max()),
                               osqp_max_rel_diff=float(np.abs(y - yo).max() / np.abs(y).max()), osqp_status=st)
                fb.append(row)
    fb = pd.DataFrame(fb)
    fb.to_csv(out_path(OUT / "fallback_samples.csv", args.overwrite), index=False, float_format="%.6g")
    print(f"fallback samples: {len(fb)}")
    if len(fb):
        print(fb.to_string(index=False))


class ACI:
    """The ACI recursion of prob_samples.py (main mode), both levels and gammas."""

    def __init__(self, hist, n):
        self.hist, self.pending, self.state = hist, {}, []
        self.alpha = {(lv, g): np.full((n, H_MAIN), LEVELS[lv]) for lv in LEVELS for g in GAMMAS}

    def update(self, t):
        if t not in self.hist.index:
            return
        y = self.hist.loc[t].to_numpy()
        for h in range(1, H_MAIN + 1):
            o_prev = str(pd.Period(t, "M") - h)
            for (lv, g), (lo, hi) in self.pending.pop((o_prev, h), {}).items():
                err = ((y < lo) | (y > hi)).astype(float)
                self.alpha[(lv, g)][:, h - 1] = np.clip(self.alpha[(lv, g)][:, h - 1] + g * (LEVELS[lv] - err), *ALPHA_CLIP)

    def intervals(self, o, Xsorted, uids, targets):
        rows = []
        for h in range(Xsorted.shape[0]):
            per_key = {}
            for (lv, g), a_ in self.alpha.items():
                a = a_[:, h].copy()
                lo, hi = sample_quantile(Xsorted[h], a / 2), sample_quantile(Xsorted[h], 1 - a / 2)
                per_key[(lv, g)] = (lo, hi)
                rows.append(pd.DataFrame({"unique_id": uids, "origin": o, "horizon": h + 1, "target": targets[h],
                                          "level": lv, "gamma": g, "alpha": a, "lo": lo, "hi": hi}))
            self.pending[(o, h + 1)] = per_key
        return rows

    def save_state(self, label, uids):
        for (lv, g), a in self.alpha.items():
            for h in range(a.shape[1]):
                self.state.append(pd.DataFrame({"state_at": label, "unique_id": uids, "horizon": h + 1, "level": lv,
                                                "gamma": g, "alpha": a[:, h]}))


def part_score(args):
    track = args.track
    smp = Sampler(track, args.results, args.clean)
    uids, hist = smp.uids, smp.hist
    P0 = Path(args.results) / "prob_main" / f"track={track}"
    D = PROB / f"track={track}"
    D.mkdir(parents=True, exist_ok=True)
    crps_st = pd.read_parquet(P0 / "crps.parquet")
    aci_ex, aci_z = ACI(hist, smp.n), ACI(hist, smp.n)
    q_rows, crps_rows, joint_rows, coh_rows, aci_rows, aci_z_rows = [], [], [], [], [], []
    for o in ORIGINS:
        t0 = time.perf_counter()
        _, Xf, A, Acheck, L, dead, targets, scale = smp.origin(o)
        Xz = np.maximum(Xf, 0.0)
        Xq = np.load(OUT / "samples" / f"track={track}" / f"origin={o}.npy")
        ok = np.isfinite(scale) & (scale > 0)
        vm = np.isin(np.flatnonzero(ok), smp.vs_idx)
        qs = np.quantile(Xq, STORE_Q, axis=1)                                   # (13, H, n)
        chk = 0.0
        for h in range(Xq.shape[0]):
            d = pd.DataFrame(qs[:, h, :].T, columns=QNAMES)
            d.insert(0, "unique_id", uids)
            d.insert(1, "origin", o)
            d.insert(2, "horizon", h + 1)
            d.insert(3, "target", targets[h])
            d.insert(4, "kind", "rec")
            q_rows.append(d)
            Cy = np.abs(Xq[h] @ Acheck.T).max(axis=1)
            coh_rows.append({"origin": o, "horizon": h + 1, "target": targets[h],
                             "max_abs_Cy": float(Cy.max()), "max_violation_pct": float((100 * Cy / Xq[h][:, smp.iw]).max()),
                             "share_values_zero_exact": float((Xq[h] == 0).mean()),
                             "share_values_zero_zeroed": float((Xz[h] == 0).mean()),
                             "mean_abs_diff_exact_zeroed_scaled": float(np.nanmean(np.abs(Xq[h] - Xz[h])[:, ok] / scale[ok]))})
            t = targets[h]
            if t not in hist.index:
                continue
            y = hist.loc[t].to_numpy()
            ref = crps_st[(crps_st.origin == o) & (crps_st.horizon == h + 1) & (crps_st.kind == "rec")] \
                .set_index("unique_id").crps.loc[uids].to_numpy()
            chk = max(chk, float(np.abs(crps_fair(Xz[h], y) - ref).max()))
            crps_rows.append(pd.DataFrame({"unique_id": uids, "origin": o, "horizon": h + 1, "target": t, "kind": "rec",
                                           "crps": crps_fair(Xq[h], y)}))
            Xs, ys = Xq[h][:, ok] / scale[ok], y[ok] / scale[ok]
            joint_rows.append({"origin": o, "horizon": h + 1, "target": t, "kind": "rec", "es": energy_score(Xs, ys),
                               "n_es": int(ok.sum()), "vs": variogram_score(Xs[:, vm], ys[vm]), "n_vs": int(vm.sum())})
        for aci, X, out in [(aci_ex, Xq, aci_rows), (aci_z, Xz, aci_z_rows)]:
            aci.update(o)
            if o == ORIGINS[-1]:
                aci.save_state(o, uids)
            out += aci.intervals(o, np.sort(X, axis=1), uids, targets)
        print(f"score {track} {o}: {time.perf_counter() - t0:.0f} s, zeroed CRPS check {chk:.1e}", flush=True)
        assert chk < 1e-6, chk
    for aci in (aci_ex, aci_z):
        for t in pd.period_range(pd.Period(ORIGINS[-1], "M") + 1, "2026-06", freq="M").astype(str):
            aci.update(t)
        aci.save_state("2026-06", uids)
    # check: the ACI code reproduces the stored zeroed-sample results (results/prob_main). Coverage events with y on an
    # interval bound (y = 0 against a bound of +-1e-13 or 1e-21 left by the projection) are ties decided by
    # floating-point noise, so the zeroed samples regenerated here can flip such an event and the alpha path then differs
    # from the stored one for that series. The check therefore has three parts: (a) the recursion replayed on the stored
    # zeroed-sample intervals reproduces the stored alpha and state exactly; (b) wherever alpha agrees, the zeroed
    # intervals equal the stored ones; (c) every first divergence of alpha is caused by a tie event (|y - bound| < TIE_TOL
    # in the stored or the regenerated interval).
    az = pd.concat(aci_z_rows, ignore_index=True)
    st = pd.read_parquet(P0 / "aci_intervals.parquet")
    keys = ["unique_id", "origin", "horizon", "level", "gamma"]
    m = az.merge(st, on=keys, suffixes=("", "_st"), validate="one_to_one")
    assert len(m) == len(az) == len(st)
    rep = ACI(hist, smp.n)                                                    # (a) replay on the stored intervals
    replay_chk = 0.0
    for o in ORIGINS:
        rep.update(o)
        so = st[st.origin == o]
        for (lv, g), a in rep.alpha.items():
            sa = so[(so.level == lv) & (so.gamma == g)].pivot(index="unique_id", columns="horizon", values="alpha").loc[uids]
            replay_chk = max(replay_chk, float(np.abs(a - sa.to_numpy()).max()))
            for h in range(H_MAIN):
                sh = so[(so.level == lv) & (so.gamma == g) & (so.horizon == h + 1)].set_index("unique_id").loc[uids]
                rep.pending.setdefault((o, h + 1), {})[(lv, g)] = (sh.lo.to_numpy(), sh.hi.to_numpy())
        if o == ORIGINS[-1]:
            rep.save_state(o, uids)
    for t in pd.period_range(pd.Period(ORIGINS[-1], "M") + 1, "2026-06", freq="M").astype(str):
        rep.update(t)
    rep.save_state("2026-06", uids)
    sz = pd.concat(rep.state, ignore_index=True).merge(pd.read_parquet(P0 / "aci_state.parquet"),
                                                       on=["state_at", "unique_id", "horizon", "level", "gamma"])
    state_chk = float(np.abs(sz.alpha_x - sz.alpha_y).max())
    same = (m.alpha - m.alpha_st).abs() < 1e-12                               # (b)
    aci_chk = float(max(np.abs(m.lo - m.lo_st)[same].max(), np.abs(m.hi - m.hi_st)[same].max()))
    first = m[~same].sort_values("origin").groupby(["unique_id", "horizon", "level", "gamma"], as_index=False).first()
    first["o_prev"] = [str(pd.Period(o, "M") - h) for o, h in zip(first.origin, first.horizon)]
    ev = first.merge(m, left_on=["unique_id", "o_prev", "horizon", "level", "gamma"], right_on=keys,
                     suffixes=("_div", ""))                                   # (c) the event that updated alpha
    ev["y"] = [hist.loc[o, u] for o, u in zip(ev.origin_div, ev.unique_id)]
    gap = np.minimum.reduce([np.abs(ev.y - ev[c]) for c in ["lo", "hi", "lo_st", "hi_st"]])
    n_untied = int((gap >= TIE_TOL).sum())
    print(f"ACI check: replay on stored intervals {replay_chk:.1e}, state {state_chk:.1e} ({len(sz)} rows); "
          f"intervals where alpha agrees {aci_chk:.1e}; rows with alpha differing {int((~same).sum())} of {len(m)} "
          f"({len(first)} series x horizon x level x gamma), first divergences not caused by a tie: {n_untied}")
    assert replay_chk < 1e-12 and state_chk < 1e-12 and aci_chk < 1e-6 and n_untied == 0
    ev.drop(columns=["origin"]).rename(columns={"origin_div": "origin_alpha_differs"}) \
        .to_csv(out_path(D / "aci_zeroed_divergences.csv", args.overwrite), index=False, float_format="%.6g")
    ties = []                                                                 # tie events in both sample versions
    for version, rows in [("exact", aci_rows), ("zeroed_regenerated", aci_z_rows), ("zeroed", [st])]:
        iv = pd.concat(rows, ignore_index=True)
        iv = iv[iv.target.isin(hist.index)]
        y = hist.rename_axis(index="target", columns="unique_id").stack().rename("y").reset_index()
        iv = iv.merge(y, on=["target", "unique_id"])
        tie = (np.abs(iv.y - iv.lo) < TIE_TOL) | (np.abs(iv.y - iv.hi) < TIE_TOL)
        g = iv.assign(tie=tie, y0=iv.y == 0).groupby(["level", "gamma"])
        ties.append(g.agg(events=("tie", "size"), tie_events=("tie", "sum"), y_zero_events=("y0", "sum"))
                    .reset_index().assign(version=version))
    pd.concat(ties, ignore_index=True).to_csv(out_path(D / "aci_tie_events.csv", args.overwrite), index=False)

    base_q = pd.read_parquet(P0 / "quantiles.parquet", filters=[("kind", "==", "base")])
    pd.concat([base_q] + q_rows, ignore_index=True).to_parquet(out_path(D / "quantiles.parquet", args.overwrite), index=False)
    pd.concat([crps_st[crps_st.kind == "base"]] + crps_rows, ignore_index=True) \
        .to_parquet(out_path(D / "crps.parquet", args.overwrite), index=False)
    j0 = pd.read_csv(P0 / "joint_scores.csv")
    pd.concat([j0[j0.kind == "base"], pd.DataFrame(joint_rows)], ignore_index=True) \
        .to_csv(out_path(D / "joint_scores.csv", args.overwrite), index=False, float_format="%.8g")
    pd.concat(aci_rows, ignore_index=True).to_parquet(out_path(D / "aci_intervals.parquet", args.overwrite), index=False)
    pd.concat(aci_ex.state, ignore_index=True).to_parquet(out_path(D / "aci_state.parquet", args.overwrite), index=False)
    pd.DataFrame(coh_rows).to_csv(out_path(D / "coherence.csv", args.overwrite), index=False, float_format="%.6g")
    pd.DataFrame([{"track": track, "zeroed_crps_max_abs_diff": chk, "aci_replay_alpha_max_abs_diff": replay_chk,
                   "aci_replay_state_max_abs_diff": state_chk, "zeroed_aci_intervals_max_abs_diff_alpha_equal": aci_chk,
                   "zeroed_aci_rows_alpha_differs": int((~same).sum()), "zeroed_aci_rows": len(m),
                   "first_divergences": len(first), "first_divergences_not_tie": n_untied}]) \
        .to_csv(out_path(D / "checks.csv", args.overwrite), index=False, float_format="%.6g")


def part_compare(args):
    RES = Path(args.results)
    pd.set_option("display.width", 250)
    meta = pd.read_parquet(RES / "base_forecasts_v2" / "series_meta.parquet")
    eco = meta[meta.unit_type == "economy"].copy()
    eco["bgroup"] = eco.measure.map(BASIS_GROUP).fillna("components")
    econ = sorted(eco.unique_id)
    Y = load_history(args.clean)
    yl = Y.stack().rename("y").reset_index().rename(columns={"period": "target", "uid": "unique_id"})
    sc = origin_scales(Y)[["unique_id", "origin", "scale1"]]
    src = {"zeroed": (RES / "prob_main", "eval_prob_nested"),
           "exact": (PROB, "eval_prob_exact_nested")}

    # ---- Table 4 quantities, side by side
    t4 = pd.concat([t4_table(RES, s).assign(samples=k) for k, (_, s) in src.items()], ignore_index=True)
    st4 = pd.read_csv(RES / "paper" / "T4_probabilistic.csv")
    e = t4[t4.samples == "exact"].drop(columns="samples").reset_index(drop=True)      # the paper's T4: exact samples
    assert np.allclose(e[st4.columns[2:]].to_numpy(float), st4[st4.columns[2:]].to_numpy(float), rtol=1e-5)
    j = {k: pd.read_csv(RES / s / "summary_joint.csv") for k, (_, s) in src.items()}
    for k in src:
        jj = j[k][(j[k].window == "main") & (j[k].kind == "rec")].set_index(["track", "horizon"])
        sel = t4.samples == k
        t4.loc[sel, "ES_mean_rec"] = [jj.loc[(tr, str(h)), "ES"] for tr, h in zip(t4[sel].track, t4[sel].horizon)]
        t4.loc[sel, "VS_mean_rec"] = [jj.loc[(tr, str(h)), "VS"] for tr, h in zip(t4[sel].track, t4[sel].horizon)]
    cols = ["sCRPS_rec_median", "sCRPS_rec_weighted", "cov80_uncal", "cov95_uncal", "cov80_aci", "cov95_aci",
            "sWinkler80_aci", "sWinkler95_aci", "ES_median_rec", "VS_median_rec", "ES_mean_rec", "VS_mean_rec"]
    long = t4.melt(id_vars=["track", "horizon", "samples"], value_vars=cols, var_name="quantity")
    side = long.pivot_table(index=["quantity", "track", "horizon"], columns="samples", values="value").reset_index()
    side["exact_minus_zeroed"] = side.exact - side.zeroed
    side["rel_change_pct"] = 100 * (side.exact / side.zeroed - 1)
    t4.to_csv(out_path(OUT / "T4_zeroed_and_exact.csv", args.overwrite), index=False, float_format="%.6g")
    side.to_csv(out_path(OUT / "T4_side_by_side.csv", args.overwrite), index=False, float_format="%.6g")
    print(side[side.horizon == "all"].to_string(index=False))

    # ---- coverage: mean over series (T4 definition), overall and residence totals
    cov = []
    for k, (_, s) in src.items():
        iv = pd.read_csv(RES / s / "summary_intervals.csv")
        iv = iv[(iv.window == "main") & (iv.level == "economies") & iv.metric.isin(["cov80", "cov95"])
                & iv.interval.isin(["uncal_rec", "aci_g0.05", "aci_g0.01"])]
        for grp, sel in [("all series", iv.grouping == "all"), ("residence totals", (iv.grouping == "measure") & (iv.grp == "RES_TOTAL_SYS"))]:
            d = iv[sel & iv.horizon.astype(str).isin(["1", "3", "6", "12", "all"])]
            cov.append(d[["track", "interval", "metric", "horizon", "n_series", "mean"]].assign(samples=k, set=grp))
    cov = pd.concat(cov, ignore_index=True).pivot_table(index=["set", "track", "interval", "metric", "horizon"],
                                                        columns="samples", values="mean").reset_index()
    cov["exact_minus_zeroed"] = cov.exact - cov.zeroed
    cov.to_csv(out_path(OUT / "coverage_mean_over_series.csv", args.overwrite), index=False, float_format="%.6g")

    # ---- coverage: pooled hits / cells (as coverage_wilson.csv) with the circular block bootstrap of coverage_block_bootstrap.csv, paired over the two versions
    pooled = []
    for tr in TRACKS:
        hits = {}
        for k, (P, _) in src.items():
            a = pd.read_parquet(P / f"track={tr}" / "aci_intervals.parquet")
            q = pd.read_parquet(P / f"track={tr}" / "quantiles.parquet",
                                columns=["unique_id", "origin", "horizon", "target", "kind", "q0.025", "q0.1", "q0.9", "q0.975"],
                                filters=[("kind", "==", "rec")])
            fr = []
            for lv, (ql, qh) in {80: ("q0.1", "q0.9"), 95: ("q0.025", "q0.975")}.items():
                fr.append(q[["unique_id", "origin", "horizon", "target", ql, qh]].rename(columns={ql: "lo", qh: "hi"})
                          .assign(level=lv, interval="uncal_rec"))
                fr.append(a[(a.gamma == 0.05) & (a.level == lv)][["unique_id", "origin", "horizon", "target", "lo", "hi"]]
                          .assign(level=lv, interval="aci_g0.05"))
            iv = pd.concat(fr, ignore_index=True)
            iv = iv[(iv.origin >= CAL_START) & iv.unique_id.isin(econ)]
            iv["lo"] = iv.lo.clip(lower=0)
            iv = nest(iv, ["unique_id", "origin", "horizon", "interval"])
            iv = iv.merge(yl, on=["unique_id", "target"])
            iv = iv[WINDOWS["main"](iv.target)]
            iv["hit"] = ((iv.y >= iv.lo) & (iv.y <= iv.hi)).astype(float)
            hits[k] = iv.merge(eco[["unique_id", "bgroup"]], on="unique_id")
        origins = sorted(hits["zeroed"].origin.unique())
        rng = np.random.default_rng(SEED)
        idx = [mbb_indices(len(origins), BLOCK, rng) for _ in range(REPS)]
        for (it, lv), _ in hits["zeroed"].groupby(["interval", "level"]):
            for grp in ["all series", "residence total"]:
                est = {}
                for k, iv in hits.items():
                    d = iv[(iv.interval == it) & (iv.level == lv)]
                    if grp != "all series":
                        d = d[d.bgroup == grp]
                    agg = d.groupby("origin").hit.agg(["sum", "size"]).reindex(origins)
                    s_, n_ = agg["sum"].to_numpy(), agg["size"].to_numpy()
                    est[k] = (s_.sum() / n_.sum(), np.array([s_[i].sum() / n_[i].sum() for i in idx]), int(n_.sum()))
                dd = est["exact"][1] - est["zeroed"][1]
                pooled.append({"track": tr, "interval": it, "level": lv, "set": grp, "n_cells": est["zeroed"][2],
                               **{f"{k}_{c}": v for k in src for c, v in
                                  [("coverage", est[k][0]), ("boot_lo95", np.percentile(est[k][1], 2.5)),
                                   ("boot_hi95", np.percentile(est[k][1], 97.5))]},
                               "exact_minus_zeroed": est["exact"][0] - est["zeroed"][0],
                               "diff_boot_lo95": np.percentile(dd, 2.5), "diff_boot_hi95": np.percentile(dd, 97.5)})
    pooled = pd.DataFrame(pooled)
    pooled.to_csv(out_path(OUT / "coverage_pooled_block_bootstrap.csv", args.overwrite), index=False, float_format="%.6g")
    print(pooled.to_string(index=False))

    # ---- FM - STAT weighted sCRPS with the origin-block bootstrap
    cnt = mbb_counts(len(ORIGINS), BLOCK, REPS, np.random.default_rng(SEED))      # as benchmarks.py
    oi = {o: i for i, o in enumerate(ORIGINS)}
    ei = {u: i for i, u in enumerate(econ)}
    w_e = meta.set_index("unique_id").loc[econ, "weight_2022"].to_numpy()

    def sums(d, hsel):
        d = d[d.unique_id.isin(econ) & d.sae.notna()]
        if hsel is not None:
            d = d[d.horizon == hsel]
        Ssum, N = np.zeros((len(econ), len(ORIGINS))), np.zeros((len(econ), len(ORIGINS)))
        np.add.at(Ssum, (d.unique_id.map(ei).to_numpy(), d.origin.map(oi).to_numpy()), d.sae.to_numpy())
        np.add.at(N, (d.unique_id.map(ei).to_numpy(), d.origin.map(oi).to_numpy()), 1.0)
        return Ssum, N

    def wmean(Ssum, N, c, keep):
        with np.errstate(invalid="ignore", divide="ignore"):
            m = (Ssum[keep] @ c.T) / (N[keep] @ c.T)
        return (w_e[keep][:, None] * m).sum(axis=0) / w_e[keep].sum()

    rows, draws = [], {}
    ones = np.ones((1, len(ORIGINS)))
    for k, (P, _) in src.items():
        cr = {}
        for tr in TRACKS:
            c = pd.read_parquet(P / f"track={tr}" / "crps.parquet", filters=[("kind", "==", "rec")]) \
                .merge(sc, on=["unique_id", "origin"])
            c = c[WINDOWS["main"](c.target)]
            c["sae"] = c.crps / c.scale1
            cr[tr] = c[["unique_id", "origin", "horizon", "sae"]]
        for hsel in [None, 1, 3, 6, 12]:
            Sa, Na = sums(cr["FM"], hsel)
            Sb, Nb = sums(cr["STAT"], hsel)
            with np.errstate(invalid="ignore", divide="ignore"):
                keep = np.isfinite(Sa.sum(1) / Na.sum(1)) & np.isfinite(Sb.sum(1) / Nb.sum(1))
            pa, pb = wmean(Sa, Na, ones, keep)[0], wmean(Sb, Nb, ones, keep)[0]
            dd = wmean(Sa, Na, cnt, keep) - wmean(Sb, Nb, cnt, keep)
            ok = np.isfinite(dd)
            rows.append({"samples": k, "horizon": "all" if hsel is None else hsel, "FM": pa, "STAT": pb,
                         "difference": pa - pb, "rel_difference_pct": 100 * (pa / pb - 1),
                         "lo95": np.percentile(dd[ok], 2.5), "hi95": np.percentile(dd[ok], 97.5),
                         "share_below_zero": float((dd[ok] < 0).mean()), "n_series": int(keep.sum()),
                         "n_resamples": int(ok.sum())})
            if hsel is None:
                draws[k] = dd
    bt = pd.DataFrame(rows)
    ref = pd.read_csv(RES / "inference" / "block_bootstrap.csv")
    ref = ref[ref.comparison.str.startswith("weighted sCRPS")]
    ref = ref.set_index(ref.horizon.astype(str))
    z = bt[bt.samples == "zeroed"]
    z = z.set_index(z.horizon.astype(str))
    assert np.allclose(z.loc[ref.index, ["difference", "lo95", "hi95"]].to_numpy(),
                       ref[["difference", "lo95", "hi95"]].to_numpy(), rtol=1e-5, atol=1e-8)
    print("check: zeroed FM - STAT bootstrap reproduces results/inference/block_bootstrap.csv")
    bt.to_csv(out_path(OUT / "fm_minus_stat_crps_bootstrap.csv", args.overwrite), index=False, float_format="%.6g")
    pd.DataFrame(draws).to_csv(out_path(OUT / "fm_minus_stat_crps_bootstrap_draws.csv", args.overwrite), index=False,
                               float_format="%.6g")
    print(bt.to_string(index=False))

    # ---- where the change comes from: weighted sCRPS by origin (all horizons in the main window) next to the
    # negative values of the free projections, and each series' contribution to the change of the weighted sCRPS
    neg = pd.read_csv(OUT / "negative_counts.csv")
    neg = neg[neg.main_window].groupby(["track", "origin"]).agg(
        mean_negative_values_per_sample=("mean_negative_values_per_sample", "mean"),
        max_negative_values_per_sample=("max_negative_values_per_sample", "max")).reset_index()
    by_o, diag_rows = [], []
    for tr in TRACKS:
        per = {}
        for k, (P, _) in src.items():
            c = pd.read_parquet(P / f"track={tr}" / "crps.parquet", filters=[("kind", "==", "rec")]) \
                .merge(sc, on=["unique_id", "origin"])
            c = c[WINDOWS["main"](c.target)]
            c["sae"] = c.crps / c.scale1
            S_, N_ = sums(c[["unique_id", "origin", "horizon", "sae"]], None)
            with np.errstate(invalid="ignore", divide="ignore"):
                keep = np.isfinite(S_.sum(1) / N_.sum(1))
            per[k] = wmean(S_, N_, np.eye(len(ORIGINS)), keep)
        by_o.append(pd.DataFrame({"track": tr, "origin": ORIGINS, "zeroed": per["zeroed"], "exact": per["exact"]}))
        # diagnostic (not a protocol change): the all-horizon weighted sCRPS with and without origin 2024-09, the
        # origin of the exploding ETS intervals
        for k, (P, _) in src.items():
            c = pd.read_parquet(P / f"track={tr}" / "crps.parquet", filters=[("kind", "==", "rec")]) \
                .merge(sc, on=["unique_id", "origin"])
            c = c[WINDOWS["main"](c.target)]
            c["sae"] = c.crps / c.scale1
            S_, N_ = sums(c[["unique_id", "origin", "horizon", "sae"]], None)
            with np.errstate(invalid="ignore", divide="ignore"):
                keep = np.isfinite(S_.sum(1) / N_.sum(1))
            sel = np.array([[o != "2024-09" for o in ORIGINS]], dtype=float)
            diag_rows.append({"track": tr, "samples": k, "all_origins": wmean(S_, N_, ones, keep)[0],
                              "without_2024_09": wmean(S_, N_, sel, keep)[0]})
    by_o = pd.concat(by_o, ignore_index=True).merge(neg, on=["track", "origin"], how="left")
    by_o["exact_minus_zeroed"] = by_o.exact - by_o.zeroed
    by_o.to_csv(out_path(OUT / "crps_change_by_origin.csv", args.overwrite), index=False, float_format="%.6g")
    wo = pd.DataFrame(diag_rows)
    wo.to_csv(out_path(OUT / "crps_without_2024-09.csv", args.overwrite), index=False, float_format="%.6g")
    print(wo.to_string(index=False))
    print(by_o.to_string(index=False))
    per_s = []
    for k, (_, s) in src.items():
        c = pd.read_parquet(RES / s / "crps_per_series.parquet")
        c = c[(c.kind == "rec") & (c.window == "main") & (c.horizon == "all") & (c.unit_type == "economy")]
        per_s.append(c[["unique_id", "track", "measure", "weight_2022", "sCRPS"]].rename(columns={"sCRPS": k}))
    per_s = per_s[0].merge(per_s[1].drop(columns=["measure", "weight_2022"]), on=["unique_id", "track"])
    per_s["contribution"] = per_s.weight_2022 * (per_s.exact - per_s.zeroed) \
        / per_s.groupby("track").weight_2022.transform("sum")
    per_s.sort_values(["track", "contribution"], ascending=[True, False]) \
        .to_csv(out_path(OUT / "crps_change_by_series.csv", args.overwrite), index=False, float_format="%.6g")
    print(per_s.groupby("track").agg(change=("contribution", "sum"), series_worse=("contribution", lambda v: int((v > 0).sum())),
                                     series_better=("contribution", lambda v: int((v < 0).sum()))).to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", required=True, choices=["count", "estimate", "qp", "check", "score", "compare"])
    ap.add_argument("--track", choices=list(TRACKS))
    ap.add_argument("--clean", default="data/clean")
    ap.add_argument("--results", default="results")
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    faulthandler.enable(sys.stderr)                                       # a native crash prints its stack to the log
    log = RunLog("exact_eval.py", vars(args))
    OUT.mkdir(parents=True, exist_ok=True)
    {"count": part_count, "estimate": part_estimate, "qp": part_qp, "check": part_check, "score": part_score,
     "compare": part_compare}[args.part](args)
    log.finish(f"{args.part} {args.track or ''}")


if __name__ == "__main__":
    main()
