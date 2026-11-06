"""
qp_pool.py: the exact non-negative QP of many samples in worker processes, with a time limit per sample.

Each worker is a child process that holds one constraint set (M, bounded series, quadprog matrix; see
reconcile.nullspace_setup) and solves one sample at a time, sent by the parent through a pipe. quadprog has no iteration
limit, and one sample of STAT origin 2024-09 made it run for hours, so the parent stops a worker that exceeds the time
limit (default 120 s), starts a fresh one, and passes the sample to the next stage. A worker that dies (e.g. a native
crash) is handled in the same way. Stages, in order:
  quadprog           reconcile.nonneg_nullspace
  quadprog_rescaled  the same on the sample divided by c = max |y_f| (the constraint right-hand side -y_f is divided
                     by the same c; the solution and the multipliers are multiplied back by c)
  osqp               reconcile.nonneg_nullspace_osqp: OSQP at eps_abs = eps_rel = 1e-9, polish on, then the exact
                     active-set step
A stage succeeds only if its solution passes the checks applied to every exact solution: largest constraint violation
max |C y| below VIOL_TOL_PCT % of the World residence total, no value below NEG_TOL, and the KKT conditions (every
multiplier of the active bounds >= -LAM_TOL; stationarity and complementary slackness hold by construction of the exact
step, which also rules out the unverified OSQP path). Otherwise the sample moves to the next stage; a sample that no
stage solves is returned with the last solution, flagged "unverified". Only the parent writes the progress log.
"""
import faulthandler
import multiprocessing as mp
import time
from collections import deque
from multiprocessing.connection import wait

import numpy as np
from threadpoolctl import threadpool_limits

from reconcile import NEG_TOL, nonneg_nullspace, nonneg_nullspace_osqp

STAGES = ("quadprog", "quadprog_rescaled", "osqp")
LAM_TOL = 1e-9                     # an active multiplier below -LAM_TOL: not a KKT point
VIOL_TOL_PCT = 1e-6                # largest |C y| above this % of the World residence total: not coherent
TIME_LIMIT = 120.0                 # seconds per sample and stage


def solve_stage(stage, xf, M, bnd, Cb):
    """One stage for one sample. Returns (y, size of the zero set, min multiplier of the active bounds, iterations,
    flag, note)."""
    if stage == "quadprog":
        y, fixed, lam, iters, flag = nonneg_nullspace(xf, M, bnd, Cb)
        note = ""
    elif stage == "quadprog_rescaled":
        c = float(np.abs(xf).max()) or 1.0
        y, fixed, lam, iters, flag = nonneg_nullspace(xf / c, M, bnd, Cb)
        y, lam = y * c, lam * c
        note = f"scale {c:.6g}"
    else:
        y, fixed, lam, iters, flag, status = nonneg_nullspace_osqp(xf, M, bnd)
        note = f"OSQP status {status}"
    return y, len(fixed), lam, iters, flag, note


def _worker(conn, M, bnd, Cb):
    """Child process: solve (j, stage, x_f) messages until None arrives."""
    faulthandler.enable()
    Cb = np.array(Cb)                                    # quadprog needs a writable copy
    with threadpool_limits(1):
        while True:
            msg = conn.recv()
            if msg is None:
                return
            j, stage, xf = msg
            t0 = time.perf_counter()
            try:
                conn.send((j, stage, None, solve_stage(stage, xf, M, bnd, Cb), time.perf_counter() - t0))
            except Exception as e:                       # noqa: BLE001 (any failure moves the sample on)
                conn.send((j, stage, f"{type(e).__name__}: {e}", None, time.perf_counter() - t0))


def solve_pool(cols, M, bnd, Cb, check, n_jobs, tag, progress=None, timeout=TIME_LIMIT, force_stage=None):
    """Exact QP for every column of cols (free projections, n x m) with a value below NEG_TOL; the other columns are
    already the QP solution (values in [NEG_TOL, 0) set to zero). check(y) -> dict with max_abs_Cy, max_violation_pct.
    force_stage: {column: stage} to start a column at a later stage (tests). Returns (Y, per-column diagnostics,
    attempts: one row per failed or fallback stage)."""
    ctx = mp.get_context("spawn")
    n, m = cols.shape
    Y = np.maximum(cols, 0.0)
    force_stage = force_stage or {}
    need = [j for j in range(m) if (cols[:, j] < NEG_TOL).any()]
    todo = deque((j, force_stage.get(j, "quadprog")) for j in need)
    diag = {j: {"n_neg_free": int((cols[:, j] < NEG_TOL).sum()), "qp": j in need, "solver": "", "fallback_reason": "",
                "n_zero": 0, "min_multiplier": np.nan, "iterations": 0, "flag": "", "seconds": 0.0} for j in range(m)}
    attempts, done = [], 0
    workers = {}                                         # parent end of pipe -> [process, (j, stage) or None, start]

    def log(text):
        if progress:
            with open(progress, "a") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {tag} {text}\n")

    def start():
        a, b = ctx.Pipe()
        p = ctx.Process(target=_worker, args=(b, M, bnd, Cb), daemon=True)
        p.start()
        b.close()
        workers[a] = [p, None, 0.0]

    def feed():
        for c, w in workers.items():
            if w[1] is None and todo:
                j, stage = todo.popleft()
                c.send((j, stage, np.ascontiguousarray(cols[:, j])))
                w[1], w[2] = (j, stage), time.perf_counter()

    def kill(c):
        p = workers.pop(c)[0]
        p.kill()
        p.join()
        c.close()
        start()

    def handle(j, stage, err, out, secs):
        nonlocal done
        d = diag[j]
        d["seconds"] += secs
        if err is None:
            y, nz, lam, iters, flag, note = out
            r = check(y)
            bad = [s for s, b in [(f"negative multiplier {lam:.3g}", lam < -LAM_TOL),
                                  (f"value {y.min():.3g} below NEG_TOL", y.min() < NEG_TOL),
                                  (f"violation {r['max_violation_pct']:.3g}%", not r["max_violation_pct"] < VIOL_TOL_PCT),
                                  (flag, flag.startswith("no exact active set"))] if b]
            if stage != "quadprog" or bad:
                attempts.append({"column": j, "stage": stage, "ok": not bad, "reason": "; ".join(bad), "note": note,
                                 "seconds": secs, "n_zero": nz, "min_multiplier": lam, "min_y": float(y.min()),
                                 "iterations": iters, "flag": flag, **r})
            if not bad or stage == STAGES[-1]:
                Y[:, j] = y
                d.update(solver=stage, n_zero=nz, min_multiplier=lam, iterations=iters,
                         flag=flag if not bad else (flag + "; " if flag else "") + "unverified: " + "; ".join(bad))
                done += 1
                if stage != "quadprog":
                    log(f"column {j}: solved by {stage} ({'checks passed' if not bad else 'UNVERIFIED'})")
                if done % 500 == 0:
                    log(f"{done}/{len(need)} samples solved")
                return
            err = "; ".join(bad)
        else:
            attempts.append({"column": j, "stage": stage, "ok": False, "reason": err, "note": "", "seconds": secs})
        if stage == STAGES[-1]:
            raise RuntimeError(f"{tag} column {j}: no stage gave a solution ({err})")
        nxt = STAGES[STAGES.index(stage) + 1]
        d["fallback_reason"] += ("; " if d["fallback_reason"] else "") + f"{stage}: {err}"
        log(f"column {j}: {stage} failed ({err}); next {nxt}")
        todo.append((j, nxt))

    for _ in range(min(n_jobs, len(need))):
        start()
    feed()
    try:
        while todo or any(w[1] is not None for w in workers.values()):
            busy = [c for c, w in workers.items() if w[1] is not None]
            for c in wait(busy, timeout=1.0):
                try:
                    msg = c.recv()
                except (EOFError, OSError):                   # the worker died
                    j, stage = workers[c][1]
                    secs = time.perf_counter() - workers[c][2]
                    kill(c)
                    handle(j, stage, "worker process died", None, secs)
                    continue
                workers[c][1] = None
                handle(*msg)
            now = time.perf_counter()
            for c, w in list(workers.items()):
                if w[1] is not None and now - w[2] > timeout:
                    j, stage = w[1]
                    kill(c)
                    handle(j, stage, f"time limit {timeout:.0f} s exceeded", None, now - w[2])
            feed()
    finally:
        for c, w in workers.items():
            try:
                c.send(None)
            except OSError:
                pass
        for w in workers.values():
            w[0].join(timeout=10)
            if w[0].is_alive():
                w[0].kill()
    return Y, [diag[j] for j in range(m)], attempts
