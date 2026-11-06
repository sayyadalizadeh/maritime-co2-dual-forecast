"""
sample_coherence.py: coherence of the evaluation samples and an exact-QP comparison.

The zeroed evaluation samples (prob_samples.py, main mode) are not stored; they are regenerated here exactly:
same marginals, Gaussian copula from the track's MinT W, S = 500, standard normal draws from
numpy default_rng([42, origin]) drawn horizon by horizon (common random numbers), free MinT projection. Check: the
fair CRPS of the regenerated base and zeroed samples must equal results/prob_main/track=<T>/crps.parquet.

Part violations (every evaluation cell, 31 origins x 12 horizons, both tracks): after the projection, negative values
  are set to zero (zeroed samples); for every sample the violation is max over the constraint rows of |C y| (the origin's
  rows: full-rank C with the DELTA_INT ratio r, plus the dead-series rows), in % of the sample's World residence total
  W|RES_TOTAL_SYS. Per cell: median and maximum over the 500 samples, the share of samples above 0.1%, and the share
  of samples with any value set to zero. One file per origin (results/sample_coherence/violations/), so an
  interrupted run resumes at the next origin.
Part qp (the 16 subsample cells: origins 2023-12, 2024-06, 2024-12, 2025-06 x horizons 1, 3, 6, 12, both tracks): the
  same 500 base samples are reconciled with the exact non-negative QP (reconcile.nonneg_gi via final_samples.qp_chunk:
  quadprog, Goldfarb-Idnani, then an exact projection on the active set; a sample whose free projection has no negative
  value is already the QP solution). One file per cell (results/sample_coherence/qp_cells/, not committed
  to git): an interrupted run skips the cells already saved.
Part score: zeroed against QP samples on the 16 cells: scaled CRPS (CRPS / MASE scale at the origin; per series mean
  over the cells, median and weighted over system economies; World residence total one by one), energy and
  variogram scores (scaled, as in prob_samples.py; median over the cells, and mean), uncalibrated coverage of the 80% and
  95% equal-tailed sample intervals (mean over system-economy series), and the FM - STAT difference of each score.

Usage:
  python src/sample_coherence.py --part violations --track FM
  python src/sample_coherence.py --part qp --track FM --estimate
  python src/sample_coherence.py --part qp --track FM --n-jobs 14
  python src/sample_coherence.py --part qp --track STAT --n-jobs 7 --reverse   (optional second process)
  python src/sample_coherence.py --part score
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.linalg import cholesky
from scipy.stats import norm

from evalcore import load_history, origin_scales
from final_samples import qp_chunk
from prob_samples import (S, SEED, TRACKS, VS_TOP, crps_fair, energy_score, quantile_map, sample_quantile,
                          variogram_score)
from reconcile import load_system, load_w_residuals, mint_w, origin_constraints, project, residual_matrix
from runlog import RunLog, out_path

OUT = Path("results/sample_coherence")
ORIGINS = pd.period_range("2022-12", "2025-06", freq="M").astype(str).tolist()
SUB_ORIGINS, SUB_H = ["2023-12", "2024-06", "2024-12", "2025-06"], [1, 3, 6, 12]
THRESH = 0.1                                                          # % of the World residence total


class Sampler:
    """Regenerates the zeroed evaluation samples of one track."""

    def __init__(self, track, results="results", clean="data/clean"):
        RES = Path(results)
        self.T = T = TRACKS[track]
        self.sysd = load_system(RES, clean)
        self.uids, self.n = self.sysd["uids"], self.sysd["n"]
        self.hist = self.sysd["Yh"]
        self.meta = pd.read_parquet(RES / "base_forecasts_v2" / "series_meta.parquet").set_index("unique_id").loc[self.uids]
        self.sc = origin_scales(load_history(clean)).pivot(index="origin", columns="unique_id", values="scale1")[self.uids]
        self.res, self.own, _ = load_w_residuals(RES, T["base"], T["fm_w"], 12)
        f = pd.read_parquet(RES / T["fdir"] / f"model={T['base']}" / "forecasts.parquet",
                            columns=["unique_id", "origin", "target", "horizon", "mean"] + [c for c in T["qcols"] if c != "mean"])
        self.f = f[f.unique_id.isin(self.uids) & f.origin.isin(ORIGINS)].set_index(["origin", "unique_id", "horizon"]).sort_index()
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
        H = int(fo.index.get_level_values("horizon").max())
        Q = np.sort(np.stack([fo[c].unstack("horizon").loc[self.uids].to_numpy() for c in self.T["qcols"]], axis=2), axis=2)
        rng = np.random.default_rng([SEED, int(o.replace("-", ""))])
        Xb = np.empty((H, S, self.n))
        for h in range(H):
            Xb[h] = quantile_map(rng.standard_normal((S, self.n)) @ Lc.T, self.zk, Q[:, h, :])
        Xf = project(Xb.transpose(2, 0, 1).reshape(self.n, H * S), A, L, False)[0].reshape(self.n, H, S).transpose(1, 2, 0)
        targets = [str(pd.Period(o, "M") + h) for h in range(1, H + 1)]
        return Xb, Xf, A, Acheck, L, dead, targets, self.sc.loc[o].to_numpy()


def part_violations(track, args):
    smp = Sampler(track, args.results, args.clean)
    crps_st = pd.read_parquet(Path(args.results) / "prob_main" / f"track={track}" / "crps.parquet")
    D = OUT / "violations" / f"track={track}"
    D.mkdir(parents=True, exist_ok=True)
    for o in ORIGINS:
        p = D / f"origin={o}.csv"
        if p.exists() and not args.overwrite:
            continue
        t0 = time.perf_counter()
        Xb, Xf, A, Acheck, L, dead, targets, _ = smp.origin(o)
        Xr = np.maximum(Xf, 0.0)
        Cy = np.abs(np.einsum("rn,hsn->hsr", Acheck, Xr)).max(axis=2)          # (H, S)
        v = 100 * Cy / Xr[:, :, smp.iw]
        zeroed = (Xf < 0).any(axis=2)
        rows = []
        cs = crps_st[crps_st.origin == o]
        for h in range(Xb.shape[0]):
            chk = np.nan
            t = targets[h]
            if t in smp.hist.index:
                y = smp.hist.loc[t].to_numpy()
                ref = cs[(cs.horizon == h + 1)].pivot(index="unique_id", columns="kind", values="crps").loc[smp.uids]
                chk = max(float(np.abs(crps_fair(Xb[h], y) - ref["base"].to_numpy()).max()),
                          float(np.abs(crps_fair(Xr[h], y) - ref["rec"].to_numpy()).max()))
            rows.append({"track": track, "origin": o, "horizon": h + 1, "target": t, "median_pct": np.median(v[h]),
                         "p95_pct": np.quantile(v[h], 0.95), "max_pct": v[h].max(),
                         "share_samples_gt_0.1pct": float((v[h] > THRESH).mean()),
                         "share_samples_with_zeroing": float(zeroed[h].mean()),
                         "share_values_zeroed": float((Xf[h] < 0).mean()), "crps_check_max_abs": chk})
        pd.DataFrame(rows).to_csv(p, index=False, float_format="%.6g")
        print(f"{track} {o}: {time.perf_counter() - t0:.0f} s, max crps check {np.nanmax([r_['crps_check_max_abs'] for r_ in rows]):.2e}",
              flush=True)


def part_qp(track, args):
    smp = Sampler(track, args.results, args.clean)
    D = OUT / "qp_cells" / f"track={track}"
    D.mkdir(parents=True, exist_ok=True)
    order = (lambda x: x[::-1]) if args.reverse else (lambda x: x)       # --reverse: a second process from the end
    for o in order(SUB_ORIGINS):
        todo = [h for h in order(SUB_H) if args.overwrite or not (D / f"origin={o}_h={h}.npz").exists()]
        if not todo:
            continue
        Xb, Xf, A, Acheck, L, dead, targets, _ = smp.origin(o)
        for h in todo:
            if not args.overwrite and not args.estimate and (D / f"origin={o}_h={h}.npz").exists():
                continue                                                  # saved meanwhile by another process
            t0 = time.perf_counter()
            m = 32 if args.estimate else S
            chunks = [list(range(j, min(j + 10, m))) for j in range(0, m, 10)]
            out = Parallel(n_jobs=args.n_jobs)(delayed(qp_chunk)(Xb[h - 1, c].T, Xf[h - 1, c].T, A, L, dead, Acheck)
                                               for c in chunks)
            secs = time.perf_counter() - t0
            if args.estimate:
                n_qp = sum(r_["qp"] for _, rows in out for r_ in rows)
                print(f"estimate {track} {o} h{h}: {m} samples ({n_qp} needed the QP) in {secs:.1f} s with {args.n_jobs} jobs "
                      f"-> 16 cells x {S} ~ {secs / m * S * 16 / 3600:.2f} h", flush=True)
                return
            Xq = np.hstack([Y for Y, _ in out]).T                                # (S, n)
            diag = pd.DataFrame([dict(sample=i, **r_) for i, r_ in enumerate(r for _, rows in out for r in rows)])
            np.savez_compressed(out_path(D / f"origin={o}_h={h}.npz", True), Xq=Xq)
            diag.assign(track=track, origin=o, horizon=h, seconds_cell=secs).to_csv(D / f"origin={o}_h={h}_diag.csv",
                                                                                    index=False, float_format="%.6g")
            print(f"{track} {o} h{h}: {secs:.0f} s, {int(diag.qp.sum())} of {S} samples needed the QP, "
                  f"max |Cy| {diag.max_abs_Cy.max():.2e}, min y {diag.min_y.min():.2e}", flush=True)


def part_score(args):
    rows, joint, cov = [], [], []
    for track in TRACKS:
        smp = Sampler(track, args.results, args.clean)
        eco = (smp.meta.unit_type == "economy").to_numpy()
        w = smp.meta.weight_2022.to_numpy()
        for o in SUB_ORIGINS:
            Xb, Xf, A, Acheck, L, dead, targets, scale = smp.origin(o)
            ok = np.isfinite(scale) & (scale > 0)
            for h in SUB_H:
                y = smp.hist.loc[targets[h - 1]].to_numpy()
                Xz = np.maximum(Xf[h - 1], 0.0)
                Xq = np.load(OUT / "qp_cells" / f"track={track}" / f"origin={o}_h={h}.npz")["Xq"]
                for meth, X in [("zeroed", Xz), ("exact QP", Xq)]:
                    sc_ = np.where(ok, crps_fair(X, y) / np.where(ok, scale, 1.0), np.nan)
                    rows.append(pd.DataFrame({"track": track, "method": meth, "origin": o, "horizon": h,
                                              "unique_id": smp.uids, "scrps": sc_}))
                    Xs, ys = X[:, ok] / scale[ok], y[ok] / scale[ok]
                    vm = np.isin(np.flatnonzero(ok), smp.vs_idx)
                    Cy = np.abs(X @ Acheck.T).max(axis=1)
                    joint.append({"track": track, "method": meth, "origin": o, "horizon": h, "es": energy_score(Xs, ys),
                                  "vs": variogram_score(Xs[:, vm], ys[vm]),
                                  "max_violation_pct": float((100 * Cy / X[:, smp.iw]).max())})
                    Xso = np.sort(X, axis=0)
                    for lv, a in [(80, 0.2), (95, 0.05)]:
                        lo = sample_quantile(Xso, np.full(smp.n, a / 2))
                        hi = sample_quantile(Xso, np.full(smp.n, 1 - a / 2))
                        inside = ((y >= lo) & (y <= hi))[eco]
                        cov.append({"track": track, "method": meth, "origin": o, "horizon": h, "level": lv,
                                    "coverage_economies": float(inside.mean())})
        print(f"{track}: scored", flush=True)
    sc = pd.concat(rows, ignore_index=True)
    meta = smp.meta
    summ = []
    for (tr, meth), d in sc.groupby(["track", "method"]):
        per = d.groupby("unique_id").scrps.mean()
        e = per[per.index.isin(meta.index[meta.unit_type == "economy"])].dropna()
        summ.append({"track": tr, "method": meth, "median_sCRPS": e.median(),
                     "weighted_sCRPS": np.average(e, weights=meta.loc[e.index, "weight_2022"]),
                     "world_res_total_sCRPS": per["W|RES_TOTAL_SYS"], "n_series": len(e)})
    summ = pd.DataFrame(summ)
    jd = pd.DataFrame(joint)
    js = jd.groupby(["track", "method"]).agg(es_median=("es", "median"), es_mean=("es", "mean"), vs_median=("vs", "median"),
                                             vs_mean=("vs", "mean"), max_violation_pct=("max_violation_pct", "max")).reset_index()
    cv = pd.DataFrame(cov).groupby(["track", "method", "level"]).coverage_economies.mean().unstack("level") \
        .rename(columns={80: "coverage80", 95: "coverage95"}).reset_index()
    tab = summ.merge(js, on=["track", "method"]).merge(cv, on=["track", "method"])
    diff = []
    for meth, d in tab.groupby("method"):
        a, b = d.set_index("track").loc["FM"], d.set_index("track").loc["STAT"]
        diff.append({"method": meth, **{f"FM_minus_STAT_{c}": a[c] - b[c] for c in
                                        ["median_sCRPS", "weighted_sCRPS", "es_median", "vs_median", "coverage80", "coverage95"]}})
    sc.to_csv(out_path(OUT / "qp_vs_zeroed_cells_scrps.csv", args.overwrite), index=False, float_format="%.6g")
    jd.to_csv(out_path(OUT / "qp_vs_zeroed_joint_cells.csv", args.overwrite), index=False, float_format="%.6g")
    tab.to_csv(out_path(OUT / "qp_vs_zeroed_summary.csv", args.overwrite), index=False, float_format="%.6g")
    pd.DataFrame(diff).to_csv(out_path(OUT / "qp_vs_zeroed_fm_minus_stat.csv", args.overwrite), index=False, float_format="%.6g")
    pd.set_option("display.width", 220)
    print(tab.to_string(index=False))
    print(pd.DataFrame(diff).to_string(index=False))


def summarise_violations(args):
    v = pd.concat([pd.read_csv(p) for p in sorted((OUT / "violations").glob("track=*/origin=*.csv"))], ignore_index=True)
    s = v.groupby("track").agg(cells=("origin", "size"), median_of_cell_medians_pct=("median_pct", "median"),
                               max_of_cell_medians_pct=("median_pct", "max"), max_pct=("max_pct", "max"),
                               mean_share_samples_gt_01pct=("share_samples_gt_0.1pct", "mean"),
                               max_share_samples_gt_01pct=("share_samples_gt_0.1pct", "max"),
                               cells_with_any_sample_gt_01pct=("share_samples_gt_0.1pct", lambda x: int((x > 0).sum())),
                               mean_share_samples_with_zeroing=("share_samples_with_zeroing", "mean"),
                               max_crps_check=("crps_check_max_abs", "max")).reset_index()
    byh = v.groupby(["track", "horizon"]).agg(median_pct=("median_pct", "median"), max_pct=("max_pct", "max"),
                                              share_gt_01pct=("share_samples_gt_0.1pct", "mean")).reset_index()
    s.to_csv(out_path(OUT / "violations_summary.csv", args.overwrite), index=False, float_format="%.6g")
    byh.to_csv(out_path(OUT / "violations_by_horizon.csv", args.overwrite), index=False, float_format="%.6g")
    v.to_csv(out_path(OUT / "violations_cells.csv", args.overwrite), index=False, float_format="%.6g")
    print(s.to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", required=True, choices=["violations", "qp", "score", "summary"])
    ap.add_argument("--track", choices=list(TRACKS))
    ap.add_argument("--clean", default="data/clean")
    ap.add_argument("--results", default="results")
    ap.add_argument("--n-jobs", type=int, default=14)
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--reverse", action="store_true", help="qp: process the cells in reverse order (second process)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    log = RunLog("sample_coherence.py", vars(args))
    if args.part == "violations":
        part_violations(args.track, args)
    elif args.part == "qp":
        part_qp(args.track, args)
    elif args.part == "summary":
        summarise_violations(args)
    else:
        part_score(args)
    log.finish(f"{args.part} {args.track or ''}")


if __name__ == "__main__":
    main()
