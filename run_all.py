"""
run_all.py: run the steps of the paper in order.

Each step is one script of src/ with the arguments used for the published results. Statistical steps run in the
project environment (requirements.txt); foundation-model steps (env "gpu") run in the foundation-model environment
(requirements-fm.txt), whose python is given by --gpu-python or the environment variable FM_PYTHON. Steps that share a
parallel group run at the same time; everything else runs one after the other. A step that fails stops the run.

Stages, in order:
  data        OECD file -> clean panels, checksums, structural zeros, constraint matrix
  forecasts   statistical and foundation-model base forecasts, in-sample and one-step errors, reconciliation, point
              evaluation
  prob        probabilistic samples (negatives set to zero; kept for the comparison), the tests on later-published
              data at origins 2025-06 and 2026-06, calibration origins and final samples at origin 2026-06
  skill       long-horizon forecasts at 43 origins, skill, H*, persistence with drift, the 2022 episode
  exact       exact non-negative QP of every sample (evaluation, calibration, real-time and final samples), ACI on the
              exact samples, scores, the final intervals
  analyses    supplementary analyses, conformal intervals, benchmarks and the origin-block bootstrap, constraint
              ablation, coherence of the zeroed samples, W diagnostics, post-release evaluation, vintage comparison,
              zeroed-against-exact comparison
  outputs     the paper tables, figures, supplementary tables and the table of quoted figures

From the archive (no refitting, no GPU): restore the stored outputs, then run the outputs stage:
  python src/archive.py restore
  python run_all.py --stages outputs --overwrite

Usage:
  python run_all.py --list                              # steps, environment and expected minutes
  python run_all.py --dry-run --overwrite               # print the commands
  python run_all.py --overwrite                         # every step (about 31 hours one after the other)
  python run_all.py --overwrite --skip-heavy            # skip the steps marked heavy (their outputs must exist)
  python run_all.py --overwrite --skip-gpu              # skip the foundation-model steps (outputs from the archive)
  python run_all.py --overwrite --stages exact analyses outputs
  python run_all.py --overwrite --steps prob.07 prob.08
Every script refuses to overwrite an existing output unless it is given --overwrite; run_all.py --overwrite passes the
flag to every script that accepts it.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = "data/raw/oecd_maritime_co2_raw_2026-10-31.csv"
FM_W = ["--onestep-dirs", "results/onestep_fm", "results/onestep_fm_rt"]
LONG_ORIGINS = ["2021-12", "2022-01", "2022-02", "2022-03", "2022-04", "2022-05", "2022-06", "2022-07", "2022-08",
                "2022-09", "2022-10", "2022-11", "2022-12"]
CALIB_ORIGINS = ["2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12", "2026-01", "2026-02", "2026-03",
                 "2026-04", "2026-05"]
STAT_BASES = ["AutoETS", "AutoTheta", "AutoARIMA", "SeasonalNaive"]
ALL_BASES = STAT_BASES + ["ChronosBolt", "TimesFM"]
ZEROED = ["--prob-dir", "results/prob_main", "--eval-dir", "results/eval_prob_nested"]
EXACT_RT = ["--d-form", "exact", "--nest", "--prob-dir", "results/prob_realtime_exact", "--main-prob-dir",
            "results/prob_main_exact"]

# (stage, script, arguments, environment, minutes (measured on the original machine), heavy, parallel group or None)
STEPS = [
    # ------------------------------------------------------------------ data
    ("data", "prepare_data.py", ["--oecd", RAW, "--out", "data/clean"], "venv", 1, False, None),
    ("data", "checksums.py", ["--raw", RAW, "--out", "results/checksums.csv"], "venv", 1, False, None),
    ("data", "build_panel.py", ["--suffix", "v2", "--structural-zeros"], "venv", 1, False, None),
    ("data", "constraints.py", ["--suffix", "v2"], "venv", 1, False, None),
    # ------------------------------------------------------------------ forecasts
    ("forecasts", "base_forecasts.py", ["--out", "results/base_forecasts_v2", "--suffix", "v2"], "venv", 34, True, None),
    ("forecasts", "evaluate_base.py", ["--out", "results/eval_base_v2"], "venv", 1, False, None),
    ("forecasts", "summary_weighted.py", ["--eval", "results/eval_base_v2"], "venv", 1, False, None),
    ("forecasts", "insample_residuals.py", ["--models"] + STAT_BASES, "venv", 33, True, None),
    ("forecasts", "fm_forecasts.py", ["--out", "results/base_forecasts_fm", "--models", "ChronosBolt", "TimesFM"], "gpu", 7, False, None),
    ("forecasts", "fm_forecasts.py", ["--models", "ChronosBolt", "TimesFM", "--spot-check"], "gpu", 1, False, None),
    ("forecasts", "fm_forecasts.py", ["--out", "results/base_forecasts_fm", "--models", "TimesFM25"], "gpu", 7, False, None),
    ("forecasts", "fm_forecasts.py", ["--models", "TimesFM25", "--spot-check"], "gpu", 1, False, None),
    ("forecasts", "fm_onestep.py", ["--out", "results/onestep_fm", "--models", "ChronosBolt", "TimesFM", "TimesFM25"], "gpu", 30, True, None),
    ("forecasts", "evaluate_fm.py", ["--out", "results/eval_fm_v2"], "venv", 1, False, None),
    *[("forecasts", "reconcile.py", ["--bases", b, "--tag", "v2_burnin12", "--burn-in", "12"], "venv", 75, True, "rec-burnin")
      for b in ALL_BASES],
    ("forecasts", "reconcile.py", ["--bases", "TimesFM25", "--tag", "v2_burnin12", "--burn-in", "12", "--fm-w", "AutoETS"],
     "venv", 27, True, "rec-fm"),
    *[("forecasts", "reconcile.py", ["--bases", b, "--tag", "v2_fmownw", "--fm-w", "own"], "venv", 38, True, "rec-fm")
      for b in ["ChronosBolt", "TimesFM", "TimesFM25"]],
    ("forecasts", "evaluate_fm_w.py", ["--out", "results/eval_fm_w"], "venv", 2, False, None),
    # ------------------------------------------------------------------ prob
    ("prob", "prob_samples.py", ["--track", "FM"], "venv", 6, False, "prob"),
    ("prob", "prob_samples.py", ["--track", "STAT"], "venv", 6, False, "prob"),
    ("prob", "realtime_base.py", ["--out", "results/realtime"], "venv", 1, False, None),
    ("prob", "fm_realtime.py", ["--out", "results/realtime/base_forecasts"], "gpu", 1, False, None),
    ("prob", "fm_onestep.py", ["--out", "results/onestep_fm_rt", "--models", "TimesFM25", "--start", "2025-07", "--end",
                               "2026-06", "--check-months", "2025-07", "2026-01", "2026-06"], "gpu", 4, False, None),
    ("prob", "reconcile.py", ["--bases", "AutoETS", "TimesFM25", "--origins", "2025-06", "2026-06", "--tag", "rt", "--fm-w",
                              "own", "--methods", "MinT", "--forecasts-dir", "results/realtime/base_forecasts",
                              "--residuals-dir", "results/realtime/insample_residuals"] + FM_W, "venv", 3, False, None),
    *[("prob", "prob_samples.py", ["--track", t, "--mode", "realtime", "--forecasts-dir", "results/realtime/base_forecasts",
                                   "--residuals-dir", "results/realtime/insample_residuals"] + FM_W, "venv", 1, False, None)
      for t in ["STAT", "FM"]],
    ("prob", "evaluate_prob.py", ["--nest", "--out", "results/eval_prob_nested"], "venv", 1, False, None),
    ("prob", "evaluate_realtime.py", ["--d-form", "exact", "--nest", "--out", "results/eval_realtime_zeroed_nested"], "venv", 1,
     False, None),
    ("prob", "final_base.py", ["--out", "results/final_base"], "venv", 1, False, None),
    ("prob", "fm_final.py", ["--out", "results/final_base/base_forecasts"], "gpu", 1, False, None),
    ("prob", "reconcile.py", ["--bases", "AutoETS", "TimesFM25", "--origins", "2026-06", "--tag", "final", "--fm-w", "own",
                              "--methods", "MinT", "--forecasts-dir", "results/final_base/base_forecasts",
                              "--residuals-dir", "results/final_base/insample_residuals"] + FM_W, "venv", 6, False, None),
    ("prob", "calib_base.py", [], "venv", 6, False, None),
    ("prob", "fm_calib.py", [], "gpu", 3, False, None),
    ("prob", "calib_aci.py", ["--track", "FM"], "venv", 2, False, None),
    ("prob", "calib_aci.py", ["--track", "STAT"], "venv", 2, False, None),
    ("prob", "reconcile.py", ["--bases", "AutoETS", "TimesFM25", "--origins"] + CALIB_ORIGINS +
     ["--tag", "calib", "--fm-w", "own", "--methods", "MinT", "--forecasts-dir", "results/calib/base_forecasts",
      "--residuals-dir", "results/calib/insample_residuals"] + FM_W, "venv", 60, True, None),
    ("prob", "final_samples.py", ["--track", "FM", "--n-jobs", "14"], "venv", 135, True, None),
    ("prob", "final_samples.py", ["--track", "STAT"], "venv", 1, False, None),
    ("prob", "final_bands.py", ["--track", "FM"], "venv", 1, False, None),
    ("prob", "final_bands.py", ["--track", "STAT"], "venv", 1, False, None),
    # ------------------------------------------------------------------ skill
    ("skill", "longhorizon_base.py", ["--out", "results/longhorizon"], "venv", 3, False, None),
    ("skill", "fm_longhorizon.py", ["--out", "results/longhorizon/base_forecasts"], "gpu", 3, False, None),
    *[("skill", "reconcile.py", ["--bases", b, "--origins"] + LONG_ORIGINS + ["--tag", "long", "--fm-w", "own", "--methods", "MinT",
                                                                           "--forecasts-dir", "results/longhorizon/base_forecasts",
                                                                           "--residuals-dir", "results/longhorizon/insample_residuals"],
       "venv", 16, True, "rec-long") for b in ["AutoETS", "TimesFM25"]],
    ("skill", "longhorizon_extra_base.py", [], "venv", 10, False, None),
    ("skill", "fm_longhorizon_extra.py", [], "gpu", 7, False, None),
    ("skill", "longhorizon_extra_reconcile.py", ["--base", "AutoETS"], "venv", 25, True, "rec-long-extra"),
    ("skill", "longhorizon_extra_reconcile.py", ["--base", "TimesFM25"], "venv", 25, True, "rec-long-extra"),
    ("skill", "skill.py", [], "venv", 10, False, None),
    ("skill", "skill_drift.py", [], "venv", 5, False, None),
    ("skill", "episode_2022.py", [], "venv", 2, False, None),
    # ------------------------------------------------------------------ analyses needed by the exact-sample stage
    ("analyses", "sample_coherence.py", ["--part", "violations", "--track", "FM"], "venv", 10, False, None),
    ("analyses", "sample_coherence.py", ["--part", "violations", "--track", "STAT"], "venv", 10, False, None),
    ("analyses", "sample_coherence.py", ["--part", "summary"], "venv", 1, False, None),
    ("analyses", "sample_coherence.py", ["--part", "qp", "--track", "FM", "--n-jobs", "14"], "venv", 120, True, None),
    ("analyses", "sample_coherence.py", ["--part", "qp", "--track", "STAT", "--n-jobs", "14"], "venv", 120, True, None),
    ("analyses", "sample_coherence.py", ["--part", "score"], "venv", 5, False, None),
    ("analyses", "benchmarks.py", [], "venv", 3, False, None),
    # ------------------------------------------------------------------ exact
    ("exact", "exact_eval.py", ["--part", "count"], "venv", 2, False, None),
    ("exact", "exact_eval.py", ["--part", "qp", "--track", "FM", "--n-jobs", "8"], "venv", 135, True, "qp-eval"),
    ("exact", "exact_eval.py", ["--part", "qp", "--track", "STAT", "--n-jobs", "8"], "venv", 210, True, "qp-eval"),
    ("exact", "exact_eval.py", ["--part", "check"], "venv", 1, False, None),
    ("exact", "exact_eval.py", ["--part", "score", "--track", "FM"], "venv", 6, False, None),
    ("exact", "exact_eval.py", ["--part", "score", "--track", "STAT"], "venv", 6, False, None),
    ("exact", "evaluate_prob.py", ["--nest", "--prob-dir", "results/prob_main_exact", "--out",
                                   "results/eval_prob_exact_nested"], "venv", 3, False, None),
    ("exact", "exact_eval.py", ["--part", "compare"], "venv", 3, False, None),
    *[("exact", "exact_samples.py", ["--part", "qp", "--source", so, "--track", tr, "--n-jobs", "8"], "venv", mins, True,
       f"qp-{so}") for so, tr, mins in [("calib", "FM", 50), ("calib", "STAT", 90), ("realtime", "FM", 6),
                                         ("realtime", "STAT", 12)]],
    ("exact", "exact_samples.py", ["--part", "qp", "--source", "final", "--track", "STAT", "--n-jobs", "8"], "venv", 4, True, None),
    ("exact", "exact_samples.py", ["--part", "check"], "venv", 2, False, None),
    ("exact", "exact_samples.py", ["--part", "aci", "--track", "FM"], "venv", 5, False, None),
    ("exact", "exact_samples.py", ["--part", "aci", "--track", "STAT"], "venv", 5, False, None),
    ("exact", "exact_samples.py", ["--part", "realtime", "--track", "FM"], "venv", 2, False, None),
    ("exact", "exact_samples.py", ["--part", "realtime", "--track", "STAT"], "venv", 2, False, None),
    ("exact", "evaluate_realtime.py", EXACT_RT + ["--out", "results/eval_realtime_exact_nested"], "venv", 2, False, None),
    ("exact", "evaluate_realtime.py", EXACT_RT + ["--aci-name", "aci_intervals_ext.parquet", "--out",
                                                  "results/eval_realtime_exact_ext_nested"], "venv", 2, False, None),
    *[("exact", "final_bands.py", ["--track", tr, "--samples", smp, "--aci-state", f"results/aci_exact/track={tr}/aci_state.parquet",
                                   "--out", "results/final_exact"], "venv", 1, False, None)
      for tr, smp in [("FM", "results/final/track=FM/samples_rec.npy"),
                      ("STAT", "results/exact_samples/final/track=STAT/origin=2026-06.npy")]],
    # ------------------------------------------------------------------ analyses
    ("analyses", "supplementary_analyses.py", [], "venv", 3, False, None),
    ("analyses", "supplementary_analyses.py", ["--items", "1", "4", "6", "7", "8", "10"] + ZEROED +
     ["--out", "results/supplementary_zeroed"], "venv", 3, False, None),
    ("analyses", "conformal_residence.py", [], "venv", 1, False, None),
    ("analyses", "conformal_residence.py", ZEROED + ["--out", "results/conformal_residence_zeroed"], "venv", 1, False, None),
    ("analyses", "conformal_s1.py", [], "venv", 1, False, None),
    ("analyses", "fm_postrelease.py", [], "venv", 1, False, None),
    ("analyses", "w_diagnostics.py", [], "venv", 1, False, None),
    ("analyses", "vintage.py", [], "venv", 1, False, None),
    ("analyses", "constraint_ablation.py", ["--run", "2", "--base", "AutoETS"], "venv", 15, True, "ablation"),
    ("analyses", "constraint_ablation.py", ["--run", "2", "--base", "TimesFM25"], "venv", 15, True, "ablation"),
    ("analyses", "constraint_ablation.py", ["--run", "3", "--base", "AutoETS"], "venv", 3, False, "ablation"),
    ("analyses", "constraint_ablation.py", ["--run", "3", "--base", "TimesFM25"], "venv", 3, False, "ablation"),
    ("analyses", "constraint_ablation.py", ["--score"], "venv", 2, False, None),
    ("analyses", "exact_samples.py", ["--part", "compare"], "venv", 1, False, None),
    # ------------------------------------------------------------------ outputs
    ("outputs", "paper_tables.py", [], "venv", 1, False, None),
    ("outputs", "paper_figures.py", [], "venv", 1, False, None),
    ("outputs", "check_figures.py", [], "venv", 1, False, None),
    ("outputs", "supplementary_tables.py", [], "venv", 1, False, None),
    ("outputs", "quoted_figures.py", [], "venv", 1, False, None),
]
STAGES = ["data", "forecasts", "prob", "skill", "exact", "analyses", "outputs"]


def step_ids():
    ids, count = [], {}
    for s in STEPS:
        count[s[0]] = count.get(s[0], 0) + 1
        ids.append(f"{s[0]}.{count[s[0]]:02d}")
    return ids


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overwrite", action="store_true", help="pass --overwrite to every script that accepts it")
    ap.add_argument("--skip-heavy", action="store_true", help="skip the steps marked heavy (outputs must exist)")
    ap.add_argument("--skip-gpu", action="store_true", help="skip the foundation-model steps (outputs must exist)")
    ap.add_argument("--stages", nargs="+", choices=STAGES, default=STAGES)
    ap.add_argument("--steps", nargs="+", default=None, help="step ids from --list")
    ap.add_argument("--gpu-python", default=os.environ.get("FM_PYTHON"),
                    help="python of the foundation-model environment (default: environment variable FM_PYTHON)")
    ap.add_argument("--venv-python", default=sys.executable, help="python of the project environment (default: this one)")
    args = ap.parse_args()

    ids = step_ids()
    sel = [(i, s) for i, s in zip(ids, STEPS) if s[0] in args.stages and (args.steps is None or i in args.steps)
           and not (args.skip_heavy and s[5]) and not (args.skip_gpu and s[3] == "gpu")]
    if args.list:
        print(f"{'id':14} {'env':4} {'min':>5} {'heavy':5} {'group':15} command")
        for i, (st, sc, a, env, mins, heavy, grp) in zip(ids, STEPS):
            print(f"{i:14} {env:4} {mins:5} {'yes' if heavy else '':5} {grp or '':15} src/{sc} {' '.join(a)}")
        groups = {}
        for s in STEPS:
            if s[6]:
                groups[s[6]] = max(groups.get(s[6], 0), s[4])
        par = sum(s[4] for s in STEPS if s[6] is None) + sum(groups.values())
        print(f"total about {sum(s[4] for s in STEPS) / 60:.1f} h one step at a time; about {par / 60:.1f} h with the "
              f"parallel groups")
        return
    if any(s[3] == "gpu" for _, s in sel) and not args.gpu_python and not args.dry_run:
        sys.exit("foundation-model steps selected: give --gpu-python (or set FM_PYTHON), or use --skip-gpu")

    def command(s):
        st, sc, a, env, *_ = s
        py = (args.gpu_python or "<FM_PYTHON>") if env == "gpu" else args.venv_python
        extra = ["--overwrite"] if args.overwrite and '"--overwrite"' in (ROOT / "src" / sc).read_text(encoding="utf-8") else []
        return [py, str(Path("src") / sc)] + a + extra

    k = 0
    while k < len(sel):
        grp = sel[k][1][6]
        batch = [sel[k]]
        while grp and k + len(batch) < len(sel) and sel[k + len(batch)][1][6] == grp:
            batch.append(sel[k + len(batch)])
        k += len(batch)
        cmds = [(i, command(s)) for i, s in batch]
        for i, c in cmds:
            print(f"[{i}] {' '.join(c)}", flush=True)
        if args.dry_run:
            continue
        t0 = time.perf_counter()
        procs = [subprocess.Popen(c, cwd=ROOT) for _, c in cmds]
        codes = [p.wait() for p in procs]
        print(f"    done in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
        if any(codes):
            sys.exit(f"step(s) {[i for (i, _), c in zip(cmds, codes) if c]} failed: stop")


if __name__ == "__main__":
    main()
