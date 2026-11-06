# Architecture

How the pipeline is organised: modules, configuration, the constraint matrix, the tests, and how to adapt the pipeline to another inventory or to a new OECD release. The data are described in [DATA_RULES.md](DATA_RULES.md).

## 1. Layout and principles

```
data/raw/        OECD complete CSV (read-only; from the archive or src/download_data.py)
data/clean/      panels built from it (src/prepare_data.py, src/build_panel.py)
src/             one script per step (command line, --help)
results/         CSV / parquet outputs; results/paper/ holds the paper's tables
figures/         the paper's figures
tests/           pytest suite
examples/        synthetic example
archive/         the archive of stored outputs (not in git): MANIFEST.csv + results/, data/ mirrored
release-assets/  the archive packaged as release assets (not in git): zip parts, MANIFEST.csv, SHA256SUMS.txt
run_all.py       runs every step in order
```

- **One step, one script.** Every step is a script in `src/` that reads from `data/` and `results/` and writes to `results/` or `figures/`. Scripts communicate only through files.
- **No silent overwrites.** `runlog.out_path` refuses to overwrite an existing output unless `--overwrite` is given.
- **Run log.** `runlog.RunLog` appends the script, settings, Python version and run time to `results/run_log.csv`.
- **Two environments.** Statistical work runs in the project environment (Python 3.12, `requirements.txt`). Foundation-model forecasts run in a separate environment with CUDA (`requirements-fm.txt`); those steps are marked `gpu` in `run_all.py`.
- **Archive.** Files over 10 MB are not in git. The stored outputs needed to run the tests and to regenerate the paper's outputs without refitting are in the archive, attached to the GitHub release as zip parts of less than 2 GB with a manifest and SHA-256 sums. `src/archive.py` downloads or unpacks these assets into `archive/` (`download`, `unpack`), verifies every file (size and SHA-256), restores the archive into the repository tree, and builds the assets (`package`).

## 2. Modules, from raw data to the paper's outputs

The exact command lines are in `run_all.py` (`STEPS`; `python run_all.py --list`).

| Stage | Scripts | Reads | Writes |
|---|---|---|---|
| data | `download_data.py`, `prepare_data.py`, `checksums.py`, `build_panel.py` (rule: `structural_zeros.py`), `constraints.py` (generic form: `dual_allocation.py`) | raw OECD CSV | `data/clean/`: tidy panel, area lookup, system panel (structural zeros, component-sum residence total, discrepancy node), wide panel, outside-system panel; `results/structural_zeros.csv`; `C_matrix_v2.npz`, `C_matrix_fullrank_v2.npz`, row and column legends, residuals of C on the history |
| forecasts | `base_forecasts.py`, `evaluate_base.py`, `summary_weighted.py`, `insample_residuals.py`, `fm_forecasts.py` and `fm_onestep.py` (gpu), `evaluate_fm.py`, `reconcile.py`, `evaluate_fm_w.py` (`evalcore.py`) | panels | `base_forecasts_v2/`, `base_forecasts_fm/` (31 origins 2022-12 ... 2025-06, h 1-12); `insample_residuals_v2/`, `onestep_fm/`; `reconciled_v2_burnin12/`, `reconciled_v2_fmownw/` (OLS, WLS, MinT; free, iterative and exact non-negative); `eval_base_v2/`, `eval_fm_v2/`, `eval_fm_w/` (MASE, RMSSE, DM tests, MCS) |
| prob | `prob_samples.py`, `evaluate_prob.py`, `realtime_base.py`, `fm_realtime.py`, `evaluate_realtime.py`, `final_base.py`, `fm_final.py`, `calib_base.py`, `fm_calib.py`, `calib_aci.py`, `final_samples.py`, `final_bands.py` | base forecasts, W, C | `prob_main/` (samples with negatives set to zero, kept for the comparison), `eval_prob_nested/`; `realtime/`, `reconciled_rt/`, `prob_realtime/` (origins 2025-06 and 2026-06); `calib/` (origins 2025-07 ... 2026-05); `final_base/`, `reconciled_final/`, `final/` (origin 2026-06) |
| skill | `longhorizon_base.py`, `fm_longhorizon.py`, `longhorizon_extra_base.py`, `fm_longhorizon_extra.py`, `longhorizon_extra_reconcile.py`, `skill.py`, `skill_drift.py` (functions: `skill_decay.py`), `episode_2022.py` | long-horizon forecasts, h 1-36, 43 origins 2021-12 ... 2025-06 | `reconciled_long/`, `reconciled_long_extra/`, `skill/` (s(h), H*, HAC test), `skill_drift/`, `episode_2022/` |
| exact | `exact_eval.py`, `exact_samples.py` (`qp_pool.py`), `evaluate_prob.py --prob-dir`, `evaluate_realtime.py --prob-dir`, `final_bands.py --out results/final_exact` | every sample: evaluation (31 origins x 12 horizons x 500), calibration, real-time and final | `prob_main_exact/`, `eval_prob_exact_nested/`, `exact_eval/` (solver diagnostics, zeroed against exact); `exact_samples/`, `aci_exact/` (ACI over all 42 origins), `prob_realtime_exact/`, `eval_realtime_exact_nested/`, `eval_realtime_exact_ext_nested/`, `final_exact/` |
| analyses | `supplementary_analyses.py`, `conformal_residence.py`, `conformal_s1.py`, `benchmarks.py`, `constraint_ablation.py`, `sample_coherence.py`, `w_diagnostics.py`, `fm_postrelease.py`, `vintage.py`, `exact_samples.py --part compare` | stored forecasts, samples' scores and intervals | `supplementary/`, `conformal_residence/`, `conformal_s1/`, `benchmarks/`, `inference/`, `checks/`, `ablation/`, `sample_coherence/`, `w_diagnostics/`, `fm_postrelease/`, `vintage/`, `compare/` (and `*_zeroed/` versions for the comparison) |
| outputs | `paper_tables.py`, `paper_figures.py`, `check_figures.py`, `supplementary_tables.py`, `quoted_figures.py` | the stages above | `results/paper/`, `figures/` |

**Shared modules:**
- **`reconcile.py`**: the system set-up (`load_system`, `origin_constraints`), the projection (`project`), W (`shrink_cov`, `mint_w`) and the exact non-negative QP: `nonneg_gi` (quadprog, Goldfarb-Idnani, followed by an exact projection on the active set), `nonneg_nullspace` (the same QP in the null space of the equality constraints, 10 to 40 times faster) and its OSQP fallback `nonneg_nullspace_osqp`. `qp_pool.py` solves many samples in worker processes with a time limit per sample (120 s) and fallback stages: quadprog on the rescaled sample, then OSQP at eps 1e-9 with the active set refined to the KKT point. Every solution is checked for coherence, non-negativity and the KKT conditions, and every fallback is recorded.
- **`evalcore.py`**: the evaluation windows, the MASE and RMSSE scales, the weighted summaries and the Diebold-Mariano test with the Harvey-Leybourne-Newbold correction.
- **`skill_decay.py`**: the persistence reference and the skill functions.
- **`dual_allocation.py`**: the generic constraint builder (Section 4).

## 3. Configuration

Settings are command-line arguments (each script's `--help`) and module constants; the published runs use the arguments in `run_all.py`.

| Setting | Value | Where |
|---|---|---|
| Evaluation origins and window | monthly 2022-12 to 2025-06; h 1-12; main window: targets up to 2025-12 without 2025-09; robustness: with 2025-09, and 2026-01 to 2026-06 (stress test, 2026 reclassification) | `evalcore.WINDOWS`, `reconcile.py` defaults |
| Discrepancy ratio r | mean of DELTA_INT / G(World) over the 12 months up to and including the origin; the balance row becomes E + F - (1 + r) G = 0 | `reconcile.R_WINDOW` |
| Dead series | zero in all 24 months up to the origin: fixed at zero for that origin (still scored) | `reconcile.DEAD_WINDOW` |
| Residual burn-in | the first 12 months of in-sample residuals are left out of W | `reconcile.py --burn-in` |
| W of foundation models | their own one-step forecast errors (2020-01 onward, context up to the previous month) | `reconcile.py --fm-w own` |
| Variance floor | `VAR_FLOOR` = 1e-4 (t/day)² | `reconcile.py` |
| Samples | S = 500 per cell, Gaussian copula with the MinT correlation, seed 42 | `prob_samples.py`, `final_samples.py` |
| ACI | γ = 0.05 (main), 0.01 (robustness); α starts at 0.20 (80%) and 0.05 (95%); α clipped to [0.001, 0.999]; 80% interval nested inside the 95% interval; state extended with the calibration origins 2025-07 to 2026-05 for the final intervals | `prob_samples.py` (`GAMMAS`, `LEVELS`, `ALPHA_CLIP`), `calib_aci.py`, `exact_samples.py` |
| Pooled split conformal | per horizon and measure group, finite-sample level ceil((N + 1)(1 - α)) / N | `evaluate_prob.py`, `conformal_s1.py` |
| Variogram score | p = 0.5, scaled values | `prob_samples.py` |
| Skill and H* | 43 origins, h 1-36; moving-block bootstrap over origins, block 6 (3, 12 and i.i.d. as robustness), 1,000 replications, seed 42; H* = first horizon with the 95% lower bound at or below zero | `skill.py` |
| MCS | 10% level, stationary bootstrap, 10,000 replications, seed 42 | `evaluate_fm.py`, `evaluate_fm_w.py` |
| Origin-block bootstrap | blocks of 6 consecutive origins, all series jointly, 2,000 resamples | `benchmarks.py`, `exact_eval.py` |
| Seeds | 42 in every script that draws random numbers | module constants `SEED` |

**Solver settings:**

| Setting | Value |
|---|---|
| quadprog | 0.1.14; Goldfarb-Idnani; no tolerance arguments |
| OSQP | 1.1.3; eps_abs = eps_rel = 1e-9, max_iter = 200,000, polish on |
| `NEG_TOL` | -1e-6 t/day |
| `DUAL_TOL` | 1e-8 |
| `RANK_TOL` | 1e-10 (pivoted QR) |
| time limit per sample (worker pool) | 120 s |

## 4. How the accounting structure enters C

- **The state vector.** For each month, y stacks the ten measures of every unit: the seven components A to G, the residence total `RES_TOTAL_SYS` = A+B+D+E+F and the territory totals `TER_DOM` = A+C and `TER_INT` = D+G. The units are the 110 system economies, `RESIDUAL` and `W` (World). World's discrepancy node `DELTA_INT` = E(W) + F(W) - G(W) comes last: 112 × 10 + 1 = 1,121 columns.
- **The constraints.** C y = 0 collects three kinds of rows (`src/constraints.py`; generic form in `src/dual_allocation.py`):

| Rows | Count | Equation |
|---|---:|---|
| within, per unit | 3 × 112 | `RES_TOTAL_SYS` - (A+B+D+E+F) = 0; `TER_DOM` - (A+C) = 0; `TER_INT` - (D+G) = 0 |
| aggregation, per measure | 10 | k(W) - Σ_economies k - k(Residual) = 0 |
| world balance | 2 | B(W) - C(W) = 0; E(W) + F(W) - G(W) - DELTA_INT = 0 |

- **Full rank.** C has 348 rows and rank 345. The three within rows of World follow from the other rows and are flagged as redundant; dropping them gives the full-rank matrix.
- **At each origin (`reconcile.origin_constraints`):** the DELTA_INT column is removed and the coefficient of G(W) in the balance row becomes -(1 + r); a row y_i = 0 is added for every dead series; dependent rows are dropped with a pivoted QR (345 to 348 independent rows per origin).
- **Residence and territory views.** The residence view (operator country) and the territory view (departure country) meet through the shared components A and D, and at World level through the two balance rows: what residents operate abroad (B, E, F) equals what non-residents operate in each economy (C, G), up to DELTA_INT.

`dual_allocation.build(paper_spec(economies))` returns exactly the stored C and full-rank C: same rows in the same order, same coefficients, same redundant flags (`tests/test_constraints.py`).

## 5. Adapting the pipeline to another inventory with two allocation principles

1. **Describe the structure as data** with `dual_allocation.Spec`: the units (economies modelled individually, a Residual node, World); the component measures; each total as a signed sum of components, for example {"RES_TOTAL": "A+B+D+E+F", "TER_DOM": "A+C", "TER_INT": "D+G"}; the World balance rows that tie the two principles together; World-only discrepancy columns if the published data do not balance exactly. `build(spec)` returns C, its full-rank version and the row and column legends. `examples/toy_dual_allocation.py` builds a three-economy system from synthetic flows.
2. **Prepare a wide panel** with one column per `unit|measure`, in the builder's column order (`System.labels`), as `build_panel.py` does. Fill only structural zeros that the published totals prove. Check C y = 0 on the history; `constraints.py` reports the residuals by row type.
3. **Swap the matrix and the panel.** Write them under the same file names for a new suffix. The rest of the pipeline uses only C, the panel and the series identifiers: base forecasts, W, projection, the exact QP, samples, ACI and evaluation. The discrepancy ratio and the dead-series rows are set in `reconcile.origin_constraints`; change the column names there if the discrepancy node differs.

## 6. A new OECD release

1. Download the complete CSV (`src/download_data.py --download`) to `data/raw/` under a new dated name; do not edit it. Record its checksum (`checksums.py`).
2. Rebuild `data/clean/` with `prepare_data.py` and `build_panel.py --suffix <new>`. Compare the system economies (`results/system_economies_<new>.csv`) and the structural zeros with the previous release, and the revisions of overlapping months (`vintage.py` does this for two releases).
3. Run `constraints.py --suffix <new>`. The script stops if any cell fails beyond rounding; investigate and document before going further.
4. Update the dates fixed by the protocol (last observed month, the origin of the tests on later-published data, the months scored); the protocol itself (origins, windows, horizons, weights) stays fixed.
5. Rerun from the base forecasts on, into new output folders, and score the newly published months.
6. Run the tests and compare the paper's outputs with the previous release.

## 7. Tests

`python -m pytest tests -v`. Core tests run on synthetic fixtures; stored-output tests read the archive (`./archive` or the folder in `MARITIME_ARCHIVE`) and are skipped with a message when it is absent.

| File | Kind | Checks |
|---|---|---|
| `test_synthetic.py` | core | the builder on the synthetic system (identities hold on the flows, rank and redundant rows, independence of the order of the totals); the projection is coherent and idempotent (diagonal and dense W); the exact QP satisfies the KKT conditions; `nonneg_gi`, the null-space form, the OSQP fallback and the worker pool with forced fallback stages give the same solution |
| `test_aci.py` | core + stored | the ACI update rule by hand; a replay of the recursion from the stored intervals reproduces the stored α at every origin and the stored state after 2025-06, for both tracks |
| `test_structural_zeros.py` | core + stored | the structural-zero rule on synthetic cases (a zero only when every containing total is published and adds up within 1 t or 0.01%; never for a published value; two missing parts of one total); on the stored data it reproduces `results/structural_zeros.csv` and the flags of the system panel |
| `test_constraints.py` | stored | C holds on all 90 historical months within the rounding tolerance (1/28 t per day or 0.01%); the generic builder reproduces the paper's C and full-rank C exactly, and up to row permutation when the totals are reordered |
| `test_reconciliation.py` | stored | at origin 2024-06 (AutoETS): projections satisfy every constraint row; the exact QP at h 1 and 12 satisfies the KKT conditions and equals the stored forecast; the solver stages agree |
| `test_exact_samples.py` | stored | exact samples of a calibration origin and of origin 2026-06 are coherent and non-negative; replaying ACI over all 42 origins reproduces the stored states; the final intervals use the same samples as the zeroed version (only the ACI columns differ); the intervals at origin 2026-06 use the state of the S1 intervals and T5a reads them |
