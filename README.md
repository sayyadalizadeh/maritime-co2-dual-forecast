# maritime-co2-dual-forecast

Code, stored outputs and documentation for the paper:

> Sayyad Alizadeh, Devran Yazir and Hürol Hocek. *An open framework for coherent forecasting of national maritime CO2 emissions under residence and territory accounting*. Submitted to Environmental Modelling & Software.

**Status.** The paper has been submitted to Environmental Modelling & Software. This is release v1.0 of the code, with the stored outputs attached to the GitHub release. A Zenodo archive with a DOI will follow on acceptance.

Authors:
- Sayyad Alizadeh: Department of Software Development, Faculty of Science, Karadeniz Technical University, Trabzon, Türkiye (contact: alizadeh@ktu.edu.tr)
- Devran Yazir: Surmene Faculty of Maritime Sciences, Karadeniz Technical University, Trabzon, Türkiye
- Hürol Hocek: Surmene Faculty of Maritime Sciences, Karadeniz Technical University, Trabzon, Türkiye

The OECD publishes monthly CO2 emissions of shipping for each economy on two accounting bases:
- **residence:** emissions are counted for the economy where the ship operator is resident (environmental-economic accounts);
- **territory:** they are counted for the economy where the voyage departs (greenhouse-gas inventories).

Each voyage enters both bases. Seven published components link them: three totals are sums of components, every measure adds up across economies to the World value, and two World balance conditions tie the bases together. The repository:
1. forecasts the monthly emissions of 110 economies, a Residual node and the World, 1 to 12 months ahead, with statistical models, a global LightGBM and zero-shot foundation models (Chronos-Bolt, TimesFM 2.5, TimesFM 3.0);
2. reconciles the forecasts so that every identity of both bases holds exactly, with non-negative values (an exact quadratic program);
3. builds probabilistic forecasts from reconciled samples and calibrates the intervals (adaptive conformal inference and pooled split conformal);
4. tests the forecasts on data the OECD published after the models were released;
5. measures up to which horizon the forecasts beat a persistence reference (skill and H*).

The constraint builder (`src/dual_allocation.py`) works for any inventory with two allocation principles; `examples/toy_dual_allocation.py` shows it on a synthetic three-economy system.

## Contents

```
src/             one script per step (each has --help); src/archive.py and src/download_data.py for the data
tests/           pytest suite: core tests on synthetic fixtures, and tests on the stored outputs
examples/        synthetic three-economy example of the constraint builder and the exact QP
results/paper/   the paper's tables (T1a-T6b), supplementary data files (S1, S1a, S3), the supplementary tables
                 (SUPPLEMENTARY_TABLES.md, Tables S1-S45), further tables (ADDITIONAL_TABLES.md), the table of
                 figures quoted in the text, tables.md, MANIFEST.csv
figures/         the paper's figures (Fig1, Fig3-Fig7) as vector PDF and 600 dpi TIFF, MANIFEST.csv
run_all.py       runs the steps in order (stages data, forecasts, prob, skill, exact, analyses, outputs)
ARCHITECTURE.md  modules, configuration, the constraint matrix, tests, adapting the pipeline
DATA_RULES.md    the data: source, variables, preparation rules and known issues
DATA_NOTICE.md   terms of the OECD data included in the archive
requirements.txt     project environment (Python 3.12)
requirements-fm.txt  foundation-model environment (Python 3.10, CUDA)
```

The stored outputs (forecasts, reconciled forecasts, evaluation results, samples used by the tests, final forecasts) and the raw data are not in git. They form the archive, which is attached to the GitHub release v1.0 (see Archive).

## Installation

Everything except the foundation-model forecasts runs on a CPU. The foundation-model forecasts are stored in the archive, so the GPU environment is needed only to recompute them.

**1. Project environment** (all steps except the foundation models; Python 3.12.10, versions pinned in `requirements.txt`):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

On Linux or macOS: `python3.12 -m venv .venv` and `.venv/bin/python -m pip install -r requirements.txt`. The scripts were run on Windows 11; they use only portable paths.

**2. Foundation-model environment** (optional; Chronos-Bolt, TimesFM 3.0, TimesFM 2.5). Python 3.10.20 with torch 2.11.0 for CUDA 12.8; the full package list is in `requirements-fm.txt`:

```powershell
conda create -n maritime-fm python=3.10
conda activate maritime-fm
pip install torch==2.11.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-fm.txt
```

`run_all.py` finds this environment through `--gpu-python` or the environment variable `FM_PYTHON`. The model checkpoints are downloaded from Hugging Face on first use: `amazon/chronos-bolt-base` (revision `5d9f166d`), `google/timesfm-3.0-pytorch` (revision `43046b85`; TimesFM Non-Commercial Licence v1.0, weights not redistributed) and `google/timesfm-2.5-200m-pytorch` (revision `1d952420`; Apache-2.0). Inference settings: supplementary Table S28.

## Archive

The archive (about 2.4 GB, 508 files) holds the stored outputs and the input data. It is attached to the GitHub release v1.0 as these assets:

| Asset | Content |
|---|---|
| `maritime-co2-dual-forecast-v1.0-archive-part1.zip`, `...-part2.zip` | the archive files, in two independent zip files of less than 2 GB each |
| `MANIFEST.csv` | every archive file with its size and SHA-256 |
| `SHA256SUMS.txt` | the SHA-256 of the assets above |

Download, check and unpack it into `archive/` at the repository root:

```powershell
.\.venv\Scripts\python.exe src\archive.py download      # assets into release-assets/, SHA-256 checked, unpacked into archive/
```

If the assets were downloaded by hand (from the release page) into `release-assets/`, unpack them with:

```powershell
.\.venv\Scripts\python.exe src\archive.py unpack        # checks SHA256SUMS.txt, extracts, verifies every file
```

Both commands verify every archive file against `MANIFEST.csv`; `python src/archive.py verify` repeats the check at any time. To keep the archive elsewhere, pass `--archive <folder>` or set the environment variable `MARITIME_ARCHIVE` to that folder; the tests use the same variable. `archive/` and `release-assets/` are ignored by git.

`results/paper/SUPPLEMENTARY_TABLES.md` differs from its copy in the v1.0 archive only in the author line, which was updated after the release: `src/archive.py restore` keeps the committed file, and `--overwrite` must not be used for this file, because it would bring back the old author line.

## Data

The input is the complete CSV of the OECD dataset *Maritime transport CO2 emissions (experimental)*:

| Item | Value |
|---|---|
| Dataflow | `OECD.SDD.NAD.SEEA:DSD_MARITIME_TRANSPORT@DF_MARITIME_TRANSPORT`, version 2.0 |
| Last updated by the OECD | 14 October 2026 |
| Downloaded | 31 October 2026, complete (unfiltered) CSV, 427,861 observations |
| File | `data/raw/oecd_maritime_co2_raw_2026-10-31.csv` (unchanged apart from the file name) |
| Size and SHA-256 | 200,921,743 bytes; `88fb19b9645e4452d26e88cf8891467a3732d380a54f1b445f3735849fdf6002` |

**Data notice.** The raw file is OECD data. It is redistributed here, with attribution, under section 3 ("Data", Permitted Use) of the OECD Terms and Conditions (https://www.oecd.org/en/about/terms-conditions.html). Source:

> OECD (2026), *Maritime transport CO2 emissions (experimental)*, https://data-explorer.oecd.org/vis?df[ds]=dsDisseminateFinalDMZ&df[id]=DSD_MARITIME_TRANSPORT%40DF_MARITIME_TRANSPORT&df[ag]=OECD.SDD.NAD.SEEA (accessed on 31 October 2026).

Anyone who shares or licenses work created with these data must keep this acknowledgment requirement. The MIT licence of this repository covers the code only, not the OECD data. The same notice is in `DATA_NOTICE.md`.

To get the file:
- **from the archive** (exact file): `python src/archive.py restore` copies `archive/data/raw/` into `data/raw/`; check it with `python src/download_data.py --verify`;
- **from the OECD API**: `python src/download_data.py --download`. The OECD revises this experimental dataset and publishes new releases that replace this one: a later download will not match the checksum above, and results computed from it will differ from the paper.

The vintage comparison (Tables S4-S6 and S39) also uses the October 2025 release of the same dataset (`data/external/oecd_maritime_v1_2025-10-06.xlsx`, in the archive). `DATA_RULES.md` documents the variables, the preparation rules and the known issues of the data.

## Quick start (fresh clone, no archive)

```powershell
.\.venv\Scripts\python.exe examples\toy_dual_allocation.py
.\.venv\Scripts\python.exe -m pytest tests -q
```

The example builds the constraint matrix of a synthetic three-economy system, reconciles noisy forecasts with the projection and the exact non-negative QP, and prints the constraint and KKT residuals. Without the archive, the core tests (constraint builder, projection, exact QP and its solver stages, ACI update, structural-zero rule) run on synthetic fixtures; the tests that need stored outputs are skipped with a message.

## Reproduction

### From the archive (no refitting, no GPU; about 1 minute after the download)

Get the archive first (see Archive), then:

```powershell
.\.venv\Scripts\python.exe src\archive.py verify                       # sizes and SHA-256 of every archived file
.\.venv\Scripts\python.exe -m pytest tests -q                          # all tests, including the stored-output tests
.\.venv\Scripts\python.exe src\archive.py restore                      # copy the stored outputs into results/ and data/
.\.venv\Scripts\python.exe run_all.py --stages outputs --overwrite     # regenerate every table and figure
```

The last command regenerates `results/paper/` and `figures/` from the stored evaluation results; the regenerated files equal the committed ones. Any single analysis step can also be rerun from the restored outputs, for example `run_all.py --steps skill.09 --overwrite` (see `run_all.py --list`).

### Full run

```powershell
.\.venv\Scripts\python.exe run_all.py --list                                   # steps, environment, minutes
.\.venv\Scripts\python.exe run_all.py --overwrite --gpu-python <python of the FM environment>
.\.venv\Scripts\python.exe run_all.py --overwrite --skip-gpu                   # foundation-model forecasts from the archive
```

Stages, in order: `data` (clean panels, structural zeros, constraint matrix), `forecasts` (base forecasts, in-sample and one-step errors, reconciliation, point evaluation), `prob` (samples, the tests at origins 2025-06 and 2026-06, calibration origins, final samples), `skill` (long-horizon forecasts, skill and H*, the 2022 episode), `exact` (exact non-negative QP of every sample, ACI on the exact samples, final intervals), `analyses` (supplementary analyses, benchmarks, inference, ablation, diagnostics), `outputs`. Every script refuses to overwrite an existing output unless it is given `--overwrite`, and logs its settings and run time to `results/run_log.csv`.

**Run times** on the original machine (Windows 11, 20-core laptop CPU, NVIDIA RTX 4050 Laptop GPU with 6 GB), from the run log, one step after the other:

| Steps | Hours |
|---|---:|
| all steps of the paper | 26.1 |
| without the foundation-model steps | 25.6 |
| without the exact non-negative QP of the samples | 15.2 |
| without both | 14.6 |
| outputs only, from the archive | 0.02 |

The exact QP of the samples (about 525,000 samples, worker processes with a time limit and fallback stages) takes about 10.9 hours; steps in the same parallel group of `run_all.py` run at the same time.

## Map of the paper's outputs

All tables and figures are written by `src/paper_tables.py`, `src/paper_figures.py`, `src/supplementary_tables.py` and `src/quoted_figures.py` (stage `outputs`) from the files listed, except `SUPPLEMENTARY_TABLES.md`, which is kept as submitted with the paper; `results/paper/MANIFEST.csv` and `figures/MANIFEST.csv` give the sources of each file.

### Tables

| Table | File (`results/paper/`) | Computed from | Scripts that compute the inputs |
|---|---|---|---|
| T1a-T1c Data, constraint system, data issues | `T1a_system.csv`, `T1b_coverage.csv`, `T1c_issues.csv` | `results/system_economies_v2.csv`, `C_matrix_*_v2`, `coverage_2025_v2.csv`, `structural_zeros.csv`, `episode_2022/` | `build_panel.py`, `constraints.py`, `episode_2022.py` |
| T2 Base models | `T2_base_models.csv` | `results/eval_fm_v2/` | `evaluate_fm.py` |
| T3a, T3b Reconciliation | `T3a_reconciliation_economies.csv`, `T3b_reconciliation_world.csv` | `results/eval_fm_w/summary.csv` | `reconcile.py`, `evaluate_fm_w.py` |
| T3c Incoherence of the base forecasts | `T3c_incoherence.csv`, `T3c_incoherence_cells.csv` | stored base forecasts | `base_forecasts.py`, `fm_forecasts.py` |
| T4 Probabilistic forecasts | `T4_probabilistic.csv` | `results/eval_prob_exact_nested/` | `prob_samples.py`, `exact_eval.py`, `evaluate_prob.py` |
| T5a, T5b Tests on later-published data | `T5a_realtime.csv`, `T5b_realtime_units.csv` (robustness: `*_eval_state.csv`) | `results/eval_realtime_exact_ext_nested/` (`eval_realtime_exact_nested/`) | `exact_samples.py`, `evaluate_realtime.py` |
| T5c Model release dates | `T5c_release_dates.csv` | Hugging Face API dates | - |
| T6, T6b Skill and H* | `T6_skill.csv`, `T6b_hstar_variants.csv` | `results/skill/` | `skill.py` |
| Supplementary data S1 (FM forecasts 2026-07 to 2027-06), S1a, S3 | `S1_forecasts_FM_2026-07_2027-06.csv`, `S1a_aci_floor_flags.csv`, `S3_data_dictionary.csv` | `results/final_exact/`, `results/conformal_s1/` | `final_samples.py`, `exact_samples.py`, `final_bands.py`, `conformal_s1.py` |
| Figures quoted in the text | `quoted_figures.csv`, `intl_share_by_month_with_2026.csv` | data and stored results named in each row | `quoted_figures.py` |

### Figures

| Figure | Files (`figures/`) | Content | Computed from |
|---|---|---|---|
| Fig. 1 | `Fig1.pdf`, `Fig1.tif` | World totals on the two bases and the international share | `data/clean/panel_system_wide_v2.parquet`, `results/episode_2022/` |
| Fig. 2 | - | framework diagram (not drawn by code) | - |
| Fig. 3 | `Fig3.pdf`, `Fig3.tif` | incoherence of the base forecasts by horizon | `results/paper/T3c_incoherence_cells.csv` |
| Fig. 4 | `Fig4.pdf`, `Fig4.tif` | MASE and scaled CRPS by horizon (with the STAT diagnostic without origin 2024-09) | `results/eval_fm_w/`, `results/eval_prob_exact_nested/`, `results/prob_main_exact/` |
| Fig. 5 | `Fig5.pdf`, `Fig5.tif` | coverage of the uncalibrated and ACI intervals by horizon | `results/eval_prob_exact_nested/` |
| Fig. 6 | `Fig6.pdf`, `Fig6.tif` | test from origin 2026-06: World and five economies | `results/eval_realtime_exact_ext_nested/`, `results/prob_realtime_exact/` |
| Fig. 7 | `Fig7.pdf`, `Fig7.tif` | skill against persistence and H* | `results/paper/T6_skill.csv` |

`src/check_figures.py` checks that no text is smaller than 7 pt and that the axis limit of Fig. 4 (b) hides no data (`results/figures_check.csv`).

### Supplementary tables

`results/paper/SUPPLEMENTARY_TABLES.md` holds Tables S1-S45, numbered in the order the paper cites them, with their captions and notes. Each table names its source file(s); paths are relative to the repository root, and files outside `results/paper/` come from the archive (`python src/archive.py restore`, see Reproduction). Where a table combines several files, all are listed with the part each one supplies. Tables S1 and S9 are compiled by the authors and have no source file.

| Table | Content | Computed by | Source file(s), under `results/` unless stated |
|---|---|---|---|
| S1 | Accounting concepts used in the paper and their counterparts | - | - |
| S2 | World international share of the territory total (%), by year and month | quoted_figures.py | `paper/intl_share_by_month_with_2026.csv` |
| S3 | Switzerland: months per component published / set to zero by the structural-zero rule / still missing | supplementary_analyses.py | `supplementary/switzerland_summary.csv` |
| S4 | October 2025 against October 2026 vintage: coverage of the comparison | vintage.py | `vintage/coverage.csv` |
| S5 | Revisions from the October 2025 to the October 2026 vintage (%) | vintage.py | `vintage/revision_stats.csv` |
| S6 | The ten most revised economy-months | vintage.py | `vintage/most_revised_economy_months.csv` |
| S7 | Inference settings of the foundation models (zero-shot) | supplementary_analyses.py | `supplementary/fm_settings.csv` |
| S8 | Number of origins behind each Table 4 column, by horizon (both tracks) | supplementary_analyses.py | `supplementary/T4_origins.csv` |
| S9 | Design choices fixed in advance and choices made after seeing results | - | - |
| S10 | Automated tests of the software | tests/ (pytest) | `tests/` |
| S11 | Model Confidence Set p-values of the nine base models (system economies and all 1,170 series) | supplementary_analyses.py | `supplementary/mcs_pvalues.csv` |
| S12 | Post-release evaluation: target months after each model's release available in the stored forecasts | fm_postrelease.py | `fm_postrelease/coverage.csv` |
| S13 | Post-release evaluation: MASE, median / weighted over system economies, on the same post-release target months | fm_postrelease.py | `fm_postrelease/postrelease_scores.csv` |
| S14 | Ranks of the 23 base/method combinations by weighted and median MASE in each window (1 = best) | supplementary_analyses.py | `supplementary/method_ranks.csv` |
| S15 | Spearman correlation of the method ranks between windows | supplementary_analyses.py | `supplementary/rank_correlation.csv` |
| S16 | Diebold-Mariano tests per system economy series, absolute error loss, main window: counts of significant differences... | supplementary_analyses.py | `supplementary/dm_counts.csv` |
| S17 | MASE of the ten World series, base and reconciled forecasts, main window, all horizons | paper_tables.py, benchmarks.py | `paper/T3b_reconciliation_world.csv`; `checks/world_mase_coincidence.csv` |
| S18 | Accuracy against the published residence totals, main window | benchmarks.py | `checks/published_totals_mase.csv` |
| S19 | Similarity of the one-step error correlations (W) with the rolling-origin h-step error correlations | w_diagnostics.py | `w_diagnostics/correlation_similarity.csv` |
| S20 | Friedman test and Nemenyi critical difference over base / WLS / MinT, per base, main window | supplementary_analyses.py | `supplementary/friedman_nemenyi.csv` |
| S21 | Model Confidence Set p-values, base x reconciliation method, system economies, main window | supplementary_analyses.py | `supplementary/mcs_pvalues.csv` |
| S22 | Constraint ablation: MASE of MinT exact QP under three constraint sets, system economies, main window | constraint_ablation.py | `ablation/mase_summary.csv` |
| S23 | Constraint ablation: MASE of the World series by constraint set | constraint_ablation.py | `ablation/mase_world_series.csv` |
| S24 | Constraint ablation: violations of the within-unit identities left by each set (%) | constraint_ablation.py | `ablation/violations_within_unit.csv` |
| S25 | Constraint ablation: World residence-territory gap and B(W) - C(W), % of the World total | constraint_ablation.py, quoted_figures.py | `ablation/violations_world_gap_by_horizon.csv`; `paper/quoted_figures.csv` |
| S26 | Benchmarks for reconciliation: MASE of bottom-up (BU) and combined (COMB) forecasts against base, WLS and MinT, main... | benchmarks.py, supplementary_analyses.py | `benchmarks/mase_summary.csv`; `benchmarks/mase_by_horizon.csv`; `supplementary/OLS_tracks.csv` |
| S27 | Coherence of the benchmarks: violations of the identities in % of the forecast's own World residence total, mean over... | benchmarks.py | `benchmarks/violations_by_horizon.csv` |
| S28 | Origin-block bootstrap intervals for key differences, main window, all horizons pooled | benchmarks.py, exact_eval.py | `inference/block_bootstrap.csv`; `exact_eval/fm_minus_stat_crps_bootstrap.csv` |
| S29 | Diebold-Mariano tests per system economy series, scaled CRPS loss, main window: counts of significant differences (5%) | supplementary_analyses.py | `supplementary/dm_counts.csv` |
| S30 | Pooled coverage with 95% moving-block bootstrap intervals over origins (block 6), by measure group | supplementary_analyses.py | `supplementary/coverage_block_bootstrap.csv` |
| S31 | ACI coverage by measure group and size tercile (terciles of the 1,100 series by 2022 weight) | supplementary_analyses.py | `supplementary/coverage_wilson.csv` |
| S32 | Coverage of the residence totals of the 110 system economies: pooled split conformal against ACI and uncalibrated int... | conformal_residence.py | `conformal_residence/coverage_residence_totals.csv` |
| S33 | Coverage of the residence totals by horizon | conformal_residence.py | `conformal_residence/coverage_by_horizon.csv` |
| S34 | Table 4 quantities, samples with negatives set to zero against exactly reconciled samples, all horizons | exact_eval.py | `exact_eval/T4_side_by_side.csv` |
| S35 | FM minus STAT weighted scaled CRPS, origin-block bootstrap, zeroed and exact samples | exact_eval.py | `exact_eval/fm_minus_stat_crps_bootstrap.csv` |
| S36 | Weighted scaled CRPS with and without origin 2024-09 (diagnostic only, no origin is excluded) | exact_eval.py | `exact_eval/crps_without_2024-09.csv`; `exact_eval/crps_change_by_origin.csv` |
| S37 | Coverage events with the outcome on an interval bound within 1e-9 t/day | exact_eval.py | `prob_main_exact/track=FM/aci_tie_events.csv`; `prob_main_exact/track=STAT/aci_tie_events.csv` |
| S38 | Real-time test from origin 2026-06 (targets July to September 2026): World and the five largest economies | paper_tables.py | `paper/T5b_realtime_units.csv`; `paper/T5a_realtime_eval_state.csv`; `paper/T5b_realtime_units_eval_state.csv` |
| S39 | Paired comparison of FM and STAT at the real-time origin 2026-06 (published residence totals, 110 economies, 2026-07... | vintage.py | `vintage/paired_2026-06.csv` |
| S40 | Pooled split-conformal intervals in the final forecasts: median ratio of their width to that of the ACI intervals | conformal_s1.py | `conformal_s1/width_ratio.csv` |
| S41 | H* for every variant of the skill test | skill.py | `paper/T6b_hstar_variants.csv` |
| S42 | Skill and forecast horizon against persistence and against persistence with a linear drift | skill_drift.py | `skill_drift/hstar_table.csv`; `skill_drift/skill.csv` |
| S43 | Final forecasts at origin 2026-06: share of series × horizon cells whose calibration level reached its floor (%) | paper_tables.py | `paper/S1a_aci_floor_flags.csv` |
| S44 | W by origin: shrinkage intensity, condition numbers on a unit diagonal, floored and dead series | w_diagnostics.py | `w_diagnostics/w_condition_by_origin.csv`; `w_diagnostics/zero_variance_series.csv` |
| S45 | Dead series that later become non-zero | w_diagnostics.py | `w_diagnostics/dead_series_revival_summary.csv`; `w_diagnostics/dead_series_revival.csv` |

### Additional tables

`results/paper/ADDITIONAL_TABLES.md` (written by `src/supplementary_tables.py`) holds the other tables computed for the paper, without numbers; each names its source file. They are:

- MASE of the base models and the two tracks by evaluation window, system economies, horizons 1-12 pooled
- MASE of the two tracks by horizon and window: system economies (weighted) and two World series
- Mean absolute error in kt CO2 per day: weighted over system economy series, and the World residence total
- OLS reconciliation as a benchmark: MASE over system economies, main window, all horizons
- Base models against ETS, Diebold-Mariano on absolute error, system economy series: better / worse than ETS after BH
- Probabilistic scores of the two tracks by window, system economies, horizons 1-12 pooled
- Base models' own probabilistic scores (no calibration), system economies, all horizons
- Calibration variants: coverage and scaled Winkler score of every interval type, system economies, main window
- Energy and variogram scores, base and reconciled samples, median over origin x horizon cells
- ACI state at each calibration origin: share of system economy cells with alpha at its floor 0.001 (%), gamma 0.05
- Ratio of the ACI interval width to the uncalibrated width, per cell, calibration origins
- Outside-system series (49 series of 25 economies, not reconciled), main window, all horizons
- STAT track: weighted scaled CRPS over system economies by horizon (Fig. 4b)
- STAT, reconciled samples, h = 10-12: the origins with the largest weighted cell sCRPS, and the horizon's weighted sCRPS without that origin
- Scaled CRPS of HKG RES_INT_TO at origin 2024-09, AutoETS base and reconciled samples
- Post-release evaluation: scaled CRPS, median / weighted over system economies, on the same post-release target months
- Evaluation samples with negatives set to zero (superseded by the exact samples): violations of C y = 0, % of the sample's World residence total, all 372 cells per track
- Evaluation samples with negatives set to zero: violation by horizon (median %, share of samples above 0.1%)
- The same 500 samples, set to zero against reconciled with the exact non-negative QP, 16 cells
- FM minus STAT under the two sample treatments (negative = FM better for scores)
- Exact non-negative QP of the samples: number of QPs and of samples that needed a fallback stage
- Tests on later-published data (published residence totals, 110 economies): zeroed against exact samples
- Final intervals at origin 2026-06 (data files S1, S1a): ACI recursion on zeroed against exact samples

## Licence

The MIT licence (`LICENSE`) covers the code only. The OECD data in the archive are redistributed under the OECD Terms and Conditions, with the acknowledgment requirement described in Data and in `DATA_NOTICE.md`. The model weights are not redistributed; they follow their own licences (see Installation).

## Citation and contact

Please cite the paper and this software (release v1.0); `CITATION.cff` has the metadata, and GitHub shows it under "Cite this repository". Contact: Sayyad Alizadeh, alizadeh@ktu.edu.tr.
