# Supplementary material

**An open framework for coherent forecasting of national maritime CO2 emissions under residence and territory accounting**

Sayyad Alizadeh, Devran Yazir, Hürol Hocek, Ünal Özdemir

This supplement contains Tables S1 to S45, numbered in the order in which the paper cites them. Each table names the file of the public repository that holds it in full.

**Abbreviations and conventions.** FM: the main track, TimesFM 2.5 with MinT reconciliation and W from its own one-step errors. STAT: the statistical track, ETS with MinT. Main window: target months up to December 2025, without September 2025. Weighted: weights proportional to each series' mean emissions per day in 2022, within one level. Scaled scores (MASE, sCRPS, Winkler) divide by the in-sample seasonal naive error at the origin. BH: Benjamini–Hochberg control of the false discovery rate. MCS: model confidence set. ACI: adaptive conformal inference. H*: the first horizon at which forecast skill is no longer supported. Economies are identified by ISO 3166 alpha-3 codes. Components: A, domestic voyages within the economy by resident operators; B, domestic voyages within other economies by resident operators; C, domestic voyages within the economy by non-resident operators; D, international voyages departing the economy by resident operators; E, international voyages arriving in the economy by resident operators; F, international voyages between other economies by resident operators; G, international voyages departing the economy by non-resident operators.

## Contents

- Table S1. Accounting concepts used in the paper and their counterparts
- Table S2. World international share of the territory total (%), by year and month
- Table S3. Switzerland: months per component published / set to zero by the structural-zero rule / still missing
- Table S4. October 2025 against October 2026 vintage: coverage of the comparison
- Table S5. Revisions from the October 2025 to the October 2026 vintage (%)
- Table S6. The ten most revised economy-months
- Table S7. Inference settings of the foundation models (zero-shot)
- Table S8. Number of origins behind each Table 4 column, by horizon (both tracks)
- Table S9. Design choices fixed in advance and choices made after seeing results
- Table S10. Automated tests of the software
- Table S11. Model Confidence Set p-values of the nine base models (system economies and all 1,170 series)
- Table S12. Post-release evaluation: target months after each model's release available in the stored forecasts
- Table S13. Post-release evaluation: MASE, median / weighted over system economies, on the same post-release target months
- Table S14. Ranks of the 23 base/method combinations by weighted and median MASE in each window (1 = best)
- Table S15. Spearman correlation of the method ranks between windows
- Table S16. Diebold-Mariano tests per system economy series, absolute error loss, main window: counts of significant differences (5%)
- Table S17. MASE of the ten World series, base and reconciled forecasts, main window, all horizons
- Table S18. Accuracy against the published residence totals, main window
- Table S19. Similarity of the one-step error correlations (W) with the rolling-origin h-step error correlations
- Table S20. Friedman test and Nemenyi critical difference over base / WLS / MinT, per base, main window
- Table S21. Model Confidence Set p-values, base x reconciliation method, system economies, main window
- Table S22. Constraint ablation: MASE of MinT exact QP under three constraint sets, system economies, main window
- Table S23. Constraint ablation: MASE of the World series by constraint set
- Table S24. Constraint ablation: violations of the within-unit identities left by each set (%)
- Table S25. Constraint ablation: World residence-territory gap and B(W) - C(W), % of the World total
- Table S26. Benchmarks for reconciliation: MASE of bottom-up (BU) and combined (COMB) forecasts against base, WLS and MinT, main window, all horizons
- Table S27. Coherence of the benchmarks: violations of the identities in % of the forecast's own World residence total, mean over origins
- Table S28. Origin-block bootstrap intervals for key differences, main window, all horizons pooled
- Table S29. Diebold-Mariano tests per system economy series, scaled CRPS loss, main window: counts of significant differences (5%)
- Table S30. Pooled coverage with 95% moving-block bootstrap intervals over origins (block 6), by measure group
- Table S31. ACI coverage by measure group and size tercile (terciles of the 1,100 series by 2022 weight)
- Table S32. Coverage of the residence totals of the 110 system economies: pooled split conformal against ACI and uncalibrated intervals
- Table S33. Coverage of the residence totals by horizon
- Table S34. Table 4 quantities, samples with negatives set to zero against exactly reconciled samples, all horizons
- Table S35. FM minus STAT weighted scaled CRPS, origin-block bootstrap, zeroed and exact samples
- Table S36. Weighted scaled CRPS with and without origin 2024-09 (diagnostic only, no origin is excluded)
- Table S37. Coverage events with the outcome on an interval bound within 1e-9 t/day
- Table S38. Real-time test from origin 2026-06 (targets July to September 2026): World and the five largest economies
- Table S39. Paired comparison of FM and STAT at the real-time origin 2026-06 (published residence totals, 110 economies, 2026-07 to 2026-09)
- Table S40. Pooled split-conformal intervals in the final forecasts: median ratio of their width to that of the ACI intervals
- Table S41. H* for every variant of the skill test
- Table S42. Skill and forecast horizon against persistence and against persistence with a linear drift
- Table S43. Final forecasts at origin 2026-06: share of series × horizon cells whose calibration level reached its floor (%)
- Table S44. W by origin: shrinkage intensity, condition numbers on a unit diagonal, floored and dead series
- Table S45. Dead series that later become non-zero

## Tables

### Table S1. Accounting concepts used in the paper and their counterparts

Source: OECD (2026) dataset description; Clarke et al. (2023); IMO (2020, 2023).

| concept | definition in this paper | framework | how it is used |
|---|---|---|---|
| Residence total | A + B + D + E + F: emissions of ships operated by residents of the economy | SEEA air emission accounts, maritime transport industry (ISIC H50) | Sum of components inside the constraint system; the published total is used for reporting and real-time scoring |
| Domestic navigation, territory basis | A + C: domestic voyages within the economy | UNFCCC inventories, domestic navigation (IPCC category 1.A.3.d.ii) | Forecast and reconciled as published |
| International navigation, territory basis | D + G: international voyages departing the economy | Labelled by the OECD as the UNFCCC memo item (international navigation, IPCC category 1.A.3.d.i); national inventories may compile the memo item differently, for example from fuel sales | A departure-based estimate of the memo item; forecast and reconciled as part of the territory total |
| Shipping emissions, IMO | Total and voyage-based international emissions of the Fourth IMO GHG Study | IMO GHG studies | Used only to compare the scope of the OECD estimates (Section 2.1) |
| IMO indicative checkpoints | Reduction of total annual GHG emissions from international shipping relative to 2008 | 2023 IMO Strategy | Context for monitoring (Section 1); not used in the analysis |

### Table S2. World international share of the territory total (%), by year and month

Source file: `results/paper/intl_share_by_month_with_2026.csv`

Last column: January minus the mean of the preceding December and the following February (2019: February only). 2026: January to June, the months published at download.

| year | m01 | m02 | m03 | m04 | m05 | m06 | m07 | m08 | m09 | m10 | m11 | m12 | annual mean | jan minus mean of dec feb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2019 | 69.3 | 75.1 | 74.9 | 75.0 | 74.7 | 74.1 | 73.9 | 74.0 | 74.1 | 74.3 | 74.5 | 74.5 | 74.0 | -5.8 |
| 2020 | 69.8 | 73.8 | 74.1 | 74.1 | 73.9 | 73.1 | 73.5 | 73.6 | 73.8 | 73.9 | 74.3 | 74.5 | 73.5 | -4.4 |
| 2021 | 70.1 | 74.9 | 74.5 | 75.0 | 74.8 | 74.6 | 74.4 | 74.2 | 74.5 | 74.4 | 72.1 | 65.0 | 73.2 | -4.6 |
| 2022 | 65.5 | 66.6 | 66.0 | 65.7 | 65.4 | 64.4 | 64.4 | 64.6 | 65.1 | 67.6 | 71.5 | 71.7 | 66.5 | -0.3 |
| 2023 | 68.8 | 72.8 | 73.1 | 73.3 | 73.0 | 71.9 | 71.8 | 71.9 | 70.0 | 69.0 | 62.9 | 70.5 | 70.8 | -3.4 |
| 2024 | 68.5 | 73.8 | 73.9 | 73.8 | 73.4 | 72.8 | 72.5 | 70.8 | 69.9 | 73.0 | 73.2 | 72.0 | 72.3 | -3.6 |
| 2025 | 70.4 | 74.7 | 74.5 | 74.6 | 74.2 | 73.3 | 73.5 | 73.5 | 73.8 | 73.8 | 74.1 | 73.5 | 73.7 | -3.0 |
| 2026 | 72.2 | 71.0 | 67.2 | 69.2 | 67.3 | 62.9 | – | – | – | – | – | – | 68.3 (January–June) | -0.1 |

### Table S3. Switzerland: months per component published / set to zero by the structural-zero rule / still missing

Source file: `results/supplementary/switzerland_summary.csv`

A, B, C = domestic components; D, E, F, G = international. Month-by-month detail in `results/supplementary/switzerland_components.csv`.

| year | A | B | C | D | E | F | G |
|---|---|---|---|---|---|---|---|
| 2019 | 12 / 0 / 0 | 12 / 0 / 0 | 0 / 12 / 0 | 0 / 4 / 8 | 3 / 9 / 0 | 12 / 0 / 0 | 4 / 0 / 8 |
| 2020 | 12 / 0 / 0 | 12 / 0 / 0 | 0 / 12 / 0 | 0 / 0 / 12 | 1 / 11 / 0 | 12 / 0 / 0 | 0 / 0 / 12 |
| 2021 | 12 / 0 / 0 | 12 / 0 / 0 | 0 / 12 / 0 | 0 / 0 / 12 | 0 / 12 / 0 | 12 / 0 / 0 | 0 / 0 / 12 |
| 2022 | 12 / 0 / 0 | 12 / 0 / 0 | 0 / 12 / 0 | 0 / 0 / 12 | 0 / 12 / 0 | 12 / 0 / 0 | 0 / 0 / 12 |
| 2023 | 12 / 0 / 0 | 12 / 0 / 0 | 0 / 12 / 0 | 0 / 0 / 12 | 1 / 11 / 0 | 12 / 0 / 0 | 0 / 0 / 12 |
| 2024 | 12 / 0 / 0 | 12 / 0 / 0 | 0 / 12 / 0 | 0 / 0 / 12 | 0 / 12 / 0 | 12 / 0 / 0 | 0 / 0 / 12 |
| 2025 | 0 / 0 / 12 | 0 / 0 / 12 | 0 / 0 / 12 | 0 / 0 / 12 | 0 / 0 / 12 | 0 / 0 / 12 | 0 / 0 / 12 |
| 2026 | 6 / 0 / 0 | 6 / 0 / 0 | 0 / 6 / 0 | 0 / 0 / 6 | 0 / 6 / 0 | 6 / 0 / 0 | 0 / 0 / 6 |

### Table S4. October 2025 against October 2026 vintage: coverage of the comparison

Source file: `results/vintage/coverage.csv`

| areas in 2025 file | months in 2025 file | first month | last month | areas matched by name | areas not matched | series months compared | series months without 2026 value | economies compared |
|---|---|---|---|---|---|---|---|---|
| 134 | 42 | 2022-01 | 2025-06 | 134 | 0 | 5,493 | 3 | 132 |

### Table S5. Revisions from the October 2025 to the October 2026 vintage (%)

Source file: `results/vintage/revision_stats.csv`

No value is identical to all printed digits in the two releases; for most economy-months in 2022 and 2023 the differences are below 0.001%.

Absolute revision |2026 / 2025 - 1|; the mean is signed.

| group | year | n | median abs pct | p95 abs pct | max abs pct | mean pct | share exactly equal | share abs gt 1pct |
|---|---|---|---|---|---|---|---|---|
| World | all | 42 | 0.323 | 0.555 | 0.660 | 0.351 | 0 | 0.000 |
| World | 2022 | 12 | 0.306 | 0.316 | 0.320 | 0.302 | 0 | 0.000 |
| World | 2023 | 12 | 0.386 | 0.432 | 0.438 | 0.382 | 0 | 0.000 |
| World | 2024 | 12 | 0.319 | 0.348 | 0.355 | 0.275 | 0 | 0.000 |
| World | 2025 | 6 | 0.533 | 0.651 | 0.660 | 0.540 | 0 | 0.000 |
| economies | all | 5,409 | 0.003 | 3.347 | 92.555 | 0.044 | 0 | 0.108 |
| economies | 2022 | 1,512 | 0.000 | 0.787 | 7.074 | -0.207 | 0 | 0.047 |
| economies | 2023 | 1,548 | 0.000 | 0.569 | 6.211 | -0.145 | 0 | 0.039 |
| economies | 2024 | 1,572 | 0.010 | 1.046 | 16.385 | -0.191 | 0 | 0.054 |
| economies | 2025 | 777 | 0.866 | 14.356 | 92.555 | 1.382 | 0 | 0.476 |
| other aggregates | all | 42 | 0.005 | 0.534 | 0.812 | -0.057 | 0 | 0.000 |
| other aggregates | 2022 | 12 | 0.002 | 0.008 | 0.009 | 0.002 | 0 | 0.000 |
| other aggregates | 2023 | 12 | 0.004 | 0.007 | 0.007 | 0.004 | 0 | 0.000 |
| other aggregates | 2024 | 12 | 0.015 | 0.040 | 0.044 | -0.016 | 0 | 0.000 |
| other aggregates | 2025 | 6 | 0.421 | 0.772 | 0.812 | -0.379 | 0 | 0.000 |

### Table S6. The ten most revised economy-months

Source file: `results/vintage/most_revised_economy_months.csv`

30 rows in the source file.

| area name | area code | period | co2 t 2025 | co2 t 2026 | revision pct |
|---|---|---|---|---|---|
| Aruba | ABW | 2025-03 | 1,012 | 1,949 | 92.6 |
| Aruba | ABW | 2025-04 | 903 | 1,659 | 83.8 |
| Samoa | WSM | 2025-05 | 9,399 | 13,946 | 48.4 |
| Belize | BLZ | 2025-05 | 5,508 | 8,056 | 46.3 |
| Belize | BLZ | 2025-06 | 6,318 | 9,146 | 44.8 |
| Suriname | SUR | 2025-06 | 5,658 | 7,916 | 39.9 |
| Samoa | WSM | 2025-06 | 9,517 | 12,679 | 33.2 |
| Tanzania | TZA | 2025-06 | 9,347 | 11,931 | 27.6 |
| Tanzania | TZA | 2025-05 | 9,030 | 11,440 | 26.7 |
| Gibraltar | GIB | 2025-04 | 8,183 | 10,218 | 24.9 |

### Table S7. Inference settings of the foundation models (zero-shot)

Source file: `results/supplementary/fm_settings.csv`

Environment (dl-gpu): chronos-forecasting==2.3.2, huggingface_hub==2.4.0, timesfm==3.0.2, torch==2.11.0+cu128, transformers==5.20.2.

| setting | Chronos-Bolt (base) | TimesFM 3.0 | TimesFM 2.5 (FM track) |
|---|---|---|---|
| checkpoint | amazon/chronos-bolt-base | google/timesfm-3.0-pytorch | google/timesfm-2.5-200m-pytorch |
| revision | 5d9f166d69f47aef3401367a7b842e78fe97b121 | 43046b85ec22d584a13f8098c2ed39c889e129c2 | 1d952420fba87f3c6dee4f240de0f1a0fbc790e3 |
| package | chronos-forecasting 2.3.2 (BaseChronosPipeline.predict quantiles) | timesfm 3.0.2 (TimesFM3Forecaster.predict batch) | timesfm 3.0.2 (TimesFM 2p5 200M torch.forecast) |
| max context of model | 2048 (config.json context length) | no fixed maximum in the package; context left-padded to whole input patches of 32 | max context=1024 (ForecastConfig of the model card) |
| context used | full history from 2019-01 to the origin month: 48 (2022-12) to 78 (2025-06) months at the rolling origins, 90 months at the real-time origin 2026-06, 36 to 48 months at the long-horizon origins; for the one-step errors behind W, one-step forecasts for every target month from January 2020 onwards, each from the full history before it | as for Chronos-Bolt | as for Chronos-Bolt |
| quantile head | direct quantile outputs 0.1, 0.2, ..., 0.9 (config.json); horizon 12 within the native prediction length 64, one pass | quantile outputs 0.1 ... 0.9 (config.json), output patch 64; sort quantiles=True | use continuous quantile head=True; deciles 0.1 ... 0.9 plus a mean channel; fix quantile crossing=True; max horizon=256 (one pass) |
| point forecast | median q0.5 (model mean kept as model point) | median (forecast output) | median (mean channel stored as mean head) |
| batch size | 256 series (CHRONOS BATCH) | package default per core batch size 4 | per core batch size=256 |
| other | float32, device map cuda, no covariates, no fine-tuning | package defaults: use symmetric averaging=False, make positive=False, use znorm=False, padding mode='none'; float32; no covariates (model-card settings) | normalize inputs=True, force flip invariance=True, infer is positive=True, torch compile=False (speed only), float32; no covariates |

### Table S8. Number of origins behind each Table 4 column, by horizon (both tracks)

Source file: `results/supplementary/T4_origins.csv`

sCRPS, ES and VS: origins 2022-12 ... 2025-06; coverage and Winkler: calibration origins 2023-12 ... 2025-06. Targets after 2025-12 and 2025-09 excluded.

| horizon | coverage and Winkler (uncal and ACI) | sCRPS, ES, VS |
|---|---|---|
| 1 | 19 | 31 |
| 2 | 19 | 31 |
| 3 | 18 | 30 |
| 4 | 18 | 30 |
| 5 | 18 | 30 |
| 6 | 18 | 30 |
| 7 | 17 | 29 |
| 8 | 16 | 28 |
| 9 | 15 | 27 |
| 10 | 14 | 26 |
| 11 | 13 | 25 |
| 12 | 12 | 24 |

### Table S9. Design choices fixed in advance and choices made after seeing results

Source: compiled by the authors. 'In advance' means before any model was run, unless stated otherwise.

| design choice | when it was fixed | note |
|---|---|---|
| Data rules: structural zeros, residence total as the sum of its components, exclusion of September 2025 from the main window | After the data audit, before any model was run | — |
| Constraint system, evaluation origins, horizons, windows, metrics and the 2022 weights | In advance | — |
| Statistical base models, global gradient-boosting model, reconciliation methods (OLS, WLS, MinT-shrink) | In advance | — |
| Foundation models | Chronos-Bolt and TimesFM 3.0 first; TimesFM 2.5 added after the first foundation-model results | — |
| W of the foundation models from their own one-step errors | After the first results, which used W from ETS residuals | Both versions reported |
| TimesFM 2.5 with MinT as the main track | After the results, by the lowest loss in the model confidence set | The post-release evaluation and the real-time tests are the independent checks |
| ACI with γ = 0.05 | After comparing γ = 0.05 and γ = 0.01 | Both reported |
| 80% interval nested inside the 95% interval | After the first results | Unnested results reported in the repository |
| Exact non-negative QP for all samples | For the final samples after the first evaluation; for the evaluation samples after the effect of setting negative values to zero had been checked | The shortcut that sets negative values to zero is reported for comparison (Tables S34–S36) |
| ACI state for the real-time test of June 2026 | After the exact reconciliation of all samples: the state of the final forecasts, calibrated over all origins up to May 2026 | The state of the evaluation recursion alone is reported for comparison (Table S38) |
| Forecast horizon: rule for H* | Revised after the first results; the final origin set and block bootstrap were specified before their results were seen | All variants reported (Table S41) |
| Correction (1 − d) of the published totals in the real-time test | Specified before scoring; its algebraic form was corrected after the first scoring | d = 0 also reported (Table S39) |

### Table S10. Automated tests of the software

Source file: `tests/` in the repository; 39 tests run in about one minute with the stored outputs of the archive. In a fresh copy without the archive, the 16 core tests run and the others are skipped with a message.

| test file | what it checks |
|---|---|
| test_constraints.py | The constraints hold on all 90 historical months within the rounding tolerance (1/28 t per day or 0.01%); the generic builder reproduces the paper's constraint matrix and its full-rank version exactly, and up to row permutation when the totals are reordered |
| test_reconciliation.py | WLS and MinT projections at origin 2024-06 (ETS) satisfy every constraint row and are idempotent; the exact non-negative MinT quadratic program at horizons 1 and 12 satisfies the Karush–Kuhn–Tucker conditions and equals the stored forecast, and the same conditions hold on the synthetic example; the null-space form of the quadratic program and its fallback stages give the same solution as the direct solver |
| test_aci.py | The ACI update rule against a hand calculation; a replay of the recursion from the stored intervals reproduces the stored levels at every origin and the stored state after 2025-06, for both tracks |
| test_structural_zeros.py | Seven tests of the structural-zero rule: synthetic cases (a zero only when every total containing the component is published and adds up; none when a total is missing or does not add up; the 1 t and 0.01% tolerances; two missing parts of one total; published values never marked) and, on the stored data, reproduction of the 1,684 filled values |
| test_exact_samples.py | Seven tests of the exactly reconciled evaluation samples and of the outputs built from them, including the real-time table and the final intervals |
| test_synthetic.py | The core tests that need no stored outputs: the builder on the synthetic system (identities, rank and redundant rows, order of the totals); the projection, coherent and idempotent, and the KKT conditions of the exact QP, for a diagonal and a dense W; the equivalence of the solvers (direct, null-space form, OSQP and the worker pool with its fallback stages); the ACI update by hand |

### Table S11. Model Confidence Set p-values of the nine base models (system economies and all 1,170 series)

Source file: `results/supplementary/mcs_pvalues.csv`

All 1,170 series = the 1,121 constraint-system series (1,120 series plus the world balance discrepancy series) and the 49 outside-system series.

| model | p, system economies, h all | p, system economies, h 1 | p, system economies, h 12 | p, all series, h all | p, all series, h 1 | p, all series, h 12 |
|---|---|---|---|---|---|---|
| ARIMA | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ETS | 0.0000 | 0.0000 | 0.2641 | 0.0000 | 0.0000 | 0.4799 |
| Theta | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Chronos-Bolt | 0.0000 | 0.0017 | 0.0000 | 0.0000 | 0.0017 | 0.0000 |
| LightGBM | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Drift | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Seasonal naive | 0.0000 | 0.0000 | 0.2641 | 0.0000 | 0.0000 | 0.4799 |
| TimesFM 3.0 | 0.0030 | 1.0000 | 0.0000 | 0.0129 | 1.0000 | 0.0000 |
| TimesFM 2.5 | 1.0000 | 0.1225 | 1.0000 | 1.0000 | 0.0588 | 1.0000 |

### Table S12. Post-release evaluation: target months after each model's release available in the stored forecasts

Source file: `results/fm_postrelease/coverage.csv`

'Release from' is the first target month after the model's release. The rolling sets have 1,121 series because they include the world balance discrepancy series.

| model | release from | set | n cells | n target months | n origins | n series |
|---|---|---|---|---|---|---|
| Chronos-Bolt | 2024-12 | rolling-main | 146,851 | 12 | 19 | 1,121 |
| Chronos-Bolt | 2024-12 | rolling-stress | 23,541 | 6 | 6 | 1,121 |
| TimesFM 2.5 | 2025-10 | rolling-main | 26,904 | 3 | 9 | 1,121 |
| TimesFM 2.5 | 2025-10 | rolling-stress | 23,541 | 6 | 6 | 1,121 |
| TimesFM 3.0 | 2026-09 | rolling-main | 0 | 0 | 0 | 0 |
| TimesFM 3.0 | 2026-09 | rolling-stress | 0 | 0 | 0 | 0 |
| TimesFM 2.5 | 2025-10 | realtime-sys (origin 2025-06) | 10,080 | 9 | 1 | 1,120 |
| TimesFM 2.5 | 2025-10 | realtime-pub (origins 2025-06, 2026-06; (1 − d) correction) | 1,650 | 12 | 2 | 110 |
| TimesFM 2.5 | 2025-10 | realtime-pub (origins 2025-06, 2026-06; d = 0) | 1,650 | 12 | 2 | 110 |
| Chronos-Bolt | 2024-12 | realtime | 0 | 0 | 0 | 0 |
| TimesFM 3.0 | 2026-09 | realtime | 0 | 0 | 0 | 0 |

### Table S13. Post-release evaluation: MASE, median / weighted over system economies, on the same post-release target months

Source file: `results/fm_postrelease/postrelease_scores.csv`

Foundation model = base forecasts of the model in the row; FM and STAT = reconciled tracks.

| model | set | Chronos-Bolt | ETS | FM (TimesFM 2.5 + MinT) | STAT (ETS + MinT) | TimesFM 2.5 |
|---|---|---|---|---|---|---|
| Chronos-Bolt | rolling-main | 0.707 / 0.694 | 0.764 / 0.710 | – | 0.746 / 0.664 | – |
| Chronos-Bolt | rolling-stress | 0.897 / 0.988 | 0.974 / 1.064 | – | 0.976 / 1.061 | – |
| TimesFM 2.5 | realtime-pub (origins 2025-06, 2026-06; (1 − d) correction) | – | 0.905 / 0.903 | 0.877 / 0.997 | 0.871 / 0.925 | 0.885 / 1.040 |
| TimesFM 2.5 | realtime-pub (origins 2025-06, 2026-06; d = 0) | – | 0.904 / 0.909 | 0.877 / 1.002 | 0.870 / 0.931 | 0.877 / 1.045 |
| TimesFM 2.5 | realtime-sys (origin 2025-06) | – | 0.990 / 0.967 | 0.915 / 0.944 | 0.967 / 0.960 | 0.928 / 0.978 |
| TimesFM 2.5 | rolling-main | – | 0.681 / 0.754 | 0.625 / 0.759 | 0.668 / 0.732 | 0.636 / 0.769 |
| TimesFM 2.5 | rolling-stress | – | 0.974 / 1.064 | 0.889 / 1.031 | 0.976 / 1.061 | 0.934 / 1.077 |

### Table S14. Ranks of the 23 base/method combinations by weighted and median MASE in each window (1 = best)

Source file: `results/supplementary/method_ranks.csv`

'Stress' denotes the January–June 2026 window, affected by the 2026 reclassification of voyages.

| base + method | weighted MASE, main | rank weighted, main | rank weighted, incl. 2025-09 | rank weighted, 2026 | rank median, main | rank median, incl. 2025-09 | rank median, 2026 |
|---|---|---|---|---|---|---|---|
| TimesFM 2.5 + MinT | 0.783 | 1.0 | 1.0 | 8.0 | 2.0 | 2.0 | 4.0 |
| TimesFM 2.5 + base | 0.792 | 2.0 | 2.0 | 18.0 | 3.0 | 3.0 | 12.0 |
| ETS + MinT | 0.793 | 3.0 | 3.0 | 12.0 | 10.0 | 10.0 | 18.0 |
| TimesFM 3.0 + base | 0.795 | 4.0 | 4.0 | 9.0 | 5.0 | 6.0 | 8.0 |
| TimesFM 2.5 + WLS | 0.796 | 5.0 | 5.0 | 17.0 | 1.0 | 1.0 | 9.0 |
| TimesFM 3.0 + MinT | 0.797 | 6.0 | 6.0 | 5.0 | 6.0 | 5.0 | 2.0 |
| TimesFM 3.0 + WLS | 0.802 | 7.0 | 7.0 | 6.0 | 4.0 | 4.0 | 7.0 |
| ETS + WLS | 0.822 | 8.0 | 8.0 | 15.0 | 11.0 | 11.0 | 19.0 |
| Theta + MinT | 0.825 | 9.0 | 9.0 | 10.0 | 16.0 | 16.0 | 14.0 |
| ETS + base | 0.827 | 10.0 | 10.0 | 14.0 | 13.0 | 15.0 | 17.0 |
| Chronos-Bolt + WLS | 0.835 | 11.0 | 11.0 | 2.0 | 7.0 | 7.0 | 1.0 |
| Chronos-Bolt + MinT | 0.836 | 12.0 | 12.0 | 1.0 | 9.0 | 9.0 | 3.0 |
| Chronos-Bolt + base | 0.836 | 13.0 | 13.0 | 3.0 | 8.0 | 8.0 | 5.0 |
| ARIMA + MinT | 0.855 | 14.0 | 15.0 | 4.0 | 14.0 | 12.0 | 6.0 |
| Theta + WLS | 0.855 | 15.0 | 14.0 | 16.0 | 18.0 | 18.0 | 16.0 |
| Theta + base | 0.857 | 16.0 | 16.0 | 13.0 | 19.0 | 19.0 | 15.0 |
| ARIMA + WLS | 0.860 | 17.0 | 17.0 | 7.0 | 12.0 | 13.0 | 11.0 |
| ARIMA + base | 0.866 | 18.0 | 18.0 | 11.0 | 15.0 | 14.0 | 13.0 |
| RandomWalkWithDrift+base | 0.934 | 19.0 | 19.0 | 23.0 | 23.0 | 23.0 | 23.0 |
| Seasonal naive + MinT | 0.986 | 20.0 | 20.0 | 21.5 | 21.0 | 21.0 | 21.5 |
| Seasonal naive + WLS | 0.986 | 21.0 | 21.0 | 20.0 | 21.0 | 22.0 | 20.0 |
| Seasonal naive + base | 0.986 | 22.0 | 22.0 | 21.5 | 21.0 | 20.0 | 21.5 |
| LightGBM+base | 1.028 | 23.0 | 23.0 | 19.0 | 17.0 | 17.0 | 10.0 |

### Table S15. Spearman correlation of the method ranks between windows

Source file: `results/supplementary/rank_correlation.csv`

| window 1 | window 2 | statistic | spearman rho | p value | n combos |
|---|---|---|---|---|---|
| main | main incl. 2025-09 | weighted | 0.999 | 0.0000 | 23 |
| main | main incl. 2025-09 | median | 0.993 | 0.0000 | 23 |
| main | 2026-01..06 (stress) | weighted | 0.397 | 0.0605 | 23 |
| main | 2026-01..06 (stress) | median | 0.727 | 0.0001 | 23 |
| main incl. 2025-09 | 2026-01..06 (stress) | weighted | 0.385 | 0.0693 | 23 |
| main incl. 2025-09 | 2026-01..06 (stress) | median | 0.751 | 0.0000 | 23 |

### Table S16. Diebold-Mariano tests per system economy series, absolute error loss, main window: counts of significant differences (5%)

Source file: `results/supplementary/dm_counts.csv`

HLN-corrected DM, rectangular kernel with lag h-1, t(n-1), two-sided. 'better' = first forecast better. BH = Benjamini-Hochberg at 5% within comparison x horizon (1,100 tests). Lag-0 fallback = tests where the long-run variance was not positive and the lag-0 autocovariance was used.

| comparison | horizon | n tests | n untestable | better / worse | after BH | lag-0 fallback | mean n origins |
|---|---|---|---|---|---|---|---|
| FM+base vs STAT+base | 1 | 1,100 | 0 | 123 / 7 | 7 / 0 | 0 | 31 |
| FM+base vs STAT+base | 3 | 1,100 | 0 | 122 / 17 | 21 / 2 | 5 | 30 |
| FM+base vs STAT+base | 6 | 1,100 | 0 | 190 / 29 | 93 / 10 | 59 | 30 |
| FM+base vs STAT+base | 12 | 1,100 | 0 | 86 / 62 | 48 / 35 | 225 | 24 |
| FM+rec vs FM+base | 1 | 1,100 | 0 | 45 / 27 | 0 / 1 | 0 | 31 |
| FM+rec vs FM+base | 3 | 1,100 | 0 | 42 / 32 | 6 / 4 | 3 | 30 |
| FM+rec vs FM+base | 6 | 1,100 | 0 | 67 / 63 | 14 / 25 | 45 | 30 |
| FM+rec vs FM+base | 12 | 1,100 | 0 | 113 / 56 | 54 / 22 | 272 | 24 |
| FM+rec vs STAT+rec | 1 | 1,100 | 0 | 97 / 21 | 0 / 0 | 0 | 31 |
| FM+rec vs STAT+rec | 3 | 1,100 | 0 | 89 / 22 | 9 / 0 | 6 | 30 |
| FM+rec vs STAT+rec | 6 | 1,100 | 0 | 147 / 29 | 65 / 9 | 58 | 30 |
| FM+rec vs STAT+rec | 12 | 1,100 | 1 | 91 / 51 | 48 / 26 | 274 | 24 |
| STAT+rec vs STAT+base | 1 | 1,100 | 0 | 86 / 28 | 2 / 0 | 0 | 31 |
| STAT+rec vs STAT+base | 3 | 1,100 | 0 | 148 / 44 | 45 / 5 | 3 | 30 |
| STAT+rec vs STAT+base | 6 | 1,100 | 0 | 173 / 62 | 84 / 26 | 122 | 30 |
| STAT+rec vs STAT+base | 12 | 1,100 | 0 | 51 / 68 | 19 / 18 | 303 | 24 |

### Table S17. MASE of the ten World series, base and reconciled forecasts, main window, all horizons

Source files: `results/paper/T3b_reconciliation_world.csv` (the table); `results/checks/world_mase_coincidence.csv` (the six-decimal MASE values and the correlation in the note)

Columns A to G are the World components; reconciled forecasts use the exact non-negative QP.

| track | method | A | B | C | D | E | F | G | residence total | domestic (territory) | international (territory) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FM (TimesFM 2.5) | base | 0.533 | 0.465 | 0.465 | 0.610 | 0.869 | 0.848 | 0.854 | 0.959 | 0.481 | 0.854 |
| FM (TimesFM 2.5) | WLS | 0.465 | 0.433 | 0.433 | 0.569 | 0.766 | 0.660 | 0.667 | 0.919 | 0.414 | 0.659 |
| FM (TimesFM 2.5) | MinT | 0.466 | 0.451 | 0.451 | 0.572 | 0.741 | 0.677 | 0.681 | 0.820 | 0.430 | 0.671 |
| STAT (ETS) | base | 0.905 | 0.506 | 0.506 | 0.724 | 0.744 | 0.765 | 0.751 | 0.914 | 0.562 | 0.763 |
| STAT (ETS) | WLS | 0.567 | 0.511 | 0.511 | 0.645 | 0.725 | 0.711 | 0.710 | 0.879 | 0.504 | 0.705 |
| STAT (ETS) | MinT | 0.534 | 0.497 | 0.497 | 0.600 | 0.683 | 0.683 | 0.679 | 0.803 | 0.493 | 0.671 |

World international (territory) and World G have nearly equal MASE for the FM base forecasts (0.853734 and 0.853893): G makes up about 91% of the territory international total and the errors of the two series are almost perfectly correlated (r = 0.9988).

### Table S18. Accuracy against the published residence totals, main window

Source file: `results/checks/published_totals_mase.csv`

Forecasts of the residence total (sum of components) divided by (1 − d), with d the mean relative discrepancy over the six months up to the origin, per economy; scale of the published series; 2022 weights; 112 units (110 economies, the Residual and the World), 38,192 cells.

| track | forecast | target | economies, median | economies, weighted | World | Residual |
|---|---|---|---|---|---|---|
| FM | reconciled | published total, (1 − d) | 0.809 | 0.802 | 0.831 | 1.815 |
| FM | reconciled | published total, d = 0 | 0.809 | 0.802 | 0.832 | 1.813 |
| FM | reconciled | sum of components (reference) | 0.814 | 0.802 | 0.820 | 1.777 |
| STAT | reconciled | published total, (1 − d) | 0.807 | 0.800 | 0.813 | 1.601 |
| STAT | reconciled | published total, d = 0 | 0.807 | 0.800 | 0.814 | 1.600 |
| STAT | reconciled | sum of components (reference) | 0.808 | 0.800 | 0.803 | 1.582 |
| FM | base | published total, (1 − d) | 0.824 | 0.819 | 0.971 | 1.715 |
| STAT | base | published total, (1 − d) | 0.841 | 0.845 | 0.923 | 1.586 |

### Table S19. Similarity of the one-step error correlations (W) with the rolling-origin h-step error correlations

Source file: `results/w_diagnostics/correlation_similarity.csv`

1,120 series, 626,640 off-diagonal pairs, origin 2025-06 W; centered versions in the source file.

| track | h | reference | n origins | corr offdiag | rel frobenius offdiag | mean abs offdiag ref | mean abs offdiag h |
|---|---|---|---|---|---|---|---|
| FM | 3 | one-step shrinkage correlation (W) | 30 | 0.636 | 3.567 | 0.050 | 0.221 |
| FM | 3 | one-step raw correlation | 30 | 0.636 | 1.220 | 0.130 | 0.221 |
| FM | 6 | one-step shrinkage correlation (W) | 30 | 0.532 | 4.072 | 0.050 | 0.246 |
| FM | 6 | one-step raw correlation | 30 | 0.532 | 1.469 | 0.130 | 0.246 |
| FM | 12 | one-step shrinkage correlation (W) | 24 | 0.287 | 5.389 | 0.050 | 0.311 |
| FM | 12 | one-step raw correlation | 24 | 0.287 | 2.092 | 0.130 | 0.311 |
| STAT | 3 | one-step shrinkage correlation (W) | 30 | 0.598 | 4.403 | 0.043 | 0.226 |
| STAT | 3 | one-step raw correlation | 30 | 0.598 | 1.358 | 0.126 | 0.226 |
| STAT | 6 | one-step shrinkage correlation (W) | 30 | 0.499 | 4.818 | 0.043 | 0.243 |
| STAT | 6 | one-step raw correlation | 30 | 0.499 | 1.564 | 0.126 | 0.243 |
| STAT | 12 | one-step shrinkage correlation (W) | 24 | 0.307 | 5.697 | 0.043 | 0.278 |
| STAT | 12 | one-step raw correlation | 24 | 0.307 | 1.977 | 0.126 | 0.278 |

### Table S20. Friedman test and Nemenyi critical difference over base / WLS / MinT, per base, main window

Source file: `results/supplementary/friedman_nemenyi.csv`

Blocks = system economy series; loss = MASE. The four-treatment version with OLS is in the source file.

| base | horizon | n series | friedman chi2 | friedman p | nemenyi cd | mean rank base | mean rank WLS | mean rank MinT |
|---|---|---|---|---|---|---|---|---|
| ARIMA | 1 | 1,100 | 56.9 | 0.0000 | 0.100 | 2.185 | 1.915 | 1.900 |
| ARIMA | 3 | 1,100 | 36.0 | 0.0000 | 0.100 | 2.147 | 1.918 | 1.935 |
| ARIMA | 6 | 1,100 | 24.8 | 0.0000 | 0.100 | 2.111 | 1.900 | 1.990 |
| ARIMA | 12 | 1,100 | 72.1 | 0.0000 | 0.100 | 2.049 | 1.800 | 2.151 |
| ARIMA | all | 1,100 | 73.0 | 0.0000 | 0.100 | 2.130 | 1.792 | 2.078 |
| Chronos-Bolt | 1 | 1,100 | 39.8 | 0.0000 | 0.100 | 2.096 | 1.846 | 2.057 |
| Chronos-Bolt | 3 | 1,100 | 67.0 | 0.0000 | 0.100 | 2.052 | 1.805 | 2.143 |
| Chronos-Bolt | 6 | 1,100 | 47.8 | 0.0000 | 0.100 | 2.007 | 1.849 | 2.144 |
| Chronos-Bolt | 12 | 1,100 | 12.3 | 0.0021 | 0.100 | 2.076 | 1.927 | 1.997 |
| Chronos-Bolt | all | 1,100 | 47.6 | 0.0000 | 0.100 | 2.054 | 1.834 | 2.113 |
| ETS | 1 | 1,100 | 111.5 | 0.0000 | 0.100 | 2.221 | 2.008 | 1.771 |
| ETS | 3 | 1,100 | 82.2 | 0.0000 | 0.100 | 2.197 | 1.992 | 1.811 |
| ETS | 6 | 1,100 | 101.4 | 0.0000 | 0.100 | 2.205 | 2.019 | 1.776 |
| ETS | 12 | 1,100 | 89.8 | 0.0000 | 0.100 | 1.971 | 1.814 | 2.215 |
| ETS | all | 1,100 | 85.2 | 0.0000 | 0.100 | 2.213 | 1.963 | 1.825 |
| Seasonal naive | 1 | 1,100 | 403.1 | 0.0000 | 0.100 | 2.105 | 2.365 | 1.529 |
| Seasonal naive | 3 | 1,100 | 166.9 | 0.0000 | 0.100 | 2.017 | 2.266 | 1.716 |
| Seasonal naive | 6 | 1,100 | 43.6 | 0.0000 | 0.100 | 2.012 | 2.135 | 1.854 |
| Seasonal naive | 12 | 1,100 | 57.4 | 0.0000 | 0.100 | 2.076 | 2.109 | 1.815 |
| Seasonal naive | all | 1,100 | 132.4 | 0.0000 | 0.100 | 2.045 | 2.220 | 1.735 |
| Theta | 1 | 1,100 | 177.3 | 0.0000 | 0.100 | 2.230 | 2.087 | 1.683 |
| Theta | 3 | 1,100 | 124.1 | 0.0000 | 0.100 | 2.208 | 2.050 | 1.741 |
| Theta | 6 | 1,100 | 118.9 | 0.0000 | 0.100 | 2.165 | 2.101 | 1.734 |
| Theta | 12 | 1,100 | 43.6 | 0.0000 | 0.100 | 2.010 | 1.855 | 2.135 |
| Theta | all | 1,100 | 95.5 | 0.0000 | 0.100 | 2.166 | 2.067 | 1.766 |
| TimesFM 2.5 | 1 | 1,100 | 19.5 | 0.0001 | 0.100 | 2.101 | 1.985 | 1.915 |
| TimesFM 2.5 | 3 | 1,100 | 13.5 | 0.0012 | 0.100 | 2.080 | 1.924 | 1.996 |
| TimesFM 2.5 | 6 | 1,100 | 12.3 | 0.0021 | 0.100 | 2.020 | 1.917 | 2.063 |
| TimesFM 2.5 | 12 | 1,100 | 10.7 | 0.0047 | 0.100 | 2.077 | 1.981 | 1.941 |
| TimesFM 2.5 | all | 1,100 | 10.8 | 0.0046 | 0.100 | 2.061 | 1.924 | 2.015 |
| TimesFM 3.0 | 1 | 1,100 | 12.3 | 0.0021 | 0.100 | 2.086 | 1.956 | 1.957 |
| TimesFM 3.0 | 3 | 1,100 | 18.4 | 0.0001 | 0.100 | 2.057 | 1.895 | 2.048 |
| TimesFM 3.0 | 6 | 1,100 | 19.9 | 0.0000 | 0.100 | 1.985 | 1.914 | 2.102 |
| TimesFM 3.0 | 12 | 1,100 | 5.2 | 0.0727 | 0.100 | 2.055 | 1.960 | 1.985 |
| TimesFM 3.0 | all | 1,100 | 12.5 | 0.0019 | 0.100 | 2.008 | 1.921 | 2.071 |

### Table S21. Model Confidence Set p-values, base x reconciliation method, system economies, main window

Source file: `results/supplementary/mcs_pvalues.csv`

arch 8.0.0, 90% MCS, range statistic, stationary bootstrap with mean block int(sqrt(T)) (5 for 31 origins, 4 for 24), 10,000 replications, seed 42. Loss per origin = mean scaled absolute error. (ETS W) = W estimated from ETS residuals. In the 90% set: p > 0.10.

| candidate | p, h all | p, h 1 | p, h 3 | p, h 6 | p, h 12 | mean loss all |
|---|---|---|---|---|---|---|
| TimesFM 2.5 + MinT | 1.0000 | 0.9914 | 0.9793 | 1.0000 | 1.0000 | 0.910 |
| TimesFM 2.5 + MinT (ETS W) | 0.4361 | 0.5394 | 1.0000 | 0.9215 | 0.0635 | 0.914 |
| TimesFM 2.5 + WLS (ETS W) | 0.3118 | 0.0621 | 0.1435 | 0.6575 | 0.0118 | 0.915 |
| TimesFM 2.5 + WLS | 0.3118 | 0.0459 | 0.1284 | 0.6575 | 0.0118 | 0.915 |
| TimesFM 2.5 + base | 0.0000 | 0.0015 | 0.0063 | 0.0001 | 0.0000 | 0.918 |
| TimesFM 3.0 + MinT | 0.0000 | 0.9914 | 0.0036 | 0.0000 | 0.0000 | 0.920 |
| TimesFM 3.0 + WLS (ETS W) | 0.0003 | 0.3539 | 0.2335 | 0.0800 | 0.0000 | 0.920 |
| TimesFM 3.0 + WLS | 0.0001 | 0.2950 | 0.2335 | 0.0675 | 0.0000 | 0.920 |
| TimesFM 3.0 + MinT (ETS W) | 0.0037 | 1.0000 | 0.5131 | 0.0800 | 0.0000 | 0.922 |
| TimesFM 3.0 + base | 0.0000 | 0.0000 | 0.0016 | 0.0007 | 0.0000 | 0.924 |
| Chronos-Bolt + MinT (ETS W) | 0.0000 | 0.0000 | 0.0003 | 0.0000 | 0.0000 | 0.939 |
| Chronos-Bolt + WLS | 0.0000 | 0.0000 | 0.0026 | 0.0000 | 0.0000 | 0.940 |
| Chronos-Bolt + WLS (ETS W) | 0.0000 | 0.0000 | 0.0024 | 0.0000 | 0.0000 | 0.940 |
| Chronos-Bolt + MinT | 0.0000 | 0.0180 | 0.0583 | 0.0015 | 0.0000 | 0.941 |
| Chronos-Bolt + base | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.946 |
| ETS + MinT | 0.0000 | 0.0000 | 0.0003 | 0.0000 | 0.0591 | 0.958 |
| ETS + WLS | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0597 | 0.969 |
| ETS + base | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.977 |
| ARIMA + WLS | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.980 |
| ARIMA + MinT | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.983 |
| Theta + MinT | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.985 |
| ARIMA + base | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.991 |
| Theta + WLS | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.004 |
| Theta + base | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.010 |
| Seasonal naive + WLS | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0716 | 1.104 |
| Seasonal naive + MinT | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0716 | 1.104 |
| Seasonal naive + base | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0716 | 1.104 |

### Table S22. Constraint ablation: MASE of MinT exact QP under three constraint sets, system economies, main window

Source file: `results/ablation/mase_summary.csv`

MASE summaries use the 1,100 economy series; the cell count covers all 31 × 12 = 372 origin–horizon pairs of the 1,120 system series; the MASE values use the main window.

Set 1 = full dual allocation; 2 = without the two world-balance rows; 3 = aggregation only.

| track | set | median MASE | weighted MASE | median rel to set1 | weighted rel to set1 | n series | n cells |
|---|---|---|---|---|---|---|---|
| FM | 1 | 0.779 | 0.783 | 1.000 | 1.000 | 1,100 | 416,640 |
| FM | 2 | 0.780 | 0.785 | 1.002 | 1.002 | 1,100 | 416,640 |
| FM | 3 | 0.783 | 0.798 | 1.005 | 1.020 | 1,100 | 416,640 |
| STAT | 1 | 0.818 | 0.793 | 1.000 | 1.000 | 1,100 | 416,640 |
| STAT | 2 | 0.819 | 0.793 | 1.001 | 0.999 | 1,100 | 416,640 |
| STAT | 3 | 0.832 | 0.821 | 1.017 | 1.035 | 1,100 | 416,640 |

### Table S23. Constraint ablation: MASE of the World series by constraint set

Source file: `results/ablation/mase_world_series.csv`

| unique id | FM set 1 | FM set 2 | FM set 3 | STAT set 1 | STAT set 2 | STAT set 3 |
|---|---|---|---|---|---|---|
| W/C | 0.451 | 0.441 | 0.453 | 0.497 | 0.493 | 0.510 |
| W/G | 0.681 | 0.714 | 0.738 | 0.679 | 0.691 | 0.702 |
| W/A | 0.466 | 0.462 | 0.472 | 0.534 | 0.534 | 0.579 |
| W/B | 0.451 | 0.474 | 0.459 | 0.497 | 0.497 | 0.505 |
| W/D | 0.572 | 0.579 | 0.600 | 0.600 | 0.601 | 0.638 |
| W/F | 0.677 | 0.684 | 0.724 | 0.683 | 0.681 | 0.715 |
| W/E | 0.741 | 0.742 | 0.805 | 0.683 | 0.681 | 0.713 |
| W/residence total | 0.820 | 0.802 | 0.815 | 0.803 | 0.801 | 0.855 |
| W/domestic (territory) | 0.430 | 0.423 | 0.428 | 0.493 | 0.487 | 0.503 |
| W/international (territory) | 0.671 | 0.700 | 0.739 | 0.671 | 0.680 | 0.702 |

### Table S24. Constraint ablation: violations of the within-unit identities left by each set (%)

Source file: `results/ablation/violations_within_unit.csv`

Relative violation = |identity residual| / total, %. Set 1 (not shown) and set 2 meet these identities to rounding (below 1e-9%).

| track | set | total | n cells | median rel pct | p95 rel pct | max rel pct | share cells rel gt 1pct | world mean rel pct |
|---|---|---|---|---|---|---|---|---|
| FM | 2 | residence total | 41,664 | 0.000 | 0.000 | 0.0 | 0.000 | 0.000 |
| FM | 2 | domestic (territory) | 41,664 | 0.000 | 0.000 | 0.0 | 0.000 | 0.000 |
| FM | 2 | international (territory) | 41,664 | 0.000 | 0.000 | 0.0 | 0.000 | 0.000 |
| FM | 3 | residence total | 41,664 | 2.364 | 12.899 | 71.4 | 0.745 | 1.038 |
| FM | 3 | domestic (territory) | 41,664 | 1.699 | 9.631 | 133.5 | 0.663 | 0.490 |
| FM | 3 | international (territory) | 41,664 | 0.342 | 4.677 | 50.2 | 0.254 | 0.159 |
| STAT | 2 | residence total | 41,664 | 0.000 | 0.000 | 0.0 | 0.000 | 0.000 |
| STAT | 2 | domestic (territory) | 41,664 | 0.000 | 0.000 | 0.0 | 0.000 | 0.000 |
| STAT | 2 | international (territory) | 41,664 | 0.000 | 0.000 | 0.0 | 0.000 | 0.000 |
| STAT | 3 | residence total | 41,664 | 1.830 | 14.036 | 2,978.2 | 0.666 | 0.495 |
| STAT | 3 | domestic (territory) | 41,664 | 1.638 | 19.970 | 246.0 | 0.609 | 0.859 |
| STAT | 3 | international (territory) | 41,664 | 0.374 | 11.513 | 681.7 | 0.314 | 0.210 |

### Table S25. Constraint ablation: World residence-territory gap and B(W) - C(W), % of the World total

Source files: `results/ablation/violations_world_gap_by_horizon.csv` (the rows by horizon; the 'all' rows are the mean over horizons 1-12); `results/paper/quoted_figures.csv` (the 'all' values, item 9)

Gap = domestic (territory) + international (territory) + r G - residence total. Every horizon in the source file.

| track | set | horizon | mean abs res ter gap pct | max abs res ter gap pct | mean abs B minus C pct of world |
|---|---|---|---|---|---|
| FM | 2 | 1 | 0.423 | 1.456 | 0.141 |
| FM | 2 | 3 | 0.612 | 1.616 | 0.224 |
| FM | 2 | 6 | 0.852 | 1.780 | 0.297 |
| FM | 2 | 12 | 1.105 | 2.286 | 0.393 |
| FM | 3 | 1 | 0.682 | 2.074 | 0.125 |
| FM | 3 | 3 | 0.979 | 2.533 | 0.159 |
| FM | 3 | 6 | 1.410 | 2.788 | 0.140 |
| FM | 3 | 12 | 2.025 | 3.815 | 0.151 |
| STAT | 2 | 1 | 0.309 | 1.312 | 0.152 |
| STAT | 2 | 3 | 0.381 | 1.484 | 0.154 |
| STAT | 2 | 6 | 0.471 | 1.341 | 0.161 |
| STAT | 2 | 12 | 0.384 | 1.027 | 0.122 |
| STAT | 3 | 1 | 0.477 | 2.300 | 0.103 |
| STAT | 3 | 3 | 0.587 | 1.827 | 0.209 |
| STAT | 3 | 6 | 0.565 | 1.802 | 0.194 |
| STAT | 3 | 12 | 0.511 | 1.126 | 0.089 |
| FM | 2 | all | 0.836 | – | – |
| FM | 3 | all | 1.436 | – | – |
| STAT | 2 | all | 0.435 | – | – |
| STAT | 3 | all | 0.564 | – | – |

### Table S26. Benchmarks for reconciliation: MASE of bottom-up (BU) and combined (COMB) forecasts against base, WLS and MinT, main window, all horizons

Source files: `results/benchmarks/mase_summary.csv` (the table); `results/benchmarks/mase_by_horizon.csv` (the values by horizon); `results/supplementary/OLS_tracks.csv` (the OLS values in the first note)

OLS with the exact QP, for comparison: weighted MASE 0.854 (TimesFM 2.5) and 0.852 (ETS); median 1.326 and 1.285.

BU builds every unit's totals from its seven component forecasts and the World values from the economies and the Residual. COMB is the mean of the TimesFM 2.5 and ETS base forecasts. Base forecasts are used as stored (negative ETS values kept). All ten World series and the values by horizon are in the source files.

| base | method | economies, median | economies, weighted | World residence total | World domestic (territory) | World international (territory) | World G |
|---|---|---|---|---|---|---|---|
| TimesFM 2.5 | base | 0.781 | 0.792 | 0.959 | 0.481 | 0.854 | 0.854 |
| TimesFM 2.5 | WLS | 0.779 | 0.796 | 0.919 | 0.414 | 0.659 | 0.667 |
| TimesFM 2.5 | MinT | 0.779 | 0.783 | 0.820 | 0.430 | 0.671 | 0.681 |
| TimesFM 2.5 | BU | 0.785 | 0.814 | 1.050 | 0.438 | 0.700 | 0.708 |
| ETS | base | 0.834 | 0.827 | 0.914 | 0.562 | 0.763 | 0.751 |
| ETS | WLS | 0.827 | 0.822 | 0.879 | 0.504 | 0.705 | 0.710 |
| ETS | MinT | 0.818 | 0.793 | 0.803 | 0.493 | 0.671 | 0.679 |
| ETS | BU | 0.840 | 0.842 | 0.923 | 0.505 | 0.704 | 0.711 |
| TimesFM 2.5 and ETS | COMB | 0.791 | 0.788 | 0.918 | 0.497 | 0.791 | 0.783 |

### Table S27. Coherence of the benchmarks: violations of the identities in % of the forecast's own World residence total, mean over origins

Source file: `results/benchmarks/violations_by_horizon.csv`

Within-unit and aggregation identities hold exactly in BU; for BU only the world balance is shown. For COMB, all 348 rows of the constraint matrix are checked. "Max" is the largest value over all origin × horizon cells.

| forecast | identities checked | h = 1 | h = 3 | h = 6 | h = 12 | all h | max |
|---|---|---|---|---|---|---|---|
| BU of TimesFM 2.5 | world balance (largest row) | 0.44 | 0.61 | 0.68 | 0.68 | 0.65 | 2.29 |
| BU of ETS | world balance (largest row) | 0.43 | 0.54 | 0.66 | 0.28 | 0.56 | 2.67 |
| COMB | all rows | 0.82 | 1.28 | 1.65 | 1.42 | 1.47 | 4.02 |
| TimesFM 2.5 base | all rows | 1.03 | 1.82 | 2.51 | 2.73 | 2.28 | 5.99 |
| ETS base | all rows | 1.04 | 1.53 | 1.80 | 0.60 | 1.47 | 5.22 |

### Table S28. Origin-block bootstrap intervals for key differences, main window, all horizons pooled

Source files: `results/inference/block_bootstrap.csv` (the table and the values by horizon in the note); `results/exact_eval/fm_minus_stat_crps_bootstrap.csv` (the zeroed-sample CRPS difference in the first note)

The CRPS row uses exactly reconciled samples; the shortcut that sets negative sample values to zero gives −0.008 (−0.042 to 0.020; Table S35).

Moving-block bootstrap over the 31 consecutive origins (blocks of 6, non-circular, 2,000 resamples, all series resampled jointly). Weighted means over the 1,100 economy series; reconciled forecasts are MinT with the exact QP; CRPS from the reconciled samples of Table 4. By horizon: FM − STAT (MASE) is significant only at h = 6 (−0.020, interval −0.044 to −0.001); FM reconciled − base only at h = 12 (−0.024, interval −0.035 to −0.013); STAT reconciled − base at h = 1, 3 and 6, and +0.009 (interval −0.002 to 0.020) at h = 12.

| difference | point | 95% interval | share of resamples below zero |
|---|---|---|---|
| Weighted MASE, FM − STAT | −0.0106 (−1.3%) | −0.0325 to 0.0111 | 0.82 |
| Weighted MASE, FM reconciled − FM base | −0.0098 (−1.2%) | −0.0169 to 0.0032 | 0.92 |
| Weighted MASE, STAT reconciled − STAT base | −0.0333 (−4.0%) | −0.0464 to −0.0166 | 1.00 |
| Weighted scaled CRPS, FM − STAT | −0.212 (−26%) | −0.553 to −0.016 | 0.99 |

### Table S29. Diebold-Mariano tests per system economy series, scaled CRPS loss, main window: counts of significant differences (5%)

Source file: `results/supplementary/dm_counts.csv`

HLN-corrected DM, rectangular kernel with lag h-1, t(n-1), two-sided. 'better' = first forecast better. BH = Benjamini-Hochberg at 5% within comparison x horizon (1,100 tests). Lag-0 fallback = tests where the long-run variance was not positive and γ_0 was used; not recorded for the stored CRPS tests.

| comparison | horizon | n tests | n untestable | better / worse | after BH | lag-0 fallback | mean n origins |
|---|---|---|---|---|---|---|---|
| FM+base vs STAT+base | 1 | 1,100 | 0 | 161 / 11 | 22 / 0 | not recorded | 31 |
| FM+base vs STAT+base | 3 | 1,100 | 0 | 194 / 14 | 57 / 2 | not recorded | 30 |
| FM+base vs STAT+base | 6 | 1,100 | 0 | 256 / 23 | 161 / 11 | not recorded | 30 |
| FM+base vs STAT+base | 12 | 1,100 | 0 | 243 / 36 | 156 / 19 | not recorded | 24 |
| FM+rec vs FM+base | 1 | 1,100 | 0 | 64 / 21 | 0 / 0 | not recorded | 31 |
| FM+rec vs FM+base | 3 | 1,100 | 0 | 50 / 28 | 4 / 2 | not recorded | 30 |
| FM+rec vs FM+base | 6 | 1,100 | 0 | 80 / 66 | 30 / 31 | not recorded | 30 |
| FM+rec vs FM+base | 12 | 1,100 | 0 | 120 / 54 | 55 / 18 | not recorded | 24 |
| FM+rec vs STAT+rec | 1 | 1,100 | 0 | 108 / 20 | 4 / 0 | not recorded | 31 |
| FM+rec vs STAT+rec | 3 | 1,100 | 0 | 127 / 6 | 24 / 0 | not recorded | 30 |
| FM+rec vs STAT+rec | 6 | 1,100 | 0 | 269 / 13 | 119 / 6 | not recorded | 30 |
| FM+rec vs STAT+rec | 12 | 1,100 | 1 | 174 / 18 | 71 / 11 | not recorded | 24 |
| STAT+rec vs STAT+base | 1 | 1,100 | 0 | 125 / 18 | 25 / 1 | not recorded | 31 |
| STAT+rec vs STAT+base | 3 | 1,100 | 0 | 153 / 41 | 61 / 4 | not recorded | 30 |
| STAT+rec vs STAT+base | 6 | 1,100 | 0 | 132 / 112 | 64 / 50 | not recorded | 30 |
| STAT+rec vs STAT+base | 12 | 1,100 | 0 | 42 / 67 | 24 / 24 | not recorded | 24 |

### Table S30. Pooled coverage with 95% moving-block bootstrap intervals over origins (block 6), by measure group

Source file: `results/supplementary/coverage_block_bootstrap.csv`

The intervals differ slightly from those of Table S32, which uses a circular block bootstrap with 2,000 resamples; the main text quotes Table S32.

Calibration origins 2023-12 ... 2025-06, main window. By horizon in the source file.

| track | interval | level | group | coverage | boot lo95 | boot hi95 |
|---|---|---|---|---|---|---|
| FM | ACI γ 0.05 (main) | 80 | all system economy series | 0.801 | 0.777 | 0.820 |
| FM | ACI γ 0.05 (main) | 95 | all system economy series | 0.920 | 0.909 | 0.928 |
| FM | uncalibrated, reconciled samples | 80 | all system economy series | 0.737 | 0.720 | 0.756 |
| FM | uncalibrated, reconciled samples | 95 | all system economy series | 0.894 | 0.880 | 0.905 |
| STAT | ACI γ 0.05 (main) | 80 | all system economy series | 0.802 | 0.788 | 0.819 |
| STAT | ACI γ 0.05 (main) | 95 | all system economy series | 0.923 | 0.917 | 0.930 |
| STAT | uncalibrated, reconciled samples | 80 | all system economy series | 0.763 | 0.748 | 0.783 |
| STAT | uncalibrated, reconciled samples | 95 | all system economy series | 0.903 | 0.893 | 0.912 |
| FM | ACI γ 0.05 (main) | 80 | components | 0.807 | 0.788 | 0.825 |
| FM | ACI γ 0.05 (main) | 80 | residence total | 0.750 | 0.705 | 0.782 |
| FM | ACI γ 0.05 (main) | 80 | territory measures | 0.803 | 0.775 | 0.826 |
| FM | ACI γ 0.05 (main) | 95 | components | 0.926 | 0.917 | 0.933 |
| FM | ACI γ 0.05 (main) | 95 | residence total | 0.870 | 0.841 | 0.889 |
| FM | ACI γ 0.05 (main) | 95 | territory measures | 0.924 | 0.914 | 0.933 |
| FM | uncalibrated, reconciled samples | 80 | components | 0.755 | 0.739 | 0.772 |
| FM | uncalibrated, reconciled samples | 80 | residence total | 0.656 | 0.626 | 0.678 |
| FM | uncalibrated, reconciled samples | 80 | territory measures | 0.717 | 0.696 | 0.738 |
| FM | uncalibrated, reconciled samples | 95 | components | 0.905 | 0.893 | 0.914 |
| FM | uncalibrated, reconciled samples | 95 | residence total | 0.835 | 0.813 | 0.852 |
| FM | uncalibrated, reconciled samples | 95 | territory measures | 0.885 | 0.867 | 0.901 |
| STAT | ACI γ 0.05 (main) | 80 | components | 0.807 | 0.795 | 0.824 |
| STAT | ACI γ 0.05 (main) | 80 | residence total | 0.764 | 0.745 | 0.778 |
| STAT | ACI γ 0.05 (main) | 80 | territory measures | 0.800 | 0.782 | 0.823 |
| STAT | ACI γ 0.05 (main) | 95 | components | 0.927 | 0.921 | 0.934 |
| STAT | ACI γ 0.05 (main) | 95 | residence total | 0.895 | 0.879 | 0.906 |
| STAT | ACI γ 0.05 (main) | 95 | territory measures | 0.924 | 0.916 | 0.935 |
| STAT | uncalibrated, reconciled samples | 80 | components | 0.779 | 0.764 | 0.798 |
| STAT | uncalibrated, reconciled samples | 80 | residence total | 0.716 | 0.698 | 0.730 |
| STAT | uncalibrated, reconciled samples | 80 | territory measures | 0.732 | 0.713 | 0.758 |
| STAT | uncalibrated, reconciled samples | 95 | components | 0.910 | 0.901 | 0.919 |
| STAT | uncalibrated, reconciled samples | 95 | residence total | 0.875 | 0.856 | 0.887 |
| STAT | uncalibrated, reconciled samples | 95 | territory measures | 0.891 | 0.879 | 0.909 |

### Table S31. ACI coverage by measure group and size tercile (terciles of the 1,100 series by 2022 weight)

Source file: `results/supplementary/coverage_wilson.csv`

Wilson intervals in the source file assume independent cells and are too narrow.

| track | level | group | coverage, small | coverage, medium | coverage, large |
|---|---|---|---|---|---|
| FM | 80 | components | 0.784 | 0.803 | 0.846 |
| FM | 80 | residence total | 0.722 | 0.754 | 0.760 |
| FM | 80 | territory measures | 0.763 | 0.788 | 0.828 |
| FM | 95 | components | 0.906 | 0.935 | 0.945 |
| FM | 95 | residence total | 0.852 | 0.874 | 0.876 |
| FM | 95 | territory measures | 0.897 | 0.920 | 0.935 |
| STAT | 80 | components | 0.798 | 0.802 | 0.827 |
| STAT | 80 | residence total | 0.766 | 0.767 | 0.762 |
| STAT | 80 | territory measures | 0.770 | 0.799 | 0.809 |
| STAT | 95 | components | 0.909 | 0.934 | 0.945 |
| STAT | 95 | residence total | 0.881 | 0.900 | 0.898 |
| STAT | 95 | territory measures | 0.901 | 0.922 | 0.933 |

### Table S32. Coverage of the residence totals of the 110 system economies: pooled split conformal against ACI and uncalibrated intervals

Source file: `results/conformal_residence/coverage_residence_totals.csv`

Calibration origins 2023-12 ... 2025-06, horizons 1-12, main window. Bootstrap: circular moving blocks of 6 origins, 2,000 resamples. Width ratio = median per-cell width relative to the ACI interval. World = World residence total alone.

| track | interval | level | n cells | coverage pooled | boot lo95 | boot hi95 | median width ratio to aci | world coverage |
|---|---|---|---|---|---|---|---|---|
| FM | ACI γ 0.05 (main) | 80 | 21,670 | 0.750 | 0.702 | 0.784 | 1.000 | 0.741 |
| FM | ACI γ 0.05 (main) | 95 | 21,670 | 0.870 | 0.842 | 0.889 | 1.000 | 0.812 |
| FM | pooled split conformal | 80 | 21,670 | 0.802 | 0.772 | 0.824 | 1.176 | 0.878 |
| FM | pooled split conformal | 95 | 21,670 | 0.953 | 0.941 | 0.963 | 1.515 | 1.000 |
| FM | uncalibrated, reconciled samples | 80 | 21,670 | 0.656 | 0.626 | 0.677 | 0.832 | 0.523 |
| FM | uncalibrated, reconciled samples | 95 | 21,670 | 0.835 | 0.813 | 0.852 | 0.877 | 0.746 |
| STAT | ACI γ 0.05 (main) | 80 | 21,670 | 0.764 | 0.743 | 0.778 | 1.000 | 0.761 |
| STAT | ACI γ 0.05 (main) | 95 | 21,670 | 0.895 | 0.880 | 0.906 | 1.000 | 0.812 |
| STAT | pooled split conformal | 80 | 21,670 | 0.813 | 0.787 | 0.831 | 1.121 | 0.919 |
| STAT | pooled split conformal | 95 | 21,670 | 0.954 | 0.943 | 0.961 | 1.384 | 0.995 |
| STAT | uncalibrated, reconciled samples | 80 | 21,670 | 0.716 | 0.697 | 0.730 | 0.920 | 0.492 |
| STAT | uncalibrated, reconciled samples | 95 | 21,670 | 0.875 | 0.858 | 0.887 | 0.937 | 0.731 |

### Table S33. Coverage of the residence totals by horizon

Source file: `results/conformal_residence/coverage_by_horizon.csv`

| track | interval | level | h 1 | h 3 | h 6 | h 12 |
|---|---|---|---|---|---|---|
| FM | ACI γ 0.05 (main) | 80 | 0.778 | 0.761 | 0.749 | 0.698 |
| FM | ACI γ 0.05 (main) | 95 | 0.926 | 0.883 | 0.855 | 0.852 |
| FM | pooled split conformal | 80 | 0.785 | 0.794 | 0.795 | 0.838 |
| FM | pooled split conformal | 95 | 0.942 | 0.949 | 0.954 | 0.964 |
| FM | uncalibrated, reconciled samples | 80 | 0.748 | 0.675 | 0.630 | 0.612 |
| FM | uncalibrated, reconciled samples | 95 | 0.906 | 0.846 | 0.821 | 0.816 |
| STAT | ACI γ 0.05 (main) | 80 | 0.766 | 0.763 | 0.755 | 0.805 |
| STAT | ACI γ 0.05 (main) | 95 | 0.913 | 0.895 | 0.883 | 0.910 |
| STAT | pooled split conformal | 80 | 0.791 | 0.811 | 0.809 | 0.859 |
| STAT | pooled split conformal | 95 | 0.943 | 0.954 | 0.946 | 0.973 |
| STAT | uncalibrated, reconciled samples | 80 | 0.707 | 0.701 | 0.706 | 0.778 |
| STAT | uncalibrated, reconciled samples | 95 | 0.875 | 0.866 | 0.862 | 0.902 |

### Table S34. Table 4 quantities, samples with negatives set to zero against exactly reconciled samples, all horizons

Source file: `results/exact_eval/T4_side_by_side.csv`

By horizon in the source file. Coverage = mean over series; ACI γ 0.05, nested.

| quantity | FM zeroed | FM exact | STAT zeroed | STAT exact |
|---|---|---|---|---|
| ES mean rec | 72.204 | 72.267 | 101.164 | 112.629 |
| ES median rec | 41.068 | 41.098 | 43.682 | 45.848 |
| VS mean rec | 2,927.040 | 2,921.480 | 12,831.400 | 17,928.900 |
| VS median rec | 2,464.190 | 2,468.430 | 2,743.320 | 2,954.780 |
| cov80 aci | 0.805 | 0.801 | 0.830 | 0.802 |
| cov80 uncal | 0.746 | 0.737 | 0.803 | 0.763 |
| cov95 aci | 0.923 | 0.920 | 0.935 | 0.923 |
| cov95 uncal | 0.899 | 0.894 | 0.919 | 0.903 |
| sCRPS rec median | 0.562 | 0.560 | 0.596 | 0.754 |
| sCRPS rec weighted | 0.590 | 0.591 | 0.598 | 0.803 |
| sWinkler80 aci | 3.516 | 3.520 | 3.665 | 6.229 |
| sWinkler95 aci | 5.108 | 5.142 | 5.187 | 8.552 |

### Table S35. FM minus STAT weighted scaled CRPS, origin-block bootstrap, zeroed and exact samples

Source file: `results/exact_eval/fm_minus_stat_crps_bootstrap.csv`

| samples | horizon | FM | STAT | difference | lo95 | hi95 | share below zero |
|---|---|---|---|---|---|---|---|
| zeroed | all | 0.590 | 0.598 | -0.008 | -0.042 | 0.020 | 0.692 |
| zeroed | 1 | 0.351 | 0.350 | 0.001 | -0.020 | 0.014 | 0.576 |
| zeroed | 3 | 0.494 | 0.505 | -0.011 | -0.038 | 0.007 | 0.908 |
| zeroed | 6 | 0.613 | 0.615 | -0.003 | -0.021 | 0.017 | 0.588 |
| zeroed | 12 | 0.730 | 0.715 | 0.015 | -0.032 | 0.076 | 0.301 |
| exact | all | 0.591 | 0.803 | -0.212 | -0.553 | -0.016 | 0.991 |
| exact | 1 | 0.352 | 0.353 | -0.001 | -0.025 | 0.012 | 0.692 |
| exact | 3 | 0.495 | 0.520 | -0.025 | -0.060 | -0.003 | 0.988 |
| exact | 6 | 0.614 | 0.668 | -0.055 | -0.125 | -0.006 | 0.990 |
| exact | 12 | 0.732 | 0.991 | -0.259 | -0.520 | -0.049 | 0.990 |

### Table S36. Weighted scaled CRPS with and without origin 2024-09 (diagnostic only, no origin is excluded)

Source files: `results/exact_eval/crps_without_2024-09.csv` (the table); `results/exact_eval/crps_change_by_origin.csv` (the values at origin 2024-09 alone in the note)

At origin 2024-09 alone, the weighted scaled CRPS of STAT is 0.84 with zeroed and 5.31 with exact samples.

| track | samples | all origins | without 2024 09 |
|---|---|---|---|
| FM | zeroed | 0.590 | 0.594 |
| FM | exact | 0.591 | 0.595 |
| STAT | zeroed | 0.598 | 0.590 |
| STAT | exact | 0.803 | 0.653 |

### Table S37. Coverage events with the outcome on an interval bound within 1e-9 t/day

Source files: `results/prob_main_exact/track=FM/aci_tie_events.csv` (FM rows); `results/prob_main_exact/track=STAT/aci_tie_events.csv` (STAT rows)

Coverage definition unchanged; the tie share bounds the effect of ties on coverage.

| track | version | level | γ | events | tie events | y zero events | tie share % |
|---|---|---|---|---|---|---|---|
| FM | exact | 80 | 0.010 | 416,640 | 4,488 | 6,195 | 1.08 |
| FM | exact | 80 | 0.050 | 416,640 | 4,866 | 6,195 | 1.17 |
| FM | exact | 95 | 0.010 | 416,640 | 5,717 | 6,195 | 1.37 |
| FM | exact | 95 | 0.050 | 416,640 | 5,786 | 6,195 | 1.39 |
| FM | zeroed, regenerated | 80 | 0.010 | 416,640 | 4,583 | 6,195 | 1.10 |
| FM | zeroed, regenerated | 80 | 0.050 | 416,640 | 4,906 | 6,195 | 1.18 |
| FM | zeroed, regenerated | 95 | 0.010 | 416,640 | 5,755 | 6,195 | 1.38 |
| FM | zeroed, regenerated | 95 | 0.050 | 416,640 | 5,811 | 6,195 | 1.39 |
| FM | zeroed, as stored | 80 | 0.010 | 416,640 | 4,583 | 6,195 | 1.10 |
| FM | zeroed, as stored | 80 | 0.050 | 416,640 | 4,906 | 6,195 | 1.18 |
| FM | zeroed, as stored | 95 | 0.010 | 416,640 | 5,755 | 6,195 | 1.38 |
| FM | zeroed, as stored | 95 | 0.050 | 416,640 | 5,811 | 6,195 | 1.39 |
| STAT | exact | 80 | 0.010 | 416,640 | 5,522 | 6,195 | 1.33 |
| STAT | exact | 80 | 0.050 | 416,640 | 5,591 | 6,195 | 1.34 |
| STAT | exact | 95 | 0.010 | 416,640 | 5,918 | 6,195 | 1.42 |
| STAT | exact | 95 | 0.050 | 416,640 | 5,943 | 6,195 | 1.43 |
| STAT | zeroed, regenerated | 80 | 0.010 | 416,640 | 5,518 | 6,195 | 1.32 |
| STAT | zeroed, regenerated | 80 | 0.050 | 416,640 | 5,587 | 6,195 | 1.34 |
| STAT | zeroed, regenerated | 95 | 0.010 | 416,640 | 5,924 | 6,195 | 1.42 |
| STAT | zeroed, regenerated | 95 | 0.050 | 416,640 | 5,949 | 6,195 | 1.43 |
| STAT | zeroed, as stored | 80 | 0.010 | 416,640 | 5,518 | 6,195 | 1.32 |
| STAT | zeroed, as stored | 80 | 0.050 | 416,640 | 5,587 | 6,195 | 1.34 |
| STAT | zeroed, as stored | 95 | 0.010 | 416,640 | 5,924 | 6,195 | 1.42 |
| STAT | zeroed, as stored | 95 | 0.050 | 416,640 | 5,949 | 6,195 | 1.43 |

### Table S38. Real-time test from origin 2026-06 (targets July to September 2026): World and the five largest economies

Source files: `results/paper/T5b_realtime_units.csv` (the table); `results/paper/T5a_realtime_eval_state.csv` (the coverage of the 110 economies with the state of the evaluation recursion, in the note); `results/paper/T5b_realtime_units_eval_state.csv` (the JPN and USA coverage with that state, in the note)

Published residence totals; calibrated (ACI) intervals, 80% nested inside 95%, from exactly reconciled samples. Coverage is over three months. The intervals are those of the final forecasts (ACI state over all origins up to May 2026). With the state of the evaluation recursion alone, the 80% coverage of the 110 economies is 85.8% (FM) and 79.4% (STAT), and the 80% coverage of JPN (both tracks) and of the USA (STAT) is 0.667.

| unit | MASE FM | MASE STAT | 80% coverage FM | 80% coverage STAT | 95% coverage FM | 95% coverage STAT |
|---|---|---|---|---|---|---|
| World | 0.229 | 0.260 | 1.000 | 1.000 | 1.000 | 1.000 |
| CHN | 0.462 | 0.435 | 1.000 | 1.000 | 1.000 | 1.000 |
| GRC | 0.642 | 0.353 | 1.000 | 1.000 | 1.000 | 1.000 |
| SGP | 0.534 | 0.596 | 0.667 | 0.667 | 1.000 | 1.000 |
| JPN | 0.926 | 0.744 | 1.000 | 1.000 | 1.000 | 1.000 |
| USA | 0.174 | 0.131 | 1.000 | 1.000 | 1.000 | 1.000 |

### Table S39. Paired comparison of FM and STAT at the real-time origin 2026-06 (published residence totals, 110 economies, 2026-07 to 2026-09)

Source file: `results/vintage/paired_2026-06.csv`

Per-economy MASE = mean over the three horizons. Bootstrap: 2,000 resamples of economies, seed 42.

| d variant | forecasts | median MASE FM | median MASE STAT | median diff FM minus STAT | boot95 lo | boot95 hi | share FM better | wilcoxon stat | wilcoxon p |
|---|---|---|---|---|---|---|---|---|---|
| (1 - d) correction | track (reconciled MinT) | 0.421 | 0.502 | -0.027 | -0.062 | 0.019 | 0.545 | 2,786 | 0.427 |
| (1 - d) correction | base (TimesFM 2.5 vs ETS) | 0.426 | 0.461 | 0.004 | -0.026 | 0.042 | 0.491 | 2,817 | 0.482 |
| d = 0 | track (reconciled MinT) | 0.421 | 0.502 | -0.027 | -0.062 | 0.024 | 0.545 | 2,806 | 0.462 |
| d = 0 | base (TimesFM 2.5 vs ETS) | 0.426 | 0.466 | 0.005 | -0.026 | 0.044 | 0.482 | 2,794 | 0.441 |

### Table S40. Pooled split-conformal intervals in the final forecasts: median ratio of their width to that of the ACI intervals

Source file: `results/conformal_s1/width_ratio.csv`

Ratios against the ACI intervals of the final forecasts, calibrated on exactly reconciled samples.

Same method and pool of errors as the evaluated real-time intervals (FM scaled absolute errors from the rolling origins December 2022 to June 2025, by horizon and measure group); centred on the forecast and clipped at zero. With the extended calibration origins July 2025 to May 2026 added to the pool, the ratios for the economies' residence totals become 1.02 and 1.37.

| series | cells | 80% | 95% |
|---|---|---|---|
| Residence totals, 110 system economies | 1,320 | 0.99 | 1.33 |
| World residence total | 12 | 1.10 | 1.52 |

### Table S41. H* for every variant of the skill test

Source file: `results/paper/T6b_hstar_variants.csv`

H* = first horizon at which the lower bound of s(h) (80%, 90%, 95% columns) is at or below zero, or at which the one-sided HAC test (Newey-West, lag h-1) no longer rejects at 5% ('test'). >36: the bound stays above zero up to h = 36. Main row: block 6, 95%.

| main | group | reference | variant | origins | test | FM 80% | FM 90% | FM 95% | FM test | STAT 80% | STAT 90% | STAT 95% | STAT test |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| yes | all | persistence | weighted | all origins | block 6 (main) | >36 | 23 | 23 | – | 6 | 6 | 5 | – |
| – | all | persistence | weighted | all origins | block 3 | >36 | 31 | 24 | – | 7 | 6 | 6 | – |
| – | all | persistence | weighted | all origins | block 12 | >36 | >36 | 24 | – | 8 | 7 | 6 | – |
| – | all | persistence | weighted | all origins | i.i.d. | >36 | >36 | 31 | – | 8 | 8 | 8 | – |
| – | all | persistence | weighted | all origins | HAC test, 5% | – | – | – | 31 | – | – | – | 6 |
| – | all | persistence | weighted | (i) 13 origins 2021-12..2022-12 | block 6 (main) | 8 | 8 | 7 | – | 6 | 6 | 6 | – |
| – | all | persistence | weighted | (i) 13 origins 2021-12..2022-12 | HAC test, 5% | – | – | – | 8 | – | – | – | 6 |
| – | all | persistence | weighted | (ii) without 2021-12..2022-10 | block 6 (main) | >36 | >36 | 6 | – | 6 | 5 | 4 | – |
| – | all | persistence | weighted | (ii) without 2021-12..2022-10 | HAC test, 5% | – | – | – | 34 | – | – | – | 5 |
| – | all | persistence | median | all origins | block 6 (main) | >36 | >36 | >36 | – | 5 | 5 | 4 | – |
| – | all | persistence | median | (i) 13 origins 2021-12..2022-12 | block 6 (main) | >36 | >36 | >36 | – | 5 | 5 | 4 | – |
| – | all | persistence | median | (ii) without 2021-12..2022-10 | block 6 (main) | >36 | >36 | 25 | – | 5 | 5 | 4 | – |
| – | all | seasonal naive | weighted | all origins | block 6 (main) | 12 | 12 | 11 | – | 9 | 8 | 7 | – |
| – | all | seasonal naive | weighted | all origins | HAC test, 5% | – | – | – | 11 | – | – | – | 9 |
| – | all | seasonal naive | median | all origins | block 6 (main) | >36 | >36 | 36 | – | 9 | 8 | 7 | – |
| – | residence totals | persistence | weighted | all origins | block 6 (main) | 28 | 27 | 27 | – | >36 | 27 | 27 | – |
| – | residence totals | persistence | weighted | all origins | HAC test, 5% | – | – | – | 27 | – | – | – | >36 |
| – | residence totals | seasonal naive | weighted | all origins | block 6 (main) | 10 | 10 | 10 | – | 11 | 10 | 10 | – |
| – | residence totals | seasonal naive | weighted | all origins | HAC test, 5% | – | – | – | 10 | – | – | – | 10 |
| – | territory totals | persistence | weighted | all origins | block 6 (main) | 7 | 6 | 6 | – | 5 | 4 | 4 | – |
| – | territory totals | persistence | weighted | all origins | HAC test, 5% | – | – | – | 6 | – | – | – | 4 |
| – | territory totals | seasonal naive | weighted | all origins | block 6 (main) | >36 | 24 | 23 | – | 7 | 6 | 6 | – |
| – | territory totals | seasonal naive | weighted | all origins | HAC test, 5% | – | – | – | >36 | – | – | – | 7 |

### Table S42. Skill and forecast horizon against persistence and against persistence with a linear drift

Source files: `results/skill_drift/hstar_table.csv` (the table); `results/skill_drift/skill.csv` (the intervals of s(1) in the note)

s(1) with its 95% block-bootstrap interval: FM 0.397 (0.305 to 0.478), STAT 0.408 (0.303 to 0.492).

Drift reference: the persistence reference plus h·(m1 − m2)/12, where m1 and m2 are the means of the last 12 months and of the 12 months before them; applied without clipping (4.7% of its values are negative). Otherwise as the main setting: 43 origins, moving-block bootstrap with blocks of 6 (1,000 resamples) and the one-sided test with Newey–West standard errors (HAC).

| track | reference | skill | s(1) | s(6) | s(12) | s(24) | s(36) | H* 95% | H* 90% | H* 80% | H* HAC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FM | persistence | weighted | 0.397 | 0.128 | 0.136 | 0.094 | 0.062 | 23 | 23 | >36 | 31 |
| FM | persistence | median | 0.291 | 0.092 | 0.134 | 0.130 | 0.089 | >36 | >36 | >36 | – |
| FM | persistence with drift | weighted | 0.392 | 0.171 | 0.263 | 0.374 | 0.457 | >36 | >36 | >36 | >36 |
| FM | persistence with drift | median | 0.292 | 0.165 | 0.290 | 0.477 | 0.562 | >36 | >36 | >36 | – |
| STAT | persistence | weighted | 0.408 | 0.112 | 0.034 | −0.021 | −0.132 | 5 | 6 | 6 | 6 |
| STAT | persistence | median | 0.273 | 0.038 | 0.009 | −0.057 | −0.252 | 4 | 5 | 5 | – |
| STAT | persistence with drift | weighted | 0.403 | 0.156 | 0.176 | 0.294 | 0.344 | 5 | >36 | >36 | >36 |
| STAT | persistence with drift | median | 0.274 | 0.115 | 0.187 | 0.364 | 0.398 | >36 | >36 | >36 | – |

### Table S43. Final forecasts at origin 2026-06: share of series × horizon cells whose calibration level reached its floor (%)

Source file: `results/paper/S1a_aci_floor_flags.csv`

STAT cells cover horizons 1 to 7, the horizons for which its final samples were drawn. The output file of the final forecasts contains the FM track only.

ACI with γ = 0.05; floor of the working miscoverage level 0.001. 'At floor, 95% only': the 95% interval equals the range of the 500 reconciled samples; '80% only': after nesting, the 80% interval equals the 95% interval. The world residence total is never at the floor.

| track | group | not at floor | at floor, 95% only | at floor, 80% and 95% | at floor, 80% only | cells |
|---|---|---|---|---|---|---|
| FM | residence total | 82.3 | 7.9 | 7.4 | 2.5 | 1,344 |
| FM | territory measures | 62.6 | 18.5 | 16.3 | 2.6 | 2,688 |
| FM | components (A–G) | 77.9 | 13.4 | 7.0 | 1.7 | 9,408 |
| STAT | residence total | 85.6 | 8.2 | 5.1 | 1.2 | 784 |
| STAT | territory measures | 67.9 | 17.0 | 13.2 | 2.0 | 1,568 |
| STAT | components (A–G) | 80.5 | 12.7 | 5.6 | 1.2 | 5,488 |

### Table S44. W by origin: shrinkage intensity, condition numbers on a unit diagonal, floored and dead series

Source files: `results/w_diagnostics/w_condition_by_origin.csv` (shrinkage intensity and condition numbers); `results/w_diagnostics/zero_variance_series.csv` (floored and dead series)

Raw condition numbers (about 1e14, reflecting scale) are in the source file. Floored = residual variance below 1e-4 (t/day)^2; dead = fixed at zero after 24 zero months.

| origin | FM lambda | FM cond(W), unit diag | FM cond(C W Cᵀ), unit diag | FM floored | FM dead | STAT lambda | STAT cond(W), unit diag | STAT cond(C W Cᵀ), unit diag | STAT floored | STAT dead |
|---|---|---|---|---|---|---|---|---|---|---|
| 2022-12 | 0.740 | 55.0 | 103.8 | 1 | 3 | 0.804 | 32.3 | 89.7 | 1 | 3 |
| 2023-01 | 0.738 | 54.3 | 103.0 | 1 | 3 | 0.803 | 31.8 | 127.8 | 1 | 3 |
| 2023-02 | 0.727 | 57.5 | 102.9 | 1 | 3 | 0.796 | 33.0 | 160.9 | 1 | 3 |
| 2023-03 | 0.723 | 57.9 | 102.1 | 1 | 3 | 0.793 | 32.8 | 120.7 | 1 | 3 |
| 2023-04 | 0.721 | 57.7 | 101.5 | 1 | 3 | 0.793 | 32.3 | 216.4 | 1 | 3 |
| 2023-05 | 0.719 | 57.5 | 100.9 | 1 | 3 | 0.790 | 32.3 | 119.0 | 1 | 3 |
| 2023-06 | 0.715 | 58.0 | 100.6 | 1 | 3 | 0.793 | 30.9 | 134.2 | 1 | 3 |
| 2023-07 | 0.713 | 58.3 | 100.2 | 1 | 3 | 0.792 | 30.8 | 116.3 | 1 | 3 |
| 2023-08 | 0.711 | 58.4 | 100.1 | 1 | 3 | 0.792 | 30.5 | 106.0 | 1 | 3 |
| 2023-09 | 0.704 | 60.1 | 99.7 | 1 | 2 | 0.785 | 31.8 | 118.4 | 1 | 2 |
| 2023-10 | 0.695 | 62.8 | 98.2 | 1 | 2 | 0.782 | 32.0 | 100.2 | 1 | 2 |
| 2023-11 | 0.672 | 78.9 | 94.9 | 1 | 2 | 0.755 | 44.3 | 153.8 | 1 | 2 |
| 2023-12 | 0.667 | 80.1 | 96.8 | 1 | 3 | 0.711 | 59.0 | 99.6 | 1 | 3 |
| 2024-01 | 0.667 | 77.4 | 94.8 | 1 | 2 | 0.706 | 59.0 | 97.6 | 1 | 2 |
| 2024-02 | 0.658 | 80.0 | 94.6 | 1 | 2 | 0.695 | 62.5 | 116.7 | 1 | 2 |
| 2024-03 | 0.655 | 80.0 | 94.0 | 1 | 2 | 0.694 | 62.1 | 113.9 | 1 | 2 |
| 2024-04 | 0.652 | 80.1 | 94.0 | 1 | 2 | 0.690 | 62.8 | 118.3 | 1 | 2 |
| 2024-05 | 0.649 | 80.4 | 94.5 | 1 | 2 | 0.690 | 61.9 | 115.5 | 1 | 2 |
| 2024-06 | 0.645 | 81.2 | 94.3 | 1 | 1 | 0.688 | 61.9 | 111.1 | 1 | 1 |
| 2024-07 | 0.643 | 81.2 | 94.6 | 1 | 2 | 0.687 | 62.1 | 113.6 | 1 | 2 |
| 2024-08 | 0.638 | 83.0 | 93.1 | 1 | 2 | 0.680 | 64.1 | 114.4 | 1 | 2 |
| 2024-09 | 0.633 | 84.3 | 92.2 | 1 | 2 | 0.678 | 64.2 | 120.4 | 1 | 2 |
| 2024-10 | 0.626 | 86.1 | 92.1 | 1 | 2 | 0.665 | 68.1 | 108.4 | 1 | 2 |
| 2024-11 | 0.625 | 85.9 | 91.3 | 1 | 2 | 0.664 | 67.6 | 109.4 | 1 | 2 |
| 2024-12 | 0.622 | 86.3 | 91.3 | 1 | 2 | 0.661 | 68.1 | 111.4 | 1 | 2 |
| 2025-01 | 0.625 | 83.9 | 89.8 | 1 | 2 | 0.667 | 64.8 | 123.9 | 1 | 2 |
| 2025-02 | 0.620 | 85.4 | 90.5 | 1 | 2 | 0.661 | 66.9 | 131.8 | 1 | 2 |
| 2025-03 | 0.619 | 85.4 | 90.3 | 0 | 1 | 0.658 | 67.8 | 113.1 | 0 | 1 |
| 2025-04 | 0.618 | 85.1 | 90.2 | 0 | 1 | 0.656 | 68.0 | 109.5 | 0 | 1 |
| 2025-05 | 0.616 | 85.3 | 90.0 | 0 | 0 | 0.656 | 67.6 | 122.4 | 0 | 0 |
| 2025-06 | 0.614 | 85.4 | 89.8 | 0 | 0 | 0.655 | 67.2 | 115.2 | 0 | 0 |

### Table S45. Dead series that later become non-zero

Source files: `results/w_diagnostics/dead_series_revival_summary.csv` (the table); `results/w_diagnostics/dead_series_revival.csv` (per-cell detail and the largest value in the note)

Largest value reached after revival: 15.5 t/day; per-cell detail in the second source file.

| origins | dead cells | distinct dead series | cells nonzero within 12m | distinct series nonzero within 12m | series nonzero |
|---|---|---|---|---|---|
| 31 | 65 | 5 | 49 | 5 | CPV/D MDG/B MDG/D MRT/D MRT/E |
