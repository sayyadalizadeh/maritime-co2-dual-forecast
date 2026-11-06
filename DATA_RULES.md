# Data rules

The clean files in `data/clean/` are produced from one raw download by `src/prepare_data.py` and `src/build_panel.py`. Nothing is interpolated or smoothed; the only values filled are structural zeros (Section 6).

## 1. Source

**OECD, Maritime transport CO2 emissions (experimental)**

| Item | Value |
|---|---|
| Dataflow | `OECD.SDD.NAD.SEEA:DSD_MARITIME_TRANSPORT@DF_MARITIME_TRANSPORT(2.0)` |
| Last updated by the OECD | 14 October 2026 |
| Downloaded | 31 October 2026, complete (unfiltered) CSV, 427,861 observations |
| File | `data/raw/oecd_maritime_co2_raw_2026-10-31.csv`, SHA-256 `88fb19b9645e4452d26e88cf8891467a3732d380a54f1b445f3735849fdf6002` |
| API query (monthly from 2019) | `https://sdmx.oecd.org/public/rest/data/OECD.SDD.NAD.SEEA,DSD_MARITIME_TRANSPORT@DF_MARITIME_TRANSPORT,/.M........?startPeriod=2019-01&dimensionAtObservation=AllDimensions` |
| Methodology | Clarke, Chan, Dequeljoe, Kim and Barahona (2023), *CO2 emissions from global shipping: A new experimental database*, OECD |
| Coverage | AIS-based estimates for vessels above 300 GT, from January 2019 |

The OECD revises the dataset. The vintage comparison uses the October 2025 release as well (`data/external/oecd_maritime_v1_2025-10-06.xlsx`).

## 2. Files

| File | Content | Rows |
|---|---|---|
| `oecd_maritime_co2_monthly_v1.csv` | Tidy monthly panel: one row per area × vessel type × component × month | 169,184 |
| `oecd_areas_v1.csv` | Area lookup: code, name, type, OECD membership | 219 |
| `oecd_series_summary_v1.csv` | One row per series: coverage, gaps, 2025 total, core-sample flag | 2,304 |
| `panel_system_v2.parquet`, `panel_system_wide_v2.parquet` | System panel (110 economies, Residual, World; long and wide), per-day values, with structural zeros, the component-sum residence total and the discrepancy node | - |
| `panel_outside_v2.parquet` | Published totals of the economies outside the system | - |
| `realtime_res_total_pub_2026-07_2026-09_v2.parquet` | Published residence totals of July to September 2026 (scored in the test at origin 2026-06) | - |

## 3. Variables

**oecd_maritime_co2_monthly_v1.csv**

| Variable | Meaning |
|---|---|
| `area_code` | ISO3 code, or `W` (World), `OECD` (OECD aggregate), `W_O` (Other countries) |
| `vessel` | `ALL_VESSELS` for every area; the 19 IMO ship types exist only for `W` and `OECD` |
| `component` | One of the 10 emission measures in Section 4 |
| `component_letter` | OECD letter code (A to G, or the aggregate formula) |
| `basis` | `residence`, `territory` or `component` |
| `period` | Month, `YYYY-MM` |
| `days_in_month` | Calendar days in the month |
| `co2_t` | CO2 in tonnes, as published |
| `co2_t_per_day` | `co2_t / days_in_month` (calendar-normalised value) |
| `flag_sep2025` | True for September 2025 (see Section 7) |
| `flag_after_component_coverage` | True for residence totals in months after the area's last published component (July to September 2026) |
| `flag_identity_mismatch` | True where the residence total differs from A+B+D+E+F by more than 1% |

**oecd_series_summary_v1.csv**: `start`, `end`, `n_months`, `n_missing_within_span`, `longest_gap_months`, `complete_2019_01_to_2026_06`, `co2_t_2025`, `core_series`.

## 4. Emission measures

| Code | Letter | Meaning |
|---|---|---|
| `RES_DOM_IN` | A | Domestic voyages in the country, resident operators |
| `RES_DOM_OUT` | B | Domestic voyages in other countries, resident operators |
| `NRES_DOM_IN` | C | Domestic voyages in the country, non-resident operators |
| `RES_INT_FROM` | D | International voyages departing the country, resident operators |
| `RES_INT_TO` | E | International voyages arriving in the country, resident operators |
| `RES_INT_OUT` | F | International voyages between other countries, resident operators |
| `NRES_INT_FROM` | G | International voyages departing the country, non-resident operators |
| `RES_TOTAL` | A+B+D+E+F | Residence basis: SEEA air emission accounts, maritime transport (ISIC H50); emissions allocated to the country where the operating company is resident |
| `TER_DOM` | A+C | Territory basis: domestic navigation (UNFCCC inventories) |
| `TER_INT` | D+G | Territory basis: international voyages departing the country (UNFCCC memo item) |

Inside the system the residence total is the component sum, `RES_TOTAL_SYS` = A+B+D+E+F. The published total (`RES_TOTAL_PUB`) is kept for reporting and for the tests on later-published data, where it is scored with the forecast divided by (1 - d), d being the mean relative discrepancy (published minus component sum, over published) in the 6 months up to the origin.

## 5. Preparation rules

1. Monthly data only; quarterly and annual series are not used.
2. The "All vessels except fishing" series are excluded. Fishing-vessel emissions can be rebuilt from the raw file as All vessels minus All vessels except fishing.
3. No interpolation, filling or smoothing. Missing months stay missing (except the structural zeros of Section 6).
4. OECD values are not calendar-adjusted. Monthly totals partly reflect the number of days (February is about 10% lower for that reason alone). The models use `co2_t_per_day`.
5. **Aggregates.** For territory-based measures, group totals are built by summing economies. The published OECD aggregate is not the sum of its members for these measures (in 2024 members sum to 87% of the published `TER_DOM` aggregate and 62% of `TER_INT`, driven by components C and G). For residence measures the members sum to 99% of the published aggregate. The World aggregate is consistent: economies sum to 98.8% (residence), 99.3% (`TER_DOM`) and 95.6% (`TER_INT`) of World.
6. **OECD membership:** 38 members. Six of them (Austria, Costa Rica, Czechia, Hungary, Israel, Slovak Republic) have no published residence series and only a few territory-basis months in 2026. Their residence emissions are presumably inside `W_O` (Other countries, about 0.8% of World).

## 6. The constraint system

- **System economies:** the economies with all seven components available from 2019-01 to 2026-06 after the structural-zero rule (110). The other economies enter through the Residual node; their published totals are forecast as outside-system series.
- **Structural zeros (`src/structural_zeros.py`).** A missing component value is set to zero only when every published total that contains it (`RES_TOTAL` for A, B, D, E, F; `TER_DOM` for A, C; `TER_INT` for D, G) is published in that month and equals the sum of the other published components within 0.01% or 1 t. Every such zero is logged (`results/structural_zeros.csv`). Nothing else is filled. Switzerland stays outside the system (A, B and F are unpublished for all of 2025).
- **Discrepancy node.** `DELTA_INT` = E(World) + F(World) - G(World) (about 0.1% of G) enters the system as a World-level series, so the world balance holds exactly; B(World) = C(World) holds exactly without a node. In forecasting, `DELTA_INT` is r times the reconciled G(World), r being the mean ratio `DELTA_INT` / G(World) over the 12 months up to the origin.
- **Residence total.** The published residence total differs from the component sum in some economy-months: it equals it exactly in 73.5% of economy-months, within 0.01% in 82.7% and within 1% in 99.05%; deviations are about twice as frequent in 2025 as in earlier years. The difference reaches 0.72% for World (2025-09) and 7.1% for the Residual (2026-06). It is documented, not modelled.

With these rules all constraints hold on the historical data to rounding (`src/constraints.py`).

## 7. Known data issues

- **September 2025 spike.** On the definition used in the paper (the September per-day value divided by the mean of the August and October per-day values, 110 system economies, 2025), the median ratio is 1.042 for the residence total (`RES_TOTAL_SYS`) and 1.055 for the territory total (`TER_DOM` + `TER_INT`); 58% (residence) and 64% (territory) of the economies are above 1.03. The 2019-2024 yearly medians are 0.998-1.004 (residence) and 0.976-1.006 (territory). The spike appears across all ship types, so it is most likely a processing artefact. Every evaluation is reported with and without September 2025 as a target month. Source: `results/paper/quoted_figures.csv`, item 4.
- **Domestic/international classification episodes.** Twice the OECD data move a large amount of emissions from international to domestic measures, mainly through B (`RES_DOM_OUT`) and C (`NRES_DOM_IN`, equal at World level) on one side and F (`RES_INT_OUT`) and G (`NRES_INT_FROM`) on the other, while World totals stay smooth. Figures are World per-day means over the episode against the same calendar months a year earlier (`src/episode_2022.py`, `results/episode_2022/`).
  - **2022 episode, 2021-12 to 2022-10.** The international share `TER_INT` / (`TER_DOM` + `TER_INT`) falls from 74.1% to 65.5% (minimum 64.4% in 2022-06), then recovers to 71.5% in 2022-11 and 73% in spring 2023. B = C rise by 188 kt/day (+74%), A by 45 kt/day; F falls by 167 and G by 173 kt/day; D and E hardly move. World totals rise by 2.5% on both bases, as in the surrounding months. Largest contributors: to the rise in `TER_DOM`, China (25% of the World change), the United States (10%), Japan, Panama and Korea (about 5% each), almost entirely through C; to the fall in `TER_INT`, Malaysia (26%), Australia (10%), the United States and Korea (8% each), almost entirely through G. Across economies the territory totals (`TER_DOM` + `TER_INT`) shift much more than the residence totals (sum of absolute changes 228 against 97 kt/day): the episode also moves emissions between economies on the territory basis. 11 of the 13 long-horizon origins (2021-12 to 2022-12) of the skill analysis lie inside the episode.
  - **2026 reclassification, from 2026-01.** The international share falls from 73.6% (2025-01 to 2025-06) to 68.3% (2026-01 to 2026-06), reaching 62.9% in June 2026; the shift grows during the half-year. B = C rise by 135 kt/day on average (+254 kt/day in June 2026 against June 2025), F falls by 75 (-202), G by 101 (-244), D and E by 21 and 25. World totals +1.1%. Largest contributors: China (17% of the `TER_DOM` rise) and the United Arab Emirates (21% of the `TER_INT` fall). For this reason the main evaluation window ends in 2025-12, and January to June 2026 is a separate stress test; the totals (`RES_TOTAL_SYS`, `TER_DOM` + `TER_INT`) are unaffected.
  - **Shorter dips.** The same rule (share at least 4 points below the same month of 2019) also flags 2023-09 to 2023-11 (minimum 62.9% in 2023-11) and 2024-09. Every January the share is about 3 to 6 points lower than in the neighbouring months, with B = C higher; January 2019 is affected too, so the rule does not flag 2022-01.
- **Residence total against components.** Identity A+B+D+E+F holds exactly in about 98% of country-months. Of the 273 mismatches, 193 belong to four small economies (Luxembourg, Sri Lanka, Myanmar, DPR Korea); the remaining 80 cluster in September to November 2025. The territory identities (A+C, D+G) hold exactly.
- **Release timing.** Residence totals run to September 2026; components run to June 2026.
- **Port-of-departure assignment.** In 2025 Malaysia shows 77.5 Mt on the territory basis, against 17.1 Mt for Singapore and 0.3 Mt for Hong Kong. This likely reflects how anchorages and ports are assigned (Hong Kong may be inside China). It is not handled and is reported as a limitation.
- **Short series.** 37 small areas have data only from 2026, mostly three months.
- **Ship types** are published only for World and the OECD aggregate, not by country.
- **Switzerland** has no `RES_INT_FROM` series (it has no seaports for these voyages); its other components are published.
- **Revisions.** Compared with the July 2025 release (dataflow version 1.0), overlapping monthly residence totals are almost unchanged (World revised up by 0.35% on average; median country revision 0.00%). The comparison with the October 2025 release is in supplementary Tables S46-S48.

## 8. Reproducing the files

```
python src/download_data.py --verify
python src/prepare_data.py --oecd data/raw/oecd_maritime_co2_raw_2026-10-31.csv --out data/clean
python src/checksums.py --raw data/raw/oecd_maritime_co2_raw_2026-10-31.csv --out results/checksums.csv
python src/build_panel.py --suffix v2 --structural-zeros
python src/constraints.py --suffix v2
```
