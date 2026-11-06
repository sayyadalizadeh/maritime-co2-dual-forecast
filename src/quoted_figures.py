"""
quoted_figures.py: the table of figures quoted in the paper text and the 2026 rows of the monthly world
international share table. Nothing is fitted; every value is recomputed from data/clean or read from a
stored result, and each row names its definition and source.

Items:
   1  2025 annual totals (Mt CO2, sum of the 12 published monthly values, September 2025 included as published) for
      Greece, Singapore, Switzerland and Malaysia: residence (published RES_TOTAL; component sum A+B+D+E+F where all
      components are available) and territory (A+C+D+G = TER_DOM + TER_INT, published); World: published residence
      total, component-sum residence total, territory total. A total is given only if all 12 months are published.
   2  2025 coverage of World by the 110 system economies (annual sums, as results/coverage_2025_v2.csv): residence
      (RES_TOTAL_SYS) and territory (TER_DOM + TER_INT); shares of system economy-months (2019-01 ... 2026-06) where the
      published residence total equals the component sum exactly (|relative difference| < 1e-6) and within 1%.
   3  calendar effect: World published residence total, monthly totals and per-day values each indexed to their
      2019-2025 mean = 100 ; February gap = per-day index minus total index, every year.
   4  September ratio = September per day / mean(August, October) per day, 110 system economies, residence
      (RES_TOTAL_SYS) and territory (TER_DOM + TER_INT) : median 2025, range of the yearly
      medians 2019-2024, share of economies with a ratio above 1.03 in 2025, pooled over 2019-2024 and the range of the
      yearly shares 2019-2024; World ratio 2025.
   5  residence-total discrepancy (published - component sum) / published (results/res_total_discrepancy.csv): the
      largest absolute value for World and its month; the same for Croatia and Portugal, with their months > 1%.
   6  OECD aggregate, 2024: the sum of the 38 members' published annual totals divided by the published OECD aggregate,
      TER_DOM, TER_INT and RES_TOTAL.
   7  Switzerland's published residence total 2025 as a share of the Residual residence total 2025 (RES_TOTAL_SYS and
      RES_TOTAL_PUB of the Residual node).
   8  World domestic share TER_DOM / (TER_DOM + TER_INT) in 2025-12, 2026-01, 2026-06; the move between 2025-12 and
      2026-06 in kt CO2 per day (change in TER_DOM and in TER_INT, and the share change times the 2026-06 territory
      total); the World territory total per day in those months.
   9  OLS (exact QP, nn_qp; and free) weighted and median MASE, system economies, main window, all horizons, both bases
      (results/supplementary/OLS_tracks.csv); constraint ablation: mean over horizons 1-12 of the
      per-horizon mean |World residence-territory gap| in % (results/ablation/violations_world_gap_by_horizon.csv),
      sets 2 and 3, both tracks.
  10  s(1) with its 95% interval (all origins, weighted skill against persistence, moving-block bootstrap of 6
      origins; results/skill/skill.csv), both tracks.
  11  World July 2026 forecast (FM track, S1): component-sum basis (W|RES_TOTAL_SYS, point and ACI 80% interval) and
      published basis (times 1 / (1 - d), d of the World at origin 2026-06; results/eval_realtime_exact_ext_nested).
  12  revision of the World monthly residence total, October 2025 vintage against October 2026 vintage
      (results/vintage/world_revisions.csv): minimum and maximum by year, 2022-2025.
  13  the one-step errors behind the foundation models' W (results/onestep_fm, results/onestep_fm_rt): first and
      last target month, shortest and longest context (months of history), per model and directory.
results/paper/intl_share_by_month_with_2026.csv is the monthly international share table of supplementary_analyses.py (item 13) with 2026 added
(January-June; the annual mean is over the published months, n_months gives their number). results/supplementary/intl_share_by_month.csv
(2019-2025) is not changed.

Outputs (--out, default results/paper): quoted_figures.csv, intl_share_by_month_with_2026.csv
Usage:
  python src/quoted_figures.py
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from runlog import RunLog, out_path

COMP = {"A": "RES_DOM_IN", "B": "RES_DOM_OUT", "C": "NRES_DOM_IN", "D": "RES_INT_FROM", "E": "RES_INT_TO",
        "F": "RES_INT_OUT", "G": "NRES_INT_FROM"}
MONTHLY = "data/clean/oecd_maritime_co2_monthly_v1.csv"
PANEL = "data/clean/panel_system_v2.parquet"
WIDE = "data/clean/panel_system_wide_v2.parquet"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/paper")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    log = RunLog("quoted_figures.py", vars(args))
    RES, OUT = Path(args.results), Path(args.out)
    rows = []

    def add(item, quantity, value, unit, definition, source):
        rows.append({"item": item, "quantity": quantity, "value": value, "unit": unit, "definition": definition,
                     "source": source})

    m = pd.read_csv(MONTHLY, dtype={"period": str})
    m = m[m.vessel == "ALL_VESSELS"]
    p = pd.read_parquet(PANEL)
    wide = pd.read_parquet(WIDE)

    # ---------------------------------------------------------------- 1
    def annual(area, comp, year="2025"):
        v = m[(m.area_code == area) & (m.component == comp) & (m.period.str[:4] == year)].co2_t
        return v.sum() / 1e6 if len(v) == 12 and v.notna().all() else np.nan, len(v)

    names = {"GRC": "Greece", "SGP": "Singapore", "CHE": "Switzerland", "MYS": "Malaysia"}
    for a, nm in names.items():
        res, n_res = annual(a, "RES_TOTAL")
        parts = [annual(a, COMP[k]) for k in "ABDEF"]
        csum = sum(v for v, _ in parts) if all(n == 12 for _, n in parts) else np.nan
        td, n_td = annual(a, "TER_DOM")
        ti, n_ti = annual(a, "TER_INT")
        add(1, f"{nm} residence total 2025, published", res, "Mt CO2", f"sum of published monthly RES_TOTAL ({n_res} months)", MONTHLY)
        add(1, f"{nm} residence total 2025, component sum A+B+D+E+F", csum, "Mt CO2",
            "sum over 2025 of A+B+D+E+F; NaN if a component month is unpublished "
            f"(months published: {', '.join(f'{k} {n}' for k, (_, n) in zip('ABDEF', parts))})", MONTHLY)
        add(1, f"{nm} territory total 2025 (A+C+D+G)", td + ti, "Mt CO2",
            f"sum of published monthly TER_DOM + TER_INT (months published: TER_DOM {n_td}, TER_INT {n_ti}); NaN if any "
            "month is unpublished", MONTHLY)
        if np.isnan(td + ti):
            add(1, f"{nm} TER_DOM 2025", td, "Mt CO2", f"sum of published monthly TER_DOM ({n_td} months)", MONTHLY)
    wres, _ = annual("W", "RES_TOTAL")
    wsum = sum(annual("W", COMP[k])[0] for k in "ABDEF")
    wter = annual("W", "TER_DOM")[0] + annual("W", "TER_INT")[0]
    add(1, "World residence total 2025, published", wres, "Mt CO2", "sum of published monthly RES_TOTAL of W", MONTHLY)
    add(1, "World residence total 2025, component sum A+B+D+E+F", wsum, "Mt CO2", "sum over 2025 of the published World components", MONTHLY)
    add(1, "World territory total 2025 (A+C+D+G)", wter, "Mt CO2", "sum of published monthly World TER_DOM + TER_INT", MONTHLY)

    # ---------------------------------------------------------------- 2
    y25 = p[p.period.str[:4] == "2025"].groupby(["unit_type", "unit", "measure"]).co2_t.sum()
    eco = y25.loc["economy"].unstack("measure")
    wld = y25.loc["world"].unstack("measure").iloc[0]
    n_eco = len(eco)
    add(2, f"coverage of World by the {n_eco} system economies 2025, residence", 100 * eco.RES_TOTAL_SYS.sum() / wld.RES_TOTAL_SYS,
        "%", "sum over system economies of annual RES_TOTAL_SYS / World RES_TOTAL_SYS (2025, co2_t)", PANEL)
    add(2, f"coverage of World by the {n_eco} system economies 2025, territory",
        100 * (eco.TER_DOM.sum() + eco.TER_INT.sum()) / (wld.TER_DOM + wld.TER_INT), "%",
        "sum over system economies of annual TER_DOM + TER_INT / World TER_DOM + TER_INT (2025, co2_t)", PANEL)
    add(2, f"coverage of World by the {n_eco} system economies 2025, published residence total",
        100 * eco.RES_TOTAL_PUB.sum() / wld.RES_TOTAL_PUB, "%", "as above with RES_TOTAL_PUB", PANEL)
    disc = pd.read_csv(RES / "res_total_discrepancy.csv", dtype={"period": str})
    de = disc[disc.unit_type == "economy"]
    add(2, "economy-months with published residence total = component sum, exactly", 100 * (de.rel_diff.abs() < 1e-6).mean(),
        "%", f"|published - component sum| / published < 1e-6; {len(de):,} system economy-months 2019-01 to 2026-06",
        "results/res_total_discrepancy.csv")
    add(2, "economy-months with published residence total = component sum, within 1%", 100 * (de.rel_diff.abs() <= 0.01).mean(),
        "%", "|published - component sum| / published <= 1%; same economy-months", "results/res_total_discrepancy.csv")
    add(2, "economy-months with published residence total = component sum, within 0.01%",
        100 * (de.rel_diff.abs() <= 1e-4).mean(), "%", "<= 0.01%; same economy-months", "results/res_total_discrepancy.csv")

    # ---------------------------------------------------------------- 3
    w = m[(m.area_code == "W") & (m.component == "RES_TOTAL")].set_index("period").sort_index()
    base = w.index < "2026-01"
    i_tot = w.co2_t / w.co2_t[base].mean() * 100
    i_day = w.co2_t_per_day / w.co2_t_per_day[base].mean() * 100
    gap = (i_day - i_tot)[w.index.str[5:7] == "02"]
    for per, g in gap.items():
        y = int(per[:4])
        leap = y % 4 == 0
        add(3, f"February {y} gap{' (leap year)' if leap else ''}", g, "index points",
            "per-day index minus monthly-total index of the World published residence total, both indexed to their "
            "2019-2025 mean = 100", MONTHLY)
    nl = gap[[int(x[:4]) % 4 != 0 for x in gap.index]]
    add(3, "February gap, non-leap years: min", nl.min(), "index points", f"over {', '.join(i[:4] for i in nl.index)}", MONTHLY)
    add(3, "February gap, non-leap years: max", nl.max(), "index points", f"over {', '.join(i[:4] for i in nl.index)}", MONTHLY)

    # ---------------------------------------------------------------- 4
    pd_ = p[p.measure.isin(["RES_TOTAL_SYS", "TER_DOM", "TER_INT", "RES_TOTAL_PUB"])] \
        .pivot_table(index="period", columns=["unit", "measure"], values="co2_t_per_day")
    units = sorted(p[p.unit_type == "economy"].unit.unique())
    rat = []
    for u in units + ["W"]:
        for basis, s in [("residence", pd_[(u, "RES_TOTAL_SYS")]), ("territory", pd_[(u, "TER_DOM")] + pd_[(u, "TER_INT")]),
                         ("residence (published)", pd_[(u, "RES_TOTAL_PUB")])]:
            for y in range(2019, 2026):
                den = (s[f"{y}-08"] + s[f"{y}-10"]) / 2
                rat.append({"unit": u, "basis": basis, "year": y, "ratio": s[f"{y}-09"] / den if den > 0 else np.nan})
    rat = pd.DataFrame(rat)
    for basis in ["residence", "territory", "residence (published)"]:
        r = rat[(rat.basis == basis) & (rat.unit != "W")]
        med = r.groupby("year").ratio.median()
        above = r.assign(a=r.ratio > 1.03).groupby("year").a.mean()
        early = r[r.year <= 2024]
        d = f"{len(units)} system economies; ratio = Sep per day / mean(Aug, Oct) per day; {basis} = " + \
            {"residence": "RES_TOTAL_SYS", "territory": "TER_DOM + TER_INT", "residence (published)": "RES_TOTAL_PUB"}[basis]
        add(4, f"September ratio 2025, median, {basis}", med[2025], "ratio", d, PANEL)
        add(4, f"September ratio, yearly medians 2019-2024, min, {basis}", med.loc[:2024].min(), "ratio", d, PANEL)
        add(4, f"September ratio, yearly medians 2019-2024, max, {basis}", med.loc[:2024].max(), "ratio", d, PANEL)
        add(4, f"share of economies with ratio > 1.03, 2025, {basis}", 100 * above[2025], "%", d, PANEL)
        add(4, f"share of economies with ratio > 1.03, pooled 2019-2024, {basis}", 100 * (early.ratio > 1.03).mean(), "%",
            d + "; economy-years 2019-2024 pooled", PANEL)
        add(4, f"share of economies with ratio > 1.03, yearly 2019-2024, min, {basis}", 100 * above.loc[:2024].min(), "%", d, PANEL)
        add(4, f"share of economies with ratio > 1.03, yearly 2019-2024, max, {basis}", 100 * above.loc[:2024].max(), "%", d, PANEL)
        add(4, f"World September ratio 2025, {basis}", rat[(rat.basis == basis) & (rat.unit == "W") & (rat.year == 2025)].ratio.iloc[0],
            "ratio", d.replace(f"{len(units)} system economies", "World"), PANEL)

    # ---------------------------------------------------------------- 5
    for u, nm in [("W", "World"), ("HRV", "Croatia"), ("PRT", "Portugal")]:
        d = disc[disc.unit == u].set_index("period")
        k = d.rel_diff.abs().idxmax()
        add(5, f"{nm}: largest |published - component sum| / published", 100 * d.rel_diff[k], "%",
            f"month {k}; published {d.published_t[k] / 1e6:.4f} Mt, component sum {d.component_sum_t[k] / 1e6:.4f} Mt; "
            f"months 2019-01 to 2026-06", "results/res_total_discrepancy.csv")
        big = d[d.rel_diff.abs() > 0.01]
        if u != "W":
            add(5, f"{nm}: months with |discrepancy| > 1%", len(big), "months",
                "; ".join(f"{i} {100 * v:+.2f}%" for i, v in big.rel_diff.items()) or "none", "results/res_total_discrepancy.csv")
    rk = disc[disc.unit_type == "economy"].assign(a=disc.rel_diff.abs()).groupby("unit").a.max().rank(ascending=False)
    add(5, "rank of Croatia / Portugal by largest |discrepancy| among system economies", f"{int(rk['HRV'])} / {int(rk['PRT'])}",
        "rank", f"out of {len(rk)} system economies", "results/res_total_discrepancy.csv")

    # ---------------------------------------------------------------- 6
    areas = pd.read_csv("data/clean/oecd_areas_v1.csv")
    OECD_MEMBERS = areas.area_code[areas.oecd_member].tolist()         # prepare_data.OECD_MEMBERS (38)
    assert len(OECD_MEMBERS) == 38
    m24 = m[m.period.str[:4] == "2024"]
    for comp in ["TER_DOM", "TER_INT", "RES_TOTAL"]:
        mem = m24[(m24.area_code.isin(OECD_MEMBERS)) & (m24.component == comp)].co2_t.sum()
        agg = m24[(m24.area_code == "OECD") & (m24.component == comp)].co2_t.sum()
        add(6, f"OECD members' sum / published OECD aggregate, 2024, {comp}", 100 * mem / agg, "%",
            f"sum of the {len(OECD_MEMBERS)} members' published monthly values in 2024 / sum of the published OECD aggregate "
            "(co2_t; members' unpublished months contribute nothing)", f"{MONTHLY}; data/clean/oecd_areas_v1.csv")

    # ---------------------------------------------------------------- 7
    che, _ = annual("CHE", "RES_TOTAL")
    resid = y25.loc["residual"].unstack("measure").iloc[0]
    add(7, "Switzerland's share of the Residual, residence, 2025 (Residual RES_TOTAL_SYS)", 100 * che * 1e6 / resid.RES_TOTAL_SYS,
        "%", "Switzerland's published annual RES_TOTAL / Residual RES_TOTAL_SYS (World minus the 110 system economies, "
        "component sums), 2025", f"{MONTHLY}; {PANEL}")
    add(7, "Switzerland's share of the Residual, residence, 2025 (Residual RES_TOTAL_PUB)", 100 * che * 1e6 / resid.RES_TOTAL_PUB,
        "%", "as above with the Residual's published residence total", f"{MONTHLY}; {PANEL}")

    # ---------------------------------------------------------------- 8
    td, ti = wide["W|TER_DOM"], wide["W|TER_INT"]
    sh = 100 * td / (td + ti)
    for per in ["2025-12", "2026-01", "2026-06"]:
        add(8, f"World domestic share {per}", sh[per], "%", "World TER_DOM / (TER_DOM + TER_INT), per day", WIDE)
        add(8, f"World territory total per day {per}", (td[per] + ti[per]) / 1e3, "kt CO2 per day",
            "World TER_DOM + TER_INT, co2_t_per_day", WIDE)
    add(8, "change in World TER_DOM, 2025-12 to 2026-06", (td["2026-06"] - td["2025-12"]) / 1e3, "kt CO2 per day",
        "TER_DOM per day 2026-06 minus 2025-12", WIDE)
    add(8, "change in World TER_INT, 2025-12 to 2026-06", (ti["2026-06"] - ti["2025-12"]) / 1e3, "kt CO2 per day",
        "TER_INT per day 2026-06 minus 2025-12", WIDE)
    add(8, "moved from international to domestic, 2025-12 to 2026-06", (sh["2026-06"] - sh["2025-12"]) / 100 *
        (td["2026-06"] + ti["2026-06"]) / 1e3, "kt CO2 per day",
        "change in the domestic share (2026-06 minus 2025-12) times the 2026-06 territory total per day", WIDE)

    # ---------------------------------------------------------------- 9
    s2 = pd.read_csv(RES / "supplementary" / "OLS_tracks.csv")
    s2 = s2[(s2.window == "main") & (s2.level == "economies") & (s2.grouping == "all") & (s2.horizon.astype(str) == "all")
            & (s2.metric == "MASE") & (s2.method == "OLS")]
    for _, r in s2.iterrows():
        for stat in ["weighted", "median"]:
            add(9, f"OLS {r.variant} {stat} MASE, base {r.base}", r[stat], "MASE",
                f"OLS reconciliation ({'exact non-negative QP' if r.variant == 'nn_qp' else 'free projection'}), "
                f"{int(r.n_series)} system economy series, main window, horizons 1-12", "results/supplementary/OLS_tracks.csv")
    ab = pd.read_csv(RES / "ablation" / "violations_world_gap_by_horizon.csv")
    ab = ab.groupby(["track", "set"]).mean_abs_res_ter_gap_pct.mean()
    for (tr, st), v in ab.items():
        if st != 1:
            add(9, f"ablation set {st}, mean World residence-territory gap, {tr}", v, "% of World RES_TOTAL_SYS",
                "100 x |TER_DOM + TER_INT + r G - RES_TOTAL_SYS| / RES_TOTAL_SYS (World), mean over horizons 1-12 of the "
                "per-horizon means (main window); set 2: without the world-balance rows; set 3: aggregation only",
                "results/ablation/violations_world_gap_by_horizon.csv")

    # ---------------------------------------------------------------- 10
    sk = pd.read_csv(RES / "skill" / "skill.csv")
    sk = sk[(sk.horizon == 1) & (sk.variant == "weighted") & (sk.origins == "all origins") & (sk.reference == "persistence")
            & (sk.group == "all") & (sk.bootstrap == "block6")]
    for _, r in sk.iterrows():
        d = (f"weighted skill against the persistence reference, {int(r.n_origins)} origins, moving-block "
             "bootstrap over origins, block 6")
        add(10, f"s(1), {r.track}", r.s, "skill", d, "results/skill/skill.csv")
        add(10, f"s(1) 95% interval, lower, {r.track}", r.lo95, "skill", d, "results/skill/skill.csv")
        add(10, f"s(1) 95% interval, upper, {r.track}", r.hi95, "skill", d, "results/skill/skill.csv")

    # ---------------------------------------------------------------- 11
    s1 = pd.read_csv(RES / "paper" / "S1_forecasts_FM_2026-07_2027-06.csv")
    x = s1[(s1.unit == "W") & (s1.measure == "RES_TOTAL_SYS") & (s1.month == "2026-07")].iloc[0]
    rt = pd.read_csv(RES / "eval_realtime_exact_ext_nested" / "res_total_pub_cells.csv")
    rt = rt[(rt.track == "FM") & (rt.origin == "2026-06") & (rt.unit == "W") & (rt.d_variant == "d") & (rt.horizon == 1)].iloc[0]
    assert np.isclose(rt.factor, 1 / (1 - rt.d), rtol=1e-5) and np.isclose(rt.point, x.point * rt.factor, rtol=1e-5)
    s1src = "results/paper/S1_forecasts_FM_2026-07_2027-06.csv"
    for lab, f in [("component-sum basis (RES_TOTAL_SYS)", 1.0), ("published basis (x 1/(1 - d))", rt.factor)]:
        for c in ["point", "lo80", "hi80"]:
            add(11, f"World July 2026 forecast, {lab}, {c}", x[c] * f / 1e6, "Mt CO2 per day",
                "FM track (TimesFM 2.5 + MinT, exact QP), origin 2026-06, h = 1; 80% = ACI gamma 0.05, nested" +
                (f"; d = {rt.d:.6f} (World, mean relative discrepancy 2026-01 to 2026-06)" if f != 1.0 else ""),
                s1src + ("" if f == 1.0 else "; results/eval_realtime_exact_ext_nested/res_total_pub_cells.csv"))
    add(11, "World July 2026 published residence total (outcome)", rt.y_pub / 1e6, "Mt CO2 per day",
        "published RES_TOTAL, July 2026 (test on later-published data)", "results/eval_realtime_exact_ext_nested/res_total_pub_cells.csv")

    # ---------------------------------------------------------------- 12
    wr = pd.read_csv(RES / "vintage" / "world_revisions.csv")
    for y, g in wr[wr.year.between(2022, 2025)].groupby("year"):
        for stat in ["min", "max"]:
            add(12, f"World residence total revision {y}, {stat}", getattr(g.revision_pct, stat)(), "%",
                f"(October 2026 vintage / October 2025 vintage - 1) x 100, World monthly published residence total; "
                f"{len(g)} months", "results/vintage/world_revisions.csv")

    # ---------------------------------------------------------------- 13
    for dname in ["onestep_fm", "onestep_fm_rt"]:
        for md in sorted((RES / dname).glob("model=*")):
            o = pd.read_parquet(md / "onestep.parquet", columns=["unique_id", "period", "context_end", "context_months"])
            o = o[o.period >= "2020-01"]                       # the months W uses (reconcile.ONESTEP_START)
            mdl = md.name.replace("model=", "")
            d = f"one-step forecasts of {o.unique_id.nunique()} series; context = months 2019-01 to the month before the target"
            add(13, f"{mdl} ({dname}): first target month", o.period.min(), "month", d, f"results/{dname}/{md.name}/onestep.parquet")
            add(13, f"{mdl} ({dname}): last target month", o.period.max(), "month", d, f"results/{dname}/{md.name}/onestep.parquet")
            add(13, f"{mdl} ({dname}): shortest context", int(o.context_months.min()), "months", d, f"results/{dname}/{md.name}/onestep.parquet")
            add(13, f"{mdl} ({dname}): longest context", int(o.context_months.max()), "months", d, f"results/{dname}/{md.name}/onestep.parquet")

    q = pd.DataFrame(rows)
    q["value"] = [("" if np.isnan(v) else f"{v:.6g}") if isinstance(v, (float, np.floating)) else v for v in q.value]
    q.to_csv(out_path(OUT / "quoted_figures.csv", args.overwrite), index=False, lineterminator="\n", float_format="%.6g")
    pd.set_option("display.width", 250, "display.max_colwidth", 90, "display.max_rows", 500)
    print(q[["item", "quantity", "value", "unit"]].to_string(index=False))

    # ---------------------------------------------------------------- the monthly international share table with 2026
    s = (wide["W|TER_INT"] / (wide["W|TER_DOM"] + wide["W|TER_INT"]) * 100).rename("share")
    d = s.reset_index()
    d.columns = ["period", "share"]
    d["year"], d["month"] = d.period.str[:4].astype(int), d.period.str[5:7].astype(int)
    t = d.pivot(index="year", columns="month", values="share")
    t["annual_mean"] = t.mean(axis=1)
    t["n_months"] = t[list(range(1, 13))].notna().sum(axis=1)
    t["jan_minus_mean_of_dec_feb"] = np.nan
    for y in t.index:
        dec = s.get(f"{y - 1}-12", np.nan)
        t.loc[y, "jan_minus_mean_of_dec_feb"] = t.loc[y, 1] - np.nanmean([dec, t.loc[y, 2]])
    t.columns = [f"m{c:02d}" if isinstance(c, (int, np.integer)) else c for c in t.columns]
    old = pd.read_csv(RES / "supplementary" / "intl_share_by_month.csv").set_index("year")
    new = t.loc[old.index, old.columns]
    assert np.allclose(new.to_numpy(float), old.to_numpy(float), rtol=1e-5, equal_nan=True)   # 2019-2025 unchanged
    t.reset_index().to_csv(out_path(OUT / "intl_share_by_month_with_2026.csv", args.overwrite), index=False, lineterminator="\n", float_format="%.6g")
    print(t.round(2).to_string())
    log.finish(f"{len(q)} quoted figures; share table with {int(t.loc[2026, 'n_months'])} months of 2026")


if __name__ == "__main__":
    main()
