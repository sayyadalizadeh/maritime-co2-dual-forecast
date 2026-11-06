"""
episode_2022.py: the 2022 domestic/international episode, quantified like the 2026 reclassification, and a
totals-only diagnostic of the skill with a stratified bootstrap.

International share = TER_INT / (TER_DOM + TER_INT) of World (territory basis); the residence share
(D + E + F) / RES_TOTAL_SYS is given next to it. All values are per day (kt CO2 per day).

Episodes (fixed windows):
  2022   2021-12 ... 2022-10
  2026   2026-01 ... 2026-06 (the 2026 reclassification)
Each episode is compared with the same calendar months 12 months earlier (reference window), so the seasonal
pattern cancels: change = mean over the episode months - mean over the reference months. As a data-driven check
of the dates, every month whose share is at least 4 percentage points below the same calendar month of 2019 is
listed (runs of such months in episodes_detected.csv).

Redistribution: the sum over system economies of the absolute change of RES_TOTAL_SYS and of TER_DOM + TER_INT.
Contributions: the change of every economy's TER_DOM and TER_INT (system economies, and outside-system economies
whose published series cover both windows), as a share of the World change; the 10 largest by absolute change,
with the change of the components that make them up (TER_DOM = A + C, TER_INT = D + G; system economies only).
Long-horizon origins: how many of the 13 origins 2021-12 ... 2022-12 lie inside the 2022 episode, and how many
of their origin-horizon cells (h = 1 ... 36, main-window targets) have a target inside it.

Diagnostic skill: s(h) (weighted MASE skill against the
persistence reference, system economies, 2022 weights, 43 origins for h <= 12 and 13 for h > 12, stratified bootstrap
over whole origins, 1,000 replications, seed 42), restricted to the totals:
  residence   RES_TOTAL_SYS of every system economy
  territory   TER_DOM + TER_INT of every system economy (derived from the reconciled forecasts; scale and 2022
              weight computed on the derived series)
  all         every system-economy series
The horizon at which the H* rule would put H* (first lo95 <= 0) is reported for information only.

Outputs: results/episode_2022/ (world_share_monthly.csv, episodes_detected.csv, episode_summary.csv,
  world_components.csv, top_contributors.csv, long_origins.csv, skill_totals.csv, skill_totals_summary.csv)
Usage:
  python src/episode_2022.py [--reps 1000] [--overwrite]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from evalcore import WINDOWS, load_history, origin_scales
from runlog import RunLog, out_path
from skill_decay import H_LONG, LONG_ORIGINS, MAIN_ORIGINS, MAIN_TAGS, SEED, TRACKS, cube, reference, skill_from

ORIGINS = sorted(set(LONG_ORIGINS) | set(MAIN_ORIGINS))            # 43 origins, 2021-12 ... 2025-06
MAIN_ONLY = [o for o in ORIGINS if o not in LONG_ORIGINS]           # 30 origins, 2023-01 ... 2025-06

COMP = {"A": "RES_DOM_IN", "B": "RES_DOM_OUT", "C": "NRES_DOM_IN", "D": "RES_INT_FROM", "E": "RES_INT_TO",
        "F": "RES_INT_OUT", "G": "NRES_INT_FROM"}
EPISODES = {"2022": ("2021-12", "2022-10"), "2026": ("2026-01", "2026-06")}
DIP_PP = 4.0


def months(a, b):
    return pd.period_range(a, b, freq="M").astype(str).tolist()


def back12(ms):
    return [str(pd.Period(m, "M") - 12) for m in ms]


def runs(flags):
    """Runs of consecutive True values in a boolean Series indexed by month: list of (start, end, n)."""
    out, start, prev = [], None, None
    for m, f in flags.items():
        if f and start is None:
            start = m
        if not f and start is not None:
            out.append((start, prev, len(months(start, prev))))
            start = None
        prev = m
    if start is not None:
        out.append((start, prev, len(months(start, prev))))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean", default="data/clean")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/episode_2022")
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    log = RunLog("episode_2022.py", vars(args))
    RES, OUT = Path(args.results), Path(args.out)

    def save(df, name):
        df.to_csv(out_path(OUT / name, args.overwrite), index=False, float_format="%.6g")

    hist = pd.read_parquet(Path(args.clean) / "panel_system_wide_v2.parquet")
    outside = pd.read_parquet(Path(args.clean) / "panel_outside_v2.parquet")
    meta = pd.read_parquet(RES / "base_forecasts_v2" / "series_meta.parquet")

    # ---------------------------------------------------------------- World shares, monthly
    W = pd.DataFrame({k: hist[f"W|{v}"] / 1e3 for k, v in COMP.items()})
    W["TER_DOM"], W["TER_INT"], W["RES_TOTAL_SYS"] = hist["W|TER_DOM"] / 1e3, hist["W|TER_INT"] / 1e3, hist["W|RES_TOTAL_SYS"] / 1e3
    W["share_int_territory_pct"] = W.TER_INT / (W.TER_DOM + W.TER_INT) * 100
    W["share_int_residence_pct"] = (W.D + W.E + W.F) / W.RES_TOTAL_SYS * 100
    base19 = W.loc["2019-01":"2019-12", "share_int_territory_pct"]
    W["share_2019_same_month_pct"] = [base19.iloc[pd.Period(m, "M").month - 1] for m in W.index]
    W["dev_from_2019_pp"] = W.share_int_territory_pct - W.share_2019_same_month_pct
    W["below_2019_by_4pp"] = W.dev_from_2019_pp <= -DIP_PP
    for e, (a, b) in EPISODES.items():
        W[f"in_episode_{e}"] = W.index.isin(months(a, b))
    save(W.reset_index().rename(columns={"period": "month"}), "world_share_monthly.csv")
    det = pd.DataFrame(runs(W.below_2019_by_4pp), columns=["start", "end", "n_months"])
    det["min_share_pct"] = [W.loc[r.start:r.end, "share_int_territory_pct"].min() for r in det.itertuples()]
    det["min_month"] = [W.loc[r.start:r.end, "share_int_territory_pct"].idxmin() for r in det.itertuples()]
    det["max_dev_pp"] = [W.loc[r.start:r.end, "dev_from_2019_pp"].min() for r in det.itertuples()]
    save(det, "episodes_detected.csv")

    # ---------------------------------------------------------------- episode summary and components
    summ, comp_rows = [], []
    for e, (a, b) in EPISODES.items():
        em, rm = months(a, b), back12(months(a, b))
        E_, R_ = W.loc[em], W.loc[rm]
        sh = E_.share_int_territory_pct
        summ.append({
            "episode": e, "start": a, "end": b, "n_months": len(em), "reference": f"{rm[0]} ... {rm[-1]}",
            "share_before_pct": W.loc[str(pd.Period(a, "M") - 1), "share_int_territory_pct"],
            "share_2months_before_pct": W.loc[str(pd.Period(a, "M") - 2), "share_int_territory_pct"],
            "share_mean_reference_pct": R_.share_int_territory_pct.mean(), "share_mean_episode_pct": sh.mean(),
            "share_min_pct": sh.min(), "share_min_month": sh.idxmin(), "share_max_pct": sh.max(),
            "share_after_pct": W.loc[str(pd.Period(b, "M") + 1), "share_int_territory_pct"]
            if str(pd.Period(b, "M") + 1) in W.index else np.nan,
            "dip_pp_vs_reference": sh.mean() - R_.share_int_territory_pct.mean(),
            "dTER_DOM_kt_day": E_.TER_DOM.mean() - R_.TER_DOM.mean(), "dTER_INT_kt_day": E_.TER_INT.mean() - R_.TER_INT.mean(),
            "dRES_TOTAL_SYS_kt_day": E_.RES_TOTAL_SYS.mean() - R_.RES_TOTAL_SYS.mean(),
            "dTER_TOTAL_pct": ((E_.TER_DOM + E_.TER_INT).mean() / (R_.TER_DOM + R_.TER_INT).mean() - 1) * 100,
            "dRES_TOTAL_SYS_pct": (E_.RES_TOTAL_SYS.mean() / R_.RES_TOTAL_SYS.mean() - 1) * 100,
            # domestic gain beyond the domestic share of the change of the territory total
            "moved_kt_day": (E_.TER_DOM.mean() - R_.TER_DOM.mean())
            - R_.TER_DOM.mean() / (R_.TER_DOM + R_.TER_INT).mean() * ((E_.TER_DOM + E_.TER_INT).mean() - (R_.TER_DOM + R_.TER_INT).mean()),
        })
        for k in list(COMP) + ["TER_DOM", "TER_INT", "RES_TOTAL_SYS"]:
            comp_rows.append({"episode": e, "measure": k if k not in COMP else f"{k} ({COMP[k]})",
                              "reference_kt_day": R_[k].mean(), "episode_kt_day": E_[k].mean(),
                              "change_kt_day": E_[k].mean() - R_[k].mean(),
                              "change_pct": (E_[k].mean() / R_[k].mean() - 1) * 100,
                              "change_peak_month_kt_day": W.at[sh.idxmin(), k] - W.at[str(pd.Period(sh.idxmin(), "M") - 12), k]})
    S = pd.DataFrame(summ)
    # redistribution across economies: sum over system economies of |change| of each total, against the World change
    se_ = meta[meta.unit_type == "economy"].unit.unique()
    for i, (e, (a, b)) in enumerate(EPISODES.items()):
        em, rm = months(a, b), back12(months(a, b))
        for name, cols in [("RES_TOTAL_SYS", ["RES_TOTAL_SYS"]), ("TER_TOTAL", ["TER_DOM", "TER_INT"])]:
            v = sum(hist[[f"{u}|{c}" for u in se_]].to_numpy() for c in cols)
            v = pd.DataFrame(v, index=hist.index)
            S.loc[i, f"sum_abs_econ_change_{name}_kt_day"] = (v.loc[em].mean() - v.loc[rm].mean()).abs().sum() / 1e3
    save(S, "episode_summary.csv")
    save(pd.DataFrame(comp_rows), "world_components.csv")

    # ---------------------------------------------------------------- contributions by economy
    se = meta[meta.unit_type == "economy"].unit.unique()
    op = outside.pivot_table(index="period", columns=["unit", "measure"], values="co2_t_per_day")
    top_rows = []
    for e, (a, b) in EPISODES.items():
        em, rm = months(a, b), back12(months(a, b))
        for meas, parts in [("TER_DOM", ["A", "C"]), ("TER_INT", ["D", "G"])]:
            dW = (hist.loc[em, f"W|{meas}"].mean() - hist.loc[rm, f"W|{meas}"].mean()) / 1e3
            rows = []
            for u in se:
                c = f"{u}|{meas}"
                r = {"unit": u, "unit_type": "system economy",
                     "change_kt_day": (hist.loc[em, c].mean() - hist.loc[rm, c].mean()) / 1e3,
                     "reference_kt_day": hist.loc[rm, c].mean() / 1e3}
                for p in parts:
                    cc = f"{u}|{COMP[p]}"
                    r[f"change_{p}_kt_day"] = (hist.loc[em, cc].mean() - hist.loc[rm, cc].mean()) / 1e3
                rows.append(r)
            for u in sorted({x[0] for x in op.columns if x[1] == meas}):
                v = op[(u, meas)]
                if v.reindex(em + rm).isna().any():
                    continue
                rows.append({"unit": u, "unit_type": "outside-system economy",
                             "change_kt_day": (v.loc[em].mean() - v.loc[rm].mean()) / 1e3,
                             "reference_kt_day": v.loc[rm].mean() / 1e3})
            d = pd.DataFrame(rows)
            d["change_pct_of_own"] = d.change_kt_day / d.reference_kt_day * 100
            d["share_of_world_change_pct"] = d.change_kt_day / dW * 100
            d = d.reindex(d.change_kt_day.abs().sort_values(ascending=False).index).head(10)
            d.insert(0, "rank", range(1, len(d) + 1))
            d.insert(0, "world_change_kt_day", dW)
            d.insert(0, "measure", meas)
            d.insert(0, "episode", e)
            top_rows.append(d)
    T = pd.concat(top_rows, ignore_index=True)
    save(T, "top_contributors.csv")

    # ---------------------------------------------------------------- long-horizon origins
    ep = set(months(*EPISODES["2022"]))
    main_win = WINDOWS["main"]
    lo_rows = []
    for o in LONG_ORIGINS:
        tg = pd.Series([str(pd.Period(o, "M") + h) for h in range(1, H_LONG + 1)])
        ok = main_win(tg)
        train12 = set(months(str(pd.Period(o, "M") - 11), o))
        lo_rows.append({"origin": o, "origin_in_episode": o in ep, "last12_months_in_episode": len(train12 & ep),
                        "targets_in_main_window": int(ok.sum()), "targets_in_episode": int((ok & tg.isin(ep)).sum()),
                        "targets_in_episode_h1_12": int((ok[:12] & tg[:12].isin(ep)).sum())})
    LO = pd.DataFrame(lo_rows)
    save(LO, "long_origins.csv")

    # ---------------------------------------------------------------- diagnostic skill, totals only
    Y = load_history(args.clean)
    Y = Y[[c for c in Y.columns if not c.startswith("OUT:") and c != "W|DELTA_INT"]]
    m_eco = meta[meta.unit_type == "economy"].set_index("unique_id")
    res_ids = [f"{u}|RES_TOTAL_SYS" for u in se]
    ter_ids = [f"{u}|TER_DOM+TER_INT" for u in se]
    all_ids = list(m_eco.index)
    Y = pd.concat([Y, pd.DataFrame({f"{u}|TER_DOM+TER_INT": Y[f"{u}|TER_DOM"] + Y[f"{u}|TER_INT"] for u in se})], axis=1)
    Y.columns.name = "uid"                                                   # as in load_history (origin_scales)
    uids = all_ids + ter_ids
    Y = Y[uids]
    wts = np.concatenate([m_eco.weight_2022.loc[all_ids].to_numpy(), Y.loc["2022-01":"2022-12", ter_ids].mean().to_numpy()])
    sc = origin_scales(Y)[["unique_id", "origin", "scale1"]]
    y_long = Y.stack().rename("y").reset_index()
    y_long.columns = ["target", "unique_id", "y"]

    def scaled(f, col):
        d = f.merge(y_long, on=["unique_id", "target"]).merge(sc, on=["unique_id", "origin"])
        d = d[main_win(d.target)]
        d["sae"] = (d.y - d[col]).abs() / d.scale1
        d.loc[~np.isfinite(d.sae), "sae"] = np.nan
        return cube(d, "sae", ORIGINS, H_LONG, uids)

    Ar = scaled(reference(Y, ORIGINS, H_LONG), "ref")
    rng = np.random.default_rng(SEED)                                    # stratified bootstrap: long and main origins
    i_long = np.array([ORIGINS.index(o) for o in LONG_ORIGINS])
    i_main = np.array([ORIGINS.index(o) for o in MAIN_ONLY])
    boot_cnt = np.zeros((args.reps, len(ORIGINS)))
    for b in range(args.reps):
        np.add.at(boot_cnt[b], rng.choice(i_long, size=len(i_long)), 1)
        np.add.at(boot_cnt[b], rng.choice(i_main, size=len(i_main)), 1)
    ones = np.ones(len(ORIGINS))
    groups = {"residence (RES_TOTAL_SYS)": np.isin(uids, res_ids), "territory (TER_DOM + TER_INT)": np.isin(uids, ter_ids),
              "all": np.isin(uids, all_ids)}
    sk_rows, sum_rows = [], []
    flt = [("method", "==", "MinT"), ("variant", "==", "nn_qp"),
           ("unique_id", "in", all_ids)]
    for tr, base in TRACKS.items():
        cols = ["unique_id", "origin", "target", "horizon", "value"]
        rl = pd.read_parquet(RES / "reconciled_long" / f"base={base}" / "reconciled.parquet", filters=flt)[cols]
        rm = pd.read_parquet(RES / f"reconciled_{MAIN_TAGS[tr]}" / f"base={base}" / "reconciled.parquet", filters=flt)[cols]
        rec = pd.concat([rl, rm[rm.origin.isin(MAIN_ONLY)]], ignore_index=True)
        t = rec[rec.unique_id.str.endswith(("|TER_DOM", "|TER_INT"))].copy()
        t["unique_id"] = t.unique_id.str.split("|").str[0] + "|TER_DOM+TER_INT"
        rec = pd.concat([rec, t.groupby(["unique_id", "origin", "target", "horizon"], as_index=False).value.sum()])
        assert sorted(rec.origin.unique()) == ORIGINS
        At = scaled(rec, "value")
        for g, msk in groups.items():
            s, mt, mr = skill_from(At[:, :, msk], Ar[:, :, msk], ones, wts[msk])
            bs = np.vstack([skill_from(At[:, :, msk], Ar[:, :, msk], c, wts[msk])[0] for c in boot_cnt])
            lo, hi = np.nanpercentile(bs, [2.5, 97.5], axis=0)
            below = np.flatnonzero(lo <= 0)
            for h in range(H_LONG):
                sk_rows.append({"track": tr, "group": g, "horizon": h + 1, "s": s[h], "lo95": lo[h], "hi95": hi[h],
                                "n_series": int(msk.sum())})
            sum_rows.append({"track": tr, "group": g, "s1": s[0], "s6": s[5], "s12": s[11], "s13": s[12], "s24": s[23],
                             "s36": s[35], "first_h_lo95_le_0": int(below[0]) + 1 if below.size else np.nan,
                             "n_h_lo95_le_0": int(below.size)})
    SK, SS = pd.DataFrame(sk_rows), pd.DataFrame(sum_rows)
    # H* of the stratified-bootstrap skill of all series (first horizon with the lower bound at or below zero)
    SS["Hstar_used"] = SS.track.map(SS[SS.group == "all"].set_index("track").first_h_lo95_le_0).astype(int)
    save(SK, "skill_totals.csv")
    save(SS, "skill_totals_summary.csv")

    pd.set_option("display.width", 220)
    print(det.to_string(index=False))
    print(S.T.to_string())
    print(pd.DataFrame(comp_rows).to_string(index=False))
    print(T.to_string(index=False))
    print(LO.to_string(index=False))
    print(SS.to_string(index=False))
    log.finish(f"2022 episode: {len(em)} months; skill diagnostic for {len(groups)} groups")


if __name__ == "__main__":
    main()
