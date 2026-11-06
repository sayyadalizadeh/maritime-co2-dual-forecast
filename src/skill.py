"""
skill.py: forecast skill and H* (all origins 2021-12 ... 2025-06).

Skill (functions of skill_decay.py): s(h) = 1 - wMASE_track(h) / wMASE_ref(h) over system
economies, 2022 weights, reconciled MinT exact-QP point forecasts. Per-series MASE at h = mean scaled absolute
error over the origins whose target lies in the main window (<= 2025-12, without 2025-09).
Origins: the 43 origins 2021-12 ... 2025-06, h 1-36 at every origin: 2021-12 ... 2022-12 from the long-horizon run
(results/reconciled_long), 2023-01 ... 2025-06 from the main reconciliation runs (h 1-12) and the extra long-horizon
run (results/reconciled_long_extra, h 13-36). At each horizon every origin whose target is in the main window is
used, so the number of origins falls from 43 (h = 1) to 12 (h = 36), about one origin per horizon (2025-09 excluded).
Intervals: moving-block bootstrap over the 43 consecutive origins (blocks of consecutive origins drawn with
  replacement, start uniform on 1 ... 43 - l + 1, concatenated and cut to 43), block length l = 6 (main), 3 and 12, and
  the i.i.d. bootstrap (l = 1); 1,000 replications, seed 42. Each replication gives origin counts, used at every
  horizon.
H*: first h with the bootstrap lower bound <= 0, at the 95% (main), 90% and 80% levels (lower bound = the 2.5th,
  5th, 10th percentile). HAC test: per origin o, d_o(h) = wMASE_ref(o, h) - wMASE_track(o, h)
  (weighted mean over the system economies of the scaled errors at that origin); one-sided test of H0: E d <= 0 with
  a Newey-West (Bartlett) variance, lag h - 1 (capped at n - 2), t-statistic against t(n - 1); H* = first h with
  p >= 0.05. Alternative reference: seasonal naive, y(o + h - 12 ceil(h / 12)). Median-based skill:
  1 - median MASE_track / median MASE_ref. Sensitivity runs: (i) the 13 origins 2021-12 ... 2022-12 at every
  horizon; (ii) all origins except 2021-12 ... 2022-10 (the 2022 episode). Diagnostic: residence totals
  (RES_TOTAL_SYS) and territory totals (TER_DOM + TER_INT) of the system economies separately (weights: 2022 weight
  of RES_TOTAL_SYS; sum of the TER_DOM and TER_INT weights).
Outputs: results/skill/ (skill.csv, hstar.csv, hac.csv)
Usage:
  python src/skill.py [--reps 1000]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from evalcore import WINDOWS, load_history, origin_scales
from runlog import RunLog, out_path
from skill_decay import LONG_ORIGINS, MAIN_TAGS, SEED, TER_TOTAL, TRACKS, add_ter_total, cube, reference, skill_from

H = 36
ORIGINS = pd.period_range("2021-12", "2025-06", freq="M").astype(str).tolist()          # 43
EPISODE = pd.period_range("2021-12", "2022-10", freq="M").astype(str).tolist()
BLOCKS = {"block6": 6, "block3": 3, "block12": 12, "iid": 1}
LEVELS = {95: 2.5, 90: 5.0, 80: 10.0}


def mbb_counts(n, l, reps, rng):
    """Origin counts [reps, n] of a moving-block bootstrap with block length l (l = 1: i.i.d.)."""
    k = int(np.ceil(n / l))
    cnt = np.zeros((reps, n))
    for b in range(reps):
        starts = rng.integers(0, n - l + 1, size=k)
        idx = (starts[:, None] + np.arange(l)[None, :]).ravel()[:n]
        np.add.at(cnt[b], idx, 1)
    return cnt


def seasonal_naive(Y, origins, H):
    idx = list(Y.index)
    V = Y.to_numpy(float)
    rows = []
    for o in origins:
        p = idx.index(o)
        for h in range(1, H + 1):
            lag = 12 * int(np.ceil(h / 12))
            rows.append(pd.DataFrame({"unique_id": Y.columns, "origin": o, "target": str(pd.Period(o, "M") + h),
                                      "horizon": h, "ref": V[p + h - lag]}))
    return pd.concat(rows, ignore_index=True)


def first_le0(lo):
    b = np.flatnonzero(~(lo > 0))
    return int(b[0]) + 1 if b.size else H + 1


def nw_test(d, lag):
    """One-sided t-test of H0: mean(d) <= 0 with a Newey-West (Bartlett) variance."""
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 3:
        return np.nan, np.nan, n
    lag = min(lag, n - 2)
    e = d - d.mean()
    v = e @ e / n
    for j in range(1, lag + 1):
        v += 2 * (1 - j / (lag + 1)) * (e[j:] @ e[:-j]) / n
    se = np.sqrt(max(v, 1e-300) / n)
    t = d.mean() / se
    return float(t), float(stats.t.sf(t, df=n - 1)), n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean", default="data/clean")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/skill")
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    log = RunLog("skill.py", vars(args))
    RES, OUT = Path(args.results), Path(args.out)

    def save(df, name):
        df.to_csv(out_path(OUT / name, args.overwrite), index=False, float_format="%.6g")

    meta = pd.read_parquet(RES / "base_forecasts_v2" / "series_meta.parquet").set_index("unique_id")
    Y = load_history(args.clean)
    Y = Y[[c for c in Y.columns if not c.startswith("OUT:") and c != "W|DELTA_INT"]]
    Y[TER_TOTAL] = Y["W|TER_DOM"] + Y["W|TER_INT"]
    econ = sorted({u.split("|")[0] for u in Y.columns if u in meta.index and meta.at[u, "unit_type"] == "economy"})
    for u in econ:                                                    # diagnostic: territory totals per economy
        Y[f"{u}|TER_TOTAL"] = Y[f"{u}|TER_DOM"] + Y[f"{u}|TER_INT"]
    uids = list(Y.columns)
    sc = origin_scales(Y)
    y_long = Y.stack().rename("y").reset_index()
    y_long.columns = ["target", "unique_id", "y"]
    wts = np.array([meta.at[u, "weight_2022"] if u in meta.index else np.nan for u in uids])
    for i, u in enumerate(uids):
        if u.endswith("|TER_TOTAL"):
            a = u.split("|")[0]
            wts[i] = meta.at[f"{a}|TER_DOM", "weight_2022"] + meta.at[f"{a}|TER_INT", "weight_2022"]
    is_eco_sys = np.array([u in meta.index and meta.at[u, "unit_type"] == "economy" for u in uids])
    groups = {"all": is_eco_sys,
              "residence totals": np.array([u.split("|")[1] == "RES_TOTAL_SYS" and u.split("|")[0] in econ for u in uids]),
              "territory totals": np.array([u.endswith("|TER_TOTAL") and u.split("|")[0] in econ for u in uids])}
    main_win = WINDOWS["main"]

    def cubes(f, col):
        d = f.merge(y_long, on=["unique_id", "target"]).merge(sc, on=["unique_id", "origin"])
        d = d[main_win(d.target)]
        d["sae"] = (d.y - d[col]).abs() / d.scale1
        d["sse"] = (d.y - d[col]) ** 2 / d.scale2
        for c in ("sae", "sse"):
            d.loc[~np.isfinite(d[c]), c] = np.nan
        return cube(d, "sae", ORIGINS, H, uids), cube(d, "sse", ORIGINS, H, uids)

    refs = {"persistence": cubes(reference(Y, ORIGINS, H), "ref"),
            "seasonal naive": cubes(seasonal_naive(Y, ORIGINS, H), "ref")}
    rng = np.random.default_rng(SEED)
    boot = {k: mbb_counts(len(ORIGINS), l, args.reps, rng) for k, l in BLOCKS.items()}
    sets = {"all origins": np.ones(len(ORIGINS)),
            "(i) 13 origins 2021-12..2022-12": np.array([o in LONG_ORIGINS for o in ORIGINS], float),
            "(ii) without 2021-12..2022-10": np.array([o not in EPISODE for o in ORIGINS], float)}

    skill_rows, hs_rows, hac_rows = [], [], []
    for tr, base in TRACKS.items():
        cols = ["unique_id", "origin", "target", "horizon", "value"]
        flt = [("method", "==", "MinT"), ("variant", "==", "nn_qp")]
        rl = pd.read_parquet(RES / "reconciled_long" / f"base={base}" / "reconciled.parquet", filters=flt)[cols]
        rm = pd.read_parquet(RES / f"reconciled_{MAIN_TAGS[tr]}" / f"base={base}" / "reconciled.parquet", filters=flt)[cols]
        rn = pd.read_parquet(RES / "reconciled_long_extra" / f"base={base}" / "reconciled.parquet", filters=flt)[cols]
        late = [o for o in ORIGINS if o not in LONG_ORIGINS]
        rec = pd.concat([rl, rm[rm.origin.isin(late)], rn[rn.origin.isin(late)]], ignore_index=True)
        rec = rec[rec.unique_id != "W|DELTA_INT"]
        assert not rec.duplicated(["unique_id", "origin", "horizon"]).any()
        assert rec.groupby("origin").horizon.nunique().eq(H).all() and rec.origin.nunique() == len(ORIGINS)
        rec = add_ter_total(rec, "value")
        t = rec[rec.unique_id.str.endswith(("|TER_DOM", "|TER_INT"))].copy()
        t["unit"] = t.unique_id.str.split("|").str[0]
        t = t[t.unit.isin(econ)].groupby(["unit", "origin", "target", "horizon"], as_index=False).value.sum()
        rec = pd.concat([rec, t.assign(unique_id=t.unit + "|TER_TOTAL").drop(columns="unit")], ignore_index=True)
        At, Bt = cubes(rec, "value")
        for gname, g in groups.items():
            for ref, (Ar, Br) in refs.items():
                for oset, mask in sets.items():
                    if (gname != "all" or ref != "persistence") and oset != "all origins":
                        continue
                    for variant, med in [("weighted", False), ("median", True)]:
                        if med and (gname != "all"):
                            continue
                        s, mt, mr = skill_from(At[:, :, g], Ar[:, :, g], mask, wts[g], med)
                        okc = np.isfinite(At[:, :, g]) & np.isfinite(Ar[:, :, g])
                        n_or = (okc.any(axis=2) * (mask[None, :] > 0)).sum(axis=1)
                        blist = BLOCKS if (gname == "all" and ref == "persistence" and variant == "weighted"
                                           and oset == "all origins") else {"block6": 6}
                        for bname in blist:
                            bs = np.vstack([skill_from(At[:, :, g], Ar[:, :, g], c * mask, wts[g], med)[0]
                                            for c in boot[bname]])
                            row_h = {"track": tr, "group": gname, "reference": ref, "variant": variant,
                                     "origins": oset, "bootstrap": bname}
                            pct = {lv: np.nanpercentile(bs, p, axis=0) for lv, p in LEVELS.items()}
                            hi95 = np.nanpercentile(bs, 97.5, axis=0)
                            for lv in LEVELS:
                                hs_rows.append({**row_h, "level": lv, "Hstar": first_le0(pct[lv])})
                            for h in range(H):
                                skill_rows.append({**row_h, "horizon": h + 1, "s": s[h], "lo80": pct[80][h],
                                                   "lo90": pct[90][h], "lo95": pct[95][h], "hi95": hi95[h],
                                                   "n_origins": int(n_or[h])})
                            if row_h == {"track": tr, "group": "all", "reference": "persistence", "variant": "weighted",
                                         "origins": "all origins", "bootstrap": "block6"}:
                                main_s, main_lo = s, pct
                        # ---- HAC test, per-origin weighted MASE difference
                        if variant == "weighted":
                            ok = np.isfinite(At[:, :, g]) & np.isfinite(Ar[:, :, g])
                            ww = np.where(ok, wts[g][None, None, :], 0.0)
                            with np.errstate(invalid="ignore", divide="ignore"):
                                wt_ = (np.where(ok, At[:, :, g], 0) * ww).sum(axis=2) / ww.sum(axis=2)
                                wr_ = (np.where(ok, Ar[:, :, g], 0) * ww).sum(axis=2) / ww.sum(axis=2)
                            hac_h = H + 1
                            for h in range(H):
                                dd = (wr_[h] - wt_[h])[mask > 0]
                                tt, p, nn = nw_test(dd, h)
                                hac_rows.append({"track": tr, "group": gname, "reference": ref, "origins": oset,
                                                 "horizon": h + 1, "mean_diff": np.nanmean(dd), "t": tt, "p_one_sided": p,
                                                 "n_origins": nn, "lag": min(h, max(nn - 2, 0))})
                                if hac_h == H + 1 and not (p < 0.05):
                                    hac_h = h + 1
                            hs_rows.append({"track": tr, "group": gname, "reference": ref, "variant": "weighted",
                                            "origins": oset, "bootstrap": "HAC test (Newey-West, lag h-1)", "level": 95,
                                            "Hstar": hac_h})
        hs95, hs90 = first_le0(main_lo[95]), first_le0(main_lo[90])
        print(f"{tr}: H*95 {hs95}, H*90 {hs90}", flush=True)
    skill = pd.DataFrame(skill_rows)
    save(skill, "skill.csv")
    save(pd.DataFrame(hs_rows), "hstar.csv")
    save(pd.DataFrame(hac_rows), "hac.csv")
    hs = pd.DataFrame(hs_rows)
    pd.set_option("display.width", 220)
    print(hs.pivot_table(index=["group", "reference", "variant", "origins", "bootstrap"], columns=["track", "level"],
                         values="Hstar").to_string())
    log.finish("skill and H*")


if __name__ == "__main__":
    main()
