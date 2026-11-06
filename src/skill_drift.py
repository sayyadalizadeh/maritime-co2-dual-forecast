"""
skill_drift.py: skill and H* against a persistence-with-drift reference.

Reference: the persistence reference (skill_decay.reference: trailing 12-month mean x seasonal profile, both
from data up to the origin) plus a linear drift per month ahead,
    ref_drift(o, h) = ref(o, h) + h (m1 - m2) / 12,
where m1 is the mean of the 12 months up to and including the origin and m2 the mean of the 12 months before
those. No clipping at zero (the definition is applied as written; the number of negative reference values is reported).
Everything else is the main setting of skill.py: the 43 origins 2021-12 ... 2025-06 with h 1-36 (MinT exact
QP, both tracks), at each horizon every origin whose target is in the main window, system economies, 2022 weights;
s(h) = 1 - wMASE_track(h) / wMASE_ref(h), and the median-based skill 1 - median MASE_track / median MASE_ref;
moving-block bootstrap over the 43 consecutive origins, block length 6, 1,000 replications, seed 42 (the same origin
counts as the main run, skill.py); H* = first h with the lower bound <= 0, at the 95% (main), 90% and 80%
levels; the HAC test of skill.py (weighted skill).
The persistence reference is recomputed with the same code as a check: s(h) and H* must equal results/skill.

Outputs: results/skill_drift/ (skill.csv, hstar.csv, hstar_table.csv, check_persistence.csv, reference_negative.csv)
Usage:
  python src/skill_drift.py [--reps 1000]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from evalcore import WINDOWS, load_history, origin_scales
from skill import BLOCKS, EPISODE, H, LEVELS, ORIGINS, first_le0, mbb_counts, nw_test
from runlog import RunLog, out_path
from skill_decay import LONG_ORIGINS, MAIN_TAGS, SEED, TRACKS, cube, reference, skill_from


def reference_drift(Y, origins, H):
    """Persistence reference plus h (m1 - m2) / 12. Long frame with columns ref and drift."""
    ref = reference(Y, origins, H)
    idx = list(Y.index)
    V = Y.to_numpy(float)
    slope = {}
    for o in origins:
        p = idx.index(o)
        slope[o] = (V[p - 11: p + 1].mean(axis=0) - V[p - 23: p - 11].mean(axis=0)) / 12
    s = pd.concat([pd.DataFrame({"unique_id": Y.columns, "origin": o, "slope": v}) for o, v in slope.items()])
    ref = ref.merge(s, on=["unique_id", "origin"])
    ref["drift"] = ref.ref + ref.horizon * ref.slope
    return ref


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean", default="data/clean")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/skill_drift")
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    log = RunLog("skill_drift.py", vars(args))
    RES, OUT = Path(args.results), Path(args.out)

    def save(df, name):
        df.to_csv(out_path(OUT / name, args.overwrite), index=False, float_format="%.6g")

    meta = pd.read_parquet(RES / "base_forecasts_v2" / "series_meta.parquet").set_index("unique_id")
    Y = load_history(args.clean)
    uids = [c for c in Y.columns if c in meta.index and meta.at[c, "unit_type"] == "economy"]   # system economies
    Y = Y[uids]
    sc = origin_scales(Y)
    y_long = Y.stack().rename("y").reset_index()
    y_long.columns = ["target", "unique_id", "y"]
    wts = meta.loc[uids, "weight_2022"].to_numpy()

    def cubes(f, col):
        d = f.merge(y_long, on=["unique_id", "target"]).merge(sc, on=["unique_id", "origin"])
        d = d[WINDOWS["main"](d.target)]
        d["sae"] = (d.y - d[col]).abs() / d.scale1
        d.loc[~np.isfinite(d.sae), "sae"] = np.nan
        return cube(d, "sae", ORIGINS, H, uids)

    ref = reference_drift(Y, ORIGINS, H)
    refs = {"persistence": cubes(ref, "ref"), "persistence with drift": cubes(ref, "drift")}
    neg = ref[ref.drift < 0]
    save(pd.DataFrame([{"n_reference_values": len(ref), "n_negative_drift": len(neg),
                        "share_negative_drift": len(neg) / len(ref), "n_series_with_negative": neg.unique_id.nunique()}]),
         "reference_negative.csv")
    rng = np.random.default_rng(SEED)
    cnt = mbb_counts(len(ORIGINS), BLOCKS["block6"], args.reps, rng)       # first draw of skill.py: identical
    mask = np.ones(len(ORIGINS))

    skill_rows, hs_rows = [], []
    for tr, base in TRACKS.items():
        cols = ["unique_id", "origin", "target", "horizon", "value"]
        flt = [("method", "==", "MinT"), ("variant", "==", "nn_qp")]
        rl = pd.read_parquet(RES / "reconciled_long" / f"base={base}" / "reconciled.parquet", filters=flt)[cols]
        rm = pd.read_parquet(RES / f"reconciled_{MAIN_TAGS[tr]}" / f"base={base}" / "reconciled.parquet", filters=flt)[cols]
        rn = pd.read_parquet(RES / "reconciled_long_extra" / f"base={base}" / "reconciled.parquet", filters=flt)[cols]
        late = [o for o in ORIGINS if o not in LONG_ORIGINS]
        rec = pd.concat([rl, rm[rm.origin.isin(late)], rn[rn.origin.isin(late)]], ignore_index=True)
        rec = rec[rec.unique_id.isin(uids)]
        assert not rec.duplicated(["unique_id", "origin", "horizon"]).any() and rec.origin.nunique() == len(ORIGINS)
        At = cubes(rec, "value")
        for rname, Ar in refs.items():
            ok = np.isfinite(At) & np.isfinite(Ar)
            n_or = ok.any(axis=2).sum(axis=1)
            for variant, med in [("weighted", False), ("median", True)]:
                s, _, _ = skill_from(At, Ar, mask, wts, med)
                bs = np.vstack([skill_from(At, Ar, c, wts, med)[0] for c in cnt])
                pct = {lv: np.nanpercentile(bs, p, axis=0) for lv, p in LEVELS.items()}
                hi95 = np.nanpercentile(bs, 97.5, axis=0)
                row = {"track": tr, "reference": rname, "variant": variant}
                for lv in LEVELS:
                    hs_rows.append({**row, "test": "block-6 bootstrap", "level": lv, "Hstar": first_le0(pct[lv])})
                skill_rows += [{**row, "horizon": h + 1, "s": s[h], "lo80": pct[80][h], "lo90": pct[90][h],
                                "lo95": pct[95][h], "hi95": hi95[h], "n_origins": int(n_or[h])} for h in range(H)]
                if variant == "weighted":                                   # HAC test
                    ww = np.where(ok, wts[None, None, :], 0.0)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        wt_ = (np.where(ok, At, 0) * ww).sum(axis=2) / ww.sum(axis=2)
                        wr_ = (np.where(ok, Ar, 0) * ww).sum(axis=2) / ww.sum(axis=2)
                    hac = H + 1
                    for h in range(H):
                        _, p, _ = nw_test(wr_[h] - wt_[h], h)
                        if not (p < 0.05):
                            hac = h + 1
                            break
                    hs_rows.append({**row, "test": "HAC test (Newey-West, lag h-1)", "level": 95, "Hstar": hac})
        print(f"{tr} done", flush=True)
    skill = pd.DataFrame(skill_rows)
    hs = pd.DataFrame(hs_rows)
    save(skill, "skill.csv")
    save(hs, "hstar.csv")

    # ---- check: the persistence reference reproduces the stored skill and H* (skill.py)
    st = pd.read_csv(RES / "skill" / "skill.csv")
    st = st[(st.group == "all") & (st.reference == "persistence") & (st.origins == "all origins") & (st.bootstrap == "block6")]
    hst = pd.read_csv(RES / "skill" / "hstar.csv")
    hst = hst[(hst.group == "all") & (hst.reference == "persistence") & (hst.origins == "all origins")
              & hst.bootstrap.isin(["block6", "HAC test (Newey-West, lag h-1)"])]
    chk = []
    for (tr, v), g in skill[skill.reference == "persistence"].groupby(["track", "variant"]):
        m = g.merge(st[(st.track == tr) & (st.variant == v)], on="horizon", suffixes=("", "_stored"))
        hm = hs[(hs.track == tr) & (hs.variant == v) & (hs.reference == "persistence")]
        hsd = hst[(hst.track == tr) & (hst.variant == v)]
        hb = {lv: int(hsd[(hsd.bootstrap == "block6") & (hsd.level == lv)].Hstar.iloc[0]) for lv in LEVELS}
        chk.append({"track": tr, "variant": v, "max_abs_diff_s": float((m.s - m.s_stored).abs().max()),
                    "max_abs_diff_lo95": float((m.lo95 - m.lo95_stored).abs().max()),
                    "Hstar_equal": all(int(hm[(hm.test == "block-6 bootstrap") & (hm.level == lv)].Hstar.iloc[0]) == hb[lv]
                                       for lv in LEVELS)})
    chk = pd.DataFrame(chk)
    save(chk, "check_persistence.csv")
    assert (chk.max_abs_diff_s < 1e-6).all() and (chk.max_abs_diff_lo95 < 1e-6).all() and chk.Hstar_equal.all(), chk

    # ---- one table: main values next to the drift reference
    rows = []
    for (tr, rname, v), g in skill.groupby(["track", "reference", "variant"], sort=False):
        h_ = hs[(hs.track == tr) & (hs.reference == rname) & (hs.variant == v)]
        r = {"track": tr, "reference": rname + (" (main)" if rname == "persistence" else ""), "skill": v}
        for h in (1, 6, 12, 24, 36):
            r[f"s({h})"] = float(g.loc[g.horizon == h, "s"].iloc[0])
        for lv in (95, 90, 80):
            r[f"Hstar_{lv}"] = int(h_[(h_.test == "block-6 bootstrap") & (h_.level == lv)].Hstar.iloc[0])
        hac = h_[h_.test.str.startswith("HAC")]
        r["Hstar_HAC"] = int(hac.Hstar.iloc[0]) if len(hac) else np.nan
        rows.append(r)
    tab = pd.DataFrame(rows)
    save(tab, "hstar_table.csv")
    pd.set_option("display.width", 220)
    print(tab.to_string(index=False))
    log.finish("skill against persistence with drift")


if __name__ == "__main__":
    main()
