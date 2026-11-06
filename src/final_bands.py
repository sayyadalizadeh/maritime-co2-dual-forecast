"""
final_bands.py: calibrated final forecasts at origin 2026-06.

For each track, from the reconciled samples of final_samples.py (horizons 1 ... final_samples.FINAL_H):
  quantiles   0.025, 0.05, 0.1, 0.2, ..., 0.9, 0.95, 0.975 of the reconciled samples (uncalibrated)
  intervals   80% and 95% ACI intervals with the extended state (calib_aci.py, state 2026-06_ext; gamma 0.05, the main
              setting; gamma 0.01 kept as a robustness column): [Q(alpha/2), Q(1 - alpha/2)] of the
              reconciled samples, then nesting: the 80% interval is clipped to lie inside the 95% interval.
  point       the track's reconciled MinT exact-QP point forecast (results/reconciled_final).
  aci_at_floor which ACI levels (gamma 0.05) have alpha at its floor ALPHA_CLIP[0] = 0.001: none, 95,
              80_and_95, or 80 (80% level only; after nesting its interval equals the 95% interval). At the floor
              the interval is (almost) the range of the samples.

Outputs (results/final/track=<FM|STAT>/): forecasts.parquet (point, quantiles, calibrated intervals; n x H rows),
  calibration_summary.csv
--samples, --aci-state and --out take the reconciled samples, the ACI state and the output folder from
elsewhere (exact samples and the ACI state of the exact recursion: results/final_exact, the paper's version).

Usage:
  python src/final_bands.py --track FM
  python src/final_bands.py --track FM --samples results/final/track=FM/samples_rec.npy
         --aci-state results/aci_exact/track=FM/aci_state.parquet --out results/final_exact
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from prob_samples import ALPHA_CLIP, QNAMES, STORE_Q, sample_quantile
from reconcile import load_system
from runlog import RunLog, out_path

ORIGIN = "2026-06"
TRACK_BASE = {"FM": ("TimesFM25", "main"), "STAT": ("AutoETS", "STAT")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", required=True, choices=list(TRACK_BASE))
    ap.add_argument("--clean", default="data/clean")
    ap.add_argument("--results", default="results")
    ap.add_argument("--samples", default=None, help="reconciled samples (H, S, n) (default results/final/track=T/samples_rec.npy)")
    ap.add_argument("--aci-state", default=None, help="ACI state with state_at 2026-06_ext (default results/calib/aci/track=T)")
    ap.add_argument("--out", default=None, help="output folder (default results/final); files go to <out>/track=T")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    log = RunLog("final_bands.py", vars(args))
    RES = Path(args.results)
    OUT = (Path(args.out) if args.out else RES / "final") / f"track={args.track}"
    OUT.mkdir(parents=True, exist_ok=True)
    base, bvar = TRACK_BASE[args.track]

    uids = load_system(RES, args.clean)["uids"]
    X = np.load(Path(args.samples) if args.samples else OUT / "samples_rec.npy")                     # (H, S, n)
    H = X.shape[0]
    Xs = np.sort(X, axis=1)
    targets = [str(pd.Period(ORIGIN, "M") + h) for h in range(1, H + 1)]
    pt = pd.read_parquet(RES / "reconciled_final" / f"base={base}" / "reconciled.parquet",
                         filters=[("method", "==", "MinT"), ("variant", "==", "nn_qp"), ("origin", "==", ORIGIN)])
    pt = pt[pt.horizon <= H].pivot(index="horizon", columns="unique_id", values="value")[uids].to_numpy()    # (H, n)
    st = pd.read_parquet(Path(args.aci_state) if args.aci_state else RES / "calib" / "aci" / f"track={args.track}" / "aci_state.parquet")
    st = st[st.state_at == "2026-06_ext"]

    rows = []
    for h in range(H):
        d = pd.DataFrame({"unique_id": uids, "origin": ORIGIN, "horizon": h + 1, "target": targets[h], "point": pt[h]})
        q = np.quantile(X[h], STORE_Q, axis=0)
        for j, c in enumerate(QNAMES):
            d[c] = q[j]
        for g in (0.05, 0.01):
            iv = {}
            for lv in (95, 80):
                a = st[(st.level == lv) & (st.gamma == g) & (st.horizon == h + 1)].set_index("unique_id").alpha.loc[uids].to_numpy()
                iv[lv] = (sample_quantile(Xs[h], a / 2), sample_quantile(Xs[h], 1 - a / 2), a)
            sfx = "" if g == 0.05 else "_g0.01"
            d[f"alpha95{sfx}"], d[f"lo95{sfx}"], d[f"hi95{sfx}"] = iv[95][2], iv[95][0], iv[95][1]
            d[f"alpha80{sfx}"] = iv[80][2]
            d[f"lo80_raw{sfx}"], d[f"hi80_raw{sfx}"] = iv[80][0], iv[80][1]
            d[f"lo80{sfx}"] = np.maximum(iv[80][0], iv[95][0])                 # nesting
            d[f"hi80{sfx}"] = np.minimum(iv[80][1], iv[95][1])
        rows.append(d)
    fc = pd.concat(rows, ignore_index=True)
    fc["nesting_clipped"] = (fc.lo80 != fc.lo80_raw) | (fc.hi80 != fc.hi80_raw)
    fc["point_outside_95"] = (fc.point < fc.lo95) | (fc.point > fc.hi95)
    f95, f80 = fc.alpha95 <= ALPHA_CLIP[0] + 1e-12, fc.alpha80 <= ALPHA_CLIP[0] + 1e-12
    fc["aci_at_floor"] = np.select([f95 & f80, f95, f80], ["80_and_95", "95", "80"], "none")
    fc.to_parquet(out_path(OUT / "forecasts.parquet", args.overwrite), index=False)
    summ = fc.groupby("horizon").agg(alpha80_median=("alpha80", "median"), alpha95_median=("alpha95", "median"),
                                     share_nesting_clipped=("nesting_clipped", "mean"),
                                     share_point_outside_95=("point_outside_95", "mean")).reset_index()
    summ.to_csv(out_path(OUT / "calibration_summary.csv", args.overwrite), index=False, float_format="%.4f")
    print(summ.to_string(index=False))
    log.finish(f"{args.track}: H = {H}, {len(fc)} forecast rows")


if __name__ == "__main__":
    main()
