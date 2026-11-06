"""
summary_weighted.py: re-summarises the evaluation of the statistical base forecasts. Nothing is refitted: the
per-series metrics of evaluate_base.py
(results/eval_base_v2/metrics_per_series.parquet) are summarised again, with weighted means
computed within one level at a time (system economies; outside-system series) and the
World and Residual series reported one by one.

Output: results/eval_base_v2/summary_weighted.csv

Usage:
  python src/summary_weighted.py [--eval results/eval_base_v2] [--overwrite]
"""
import argparse
from pathlib import Path

import pandas as pd

from evalcore import summarise_levels
from runlog import RunLog, out_path

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--eval", default="results/eval_base_v2")
ap.add_argument("--overwrite", action="store_true")
args = ap.parse_args()
log = RunLog("summary_weighted.py", vars(args))
EV = Path(args.eval)

ps = pd.read_parquet(EV / "metrics_per_series.parquet")
metrics = ["MASE", "RMSSE", "sCRPS", "cov80", "cov95", "sWinkler95"]
s = summarise_levels(ps, ["model"], metrics=metrics)
s.to_csv(out_path(EV / "summary_weighted.csv", args.overwrite), index=False, float_format="%.5g")
print(f"rows: {len(s)} | levels: {s.level.value_counts().to_dict()}")
log.finish(f"{len(s)} rows")
