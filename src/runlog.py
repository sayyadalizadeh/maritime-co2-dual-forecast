"""
runlog.py: small helpers shared by the scripts in src/.

- RunLog: records start time, settings and run time of a script and appends one
  row to results/run_log.csv when the run finishes.
- out_path: refuses to overwrite an existing output file unless overwrite=True,
  so that earlier results are never replaced silently.
"""
import csv
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_LOG = ROOT / "results" / "run_log.csv"


class RunLog:
    def __init__(self, script, settings):
        self.script = script
        self.settings = dict(settings)
        self.t0 = time.perf_counter()
        self.started = datetime.now().isoformat(timespec="seconds")
        print(f"[{self.script}] start {self.started}  settings={json.dumps(self.settings, default=str)}")

    def finish(self, note=""):
        seconds = time.perf_counter() - self.t0
        RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
        new = not RUN_LOG.exists()
        with open(RUN_LOG, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["script", "started", "seconds", "python", "settings", "note"])
            w.writerow([self.script, self.started, f"{seconds:.1f}", platform.python_version(),
                        json.dumps(self.settings, default=str), note])
        print(f"[{self.script}] done in {seconds:.1f} s")


def out_path(path, overwrite=False):
    """Return path as a Path; stop if it exists and overwrite is not allowed."""
    p = Path(path)
    if p.exists() and not overwrite:
        sys.exit(f"Output {p} already exists. Version the output or pass --overwrite.")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
