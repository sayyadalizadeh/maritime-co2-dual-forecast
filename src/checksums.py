"""
checksums.py: SHA-256 checksums of the raw OECD file and every file in data/clean.

Output: results/checksums.csv (file, bytes, sha256)

Usage:
  python src/checksums.py [--raw data/raw/oecd_maritime_co2_raw_2026-10-31.csv] [--clean data/clean]
                          [--out results/checksums.csv] [--overwrite]
"""
import argparse
import csv
import hashlib
from pathlib import Path

from runlog import RunLog, out_path

ap = argparse.ArgumentParser()
ap.add_argument("--raw", default="data/raw/oecd_maritime_co2_raw_2026-10-31.csv")
ap.add_argument("--clean", default="data/clean")
ap.add_argument("--out", default="results/checksums.csv")
ap.add_argument("--overwrite", action="store_true", help="allow overwriting an existing output")
args = ap.parse_args()
log = RunLog("checksums.py", vars(args))


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


files = [Path(args.raw)] + sorted(p for p in Path(args.clean).iterdir() if p.is_file())
with open(out_path(args.out, args.overwrite), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["file", "bytes", "sha256"])
    for p in files:
        w.writerow([p.as_posix(), p.stat().st_size, sha256(p)])
        print(p.as_posix(), p.stat().st_size)
log.finish(f"{len(files)} files")
