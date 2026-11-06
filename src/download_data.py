"""
download_data.py: get the OECD input file and verify its SHA-256.

The paper uses the complete CSV of the OECD dataset "Maritime transport CO2 emissions (experimental)"
(dataflow OECD.SDD.NAD.SEEA:DSD_MARITIME_TRANSPORT@DF_MARITIME_TRANSPORT, version 2.0), last updated by the OECD on
14 October 2026 and downloaded on 31 October 2026:

  data/raw/oecd_maritime_co2_raw_2026-10-31.csv   200,921,743 bytes   SHA-256 REFERENCE_SHA256 (below)

Two ways to get it:
  1. from the archive of this repository (archive/data/raw/, see README): the exact file; then
       python src/download_data.py --verify
  2. from the OECD SDMX API (complete download, CSV with labels, the layout prepare_data.py reads):
       python src/download_data.py --download
     The OECD revises the data and publishes new releases: a download made later is a later release, its checksum
     will not match, and results computed from it will differ from the paper. The API output may also differ in row
     order from the file downloaded through the OECD Data Explorer. The paper's results are reproduced exactly only
     from the archived file.

Usage:
  python src/download_data.py --verify [--file data/raw/oecd_maritime_co2_raw_2026-10-31.csv]
  python src/download_data.py --download [--out data/raw]
"""
import argparse
import datetime as dt
import hashlib
import sys
import urllib.request
from pathlib import Path

REFERENCE_FILE = "data/raw/oecd_maritime_co2_raw_2026-10-31.csv"
REFERENCE_BYTES = 200_921_743
REFERENCE_SHA256 = "88fb19b9645e4452d26e88cf8891467a3732d380a54f1b445f3735849fdf6002"
API = ("https://sdmx.oecd.org/public/rest/data/OECD.SDD.NAD.SEEA,DSD_MARITIME_TRANSPORT@DF_MARITIME_TRANSPORT,/all"
       "?dimensionAtObservation=AllDimensions&format=csvfilewithlabels")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(p):
    p = Path(p)
    if not p.exists():
        sys.exit(f"{p} not found: copy it from the archive (archive/data/raw/) or use --download")
    size, digest = p.stat().st_size, sha256(p)
    ok = size == REFERENCE_BYTES and digest == REFERENCE_SHA256
    print(f"{p}: {size:,} bytes, SHA-256 {digest}")
    print("matches the file used in the paper" if ok else
          "DOES NOT match the file used in the paper (a different OECD release or a modified file)")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--verify", action="store_true")
    g.add_argument("--download", action="store_true")
    ap.add_argument("--file", default=REFERENCE_FILE, help="file to verify")
    ap.add_argument("--out", default="data/raw", help="folder of the downloaded file")
    args = ap.parse_args()
    if args.verify:
        sys.exit(0 if verify(args.file) else 1)
    out = Path(args.out) / f"oecd_maritime_co2_raw_{dt.date.today().isoformat()}.csv"
    if out.exists():
        sys.exit(f"{out} exists; not overwritten")
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {API}\n  -> {out} (about 200 MB)")
    req = urllib.request.Request(API, headers={"User-Agent": "maritime-co2-dual-forecast"})
    with urllib.request.urlopen(req, timeout=600) as r, open(out, "wb") as fh:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
    verify(out)
    print("Use it with: python src/prepare_data.py --oecd", out.as_posix(), "--out data/clean")


if __name__ == "__main__":
    main()
