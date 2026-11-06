"""
archive.py: the archive of stored outputs (archive/): download it from the GitHub release, unpack it from local zip
files, verify it, restore it into the repository tree, rebuild its manifest, or package it as release assets.

The archive mirrors the repository tree: archive/results/... and archive/data/..., with archive/MANIFEST.csv listing
every file with its size in bytes and its SHA-256. It is published as assets of the GitHub release v1.0:
  maritime-co2-dual-forecast-v1.0-archive-part<k>.zip   the archive files, split into independent zip files of less
                                                        than 2 GB each (paths relative to the archive folder)
  MANIFEST.csv                                          every archive file with its size and SHA-256
  SHA256SUMS.txt                                        the SHA-256 of every asset above (sha256sum format)

Actions:
  download  download the assets of the release into --assets (default release-assets/), check their SHA-256 against
            SHA256SUMS.txt, then unpack them as below; assets already present with the right SHA-256 are not
            downloaded again
  unpack    build the archive folder from local assets in --assets: check every asset against SHA256SUMS.txt,
            extract the zip files, copy MANIFEST.csv, then verify every file against the manifest
  verify    check every file of the manifest (size and SHA-256)
  restore   copy the archived files into the repository (results/, data/), after verifying them; files that already
            exist with the same SHA-256 are left alone, files that exist with other content are reported and kept
            (use --overwrite to replace them)
  manifest  rewrite MANIFEST.csv of the archive folder from the files in it
  package   write the release assets (zip parts, MANIFEST.csv, SHA256SUMS.txt) into --assets from the archive folder,
            after verifying it; the zip files are deterministic (sorted entries, fixed time stamps)

The archive folder is --archive, else the environment variable MARITIME_ARCHIVE, else archive/ at the repository root.

Usage:
  python src/archive.py download [--assets release-assets] [--url URL]
  python src/archive.py unpack   [--assets release-assets]
  python src/archive.py verify
  python src/archive.py restore  [--overwrite]
  python src/archive.py manifest
  python src/archive.py package  [--assets release-assets] [--overwrite]
"""
import argparse
import hashlib
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
VERSION = "v1.0"
RELEASE_URL = f"https://github.com/sayyadalizadeh/maritime-co2-dual-forecast/releases/download/{VERSION}/"
PART_NAME = "maritime-co2-dual-forecast-{version}-archive-part{k}.zip"
PART_LIMIT = 1_900_000_000          # bytes of archive files per zip part (uncompressed), so every part is below 2 GB
ZIP_TIME = (2026, 11, 6, 0, 0, 0)   # fixed time stamp of every zip entry, so that the zip files are reproducible
SUMS, MANIFEST = "SHA256SUMS.txt", "MANIFEST.csv"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(arch):
    man = pd.read_csv(arch / MANIFEST)
    bad = []
    for r in man.itertuples():
        p = arch / r.file
        if not p.exists():
            bad.append((r.file, "missing"))
        elif p.stat().st_size != r.bytes:
            bad.append((r.file, "size differs"))
        elif sha256(p) != r.sha256:
            bad.append((r.file, "SHA-256 differs"))
    return man, bad


def report_verify(arch):
    man, bad = verify(arch)
    for f, why in bad:
        print(f"  {why}: {f}")
    print(f"{len(man)} files in the manifest, {man.bytes.sum() / 1e9:.3f} GB; {len(bad)} problems")
    if bad:
        sys.exit(1)
    return man


def read_sums(assets):
    """SHA256SUMS.txt -> {asset name: sha256}."""
    p = assets / SUMS
    if not p.exists():
        sys.exit(f"{p} not found")
    sums = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            h, name = line.split(maxsplit=1)
            sums[name.strip().lstrip("*")] = h.lower()
    return sums


def check_assets(assets, sums):
    for name, h in sums.items():
        p = assets / name
        if not p.exists():
            sys.exit(f"asset missing: {p}")
        if sha256(p) != h:
            sys.exit(f"SHA-256 of {p} differs from {SUMS}")
        print(f"  ok  {name}  ({p.stat().st_size / 1e6:,.1f} MB)")


def safe_target(arch, name):
    """Target path of a zip entry inside the archive folder; refuses absolute paths and '..'."""
    pp = PurePosixPath(name)
    if pp.is_absolute() or ".." in pp.parts or "\\" in name or ":" in name:
        sys.exit(f"unsafe path in zip file: {name}")
    return arch.joinpath(*pp.parts)


def unpack(assets, arch, overwrite):
    sums = read_sums(assets)
    print(f"checking {len(sums)} assets in {assets}")
    check_assets(assets, sums)
    arch.mkdir(parents=True, exist_ok=True)
    n = same = 0
    for name in sorted(s for s in sums if s.endswith(".zip")):
        with zipfile.ZipFile(assets / name) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                dst = safe_target(arch, info.filename)
                if dst.exists() and not overwrite:
                    same += 1                      # checked against the manifest below
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, open(dst, "wb") as out:
                    shutil.copyfileobj(src, out, 1 << 20)
                n += 1
        print(f"  extracted {name}")
    shutil.copyfile(assets / MANIFEST, arch / MANIFEST)
    print(f"unpacked into {arch}: {n} files written, {same} already present")
    report_verify(arch)


def download(url, assets):
    assets.mkdir(parents=True, exist_ok=True)

    def get(name):
        tmp = assets / (name + ".download")
        req = urllib.request.Request(url + name, headers={"User-Agent": "maritime-co2-dual-forecast"})
        with urllib.request.urlopen(req) as r, open(tmp, "wb") as out:
            total, done, step = int(r.headers.get("Content-Length") or 0), 0, 0
            for chunk in iter(lambda: r.read(1 << 20), b""):
                out.write(chunk)
                done += len(chunk)
                if done >= step:
                    print(f"  {name}: {done / 1e6:,.0f} of {total / 1e6:,.0f} MB", flush=True)
                    step += 200_000_000
        tmp.replace(assets / name)

    print(f"downloading {SUMS} from {url}")
    get(SUMS)
    for name, h in read_sums(assets).items():
        p = assets / name
        if p.exists() and sha256(p) == h:
            print(f"  already downloaded: {name}")
            continue
        get(name)
        if sha256(p) != h:
            sys.exit(f"SHA-256 of the downloaded {name} differs from {SUMS}; delete it and try again")


def package(arch, assets, overwrite):
    man = report_verify(arch)
    assets.mkdir(parents=True, exist_ok=True)
    old = [p for p in assets.iterdir() if p.is_file() and not p.name.startswith("RELEASE_NOTES")]
    if old and not overwrite:
        sys.exit(f"{assets} already holds {len(old)} files; use --overwrite")
    for p in old:
        p.unlink()
    # split the files, in manifest order (sorted paths), into parts of at most PART_LIMIT bytes
    parts, cur, size = [], [], 0
    for r in man.sort_values("file").itertuples():
        if r.bytes > PART_LIMIT:
            sys.exit(f"{r.file} alone exceeds the part limit")
        if cur and size + r.bytes > PART_LIMIT:
            parts.append(cur)
            cur, size = [], 0
        cur.append(r.file)
        size += r.bytes
    parts.append(cur)
    names = []
    for k, files in enumerate(parts, start=1):
        name = PART_NAME.format(version=VERSION, k=k)
        with zipfile.ZipFile(assets / name, "w", zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as zf:
            for f in files:
                info = zipfile.ZipInfo(f, date_time=ZIP_TIME)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                with open(arch / f, "rb") as src, zf.open(info, "w", force_zip64=True) as out:
                    shutil.copyfileobj(src, out, 1 << 20)
        b = (assets / name).stat().st_size
        if b >= 2_000_000_000:
            sys.exit(f"{name} is {b:,} bytes, not below 2 GB; lower PART_LIMIT")
        print(f"  {name}: {len(files)} files, {b / 1e9:.3f} GB")
        names.append(name)
    shutil.copyfile(arch / MANIFEST, assets / MANIFEST)
    names.append(MANIFEST)
    lines = [f"{sha256(assets / n)}  {n}" for n in names]
    (assets / SUMS).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {len(names)} assets and {SUMS} to {assets}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["download", "unpack", "verify", "restore", "manifest", "package"])
    # default: the environment variable MARITIME_ARCHIVE (as for the tests), else archive/ at the repository root
    ap.add_argument("--archive", default=os.environ.get("MARITIME_ARCHIVE", str(ROOT / "archive")))
    ap.add_argument("--assets", default=str(ROOT / "release-assets"), help="folder of the release assets")
    ap.add_argument("--url", default=RELEASE_URL, help="download location of the release assets")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    arch, assets = Path(args.archive).resolve(), Path(args.assets).resolve()

    if args.action == "manifest":
        files = sorted(p for p in arch.rglob("*") if p.is_file() and p.name != MANIFEST)
        man = pd.DataFrame([{"file": p.relative_to(arch).as_posix(), "bytes": p.stat().st_size, "sha256": sha256(p)}
                            for p in files])
        man.to_csv(arch / MANIFEST, index=False, lineterminator="\n")
        print(f"{len(man)} files, {man.bytes.sum() / 1e9:.3f} GB")
        return
    if args.action == "package":
        package(arch, assets, args.overwrite)
        return
    if args.action == "download":
        download(args.url if args.url.endswith("/") else args.url + "/", assets)
        unpack(assets, arch, args.overwrite)
        return
    if args.action == "unpack":
        unpack(assets, arch, args.overwrite)
        return

    man = report_verify(arch)
    if args.action == "verify":
        return
    copied = same = kept = 0
    for r in man.itertuples():
        src, dst = arch / r.file, ROOT / r.file
        if dst.exists():
            if dst.stat().st_size == r.bytes and sha256(dst) == r.sha256:
                same += 1
                continue
            if not args.overwrite:
                print(f"  exists with other content (kept; --overwrite replaces it): {r.file}")
                kept += 1
                continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        copied += 1
    print(f"restored into {ROOT}: {copied} copied, {same} already identical, {kept} kept")


if __name__ == "__main__":
    main()
