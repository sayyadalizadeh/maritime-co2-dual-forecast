"""
check_figures.py: checks that the paper figures (src/paper_figures.py) meet the print requirements and that the axis
limit of Fig4 (b) hides no data.

The figures are drawn in memory (nothing is written by paper_figures.py). For every figure:
  - internal terms: no variable or file name, "rho", "gamma", "AutoETS" or "main window" in any text (tick labels
    included);
  - print size (mm) and the smallest font size (pt); every text must be at least 7 pt;
  - the exported files in figures/: PDF page size, TIFF pixel size, resolution and file size.
Fig4: the figure must contain every plotted array (line and marker coordinates, filled areas, bars; multiset
comparison) of the same figure drawn with the full axis and without the diagnostic line, so the axis limit of panel
(b) hides nothing; the extra arrays are the diagnostic line and the off-scale marker.
Output: results/figures_check.csv.

Usage:
  python src/check_figures.py [--overwrite]
"""
import argparse
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from PIL import Image

import paper_figures as new
from runlog import RunLog, out_path

FIGURES = ["Fig1", "Fig3", "Fig4", "Fig5", "Fig6", "Fig7"]
# internal names that must not appear in the figures (variable and file names)
FORBIDDEN = re.compile(r"\bD\d{1,2}\b|RES_|TER_|NRES_|DELTA|flag_|sep2025|\brho\b|gamma|AutoETS|artefact|track\b|"
                       r"main window|\.csv|\.parquet|results/|episode \(", re.IGNORECASE)


def data_arrays(fig):
    """Every plotted array of a figure, as hashable keys (kind, shape, rounded values)."""
    out = []
    for ax in fig.axes:
        arrs = [("line", l_.get_xydata()) for l_ in ax.lines]
        for c in ax.collections:
            if isinstance(c, LineCollection):
                seg = c.get_segments()
                arrs.append(("segments", np.concatenate(seg) if seg else np.empty((0, 2))))
            else:
                ps = [p.vertices for p in c.get_paths()]
                arrs.append(("area", np.concatenate(ps) if ps else np.empty((0, 2))))
        for p in ax.patches:
            if isinstance(p, Rectangle):
                arrs.append(("rect", np.array([p.get_x(), p.get_y(), p.get_width(), p.get_height()])))
            else:
                arrs.append(("patch", p.get_path().vertices))
        for kind, a in arrs:
            a = np.round(np.asarray(a, dtype=float), 9)
            out.append((kind, a.shape, a.tobytes()))
    return Counter(out)


def font_sizes(fig):
    """Font sizes of every visible text of a figure."""
    return [t.get_fontsize() for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean", default="data/clean")
    ap.add_argument("--results", default="results")
    ap.add_argument("--figures", default="figures")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    log = RunLog("check_figures.py", vars(args))
    figs = {}

    def keep(fig, name):
        figs[name] = fig

    new.build(args, keep)
    # Fig4 with the full axis and without the diagnostic line, as the reference of the axis-limit check
    res = Path(args.results)
    sh = pd.read_csv(res / "eval_fm_w" / "summary_by_horizon.csv")
    sh = sh[(sh.window == "main") & (sh.level == "economies") & (sh.grouping == "all") & (sh.metric == "MASE")]
    new.fig4(res, sh, new.SRC, keep, "_full_axis")

    rows = []
    for name in FIGURES:
        fn = figs[name]
        sizes = font_sizes(fn)
        r = {"figure": name, "n_arrays": sum(data_arrays(fn).values()),
             "internal_terms": " | ".join(sorted({m.group(0) for t in fn.findobj(Text) for m in FORBIDDEN.finditer(t.get_text())})),
             "min_font_pt": min(sizes),
             "width_mm": round(fn.get_size_inches()[0] * 25.4, 2), "height_mm": round(fn.get_size_inches()[1] * 25.4, 2)}
        if name == "Fig4":
            dr, dn = data_arrays(figs["Fig4_full_axis"]), data_arrays(fn)
            r.update({"arrays_only_full_axis": sum((dr - dn).values()), "arrays_only_limited": sum((dn - dr).values())})
        pdf, tif = Path(args.figures) / f"{name}.pdf", Path(args.figures) / f"{name}.tif"
        if pdf.exists():
            mb = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)\s*\]", pdf.read_bytes())
            r["pdf_mm"] = f"{float(mb.group(1)) / 72 * 25.4:.1f} x {float(mb.group(2)) / 72 * 25.4:.1f}"
            r["pdf_MB"] = round(pdf.stat().st_size / 1e6, 3)
        if tif.exists():
            im = Image.open(tif)
            r.update({"tiff_px": f"{im.size[0]} x {im.size[1]}", "tiff_dpi": round(float(im.info["dpi"][0])),
                      "tiff_mode": im.mode, "tiff_MB": round(tif.stat().st_size / 1e6, 2)})
        rows.append(r)
    plt.close("all")
    df = pd.DataFrame(rows)
    df.to_csv(out_path(res / "figures_check.csv", args.overwrite), index=False)
    print(df.to_string(index=False))
    ok = ((df.min_font_pt >= 7).all() and (df.internal_terms == "").all()
          and (df.arrays_only_full_axis.dropna() == 0).all())
    print("FONTS >= 7 pt, NO INTERNAL TERMS, FIG4 AXIS LIMIT HIDES NO DATA" if ok else "CHECK FAILED")
    log.finish("ok" if ok else "failed")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
