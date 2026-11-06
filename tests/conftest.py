"""
Shared set-up of the test suite.

Two kinds of tests:
  - core tests run on small synthetic fixtures (the three-economy example of examples/toy_dual_allocation.py and
    hand-made cases) and need no stored outputs: they run in a fresh clone;
  - stored-output tests read the archive of stored outputs. The archive mirrors the repository tree (results/,
    data/clean/) and is looked for in the folder given by the environment variable MARITIME_ARCHIVE, else in
    ./archive (the archive folder, see README). Without the archive these tests are skipped with a message. After a
    full run of the pipeline they can read the repository's own outputs (set MARITIME_ARCHIVE to the repository
    root).

Run from the repository root:
  python -m pytest tests -v
"""
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
ARCHIVE = Path(os.environ.get("MARITIME_ARCHIVE", ROOT / "archive")).resolve()
RES, CLEAN = ARCHIVE / "results", ARCHIVE / "data" / "clean"


def need(*paths):
    """Skip the test if a stored output is missing."""
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        pytest.skip(f"stored outputs not found in the archive folder {ARCHIVE} (download the archive into ./archive or "
                    "set MARITIME_ARCHIVE; see README): " + ", ".join(missing[:3]) + (" ..." if len(missing) > 3 else ""))


def load_toy():
    """The module examples/toy_dual_allocation.py."""
    spec_ = importlib.util.spec_from_file_location("toy", ROOT / "examples" / "toy_dual_allocation.py")
    toy = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(toy)
    return toy


def toy_spec(toy, totals=None):
    """The three-economy system of the synthetic example (economies, Residual, World, the paper's identities)."""
    from dual_allocation import PAPER_LETTERS, Spec
    return Spec(economies=toy.UNITS[:3], residual="REST", world="W", components=list(PAPER_LETTERS.values()),
                totals=totals or {"RES_TOTAL_SYS": "A+B+D+E+F", "TER_DOM": "A+C", "TER_INT": "D+G"},
                letters=PAPER_LETTERS, balances={"b1": "B-C", "b2": "E+F-G-DELTA_INT"}, world_only=["DELTA_INT"])


@pytest.fixture(scope="session")
def toy_system():
    """Synthetic system: the built constraint matrices and coherent flows y (C y = 0) from random voyage flows."""
    from dual_allocation import build
    toy = load_toy()
    s = build(toy_spec(toy))
    rng = np.random.default_rng(1)
    n = len(toy.UNITS)
    Dom = rng.gamma(2.0, 1.0, (n, n)) * 10
    Int = rng.gamma(2.0, 1.0, (n, n, n)) * 10
    Int[:, np.arange(n), np.arange(n)] = 0.0
    y = toy.vector(s, toy.components(Dom, Int))
    return {"toy": toy, "spec": s, "y": y, "C": s.C.toarray(), "A": s.C_fullrank.toarray()}
