"""Run the LOCI estimator of Immer et al. (ICML 2023) from the authors' public code.

The source files are fetched at run time from https://github.com/AlexImmer/loci into
``data/loci_src`` (they are not redistributed with this repository).  Requires torch,
scikit-learn, tueplots and gin-config.  ``loci_score(x, y)`` > 0 means x -> y.
"""
from __future__ import annotations

import pathlib
import sys
import urllib.request

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "loci_src"
BASE = "https://raw.githubusercontent.com/AlexImmer/loci/main/"
FILES = ["causa/__init__.py", "causa/loci.py", "causa/het_ridge.py", "causa/hsic.py", "causa/ml.py", "causa/utils.py"]


def _fetch():
    for f in FILES:
        p = SRC / f
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(urllib.request.urlopen(BASE + f, timeout=60).read())
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))


def loci_score(x, y, neural_network=True, n_steps=None, seed=0):
    _fetch()
    import torch
    from causa.loci import loci
    torch.set_num_threads(1)
    x = np.asarray(x, float); y = np.asarray(y, float)
    x = (x - x.mean()) / (x.std() + 1e-12); y = (y - y.mean()) / (y.std() + 1e-12)
    return float(loci(x, y, independence_test=True, neural_network=neural_network, n_steps=n_steps))
