"""Experiment 16: relative-fit direction certificate on simulated pairs.

Nine families with known direction, fitted with the frozen settings of
sprint_cd.relative_fit.  Reports, per family, how often each certificate commits
and how often a committed direction is wrong.  The Gaussian-linear family has no
identified direction and is a negative control.
"""
from __future__ import annotations

import argparse
import sys
import pathlib
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import save_json, wilson_interval  # noqa: E402
from sprint_cd.relative_fit import certify_direction  # noqa: E402

FAMILIES = ["lin_laplace", "lin_unif", "lin_gauss", "anm_tanh", "anm_sq", "anm_sin",
            "het", "het_strong", "het_lap"]


def noise(kind, n, rng):
    return {"gauss": lambda: rng.normal(size=n), "laplace": lambda: rng.laplace(size=n),
            "unif": lambda: rng.uniform(-1.7, 1.7, size=n), "t5": lambda: rng.standard_t(5, size=n)}[kind]()


def gen(fam, n, rng):
    if fam == "lin_laplace":
        x = noise("laplace", n, rng); y = 0.8 * x + noise("laplace", n, rng)
    elif fam == "lin_unif":
        x = noise("unif", n, rng); y = 0.8 * x + noise("unif", n, rng)
    elif fam == "lin_gauss":
        x = noise("gauss", n, rng); y = 0.8 * x + noise("gauss", n, rng)
    elif fam == "anm_tanh":
        x = rng.normal(size=n); y = np.tanh(2 * x) + 0.3 * noise("laplace", n, rng)
    elif fam == "anm_sq":
        x = rng.normal(size=n); y = x ** 2 + 0.5 * noise("t5", n, rng)
    elif fam == "anm_sin":
        x = rng.uniform(-3, 3, size=n); y = np.sin(x) + 0.3 * noise("gauss", n, rng)
    elif fam == "het":
        x = rng.normal(size=n); y = 0.7 * x + np.sqrt(0.2 + 0.8 * x ** 2) * noise("gauss", n, rng)
    elif fam == "het_strong":
        x = rng.normal(size=n); y = 0.3 * x + np.exp(0.8 * x) * 0.4 * noise("gauss", n, rng)
    else:
        x = rng.uniform(-2, 2, size=n); y = np.sin(x) + np.sqrt(0.1 + 0.5 * x ** 2) * noise("laplace", n, rng)
    return x, y


def one(a):
    fam, n, seed = a
    rng = np.random.default_rng(seed)
    x, y = gen(fam, n, rng)
    flip = rng.random() < 0.5
    if flip:
        x, y = y, x
    r = certify_direction(x, y, seed=seed)
    return fam, ("y->x" if flip else "x->y"), r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    out = {"config": vars(a), "families": {}}
    with Pool(a.workers) as p:
        for fam in FAMILIES:
            res = p.map(one, [(fam, a.n, 1000 * k + 11) for k in range(a.reps)])
            row = {}
            for kind in ("agree", "gg", "ggh"):
                dec = [r for r in res if r[2][kind] is not None]
                ok = sum(r[2][kind] == r[1] for r in dec)
                row[kind] = {"decided": len(dec), "correct": ok, "wrong": len(dec) - ok,
                             "coverage": len(dec) / a.reps,
                             "wrong_rate_ci": list(wilson_interval(len(dec) - ok, a.reps))}
            for f in ("forced_gg", "forced_ggh"):
                row[f] = sum(r[2][f] == r[1] for r in res) / a.reps
            out["families"][fam] = row
            print(fam, {k: (v["decided"], v["wrong"]) if isinstance(v, dict) else v for k, v in row.items()}, flush=True)
            save_json("exp16_pair_orientation_sim", out)


if __name__ == "__main__":
    main()
