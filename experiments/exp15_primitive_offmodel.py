"""Experiment 15: calibration of the adjacency primitive off its model.

The exact guarantee of the safe-linear primitive needs Gaussian errors.  Under a
linear SEM with other noise laws, or with conditional heteroscedasticity, the
d-separation null is a zero partial regression coefficient and the guarantee is
only asymptotic.  This experiment measures what that costs: the probability that
the running maximum of the e-process ever reaches 1/alpha when the null holds,
monitored after every observation from n = 50 to 1000.

Two primitives are compared: the Gaussian Bayes-factor primitive used by CERT-CD,
and the heteroscedasticity-robust 't' e-process of Lindon et al. (arXiv:2210.08589,
Thm 5.4), monitored every 5 observations because it refits at each time.
"""
from __future__ import annotations

import argparse
import sys
import pathlib
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import running_cross_products, residual_moments_at, save_json, wilson_interval  # noqa: E402
from sprint_cd.eprocess.safe_linear import safe_linear_log_e  # noqa: E402
from sprint_cd.stats import ResidualMoments  # noqa: E402

LAWS = ["gauss", "laplace", "t2.5", "lognormal"]
SCEN = ["S0", "chain", "het"]
G_BF = 0.25
G_ROB = 10.0


def noise(law, n, rng):
    return {"gauss": lambda: rng.normal(size=n), "laplace": lambda: rng.laplace(size=n),
            "t2.5": lambda: rng.standard_t(2.5, size=n),
            "lognormal": lambda: rng.lognormal(size=n) - np.exp(0.5)}[law]()


def data(law, scen, seed, nmax):
    rng = np.random.default_rng(seed)
    if scen == "S0":
        return np.column_stack([noise(law, nmax, rng), noise(law, nmax, rng)]), ()
    if scen == "het":
        s = rng.normal(size=nmax)
        h = np.sqrt(1 + s ** 2)
        return np.column_stack([h * noise(law, nmax, rng), h * noise(law, nmax, rng), s]), (2,)
    s = noise(law, nmax, rng)
    return np.column_stack([.8 * s + noise(law, nmax, rng), .8 * s + noise(law, nmax, rng), s]), (2,)


def robust_loge(y, x, Z, g):
    n = len(y)
    A = np.column_stack([np.ones(n), Z]) if Z.shape[1] else np.ones((n, 1))
    k = A.shape[1] + 1
    nu = n - k
    if nu <= 2:
        return 0.0
    cx, *_ = np.linalg.lstsq(A, x, rcond=None)
    rt = x - A @ cx
    sxx = rt @ rt
    if sxx <= 0:
        return 0.0
    b = (rt @ y) / sxx
    B = np.column_stack([A, x])
    cy, *_ = np.linalg.lstsq(B, y, rcond=None)
    e = y - B @ cy
    V = np.sum(e ** 2 * rt ** 2) / sxx ** 2
    Q = b * b / V
    return 0.5 * np.log(g / (g + n)) - (nu + 1) / 2 * (np.log1p(g * Q / ((g + n) * nu)) - np.log1p(Q / nu))


def one(a):
    law, scen, seed, nmax = a
    D, cond = data(law, scen, seed, nmax)
    cs, cc = running_cross_products(D)
    mb = -np.inf
    for t in range(49, nmax):
        sxx, sxy, syy, dof = residual_moments_at(cs, cc, t, 0, 1, cond)
        if dof > 0:
            mb = max(mb, safe_linear_log_e(ResidualMoments(sxx, sxy, syy, dof, t + 1), G_BF))
    y, x, Z = D[:, 0], D[:, 1], D[:, 2:]
    mr = -np.inf
    for n in range(40, nmax + 1, 5):
        mr = max(mr, robust_loge(y[:n], x[:n], Z[:n], G_ROB))
    return mb, mr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--nmax", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    out = {"config": vars(a), "g_bf": G_BF, "g_robust": G_ROB, "cells": []}
    with Pool(a.workers) as p:
        for scen in SCEN:
            for law in LAWS:
                m = np.array(p.map(one, [(law, scen, 10 * k + 3, a.nmax) for k in range(a.reps)]))
                cell = {"scenario": scen, "law": law}
                for name, col in (("bayes_factor", 0), ("robust_t", 1)):
                    for alpha in (0.05, 0.01):
                        k = int(np.sum(m[:, col] >= np.log(1 / alpha)))
                        lo, hi = wilson_interval(k, a.reps)
                        cell[f"{name}_a{alpha}"] = {"k": k, "rate": k / a.reps, "ci": [lo, hi]}
                out["cells"].append(cell)
                print(scen, law, {k: v["rate"] for k, v in cell.items() if isinstance(v, dict)}, flush=True)
                save_json("exp15_primitive_offmodel", out)


if __name__ == "__main__":
    main()
