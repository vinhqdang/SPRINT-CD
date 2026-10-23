"""Experiment 25: the nonparametric certificate on the exp13 and exp17 DAG data.

Written before running.  Regenerates exactly the data of exp13 (150 DAGs, d = 8,
Laplace and Gaussian, n up to 6400, seed 20261022) and of the exp17 scaling
suite (d = 4, 8, 12, 16, 60 DAGs, n up to 3200, seed 20261024) and runs the
nonparametric certificate (k = 2, alpha_A = .05, same family-wise split) on the
same prefixes, so it can be compared with the stored estimates of CERT-CD
(linear primitive), PC, GES and DirectLiNGAM.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json, pathlib, sys, warnings
from multiprocessing import Pool
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from sprint_cd.npadj import NPAdjacency
from sprint_cd.simulate import random_dag
import exp17_scaling_ablation as e17

B13, D13, NMAX13 = 20261022, 8, 6400
GRID13 = [100, 200, 400, 800, 1600, 3200, 6400]


def stream(Xs, d, grid):
    a = NPAdjacency(d, 0.05 / (d * (d - 1) / 2), k=2, warm=100)
    cur, out = 0, []
    for n in grid:
        for s in range(cur, n, 100):
            a.update(Xs[s:min(s + 100, n)])
        cur = n
        out.append([list(p) for p in a.certified_edges()])
    return out


def job13(a):
    r, ci, cond = a
    warnings.filterwarnings("ignore")
    sem = random_dag(D13, 0.25, np.random.default_rng([B13, r, 0]), coef_low=0.5, coef_high=1.2)
    X = sem.sample(NMAX13, np.random.default_rng([B13, r, 1 + ci]), noise=cond)
    Xs = (X - X[:100].mean(0)) / X[:100].std(0, ddof=1)
    return {"r": r, "cond": cond, "edges": stream(Xs, D13, GRID13)}


def job17(a):
    d, r = a
    warnings.filterwarnings("ignore")
    ep = min(0.5, 1.75 / (d - 1))
    Xs, truth = e17._data(d, ep, r, 1)
    return {"d": d, "r": r, "edges": stream(Xs, d, e17.GRID)}


def main():
    with Pool(4) as p:
        r13 = p.map(job13, [(r, ci, c) for r in range(150) for ci, c in enumerate(("laplace", "gaussian"))], chunksize=4)
        r17 = p.map(job17, [(d, r) for d in e17.SIZES for r in range(60)], chunksize=2)
    (HERE.parent / "results" / "exp25_nonparam_monitoring.json").write_text(json.dumps({"exp13": r13, "exp17": r17}))


if __name__ == "__main__":
    main()
