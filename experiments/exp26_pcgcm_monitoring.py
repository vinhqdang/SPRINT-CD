"""Experiment 26: PC with the same nonparametric test, under monitoring.

Written before running.  Regenerates the exp13 data (150 DAGs, d = 8, Laplace and
Gaussian errors, n up to 6400, seed 20261022) and runs PC (depth <= 3) with the
fixed-sample nonparametric conditional-independence test of sprint_cd.pc_gcm at
alpha = .05 and .01 on every prefix, so that it is compared with CERT-CD and
the other methods on identical data.  Reported as in exp13: P(a false edge is ever
in an estimate), precision and recall at each n.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json, pathlib, sys, warnings
from multiprocessing import Pool
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from sprint_cd.pc_gcm import pc_gcm_skeleton
from sprint_cd.simulate import random_dag

B, D = 20261022, 8
GRID = [100, 200, 400, 800, 1600, 3200, 6400]


def job(a):
    r, ci, cond = a
    warnings.filterwarnings("ignore")
    sem = random_dag(D, 0.25, np.random.default_rng([B, r, 0]), coef_low=0.5, coef_high=1.2)
    X = sem.sample(6400, np.random.default_rng([B, r, 1 + ci]), noise=cond)
    Xs = (X - X[:100].mean(0)) / X[:100].std(0, ddof=1)
    out = {"r": r, "cond": cond}
    for al in (0.05, 0.01):
        out[f"pcgcm_{al}"] = [pc_gcm_skeleton(Xs[:n], al, max_depth=3) for n in GRID]
    return out


def main():
    jobs = [(r, ci, c) for r in range(150) for ci, c in enumerate(("laplace", "gaussian"))]
    with Pool(4) as p:
        recs = p.map(job, jobs, chunksize=4)
    (HERE.parent / "results" / "exp26_pcgcm_monitoring.json").write_text(json.dumps({"grid": GRID, "records": recs}))


if __name__ == "__main__":
    main()
