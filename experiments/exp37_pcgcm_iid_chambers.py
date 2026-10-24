"""Experiment 37: PC with the nonparametric test on the two i.i.d. chamber recordings (exp20 subsamples).
Written before running; fills the one missing cell of the graph benchmark table."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json, pathlib, sys, warnings
from multiprocessing import Pool
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from sprint_cd.pc_gcm import pc_gcm_skeleton
import exp20_causal_chamber as e20


def one(job):
    di, r = job
    warnings.filterwarnings("ignore")
    X, var, edges = e20.load(*e20.DATASETS[di])
    rng = np.random.default_rng([e20.SEED, di, r])
    X = X[rng.permutation(len(X))[:e20.NMAX]]
    Z = (X - X[:100].mean(0)) / (X[:100].std(0, ddof=1) + 1e-12)
    return {"ds": di, "r": r, "a05": pc_gcm_skeleton(Z, .05, 3), "a01": pc_gcm_skeleton(Z, .01, 3)}


def main():
    for di in range(len(e20.DATASETS)):
        e20.load(*e20.DATASETS[di])
    with Pool(4) as p:
        r = p.map(one, [(di, r) for di in range(len(e20.DATASETS)) for r in range(10)], chunksize=1)
    (HERE.parent / "results" / "exp37_pcgcm_iid_chambers.json").write_text(json.dumps({"records": r}))


if __name__ == "__main__":
    main()
