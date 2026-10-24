"""Experiment 43: BF-BIC with BOSS (arXiv 2510.04276) on the graph benchmark.

Written and committed before running.  Frozen: Legendre degree p = 3, penalty discount 1, at most
5 parents, one random start.  Same data and rows as exp36: Sachs (853 cells, log scale), the 2 x 10
subsamples of 4000 shuffled rows of the i.i.d. Causal Chamber recordings, and the first 4000 rows
in time order of the 19 untouched recordings.  Skeleton only.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import itertools, json, pathlib, sys, warnings
from multiprocessing import Pool
import numpy as np
import joblib

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from sprint_cd.bfbic_boss import bfbic_boss_skeleton
from sprint_cd.learned import pair_features
from sprint_cd.notears import notears_linear
from sprint_cd.npadj import NPAdjacency
from sprint_cd.pc_gcm import pc_gcm_skeleton
import exp10_sachs as e10
import exp20_causal_chamber as e20
import exp27_chamber_fresh as e27
from exp20_causal_chamber import pc, ges, lingam


def nt_edges(Z):
    W = notears_linear(Z)
    d = Z.shape[1]
    return [[i, j] for i, j in itertools.combinations(range(d), 2) if W[i, j] != 0 or W[j, i] != 0]


def sachs():
    warnings.filterwarnings("ignore")
    X, names = e10.load_sachs()
    X = np.log(np.maximum(X, 1e-3))
    rng = np.random.default_rng(20261600)
    X = X[rng.permutation(len(X))]
    Z = (X - X.mean(0)) / X.std(0)
    d = Z.shape[1]
    idx = {n: k for k, n in enumerate(names)}
    truth = sorted({tuple(sorted((idx[a], idx[b]))) for a, b in e10.CONSENSUS})

    def npc():
        a = NPAdjacency(d, 0.05 / (d * (d - 1) / 2), k=3, warm=100)
        for s in range(0, len(Z), 50):
            a.update(Z[s:s + 50])
        return [list(q) for q in a.certified_edges()]

    out = {"BF-BIC+BOSS": e27.guarded(lambda: bfbic_boss_skeleton(Z), 1800)[0]}
    return {"names": names, "truth": [list(t) for t in truth], "edges": out}


def chamber_iid(job):
    di, r = job
    warnings.filterwarnings("ignore")
    X, var, edges = e20.load(*e20.DATASETS[di])
    rng = np.random.default_rng([e20.SEED, di, r])
    X = X[rng.permutation(len(X))[:e20.NMAX]]
    Z = (X - X[:100].mean(0)) / (X[:100].std(0, ddof=1) + 1e-12)
    v, e = e27.guarded(lambda: bfbic_boss_skeleton(Z), 1800)
    return {"kind": "iid", "ds": di, "r": r, "edges": v}


def chamber_walk(idx):
    warnings.filterwarnings("ignore")
    spec = e27.specs()[idx]
    X, var, edges = e27.load(spec)
    XA = X[:4000]
    Z = (XA - XA[:100].mean(0)) / (XA[:100].std(0, ddof=1) + 1e-12)
    v, e = e27.guarded(lambda: bfbic_boss_skeleton(Z), 1800)
    return {"kind": "walk", "spec": list(spec), "edges": v}


def main():
    for di in range(len(e20.DATASETS)):
        e20.load(*e20.DATASETS[di])
    n = len(e27.specs())
    res = {"sachs": sachs()}
    with Pool(4) as p:
        r1 = p.map(chamber_iid, [(di, r) for di in range(len(e20.DATASETS)) for r in range(10)], chunksize=1)
        r2 = p.map(chamber_walk, range(n), chunksize=1)
    res["chambers"] = r1 + r2
    (HERE.parent / "results" / "exp43_bfbic_boss.json").write_text(json.dumps(res))


if __name__ == "__main__":
    main()
