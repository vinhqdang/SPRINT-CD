"""Experiment 45: published methods with public code on the graph benchmark.

Written and committed before running.  Methods (library defaults, see sprint_cd/sota_wrappers.py):
AVICI (pretrained scm-v0, edge probability >= 0.5), GOLEM, NOTEARS-MLP, GraN-DAG, DAG-GNN (gCastle),
GRaSP and BOSS (causal-learn, BIC), RESIT and CAM-UV (lingam).  Data and reference graphs as in exp36:
Sachs (853 cells, log scale), the 2 x 10 subsamples of the i.i.d. Chamber recordings and the 19 untouched
recordings; every method sees at most 2000 rows (the first 2000 in time order for the recordings with
serial dependence, a random 2000 of the shuffled rows otherwise), standardised on the first 100 rows
(Sachs: all rows).  A method that exceeds 900 s on a data set is recorded as a timeout and is excluded from
that cell's mean.  Skeleton only.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
import json, pathlib, sys, warnings
from multiprocessing import Pool
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from sprint_cd.sota_wrappers import METHODS
import exp10_sachs as e10
import exp20_causal_chamber as e20
import exp27_chamber_fresh as e27

NMAX = 2000


def run_all(Z):
    import torch; torch.set_num_threads(1)
    out = {}
    for name, fn in METHODS.items():
        out[name], err = e27.guarded(lambda: fn(Z), 900)
        if err: out[name + " err"] = err
    return out


def sachs(_):
    warnings.filterwarnings("ignore")
    X, names = e10.load_sachs(); X = np.log(np.maximum(X, 1e-3))
    X = X[np.random.default_rng(20261600).permutation(len(X))]
    Z = (X - X.mean(0)) / X.std(0)
    idx = {n: k for k, n in enumerate(names)}
    truth = sorted({tuple(sorted((idx[a], idx[b]))) for a, b in e10.CONSENSUS})
    return {"kind": "sachs", "truth": [list(t) for t in truth], "edges": run_all(Z)}


def chamber_iid(job):
    warnings.filterwarnings("ignore")
    di, r = job
    X, var, edges = e20.load(*e20.DATASETS[di])
    rng = np.random.default_rng([e20.SEED, di, r])
    X = X[rng.permutation(len(X))[:NMAX]]
    Z = (X - X[:100].mean(0)) / (X[:100].std(0, ddof=1) + 1e-12)
    return {"kind": "iid", "ds": di, "r": r, "edges": run_all(Z)}


def chamber_walk(idx):
    warnings.filterwarnings("ignore")
    spec = e27.specs()[idx]
    X, var, edges = e27.load(spec)
    XA = X[:NMAX]
    Z = (XA - XA[:100].mean(0)) / (XA[:100].std(0, ddof=1) + 1e-12)
    return {"kind": "walk", "spec": list(spec), "edges": run_all(Z)}


def _call(a):
    return globals()[a[0]](a[1])


def main():
    for di in range(len(e20.DATASETS)): e20.load(*e20.DATASETS[di])
    n = len(e27.specs())
    jobs = [("sachs", 0)] + [("chamber_iid", (di, r)) for di in range(len(e20.DATASETS)) for r in range(10)] + [("chamber_walk", i) for i in range(n)]
    part = HERE.parent / "results" / "exp45_sota_graphs.partial.json"
    recs = json.loads(part.read_text())["records"] if part.exists() else []
    todo = jobs[len(recs):]
    with Pool(4) as p:
        for s in range(0, len(todo), 4):
            recs.extend(p.map(_call, todo[s:s + 4], chunksize=1))
            part.write_text(json.dumps({"records": recs})); print(len(recs), "/", len(jobs), flush=True)
    (HERE.parent / "results" / "exp45_sota_graphs.json").write_text(json.dumps({"records": recs}))
    part.unlink()


if __name__ == "__main__":
    main()
