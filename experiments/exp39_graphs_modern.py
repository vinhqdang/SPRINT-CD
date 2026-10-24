"""Experiment 39: DAGMA and PCMCI+ added to the graph benchmark.

Written and committed before running.  DAGMA (linear, l2 loss, lambda1 = .03, threshold .3;
Bello et al., NeurIPS 2022, package dagma) and PCMCI+ (tigramite, ParCorr, tau in {0, 1},
pc_alpha .01; Runge, UAI 2020; skeleton from the contemporaneous links) on the same data as
exp36: the Sachs cells (log-transformed, rows in a random order), the 2 x 10 subsamples of
4000 shuffled rows of the two i.i.d. chamber recordings, and the first 4000 rows in time order of
the 19 new chamber recordings.  Skeleton only.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import itertools, json, pathlib, sys, warnings
from multiprocessing import Pool
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import exp10_sachs as e10
import exp20_causal_chamber as e20
import exp27_chamber_fresh as e27


def dagma_edges(Z):
    from dagma.linear import DagmaLinear
    W = DagmaLinear(loss_type="l2", verbose=False).fit(Z, lambda1=0.03)
    d = Z.shape[1]
    return [[i, j] for i, j in itertools.combinations(range(d), 2) if abs(W[i, j]) > 0.3 or abs(W[j, i]) > 0.3]


def pcmci_edges(Z):
    from tigramite import data_processing as pp
    from tigramite.independence_tests.parcorr import ParCorr
    from tigramite.pcmci import PCMCI
    res = PCMCI(dataframe=pp.DataFrame(Z), cond_ind_test=ParCorr(), verbosity=0).run_pcmciplus(tau_min=0, tau_max=1, pc_alpha=0.01)
    g = res["graph"]
    d = Z.shape[1]
    return [[i, j] for i, j in itertools.combinations(range(d), 2) if g[i, j, 0] != "" or g[j, i, 0] != ""]


def run(Z):
    out = {}
    for name, fn in (("DAGMA", dagma_edges), ("PCMCI+", pcmci_edges)):
        out[name], _ = e27.guarded(lambda: fn(Z), 900)
    return out


def sachs():
    warnings.filterwarnings("ignore")
    X, names = e10.load_sachs(); X = np.log(np.maximum(X, 1e-3))
    X = X[np.random.default_rng(20261600).permutation(len(X))]
    return {"kind": "sachs", "edges": run((X - X.mean(0)) / X.std(0))}


def iid(job):
    di, r = job
    warnings.filterwarnings("ignore")
    X, var, edges = e20.load(*e20.DATASETS[di])
    X = X[np.random.default_rng([e20.SEED, di, r]).permutation(len(X))[:e20.NMAX]]
    return {"kind": "iid", "ds": di, "r": r, "edges": run((X - X[:100].mean(0)) / (X[:100].std(0, ddof=1) + 1e-12))}


def walk(idx):
    warnings.filterwarnings("ignore")
    spec = e27.specs()[idx]; X, var, edges = e27.load(spec); XA = X[:4000]
    return {"kind": "walk", "spec": list(spec), "edges": run((XA - XA[:100].mean(0)) / (XA[:100].std(0, ddof=1) + 1e-12))}


def main():
    for di in range(len(e20.DATASETS)): e20.load(*e20.DATASETS[di])
    n = len(e27.specs())
    res = {"sachs": sachs()}
    with Pool(4) as p:
        r = p.map(iid, [(di, r) for di in range(len(e20.DATASETS)) for r in range(10)], chunksize=1)
        r += p.map(walk, range(n), chunksize=1)
    res["chambers"] = r
    (HERE.parent / "results" / "exp39_graphs_modern.json").write_text(json.dumps(res))


if __name__ == "__main__":
    main()
