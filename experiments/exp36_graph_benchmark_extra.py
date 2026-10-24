"""Experiment 36: Sachs data and NOTEARS, added to the graph benchmark.

Written and committed before running.  (a) The observational Sachs data (853 cells, 11 proteins,
log-transformed as in exp10), reference = the 17-arc consensus network, all methods:
PC (.05, .01), GES-BIC, DirectLiNGAM, NOTEARS-linear (lambda .1, threshold .3), PC with the
nonparametric test (.05, .01), BTAD (K = min(60, n // 50)), the frozen LRG model, CERT-CD
nonparametric (k = 3, rows streamed in a random order).  (b) NOTEARS-linear on the Causal Chamber
data of the other experiments: the 2 x 10 subsamples of 4000 shuffled rows of the i.i.d.
recordings (exp20) and the first 4000 rows in time order of the 19 untouched recordings (exp30).
Skeleton only.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import itertools, json, pathlib, sys, warnings
from multiprocessing import Pool
import numpy as np
import joblib

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from sprint_cd.btad import btad_skeleton
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
    mdl = joblib.load(HERE.parent / "results" / "lrg_model.joblib")
    pairs, F = pair_features(Z)
    p = mdl["model"].predict_proba(F)[:, 1]

    def npc():
        a = NPAdjacency(d, 0.05 / (d * (d - 1) / 2), k=3, warm=100)
        for s in range(0, len(Z), 50):
            a.update(Z[s:s + 50])
        return [list(q) for q in a.certified_edges()]

    out = {}
    for name, fn in (("PC a=0.05", lambda: pc(Z, .05)), ("PC a=0.01", lambda: pc(Z, .01)), ("GES-BIC", lambda: ges(Z)),
                     ("DirectLiNGAM", lambda: lingam(Z)), ("NOTEARS", lambda: nt_edges(Z)),
                     ("PC-GCM a=0.05", lambda: pc_gcm_skeleton(Z, .05, 3)), ("PC-GCM a=0.01", lambda: pc_gcm_skeleton(Z, .01, 3)),
                     ("BTAD", lambda: btad_skeleton(Z, K=min(60, len(Z) // 50), alpha=0.2, max_blanket=8)),
                     ("CERT-CD nonparametric k=3", npc)):
        out[name], _ = e27.guarded(fn, 900)
    out["LRG"] = [list(map(int, q)) for q, pr in zip(pairs, p) if pr >= mdl["threshold"]]
    return {"names": names, "truth": [list(t) for t in truth], "edges": out}


def chamber_iid(job):
    di, r = job
    warnings.filterwarnings("ignore")
    X, var, edges = e20.load(*e20.DATASETS[di])
    rng = np.random.default_rng([e20.SEED, di, r])
    X = X[rng.permutation(len(X))[:e20.NMAX]]
    Z = (X - X[:100].mean(0)) / (X[:100].std(0, ddof=1) + 1e-12)
    v, e = e27.guarded(lambda: nt_edges(Z), 900)
    return {"kind": "iid", "ds": di, "r": r, "edges": v}


def chamber_walk(idx):
    warnings.filterwarnings("ignore")
    spec = e27.specs()[idx]
    X, var, edges = e27.load(spec)
    XA = X[:4000]
    Z = (XA - XA[:100].mean(0)) / (XA[:100].std(0, ddof=1) + 1e-12)
    v, e = e27.guarded(lambda: nt_edges(Z), 900)
    return {"kind": "walk", "spec": list(spec), "edges": v}


def main():
    for di in range(len(e20.DATASETS)):
        e20.load(*e20.DATASETS[di])
    n = len(e27.specs())
    res = {"sachs": sachs()}
    with Pool(4) as p:
        r1 = p.map(chamber_iid, [(di, r) for di in range(len(e20.DATASETS)) for r in range(10)], chunksize=1)
        r2 = p.map(chamber_walk, range(n), chunksize=1)
    res["notears_chambers"] = r1 + r2
    (HERE.parent / "results" / "exp36_graph_benchmark_extra.json").write_text(json.dumps(res))


if __name__ == "__main__":
    main()
