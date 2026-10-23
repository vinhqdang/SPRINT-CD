"""Experiment 23: nonparametric adjacency certificate on the Causal Chambers.

Post-hoc redesign, disclosed.  exp20 showed that the linear certificate (like
every comparator) reports edges that the chamber graph does not contain.  This
script (a) runs the nonparametric certificate of sprint_cd.npadj on the SAME
subsamples as exp20, and (b) builds a data-based reference set V of
non-adjacent pairs for which the conditional independence implied by the graph
(given the parents of the non-descendant, additive spline regression) is
rejected on 20000 rows at |z| > 5.  These are dependences the graph itself cannot
explain.  A certified edge is then scored three ways: strict (must be a graph
edge), adjusted (a graph edge or a pair in V).  The same scoring is applied to
the exp20 comparators.  V does not depend on any monitored algorithm.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import itertools, json, pathlib, sys, warnings
from multiprocessing import Pool
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from exp20_causal_chamber import DATASETS, GRID, NMAX, SEED, load
from sprint_cd.npadj import NPAdjacency

BATCH, WARM = 100, 100


def _basis(Z):
    cols = [np.ones(len(Z))]
    for a in range(Z.shape[1]):
        x = np.clip(Z[:, a], *np.quantile(Z[:, a], [.01, .99]))
        q = np.quantile(x, [.2, .4, .6, .8])
        cols += [x, x ** 2, x ** 3] + [np.maximum(x - t, 0) ** 3 for t in q]
    return np.column_stack(cols)


def _resid(y, Z):
    B = _basis(Z) if Z.shape[1] else np.ones((len(y), 1))
    pen = np.eye(B.shape[1]); pen[0, 0] = 0
    return y - B @ np.linalg.solve(B.T @ B + pen, B.T @ y)


def reference_set(di):
    X, var, edges = load(*DATASETS[di])
    d = len(var)
    rng = np.random.default_rng(1)
    X = X[rng.permutation(len(X))[:20000]]
    X = (X - X.mean(0)) / X.std(0)
    A = np.zeros((d, d), bool)
    for a, b in edges:
        A[a, b] = True
    R = A.copy()
    for _ in range(d):
        R = R | (R.astype(int) @ A.astype(int) > 0)
    V = []
    for i, j in itertools.combinations(range(d), 2):
        if A[i, j] or A[j, i]:
            continue
        if R[i, j]:
            i, j = j, i
        if R[i, j]:
            continue
        pa = list(np.flatnonzero(A[:, i]))
        z = _resid(X[:, i], X[:, pa]) * _resid(X[:, j], X[:, pa])
        t = z.mean() / z.std() * np.sqrt(len(z))
        if abs(t) > 5:
            V.append([min(i, j), max(i, j), float(t)])
    return V


def one(job):
    di, r = job
    warnings.filterwarnings("ignore")
    X, var, edges = load(*DATASETS[di])
    rng = np.random.default_rng([SEED, di, r])
    X = X[rng.permutation(len(X))[:NMAX]]
    Xs = (X - X[:100].mean(0)) / (X[:100].std(0, ddof=1) + 1e-12)
    d = Xs.shape[1]
    a = NPAdjacency(d, 0.05 / (d * (d - 1) / 2), k=2, warm=WARM)
    cur, out = 0, []
    for n in GRID:
        for s in range(cur, n, BATCH):
            a.update(Xs[s:min(s + BATCH, n)])
        cur = n
        out.append([list(p) for p in a.certified_edges()])
    return {"ds": di, "r": r, "nonparam": out}


def main(reps=10):
    with Pool(4) as p:
        recs = p.map(one, [(di, r) for di in range(len(DATASETS)) for r in range(reps)], chunksize=1)
        V = p.map(reference_set, range(len(DATASETS)))
    (HERE.parent / "results" / "exp23_chamber_nonparam.json").write_text(
        json.dumps({"reference_set": V, "records": recs, "grid": GRID}))


if __name__ == "__main__":
    main()
