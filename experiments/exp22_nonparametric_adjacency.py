"""Experiment 22: nonparametric sequential adjacency certificate vs the linear primitive.

Written before running.  Random DAGs (d = 8, edge probability .25) with
additive NONLINEAR mechanisms X_i = sum_{p in pa(i)} c_ip f_ip(X_p) + e_i, each
f drawn from {identity, tanh(2x), sin(1.5x), x^2 - 1, |x|}, Laplace or Gaussian
errors, n = 100 ... 3200 monitored.  Compared: CERT-CD with the linear Bayes
factor primitive (orientation off) and the nonparametric certificate with the
same level (alpha = .1, alpha_A = .05, k = 2).  Reported: P(ever a false edge),
precision and recall of the skeleton at n = 3200.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse, itertools, json, pathlib, sys, time, warnings
from multiprocessing import Pool
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from sprint_cd.certcd import CertCD, CertCDConfig
from sprint_cd.npadj import NPAdjacency

D, N, BATCH, WARM = 8, 3200, 100, 100
GRID = [100, 200, 400, 800, 1600, 3200]
FUN = [lambda x: x, lambda x: np.tanh(2 * x), lambda x: np.sin(1.5 * x),
       lambda x: x ** 2 - 1, lambda x: np.abs(x) - 0.8]


def gen(r, mech, noise, d=D, ep=0.25, n=N):
    rng = np.random.default_rng([20261040, r, 0])
    order = rng.permutation(d)
    A = np.zeros((d, d), bool)
    for a in range(d):
        for b in range(a + 1, d):
            if rng.random() < ep:
                A[order[a], order[b]] = True
    X = np.zeros((n, d))
    srng = np.random.default_rng([20261040, r, 1])
    for v in order:
        e = srng.laplace(size=n) if noise == "laplace" else srng.normal(size=n)
        x = 0.7 * e
        for p in np.flatnonzero(A[:, v]):
            f = FUN[rng.integers(len(FUN))] if mech == "nonlinear" else FUN[0]
            c = rng.choice([-1, 1]) * rng.uniform(0.6, 1.2)
            x = x + c * f(X[:, p])
        X[:, v] = x
    return X, A


def one(job):
    mech, noise, r = job
    warnings.filterwarnings("ignore")
    X, A = gen(r, mech, noise)
    X = (X - X[:100].mean(0)) / (X[:100].std(0, ddof=1) + 1e-12)
    T = {(i, j) for i, j in itertools.combinations(range(D), 2) if A[i, j] or A[j, i]}
    out = {"mech": mech, "noise": noise, "r": r, "truth": [list(e) for e in T]}
    a = CertCD(D, CertCDConfig(alpha=0.1, max_order=2, warmup=50, orient=False))
    a.warm_up(X[:50]); cur, lin = 50, []
    for n in GRID:
        for s in range(cur, n, BATCH):
            a.update(X[s:min(s + BATCH, n)])
        cur = n
        lin.append([list(p) for p in a.certified_edges()])
    b = NPAdjacency(D, 0.05 / (D * (D - 1) / 2), k=2, warm=WARM)
    cur, npz = 0, []
    for n in GRID:
        for s in range(cur, n, BATCH):
            b.update(X[s:min(s + BATCH, n)])
        cur = n
        npz.append([list(p) for p in b.certified_edges()])
    out["linear"], out["nonparam"] = lin, npz
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=60)
    a = ap.parse_args()
    jobs = [(m, nz, r) for m in ("linear", "nonlinear") for nz in ("laplace", "gaussian") for r in range(a.reps)]
    with Pool(4) as p:
        recs = p.map(one, jobs, chunksize=2)
    (HERE.parent / "results" / "exp22_nonparametric_adjacency.json").write_text(
        json.dumps({"config": {"reps": a.reps, "grid": GRID}, "records": recs}))


if __name__ == "__main__":
    main()
