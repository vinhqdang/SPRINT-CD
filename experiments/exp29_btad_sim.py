"""Experiment 29: BTAD against PC, GES, DirectLiNGAM and PC with an iid nonparametric test
on simulated nonlinear DAGs with serial dependence.

Written and committed before running.  d = 12 nodes, n = 4000 time-ordered rows, edge
probability .2, Laplace errors, nonlinear additive mechanisms (as exp22), roots are
AR(1) with coefficient phi in {0, .9, .99}.  Scenario 'plain': nothing else.  Scenario
'extras': two near-duplicate sensors (a root-plus-noise copy added as nodes 13 and 14, true edge to
the original) and a slow latent drift (AR(.999)) added to three nodes.  Truth = the DAG skeleton
including the duplicate edges; dependences through the drift count as false for every method.
BTAD settings are frozen: K = 60, alpha = .2, max blanket 8, gap .05.  R = 40 DAGs per cell.
Metrics: skeleton precision, recall, F1.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import itertools, json, pathlib, sys, warnings
from multiprocessing import Pool
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from sprint_cd.btad import btad_skeleton
from sprint_cd.pc_gcm import pc_gcm_skeleton
from exp20_causal_chamber import pc, ges, lingam

FUN = [lambda x: x, lambda x: np.tanh(2 * x), lambda x: np.sin(1.5 * x), lambda x: x ** 2 - 1,
       lambda x: np.abs(x) - 0.8]
D, N = 12, 4000


def ar1(rng, n, phi, scale_noise):
    e = scale_noise
    x = np.zeros(n); x[0] = e[0]
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


def gen(r, phi, extras):
    rng = np.random.default_rng([20261200, r, int(phi * 100), int(extras)])
    order = rng.permutation(D)
    A = np.zeros((D, D), bool)
    for a in range(D):
        for b in range(a + 1, D):
            if rng.random() < 0.2:
                A[order[a], order[b]] = True
    drift_nodes = set(rng.choice(D, 3, replace=False)) if extras else set()
    u = ar1(rng, N, 0.999, rng.laplace(size=N) * 0.05) if extras else np.zeros(N)
    u = u / (u.std() + 1e-9)
    X = np.zeros((N, D))
    for v in order:
        e = rng.laplace(size=N)
        if not A[:, v].any():
            x = ar1(rng, N, phi, e * np.sqrt(max(1 - phi ** 2, 1e-3)))
            x = x / (x.std() + 1e-9)
        else:
            x = 0.7 * e
            for p in np.flatnonzero(A[:, v]):
                x = x + rng.choice([-1, 1]) * rng.uniform(0.6, 1.2) * FUN[rng.integers(5)](X[:, p])
        if v in drift_nodes:
            x = x + 0.8 * u * (x.std() + 1e-9)
        X[:, v] = x
    T = {(i, j) for i, j in itertools.combinations(range(D), 2) if A[i, j] or A[j, i]}
    if extras:
        src = rng.choice(D, 2, replace=False)
        dup = np.column_stack([X[:, s] + 0.2 * X[:, s].std() * rng.normal(size=N) for s in src])
        X = np.hstack([X, dup])
        T |= {(int(s), D + k) for k, s in enumerate(src)}
    return X, T


def prf(E, T):
    E = {tuple(sorted(p)) for p in E}
    tp, fp, fn = len(E & T), len(E - T), len(T - E)
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
    return [P, R, 2 * P * R / (P + R) if P + R else 0.0]


def job(a):
    r, phi, extras = a
    warnings.filterwarnings("ignore")
    X, T = gen(r, phi, extras)
    Z = (X - X.mean(0)) / X.std(0)
    out = {"r": r, "phi": phi, "extras": extras, "m": {}}
    for name, fn in (("BTAD", lambda Z: btad_skeleton(Z, K=60, alpha=0.2, max_blanket=8)),
                     ("PC-GCM a=0.01", lambda Z: pc_gcm_skeleton(Z, .01, 3)),
                     ("PC a=0.01", lambda Z: pc(Z, .01)), ("GES-BIC", ges), ("DirectLiNGAM", lingam)):
        try:
            out["m"][name] = prf(fn(Z), T)
        except Exception as e:
            out["m"][name] = None
    return out


def main(R=40):
    jobs = [(r, phi, ex) for ex in (0, 1) for phi in (0.0, 0.9, 0.99) for r in range(R)]
    with Pool(4) as p:
        recs = p.map(job, jobs, chunksize=2)
    (HERE.parent / "results" / "exp29_btad_sim.json").write_text(json.dumps({"records": recs}))


if __name__ == "__main__":
    main()
