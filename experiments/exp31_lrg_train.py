"""Experiment 31: train the learned replicated-gain skeleton model (LRG) on simulated data.

Training prior (random per data set, all time-ordered, n = 4000): d in {8..16}; structure
random (edge prob .15-.3) or bipartite 'device-like' (inputs -> outputs); mechanisms linear or
nonlinear additive; Laplace / Gaussian / t5 / heteroscedastic noise; roots AR(1) with
phi in {0, .5, .9, .97, .99} or a random walk; effect sizes from weak to strong; optional
latent slow drift feeding 2-5 nodes (dependence through it counts as NO edge); optional
near-duplicate sensors (true edge to the original).  Training and validation data sets use
seeds disjoint from the held-out simulation test.  The decision threshold is chosen on the
validation split to maximise F1 and then frozen.  No Causal Chamber data are used.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import itertools, json, pathlib, sys, warnings
from multiprocessing import Pool
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from sprint_cd.learned import pair_features, FEATURE_NAMES

FUN = [lambda x: x, lambda x: np.tanh(2 * x), lambda x: np.sin(1.5 * x), lambda x: x ** 2 - 1,
       lambda x: np.abs(x) - 0.8, lambda x: np.exp(-x ** 2) - 0.6, lambda x: np.sign(x) * np.sqrt(np.abs(x))]
N = 4000


def ar(rng, n, phi, noise):
    if phi >= 1.0:
        x = np.cumsum(noise)
        return x
    x = np.zeros(n); x[0] = noise[0]
    for t in range(1, n):
        x[t] = phi * x[t - 1] + noise[t]
    return x


def noise_draw(rng, kind, n):
    return {"laplace": rng.laplace(size=n), "gauss": rng.normal(size=n), "t5": rng.standard_t(5, size=n)}[kind]


def gen(seed):
    rng = np.random.default_rng([20261300, seed])
    d = int(rng.choice([8, 10, 12, 14, 16]))
    kind = str(rng.choice(["laplace", "gauss", "t5"]))
    phi = float(rng.choice([0.0, 0.5, 0.9, 0.97, 0.99, 1.0]))
    nonlin = rng.random() < 0.8
    hetero = rng.random() < 0.3
    bip = rng.random() < 0.4
    A = np.zeros((d, d), bool)
    order = rng.permutation(d)
    if bip:
        k_in = max(2, d // 3)
        ins, outs = order[:k_in], order[k_in:]
        p = rng.uniform(0.25, 0.5)
        for a in ins:
            for b in outs:
                if rng.random() < p:
                    A[a, b] = True
    else:
        p = rng.uniform(0.12, 0.3)
        for a in range(d):
            for b in range(a + 1, d):
                if rng.random() < p:
                    A[order[a], order[b]] = True
    n_lat = int(rng.choice([0, 1, 2]))
    lat = []
    for _ in range(n_lat):
        u = ar(rng, N, float(rng.choice([0.99, 0.995, 0.999])), rng.normal(size=N))
        lat.append((u / (u.std() + 1e-9), rng.choice(d, int(rng.integers(2, 6)), replace=False)))
    X = np.zeros((N, d))
    wmin = rng.uniform(0.1, 0.5)
    for v in order:
        e = noise_draw(rng, kind, N) * rng.uniform(0.3, 1.0)
        pa = np.flatnonzero(A[:, v])
        if len(pa) == 0:
            x = ar(rng, N, phi, e * (np.sqrt(max(1 - phi ** 2, 1e-3)) if phi < 1 else 1.0))
            x = (x - x.mean()) / (x.std() + 1e-9)
        else:
            x = e.copy()
            for p_ in pa:
                f = FUN[rng.integers(len(FUN))] if nonlin else FUN[0]
                x = x + rng.choice([-1, 1]) * rng.uniform(wmin, 1.3) * f(X[:, p_])
            if hetero:
                x = x * (1 + 0.5 * np.abs(X[:, pa[0]]))
        for u, nodes in lat:
            if v in nodes:
                x = x + rng.uniform(0.3, 1.0) * u * (x.std() + 1e-9)
        X[:, v] = x
    T = {(i, j) for i, j in itertools.combinations(range(d), 2) if A[i, j] or A[j, i]}
    if rng.random() < 0.5:
        k = int(rng.integers(1, 4))
        src = rng.choice(d, k, replace=False)
        dup = np.column_stack([X[:, s] + rng.uniform(0.05, 0.3) * X[:, s].std() * rng.normal(size=N) for s in src])
        X = np.hstack([X, dup])
        T |= {(int(s), d + q) for q, s in enumerate(src)}
    return X, T


def one(seed):
    warnings.filterwarnings("ignore")
    X, T = gen(seed)
    pairs, F = pair_features((X - X.mean(0)) / X.std(0))
    y = np.array([1 if tuple(sorted(p)) in T else 0 for p in pairs])
    return seed, F, y


def main(n_train=600, n_val=150, n_test=150):
    seeds = list(range(n_train + n_val + n_test))
    with Pool(4) as p:
        res = p.map(one, seeds, chunksize=4)
    np.savez_compressed(HERE.parent / "results" / "exp31_features.npz",
                        seed=np.concatenate([[s] * len(y) for s, F, y in res]),
                        F=np.vstack([F for s, F, y in res]), y=np.concatenate([y for s, F, y in res]),
                        n_train=n_train, n_val=n_val, n_test=n_test)


if __name__ == "__main__":
    main()
