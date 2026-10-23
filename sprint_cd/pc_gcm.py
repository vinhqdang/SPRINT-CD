"""PC skeleton with a fixed-sample nonparametric conditional-independence test.

Comparator for the nonparametric sequential certificate: the same additive-spline
ridge regressions and the same three residual products (r_i r_j, r_i r_j^2-type
square products), but used the way PC uses a test: an edge is DELETED as soon as
some conditioning set of size <= max_depth gives a p-value above alpha, evaluated
at one sample size.  Residuals are cross-fitted (2 folds).  Edges are removed
level by level (PC-stable).
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.stats import norm

__all__ = ["pc_gcm_skeleton"]


class _Resid:
    def __init__(self, X, folds=2, ridge=1.0, seed=0):
        n, d = X.shape
        self.X = X
        self.lo, self.hi = np.quantile(X, 0.01, axis=0), np.quantile(X, 0.99, axis=0)
        Xc = np.clip(X, self.lo, self.hi)
        self.Xc = Xc
        self.knots = [np.quantile(Xc[:, a], (0.25, 0.5, 0.75)) for a in range(d)]
        self.Y = np.hstack([X, X ** 2])
        self.fold = np.random.default_rng(seed).permutation(n) % folds
        self.folds, self.ridge = folds, ridge
        self.cache = {}

    def _design(self, S):
        Xc = self.Xc
        cols = [np.ones(len(Xc))]
        for a in S:
            x = Xc[:, a]
            cols += [x, x ** 2, x ** 3] + [np.maximum(x - q, 0.0) ** 3 for q in self.knots[a]]
        if len(S) == 2:
            a, b = Xc[:, S[0]], Xc[:, S[1]]
            cols += [a * b, a * a * b, a * b * b]
        elif len(S) > 2:
            for a, b in itertools.combinations(S, 2):
                cols.append(Xc[:, a] * Xc[:, b])
        return np.column_stack(cols)

    def get(self, S):
        if S not in self.cache:
            Z = self._design(S)
            R = np.empty_like(self.Y)
            pen = self.ridge * np.eye(Z.shape[1]); pen[0, 0] = 0
            for f in range(self.folds):
                tr, te = self.fold != f, self.fold == f
                c = np.linalg.solve(Z[tr].T @ Z[tr] + pen, Z[tr].T @ self.Y[tr])
                R[te] = self.Y[te] - Z[te] @ c
            self.cache[S] = R
        return self.cache[S]


def _pvalue(R, i, j, d):
    n = len(R)
    ps = []
    for a, b in ((i, j), (i, d + j), (j, d + i)):
        z = R[:, a] * R[:, b]
        sd = z.std(ddof=1)
        if sd < 1e-12:
            ps.append(1.0)
            continue
        t = z.mean() / (sd / np.sqrt(n))
        ps.append(2 * (1 - norm.cdf(abs(t))))
    return min(1.0, 3 * min(ps))


def pc_gcm_skeleton(X, alpha=0.05, max_depth=3, seed=0):
    X = np.asarray(X, float)
    n, d = X.shape
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    res = _Resid(X, seed=seed)
    adj = {i: set(range(d)) - {i} for i in range(d)}
    for depth in range(max_depth + 1):
        snapshot = {i: set(a) for i, a in adj.items()}
        for i, j in itertools.combinations(range(d), 2):
            if j not in adj[i]:
                continue
            cand = (snapshot[i] - {j}) | set()
            for (u, v) in ((i, j), (j, i)):
                if j not in adj[i]:
                    break
                pool = sorted(snapshot[u] - {v})
                for S in itertools.combinations(pool, depth):
                    if _pvalue(res.get(tuple(S)), i, j, d) > alpha:
                        adj[i].discard(j); adj[j].discard(i)
                        break
    return [[i, j] for i, j in itertools.combinations(range(d), 2) if j in adj[i]]
