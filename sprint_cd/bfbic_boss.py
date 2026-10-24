"""BF-BIC with BOSS: truncated basis-function BIC and a best-order score search.

Re-implementation of the nonlinear score of "Scalable Causal Discovery from Recursive Nonlinear Data via
Truncated Basis Function Scores and Tests" (arXiv:2510.04276): each continuous variable is rescaled to
[-1, 1] and expanded into the shifted Legendre polynomials P_1..P_p; the local score of a child with
parent set Z is the sum over the child's own basis columns X_j of the BIC of the regression of X_j on Z's
columns and the earlier columns X_1..X_{j-1}, BIC = 2L - c k ln N with L the Gaussian log-likelihood at the
residual variance.  The search is a best-order-score-search style permutation hill climb (each variable
in turn is moved to its best position; parents of a variable given a prefix are found by forward-backward
search on the local score).  The output is the skeleton of the best DAG.  The paper's formula for L has
a typographical inconsistency; we use the standard Gaussian log-likelihood 2L = -N ln(2 pi s2) - N.
"""
from __future__ import annotations

import itertools

import numpy as np

__all__ = ["bfbic_boss_skeleton"]


def _legendre(x, p):
    out = [x]
    P0, P1 = np.ones_like(x), x
    for n in range(2, p + 1):
        P0, P1 = P1, ((2 * n - 1) * x * P1 - (n - 1) * P0) / n
        out.append(P1)
    return np.column_stack(out)


class _Scorer:
    def __init__(self, X, p, c):
        n, d = X.shape
        lo, hi = X.min(0), X.max(0)
        U = 2 * (X - lo) / np.where(hi > lo, hi - lo, 1.0) - 1
        self.n, self.d, self.p, self.c = n, d, p, c
        self.B = [_legendre(U[:, j], p) for j in range(d)]
        self.cache = {}

    def local(self, j, parents):
        key = (j, tuple(sorted(parents)))
        if key in self.cache:
            return self.cache[key]
        n, p = self.n, self.p
        Z = [np.ones((n, 1))] + [self.B[q] for q in key[1]]
        Z = np.hstack(Z)
        total = 0.0
        cols = Z
        for t in range(p):
            y = self.B[j][:, t]
            k = cols.shape[1]
            beta, *_ = np.linalg.lstsq(cols, y, rcond=None)
            s2 = max(np.mean((y - cols @ beta) ** 2), 1e-12)
            total += -n * np.log(2 * np.pi * s2) - n - self.c * k * np.log(n)
            cols = np.hstack([cols, y[:, None]])
        self.cache[key] = total
        return total


def _best_parents(sc, j, pred, max_par):
    cur, best = [], sc.local(j, [])
    improved = True
    while improved and len(cur) < max_par:
        improved = False
        cand = [(sc.local(j, cur + [v]), v) for v in pred if v not in cur]
        if cand:
            s, v = max(cand)
            if s > best + 1e-9:
                cur.append(v); best = s; improved = True
    shrunk = True
    while shrunk and cur:
        shrunk = False
        for v in list(cur):
            rest = [u for u in cur if u != v]
            s = sc.local(j, rest)
            if s > best + 1e-9:
                cur, best, shrunk = rest, s, True
                break
    return cur, best


def _perm_score(sc, perm, max_par):
    total, parents = 0.0, {}
    for pos, j in enumerate(perm):
        P, s = _best_parents(sc, j, perm[:pos], max_par)
        parents[j] = P; total += s
    return total, parents


def bfbic_boss_skeleton(X, p=3, penalty=1.0, max_par=5, restarts=1, seed=0):
    X = np.asarray(X, float)
    sc = _Scorer(X, p, penalty)
    d = X.shape[1]
    rng = np.random.default_rng(seed)
    best_total, best_par = -np.inf, None
    for _ in range(restarts):
        perm = list(rng.permutation(d))
        cur, _p = _perm_score(sc, perm, max_par)
        improved = True
        while improved:
            improved = False
            for v in list(perm):
                base = [u for u in perm if u != v]
                best_pos, best_s = perm.index(v), cur
                for pos in range(d):
                    cand = base[:pos] + [v] + base[pos:]
                    s, _ = _perm_score(sc, cand, max_par)
                    if s > best_s + 1e-9:
                        best_s, best_pos = s, pos
                if best_pos != perm.index(v):
                    perm = base[:best_pos] + [v] + base[best_pos:]; cur = best_s; improved = True
        s, par = _perm_score(sc, perm, max_par)
        if s > best_total:
            best_total, best_par = s, par
    return [[i, j] for i, j in itertools.combinations(range(d), 2) if i in best_par[j] or j in best_par[i]]
