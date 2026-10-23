"""BTAD: block-replicated additive discovery (skeleton).

Time-ordered data are cut into K contiguous blocks (small gaps dropped between
them).  For a target j and a set S, an additive spline ridge model is fitted on all
blocks but one and scored on the held-out block; the gain of adding a variable i
is the per-block log ratio of held-out mean squared errors.  A variable is
*replicated* when the K block gains have a one-sided t statistic above a Bonferroni
threshold.  A Markov-blanket estimate per node is grown and shrunk with this test
(at most `max_blanket` members), the skeleton keeps the mutual blanket members, and
spouse candidates are removed if some small subset of the union of the blankets
removes the replicated gain in both directions.
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.stats import t as tdist

__all__ = ["btad_skeleton"]


class _Model:
    def __init__(self, X, K, gap_frac, ridge):
        n, d = X.shape
        self.X = (X - X.mean(0)) / (X.std(0) + 1e-12)
        self.lo, self.hi = np.quantile(self.X, 0.01, axis=0), np.quantile(self.X, 0.99, axis=0)
        self.Xc = np.clip(self.X, self.lo, self.hi)
        self.knots = [np.quantile(self.Xc[:, a], (0.25, 0.5, 0.75)) for a in range(d)]
        edges = np.linspace(0, n, K + 1).astype(int)
        gap = int(gap_frac * n / K)
        self.blocks = []
        for k in range(K):
            a, b = edges[k], edges[k + 1]
            self.blocks.append(np.arange(a + gap, b - gap) if b - a > 4 * gap else np.arange(a, b))
        used = np.concatenate(self.blocks)
        self.K, self.ridge, self.d = K, ridge, d
        self.rows = used
        self.pos = [np.searchsorted(used, blk) for blk in self.blocks]
        self.cache = {}

    def _design(self, S):
        Xc = self.Xc[self.rows]
        cols = [np.ones(len(Xc))]
        for a in S:
            x = Xc[:, a]
            cols += [x, x ** 2, x ** 3] + [np.maximum(x - q, 0.0) ** 3 for q in self.knots[a]]
        if len(S) >= 2:
            for a, b in itertools.combinations(S, 2):
                cols.append(Xc[:, a] * Xc[:, b])
        return np.column_stack(cols)

    def block_mse(self, j, S):
        key = (j, tuple(sorted(S)))
        if key in self.cache:
            return self.cache[key]
        Z = self._design(sorted(S))
        y = self.X[self.rows, j]
        p = Z.shape[1]
        pen = self.ridge * np.eye(p); pen[0, 0] = 0
        Gs = [Z[ix].T @ Z[ix] for ix in self.pos]
        bs = [Z[ix].T @ y[ix] for ix in self.pos]
        G, b = sum(Gs), sum(bs)
        out = np.empty(self.K)
        for k, ix in enumerate(self.pos):
            beta = np.linalg.solve(G - Gs[k] + pen, b - bs[k])
            out[k] = np.mean((y[ix] - Z[ix] @ beta) ** 2)
        self.cache[key] = out
        return out

    def tstat(self, j, S, i):
        a = self.block_mse(j, S)
        b = self.block_mse(j, list(S) + [i])
        dlt = np.log(a + 1e-12) - np.log(b + 1e-12)
        sd = dlt.std(ddof=1)
        return dlt.mean() / (sd / np.sqrt(self.K) + 1e-12)


def btad_skeleton(X, K=20, alpha=0.05, max_blanket=5, gap_frac=0.05, ridge=1.0, max_sep=2):
    X = np.asarray(X, float)
    m = _Model(X, K, gap_frac, ridge)
    d = m.d
    thr = lambda M: tdist.ppf(1 - alpha / max(M, 1), K - 1)
    blanket = {}
    for j in range(d):
        B = []
        while len(B) < max_blanket:
            cand = [i for i in range(d) if i != j and i not in B]
            ts = [m.tstat(j, B, i) for i in cand]
            best = int(np.argmax(ts))
            if ts[best] > thr(len(cand)):
                B.append(cand[best])
            else:
                break
        changed = True
        while changed and len(B) > 1:
            changed = False
            for i in list(B):
                rest = [v for v in B if v != i]
                if m.tstat(j, rest, i) <= thr(len(B)):
                    B = rest; changed = True
                    break
        blanket[j] = B
    edges = [(i, j) for i, j in itertools.combinations(range(d), 2) if i in blanket[j] and j in blanket[i]]
    kept = []
    for i, j in edges:
        pool = sorted((set(blanket[i]) | set(blanket[j])) - {i, j})
        remove = False
        for r in range(0, max_sep + 1):
            for S in itertools.combinations(pool, r):
                if m.tstat(j, list(S), i) <= thr(2) and m.tstat(i, list(S), j) <= thr(2):
                    remove = True; break
            if remove:
                break
        if not remove:
            kept.append([i, j])
    return kept
