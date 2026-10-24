"""Learned skeleton discovery from replicated-gain features (LRG).

For every pair (i, j) of a time-ordered data set we compute, in both directions and for
three conditioning sets (empty, the three strongest other predictors of the target, and the
single conditioner that most explains the pair away), the block-wise held-out gain of adding
the other variable (mean, t statistic over K contiguous blocks, fraction of positive blocks),
together with the marginal correlation, the lag-1 autocorrelations of both variables and
within-data-set ranks.  A gradient-boosting classifier trained on simulated data sets with
known skeletons (nonlinear mechanisms, serial dependence, latent drift, duplicated sensors)
maps these features to an edge probability.  The model is trained once and then applied
unchanged to new data.
"""
from __future__ import annotations

import itertools

import numpy as np

from .btad import _Model

__all__ = ["pair_features", "FEATURE_NAMES"]

K_BLOCKS = 40
POOL = 5


def _gain_stats(m, j, S, i):
    a = m.block_mse(j, S)
    b = m.block_mse(j, list(S) + [i])
    dl = np.log(a + 1e-12) - np.log(b + 1e-12)
    sd = dl.std(ddof=1)
    return dl.mean(), dl.mean() / (sd / np.sqrt(len(dl)) + 1e-12), float(np.mean(dl > 0))


def pair_features(X, K=K_BLOCKS):
    X = np.asarray(X, float)
    n, d = X.shape
    m = _Model(X, K, 0.05, 1.0)
    Z = m.X
    corr = np.corrcoef(Z.T)
    ac = np.array([np.corrcoef(Z[:-1, v], Z[1:, v])[0, 1] for v in range(d)])
    # strongest other predictors of each target (marginal replicated t)
    marg = np.zeros((d, d))
    for j in range(d):
        for i in range(d):
            if i != j:
                marg[j, i] = _gain_stats(m, j, [], i)[1]
    pairs = list(itertools.combinations(range(d), 2))
    feats, names_done = [], False
    for i, j in pairs:
        row = []
        stats_by_dir = []
        for tgt, src in ((j, i), (i, j)):
            others = [v for v in np.argsort(-marg[tgt]) if v not in (tgt, src)]
            top3 = [int(v) for v in others[:3]]
            pool = [int(v) for v in others[:POOL]]
            e0 = _gain_stats(m, tgt, [], src)
            e3 = _gain_stats(m, tgt, top3, src)
            ts = [(_gain_stats(m, tgt, [k], src), k) for k in pool]
            best = min(ts, key=lambda z: z[0][1])[0]
            stats_by_dir.append((e0, e3, best))
        for s in range(3):
            for c in range(3):
                v = [stats_by_dir[0][s][c], stats_by_dir[1][s][c]]
                row += [max(v), min(v)]
        row += [abs(corr[i, j]), min(ac[i], ac[j]), max(ac[i], ac[j]), np.log(d)]
        feats.append(row)
    F = np.array(feats)
    # within-data-set ranks of the t statistics (max over directions)
    t_cols = [2 * (3 * s + 1) for s in range(3)]  # index of max t for each set
    ranks = np.column_stack([(-F[:, c]).argsort().argsort() / max(len(F) - 1, 1) for c in t_cols])
    return pairs, np.hstack([F, ranks])


FEATURE_NAMES = [f"{s}_{k}_{ag}" for s in ("empty", "top3", "best1") for k in ("mean", "t", "pos") for ag in ("max", "min")] \
    + ["abscorr", "ac_min", "ac_max", "log_d", "rank_t_empty", "rank_t_top3", "rank_t_best1"]
