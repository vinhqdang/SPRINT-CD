"""Block-ICP: invariant causal prediction with time blocks as environments, nonlinear additive.

When actuators drift (random walks), contiguous time blocks are environments in which the
exogenous variables are shifted.  For a target j, a set S is *invariant* if the residual of an
additive spline regression of j on S has the same mean and variance in every block (F tests of
block means and of block log-variances, Bonferroni over the two).  The estimate of the parents
of j is the intersection of all invariant sets of size <= max_size among candidate variables
(Peters, Buhlmann & Meinshausen 2016, with the nonlinear invariance test replaced by this block
test).  Candidates are the pool_size variables with the largest block-replicated marginal gain for j.
The skeleton is the union of the estimated parent sets.
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy import stats

from .btad import _Model

__all__ = ["block_icp_skeleton"]


def _resid(m, j, S):
    Z = m._design(sorted(S)) if S else np.ones((len(m.rows), 1))
    y = m.X[m.rows, j]
    pen = m.ridge * np.eye(Z.shape[1]); pen[0, 0] = 0
    beta = np.linalg.solve(Z.T @ Z + pen, Z.T @ y)
    return y - Z @ beta


def _invariant(r, pos, alpha):
    groups = [r[ix] for ix in pos if len(ix) > 3]
    if len(groups) < 3:
        return True
    p_mean = stats.f_oneway(*groups).pvalue
    lv = [np.log(g ** 2 + 1e-8) for g in groups]
    p_var = stats.f_oneway(*lv).pvalue
    return min(1.0, 2 * min(p_mean, p_var)) > alpha


def block_icp_skeleton(X, K=20, alpha=0.05, max_size=3, pool_size=8, gap_frac=0.05, cand_alpha=0.2):
    m = _Model(np.asarray(X, float), K, gap_frac, 1.0)
    d = m.d
    from scipy.stats import t as tdist
    parents = {}
    for j in range(d):
        ts = {i: m.tstat(j, [], i) for i in range(d) if i != j}
        cand = [i for i, t in sorted(ts.items(), key=lambda kv: -kv[1])][:pool_size]
        accepted = []
        found = False
        for size in range(0, min(max_size, len(cand)) + 1):
            for S in itertools.combinations(cand, size):
                if _invariant(_resid(m, j, S), m.pos, alpha):
                    accepted.append(set(S)); found = True
            if found and size >= 1:
                break
        P = set.intersection(*accepted) if accepted else set()
        parents[j] = P
    return [[i, j] for i, j in itertools.combinations(range(d), 2) if i in parents[j] or j in parents[i]]
