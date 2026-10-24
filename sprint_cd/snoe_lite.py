"""Orientation of a known edge by comparing pairwise additive-noise likelihoods with parents,
following the idea of SNOE (Huang and Zhou, arXiv 2506.05590): the direction with the larger
log-likelihood of (child | cause, other neighbours) + (cause | its other neighbours) wins.
Differences from the paper (code is not public): additive spline regression with a ridge penalty
(SplineTransformer, 6 knots, cubic) instead of their GAM, Gaussian residual likelihood, every edge
scored independently instead of the sequential ranking, other neighbours taken from the supplied skeleton.
"""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import SplineTransformer


def _ll(y, Z, seed=0):
    n = len(y)
    if Z.shape[1]:
        B = SplineTransformer(n_knots=6, degree=3).fit_transform(Z)
    else:
        B = np.zeros((n, 1))
    idx = np.random.default_rng(seed).permutation(n); h = n // 2
    res = np.empty(n)
    for tr, te in ((idx[:h], idx[h:]), (idx[h:], idx[:h])):
        m = Ridge(alpha=1.0).fit(B[tr], y[tr]); res[te] = y[te] - m.predict(B[te])
    return -0.5 * n * np.log(np.var(res) + 1e-12)


def snoe_delta(X, i, j, nbr_i, nbr_j, max_n=2000, seed=0):
    """>0 means i -> j."""
    if len(X) > max_n:
        X = X[np.random.default_rng(seed).choice(len(X), max_n, replace=False)]
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-12)
    pi = [k for k in nbr_i if k != j]; pj = [k for k in nbr_j if k != i]
    fwd = _ll(Xz[:, j], Xz[:, [i] + pj], seed) + _ll(Xz[:, i], Xz[:, pi], seed)
    bwd = _ll(Xz[:, i], Xz[:, [j] + pi], seed) + _ll(Xz[:, j], Xz[:, pj], seed)
    return float(fwd - bwd)
