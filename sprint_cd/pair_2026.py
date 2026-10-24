"""Bivariate direction methods from 2026 preprints, re-implemented from their descriptions.

* ``sorted_compression`` : the parameter-free sorted-conditional bz2 compression baseline of
  "One Ruler: A Same-Hands Re-Evaluation of Bivariate Causal Direction on Tuebingen, with a
  Parameter-Free Compression Baseline" (arXiv:2606.23767).
* ``tra_delta`` / ``tra_conservative`` : Topological Residual Asymmetry (El Bouchattaoui,
  arXiv:2602.00427): cross-fitted smoothing-spline residuals, copula-standardised residual cloud,
  window-truncated total persistence of the 0-dimensional MST, direction from the sign of the
  difference; the conservative variant abstains unless a Gaussian-copula null rejects.
The papers do not give every constant (kappa, c_beta, number of folds); ours are fixed in
advance in the experiment scripts and stated there.
"""
from __future__ import annotations

import bz2

import numpy as np
from scipy.interpolate import make_smoothing_spline
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import cdist
from scipy.stats import norm, rankdata

__all__ = ["sorted_compression", "tra_delta", "tra_conservative"]


def _quant(v):
    v = np.asarray(v, float)
    lo, hi = v.min(), v.max()
    q = np.zeros(len(v)) if hi <= lo else (v - lo) / (hi - lo)
    return np.minimum((q * 256).astype(int), 255).astype(np.uint8)


def _dc(q):
    return len(bz2.compress(np.diff(q.astype(np.int16)).astype(np.int8).tobytes(), 9))


def _sc(key, val):
    order = np.argsort(key, kind="stable")
    return len(bz2.compress(np.diff(val[order].astype(np.int16)).astype(np.int8).tobytes(), 9))


def sorted_compression(x, y, max_n=8000):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) > max_n:
        idx = np.linspace(0, len(x) - 1, max_n).astype(int)
        x, y = x[idx], y[idx]
    qx, qy = _quant(x), _quant(y)
    s = _sc(qy, qx) / max(_dc(qx), 1) - _sc(qx, qy) / max(_dc(qy), 1)
    return float(s)  # > 0: x -> y


def _crossfit_resid(a, b, K, rng):
    """Out-of-fold residuals of a smoothing-spline regression of b on a."""
    n = len(a)
    fold = rng.permutation(n) % K
    r = np.empty(n)
    for k in range(K):
        tr, te = fold != k, fold == k
        order = np.argsort(a[tr]); xs, ys = a[tr][order], b[tr][order]
        keep = np.concatenate([[True], np.diff(xs) > 0])
        spl = make_smoothing_spline(xs[keep], ys[keep])
        r[te] = b[te] - spl(a[te])
    return r


def _tp(cloud, kappa, c_beta):
    n = len(cloud)
    alpha = kappa * n ** (-2.0 / 3.0)
    beta = c_beta * alpha
    D = cdist(cloud, cloud)
    T = minimum_spanning_tree(D)
    w = T.data
    psi = np.maximum(np.minimum(w, beta) - alpha, 0.0)
    return float(psi.sum() / ((n - 1) * (beta - alpha)))


def _rank01(v):
    return rankdata(v) / (len(v) + 1.0)


def tra_delta(x, y, kappa=1.0, c_beta=4.0, K=5, seed=0, max_n=1000):
    """Delta_n > 0 means x -> y (the residual cloud of y given x is the less clustered one)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    rng = np.random.default_rng(seed)
    if len(x) > max_n:
        idx = rng.choice(len(x), max_n, replace=False); x, y = x[idx], y[idx]
    x = (x - x.mean()) / (x.std() + 1e-12); y = (y - y.mean()) / (y.std() + 1e-12)
    ryx = _crossfit_resid(x, y, K, np.random.default_rng(seed))
    rxy = _crossfit_resid(y, x, K, np.random.default_rng(seed + 1))
    c_yx = np.column_stack([_rank01(x), _rank01(ryx)])
    c_xy = np.column_stack([_rank01(y), _rank01(rxy)])
    return _tp(c_yx, kappa, c_beta) - _tp(c_xy, kappa, c_beta)


def tra_conservative(x, y, kappa=1.0, c_beta=4.0, K=5, B=100, alpha=0.10, seed=0, max_n=500):
    """Returns (direction or None, delta, p_value); the null is a Gaussian copula with the empirical margins."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    rng = np.random.default_rng(seed)
    if len(x) > max_n:
        idx = rng.choice(len(x), max_n, replace=False); x, y = x[idx], y[idx]
    d0 = tra_delta(x, y, kappa, c_beta, K, seed, max_n=len(x))
    zx, zy = norm.ppf(_rank01(x)), norm.ppf(_rank01(y))
    rho = float(np.corrcoef(zx, zy)[0, 1])
    xs, ys = np.sort(x), np.sort(y)
    cnt = 0
    for b in range(B):
        g = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=len(x))
        ux, uy = norm.cdf(g[:, 0]), norm.cdf(g[:, 1])
        xb, yb = np.quantile(xs, ux), np.quantile(ys, uy)
        xb = xb + 1e-9 * rng.normal(size=len(xb)); yb = yb + 1e-9 * rng.normal(size=len(yb))
        db = tra_delta(xb, yb, kappa, c_beta, K, seed + 7 + b, max_n=len(x))
        cnt += abs(db) >= abs(d0)
    p = (1 + cnt) / (B + 1)
    return ("x->y" if d0 > 0 else "y->x") if p <= alpha else None, float(d0), float(p)
