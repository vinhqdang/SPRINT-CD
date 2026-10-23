"""Forced-choice pairwise direction baselines used for comparison.

Each function takes two 1-d arrays and returns ``"x->y"`` or ``"y->x"``.  They
always output a direction (no abstention).  Implementations are compact
re-implementations of the published rules, not the authors' code.

* ``igci``      Daniusis et al. (2010), slope-based, uniform reference.
* ``reci``      Blobaum et al. (2018), regression error: smaller error is causal.
* ``anm_hsic``  Hoyer et al. (2009) additive noise model, polynomial-spline fit,
                smaller HSIC between residual and regressor is causal.
* ``lingam_pair`` Hyvarinen & Smith (2013) first-order likelihood-ratio rule
                with the sign taken from the sample kurtosis.
"""
from __future__ import annotations

import numpy as np

__all__ = ["igci", "reci", "anm_hsic", "lingam_pair"]


def _minmax(v):
    lo, hi = v.min(), v.max()
    return (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)


def _spacing_entropy(v):
    v = np.sort(v)
    d = np.diff(v)
    d = d[d > 0]
    return float(np.mean(np.log(d))) if len(d) else 0.0


def igci(x, y):
    """Entropy form of IGCI with a uniform reference: x->y if h(y) < h(x)."""
    x, y = _minmax(np.asarray(x, float)), _minmax(np.asarray(y, float))
    return "x->y" if _spacing_entropy(y) < _spacing_entropy(x) else "y->x"


def _poly_mse(a, b, deg=3):
    A = np.vander(a, deg + 1)
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    return float(np.mean((b - A @ coef) ** 2))


def reci(x, y):
    x, y = _minmax(np.asarray(x, float)), _minmax(np.asarray(y, float))
    return "x->y" if _poly_mse(x, y) < _poly_mse(y, x) else "y->x"


def _basis(a, deg_knots=4):
    k = np.quantile(a, np.linspace(0.2, 0.8, deg_knots))
    return np.column_stack([np.ones_like(a), a, a ** 2, a ** 3]
                           + [np.maximum(a - t, 0) ** 3 for t in k])


def _hsic(a, b):
    n = len(a)
    def gram(v):
        d = (v[:, None] - v[None, :]) ** 2
        h = np.median(d[d > 0]) if np.any(d > 0) else 1.0
        return np.exp(-d / h)
    K, L = gram(a), gram(b)
    H = np.eye(n) - 1.0 / n
    return float(np.trace(K @ H @ L @ H) / n ** 2)


def anm_hsic(x, y, max_n=500, seed=0):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) > max_n:
        idx = np.random.default_rng(seed).choice(len(x), max_n, replace=False)
        x, y = x[idx], y[idx]
    x = (x - x.mean()) / (x.std() + 1e-12)
    y = (y - y.mean()) / (y.std() + 1e-12)

    def stat(a, b):
        A = _basis(a)
        coef, *_ = np.linalg.lstsq(A, b, rcond=None)
        r = b - A @ coef
        return _hsic(r / (r.std() + 1e-12), a)

    return "x->y" if stat(x, y) < stat(y, x) else "y->x"


def lingam_pair(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    x = (x - x.mean()) / (x.std() + 1e-12)
    y = (y - y.mean()) / (y.std() + 1e-12)
    rho = float(np.mean(x * y))
    R = rho * float(np.mean(x * np.tanh(y) - np.tanh(x) * y))
    kurt = 0.5 * (np.mean(x ** 4) + np.mean(y ** 4)) - 3.0
    if kurt < 0:
        R = -R
    return "x->y" if R > 0 else "y->x"
