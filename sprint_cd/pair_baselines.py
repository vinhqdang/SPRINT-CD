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

__all__ = ["igci", "reci", "anm_hsic", "lingam_pair", "scored"]


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


def scored(x, y, seed=0):
    """Direction and a confidence score for each rule (larger = more confident).

    Used to compare with the abstaining certificate at matched coverage: rank
    pairs by confidence and keep the top k.
    """
    x0, y0 = np.asarray(x, float), np.asarray(y, float)
    out = {}
    h = _spacing_entropy(_minmax(y0)) - _spacing_entropy(_minmax(x0))
    out["IGCI"] = ("x->y" if h < 0 else "y->x", abs(h))
    a, b = _poly_mse(_minmax(x0), _minmax(y0)), _poly_mse(_minmax(y0), _minmax(x0))
    out["RECI"] = ("x->y" if a < b else "y->x", abs(np.log((a + 1e-12) / (b + 1e-12))))
    xs, ys = x0, y0
    if len(xs) > 500:
        idx = np.random.default_rng(seed).choice(len(xs), 500, replace=False)
        xs, ys = xs[idx], ys[idx]
    xs = (xs - xs.mean()) / (xs.std() + 1e-12)
    ys = (ys - ys.mean()) / (ys.std() + 1e-12)

    def stat(u, v):
        A = _basis(u)
        coef, *_ = np.linalg.lstsq(A, v, rcond=None)
        r = v - A @ coef
        return _hsic(r / (r.std() + 1e-12), u)

    sx, sy = stat(xs, ys), stat(ys, xs)
    out["ANM-HSIC"] = ("x->y" if sx < sy else "y->x", abs(np.log((sx + 1e-12) / (sy + 1e-12))))
    xz = (x0 - x0.mean()) / (x0.std() + 1e-12)
    yz = (y0 - y0.mean()) / (y0.std() + 1e-12)
    R = float(np.mean(xz * yz)) * float(np.mean(xz * np.tanh(yz) - np.tanh(xz) * yz))
    if 0.5 * (np.mean(xz ** 4) + np.mean(yz ** 4)) - 3.0 < 0:
        R = -R
    out["LiNGAM-pair"] = ("x->y" if R > 0 else "y->x", abs(R))
    return out
