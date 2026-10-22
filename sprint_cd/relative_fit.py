"""Prequential relative-fit direction certificate.

For a pair (x, y) two forecasters of the joint law are fitted sequentially on
past observations only: ``x -> y`` (marginal of x times conditional of y | x)
and ``y -> x``.  The increment ``d_s`` is the difference of their log predictive
densities at observation ``s``.  An asymptotic confidence sequence for the
running mean of ``d`` (Waudby-Smith et al., Ann. Statist. 2024, Thm 2.2)
certifies ``x -> y`` when its lower end exceeds zero and ``y -> x`` when its
upper end falls below zero.

What is certified is that one factorisation predicts strictly better than the
other.  That is a statement about predictive fit, valid whether or not either
model is correct; it equals a statement about causal direction only when the
better-fitting model class is also the causal one.

Two model classes are provided.  ``gg``: generalised-Gaussian marginal and a
spline-mean generalised-Gaussian conditional with constant scale.  ``ggh``:
the same with a log-scale that is quadratic in the regressor.  The primary
verdict (``agree``) requires both classes to certify the same direction.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import lambertw

from .certificates import fit_gg_regression, gg_logpdf

__all__ = ["certify_direction", "cs_radius"]


def _basis(x: np.ndarray, knots) -> np.ndarray:
    return np.column_stack([x, x ** 2, x ** 3] + [np.maximum(x - k, 0.0) ** 3 for k in knots])


class _GG:
    het = False

    def fit(self, a, b, knots):
        self.knots = knots
        n = len(a)
        _, self.ca, self.sa, self.ka = fit_gg_regression(a, np.ones((n, 1)))
        X = np.column_stack([np.ones(n), _basis(a, knots)])
        _, self.cb, self.sb, self.kb = fit_gg_regression(b, X)
        return self

    def logp(self, a, b):
        X = np.column_stack([np.ones(len(a)), _basis(a, self.knots)])
        return (gg_logpdf(a - self.ca[0], self.sa, self.ka)
                + gg_logpdf(b - X @ self.cb, self.sb, self.kb))


class _GGH:
    het = True

    def fit(self, a, b, knots):
        self.knots = knots
        n = len(a)
        _, self.ca, self.sa, self.ka = fit_gg_regression(a, np.ones((n, 1)))
        X = np.column_stack([np.ones(n), _basis(a, knots)])
        _, c, s, k = fit_gg_regression(b, X)
        Zs = np.column_stack([np.ones(n), a, a * a])
        g = np.zeros(3)
        for _ in range(2):
            r = b - X @ c

            def nll(gg):
                ls = Zs @ gg
                return np.sum(ls + np.abs(r / (s * np.exp(ls))) ** k)

            g = minimize(nll, g, method="BFGS", options={"maxiter": 30}).x
            sig = np.exp(Zs @ g)
            sig = sig / np.exp(np.mean(Zs @ g))
            _, c, s, k = fit_gg_regression(b / sig, X / sig[:, None])
            g = g - np.mean(Zs @ g) * np.array([1.0, 0.0, 0.0])
        self.c, self.s, self.k, self.g = c, s, k, g
        return self

    def logp(self, a, b):
        X = np.column_stack([np.ones(len(a)), _basis(a, self.knots)])
        Zs = np.column_stack([np.ones(len(a)), a, a * a])
        sig = np.exp(Zs @ self.g)
        return (gg_logpdf(a - self.ca[0], self.sa, self.ka)
                + gg_logpdf((b - X @ self.c) / sig, self.s, self.k) - np.log(sig))


def cs_radius(t: float, sd: float, alpha: float, tstar: float = 200.0) -> float:
    w = lambertw(-alpha ** 2 / np.e, -1).real
    rho = np.sqrt(max((-w - 1.0) / tstar, 1e-12))
    return sd * np.sqrt(2.0 * (t * rho ** 2 + 1.0) / (t ** 2 * rho ** 2)
                        * np.log(np.sqrt(t * rho ** 2 + 1.0) / alpha))


def _run(x, y, cls, m0, alpha, batch, kx, ky, min_scored):
    n = len(x)
    ds, verdict = [], None
    s = m0
    while s < n:
        e = min(s + batch, n)
        fa = cls().fit(x[:s], y[:s], kx)
        fb = cls().fit(y[:s], x[:s], ky)
        ds.extend(fa.logp(x[s:e], y[s:e]) - fb.logp(y[s:e], x[s:e]))
        s = e
        D = np.asarray(ds)
        t = len(D)
        if t >= min_scored and verdict is None:
            mu, sd = D.mean(), D.std(ddof=1) + 1e-12
            r = cs_radius(t, sd, alpha)
            if mu - r > 0:
                verdict = "x->y"
            elif mu + r < 0:
                verdict = "y->x"
    D = np.asarray(ds)
    return verdict, ("x->y" if D.mean() > 0 else "y->x")


def certify_direction(x, y, alpha: float = 0.05, *, seed: int = 0,
                      min_warm: int = 100, warm_frac: float = 0.2,
                      batch: int = 50, min_scored: int = 60):
    """Returns a dict with the ``agree`` verdict and each class's verdict.

    ``None`` means undecided.  Pairs with fewer than ``min_warm + 100``
    observations are not scored.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    rng = np.random.default_rng(seed)
    p = rng.permutation(n)
    x, y = x[p], y[p]
    x = (x - x[:min_warm].mean()) / (x[:min_warm].std() + 1e-12)
    y = (y - y[:min_warm].mean()) / (y[:min_warm].std() + 1e-12)
    m0 = max(min_warm, int(np.ceil(warm_frac * n)))
    if n - m0 < 100:
        return None
    kx = np.quantile(x[:m0], (.2, .4, .6, .8))
    ky = np.quantile(y[:m0], (.2, .4, .6, .8))
    v1, f1 = _run(x, y, _GG, m0, alpha, batch, kx, ky, min_scored)
    v2, f2 = _run(x, y, _GGH, m0, alpha, batch, kx, ky, min_scored)
    return {"agree": v1 if v1 == v2 else None, "gg": v1, "ggh": v2,
            "forced_gg": f1, "forced_ggh": f2}
