"""NOTEARS (linear, L1-penalised least squares, acyclicity by trace-exponential).

Compact re-implementation of Zheng et al. (NeurIPS 2018) with the augmented Lagrangian and
L-BFGS-B on the positive/negative parts of W.  Returns the weighted adjacency matrix.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as sla
import scipy.optimize as sopt

__all__ = ["notears_linear"]


def notears_linear(X, lambda1=0.1, max_iter=50, h_tol=1e-8, rho_max=1e16, w_threshold=0.3):
    X = np.asarray(X, float)
    X = X - X.mean(0)
    n, d = X.shape

    def _adj(w):
        return (w[: d * d] - w[d * d:]).reshape(d, d)

    def _loss(W):
        R = X @ W - X
        return 0.5 / n * (R ** 2).sum(), 1.0 / n * X.T @ R

    def _h(W):
        E = sla.expm(W * W)
        return np.trace(E) - d, E.T * W * 2

    def _func(w):
        W = _adj(w)
        loss, G_loss = _loss(W)
        h, G_h = _h(W)
        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w.sum()
        G = G_loss + (rho * h + alpha) * G_h
        return obj, np.concatenate((G + lambda1, -G + lambda1), axis=None)

    w_est = np.zeros(2 * d * d)
    rho, alpha, h = 1.0, 0.0, np.inf
    bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]
    for _ in range(max_iter):
        w_new, h_new = None, None
        while rho < rho_max:
            sol = sopt.minimize(_func, w_est, method="L-BFGS-B", jac=True, bounds=bnds)
            w_new = sol.x
            h_new, _ = _h(_adj(w_new))
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        w_est, h = w_new, h_new
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            break
    W = _adj(w_est)
    W[np.abs(W) < w_threshold] = 0
    return W
