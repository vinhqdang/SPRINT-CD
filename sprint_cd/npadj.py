"""Nonparametric sequential adjacency certificate (sequential generalised covariance measure).

For a pair (i, j) and a conditioning set S, let r_i = X_i - E[X_i | X_S] and
r_j = X_j - E[X_j | X_S].  If S d-separates i and j then X_i is independent of
X_j given X_S and E[r_i r_j] = 0.  The regressions are fitted by flexible
(additive spline plus pairwise interaction) ridge regression on PAST rows only,
refitted on a geometric schedule, and the product z_s = r_i,s r_j,s is scored on
each new row with the fit available before it.  The running mean of z has an
asymptotic confidence sequence (Waudby-Smith et al., Ann. Statist. 2024,
Thm 2.2); the set S is *rejected* when the sequence excludes zero.  The product
of regression errors is the only bias term, so nonlinear conditional means are
handled by the flexible fit rather than assumed linear.

A pair is certified adjacent once EVERY S of size <= k has been rejected (at
any time).  If the pair is non-adjacent, one S* separates it, so the event is
contained in "S* is ever rejected", which has probability <= alpha_pair
asymptotically.  Guarantee: asymptotic, under a product-rate condition on the
regressions (Shah & Peters 2020; Waudby-Smith, Kennedy & Ramdas 2023).
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.special import lambertw

__all__ = ["NPAdjacency"]


def _rho(alpha, tstar=200.0):
    w = lambertw(-alpha ** 2 / np.e, -1).real
    return np.sqrt(max((-w - 1.0) / tstar, 1e-12))


def cs_radius_arr(t, sd, alpha, rho):
    return sd * np.sqrt(2.0 * (t * rho ** 2 + 1.0) / (t ** 2 * rho ** 2)
                        * np.log(np.sqrt(t * rho ** 2 + 1.0) / alpha))


class NPAdjacency:
    def __init__(self, d, alpha_pair, k=2, warm=100, refit_factor=1.25,
                 min_scored=60, ridge=1.0, squares=True):
        self.squares = squares
        self.n_tests = 3 if squares else 1
        alpha_pair = alpha_pair / self.n_tests
        self.d, self.alpha, self.k = d, alpha_pair, k
        self.warm, self.factor, self.min_scored, self.ridge = warm, refit_factor, min_scored, ridge
        self.rho = _rho(alpha_pair)
        self.sets = [S for r in range(k + 1) for S in itertools.combinations(range(d), r)]
        self.buf = []
        self.n = 0
        self.n_fit = 0
        self.coef = {}
        self.sz = {S: np.zeros((d, d)) for S in self.sets}
        self.sz2 = {S: np.zeros((d, d)) for S in self.sets}
        self.sq = {S: np.zeros((d, d)) for S in self.sets}     # (raw_i, square_j) products
        self.sq2 = {S: np.zeros((d, d)) for S in self.sets}
        self.t = 0
        self.ever = {S: np.zeros((d, d), bool) for S in self.sets}
        self.knots = None

    # -- basis ---------------------------------------------------------
    def _prep(self, X):
        return np.clip(X, self.lo, self.hi)

    def _design(self, Xc, S):
        cols = [np.ones(len(Xc))]
        for a in S:
            x = Xc[:, a]
            cols += [x, x ** 2, x ** 3] + [np.maximum(x - q, 0.0) ** 3 for q in self.knots[a]]
        if len(S) == 2:
            a, b = Xc[:, S[0]], Xc[:, S[1]]
            cols += [a * b, a * a * b, a * b * b]
        return np.column_stack(cols)

    def _fit_all(self):
        X = np.vstack(self.buf)
        Xc = self._prep(X)
        for S in self.sets:
            Z = self._design(Xc, S)
            p = Z.shape[1]
            pen = self.ridge * np.eye(p)
            pen[0, 0] = 0.0
            Y = np.hstack([X, X ** 2]) if self.squares else X
            self.coef[S] = np.linalg.solve(Z.T @ Z + pen, Z.T @ Y)
        self.n_fit = len(X)

    # -- streaming -----------------------------------------------------
    def update(self, batch):
        batch = np.atleast_2d(np.asarray(batch, float))
        if self.knots is None:
            if self.n + len(batch) < self.warm:
                self.buf.append(batch); self.n += len(batch); return
            self.buf.append(batch); self.n += len(batch)
            X = np.vstack(self.buf)
            self.lo, self.hi = np.quantile(X, 0.01, axis=0), np.quantile(X, 0.99, axis=0)
            Xc = self._prep(X)
            self.knots = [np.quantile(Xc[:, a], (0.25, 0.5, 0.75)) for a in range(self.d)]
            self._fit_all()
            return
        Xc = self._prep(batch)
        d = self.d
        Yb = np.hstack([batch, batch ** 2]) if self.squares else batch
        for S in self.sets:
            R = Yb - self._design(Xc, S) @ self.coef[S]
            if S:
                R[:, list(S)] = 0.0
                if self.squares:
                    R[:, [d + a for a in S]] = 0.0
            A = R[:, :d]
            self.sz[S] += A.T @ A
            self.sz2[S] += (A * A).T @ (A * A)
            if self.squares:
                B = R[:, d:]
                self.sq[S] += A.T @ B
                self.sq2[S] += (A * A).T @ (B * B)
        self.t += len(batch)
        self.buf.append(batch); self.n += len(batch)
        if self.n >= self.factor * self.n_fit:
            self._fit_all()
        if self.t >= self.min_scored:
            t = self.t
            for S in self.sets:
                m = self.sz[S] / t
                sd = np.sqrt(np.maximum(self.sz2[S] / t - m * m, 1e-12))
                rej = np.abs(m) > cs_radius_arr(t, sd, self.alpha, self.rho)
                if self.squares:
                    m2 = self.sq[S] / t
                    sd2 = np.sqrt(np.maximum(self.sq2[S] / t - m2 * m2, 1e-12))
                    r2 = np.abs(m2) > cs_radius_arr(t, sd2, self.alpha, self.rho)
                    rej = rej | r2 | r2.T
                self.ever[S] |= rej

    def adjacent(self, i, j):
        for S in self.sets:
            if i in S or j in S:
                continue
            if not self.ever[S][i, j]:
                return False
        return True

    def certified_edges(self):
        return [(i, j) for i, j in itertools.combinations(range(self.d), 2) if self.adjacent(i, j)]
