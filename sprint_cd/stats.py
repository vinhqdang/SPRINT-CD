"""Streaming sufficient statistics for linear-Gaussian causal discovery.

The central observation exploited throughout SPRINT-CD is that, for a
jointly Gaussian stream, *every* conditional-independence hypothesis
``X_i indep X_j | X_S`` has an e-value that is a deterministic function of

    (n, centred cross-product matrix C_n)

via Schur complements.  A single ``O(d^2)`` running statistic therefore
serves the whole -- combinatorially large -- hypothesis family, and
e-processes never need to be instantiated or stored per hypothesis.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["GaussianSuffStat", "ResidualMoments"]


@dataclass
class ResidualMoments:
    """Cross-products of ``x`` and ``y`` after residualising on ``Z`` (+ intercept).

    Attributes
    ----------
    sxx, sxy, syy:
        Entries of the residual cross-product matrix.
    dof:
        ``n - |S| - 1``: sample size minus the dimension of the nuisance
        block (conditioning set plus intercept).  This is the exponent
        parameter ``n - p`` of the safe linear e-value.
    n:
        Number of observations the moments are based on.
    """

    sxx: float
    sxy: float
    syy: float
    dof: int
    n: int

    @property
    def partial_corr(self) -> float:
        """Sample partial correlation ``r_{xy.S}``, clipped to [-1, 1]."""
        denom = np.sqrt(self.sxx * self.syy)
        if denom <= 0.0:
            return 0.0
        return float(np.clip(self.sxy / denom, -1.0, 1.0))


@dataclass
class GaussianSuffStat:
    """Running ``(n, sum, cross-product)`` statistics for a ``d``-variate stream.

    Rank-one updates are accumulated in place; all downstream quantities are
    derived on demand, so the memory footprint is ``O(d^2)`` regardless of
    how many hypotheses the discovery algorithm ends up touching.
    """

    d: int
    n: int = 0
    _sum: np.ndarray = field(default=None, repr=False)
    _cross: np.ndarray = field(default=None, repr=False)
    _centred_cache: np.ndarray = field(default=None, repr=False)
    _cache_n: int = field(default=-1, repr=False)

    def __post_init__(self) -> None:
        if self.d <= 0:
            raise ValueError("d must be positive")
        if self._sum is None:
            self._sum = np.zeros(self.d, dtype=float)
        if self._cross is None:
            self._cross = np.zeros((self.d, self.d), dtype=float)

    # ------------------------------------------------------------------
    # accumulation
    # ------------------------------------------------------------------
    def update(self, batch: np.ndarray) -> "GaussianSuffStat":
        """Absorb a batch of observations (``(m, d)`` or a single ``(d,)`` row)."""
        arr = np.asarray(batch, dtype=float)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.ndim != 2 or arr.shape[1] != self.d:
            raise ValueError(f"expected batch with {self.d} columns, got {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError("batch contains non-finite values")
        self.n += arr.shape[0]
        self._sum += arr.sum(axis=0)
        self._cross += arr.T @ arr
        self._centred_cache = None
        return self

    def copy(self) -> "GaussianSuffStat":
        return GaussianSuffStat(
            d=self.d, n=self.n, _sum=self._sum.copy(), _cross=self._cross.copy()
        )

    # ------------------------------------------------------------------
    # derived quantities
    # ------------------------------------------------------------------
    def centred_cross(self) -> np.ndarray:
        """Centred cross-product matrix ``C = X'X - n * mean mean'``.

        Cached between updates: a discovery sweep queries many hypotheses at a
        fixed sample size, and rebuilding the ``d x d`` matrix per query
        dominates the cost otherwise.
        """
        if self.n == 0:
            return np.zeros((self.d, self.d), dtype=float)
        if self._centred_cache is None or self._cache_n != self.n:
            self._centred_cache = self._cross - np.outer(self._sum, self._sum) / self.n
            self._cache_n = self.n
        return self._centred_cache

    def marginal_var(self, ddof: int = 1) -> np.ndarray:
        """Per-variable marginal variances (used for warm-up scale setting)."""
        if self.n - ddof <= 0:
            return np.full(self.d, np.nan)
        return np.diag(self.centred_cross()) / (self.n - ddof)

    def residual_moments(
        self, i: int, j: int, cond: tuple[int, ...] | list[int] = (), *, ridge: float = 0.0
    ) -> ResidualMoments:
        """Residual cross-products of ``X_i`` and ``X_j`` given ``X_cond``.

        Computed as the Schur complement of the centred cross-product matrix,
        which is exactly the cross-product matrix of the residuals from an
        intercept-inclusive least-squares regression on ``X_cond``.
        """
        cond = tuple(cond)
        if i == j:
            raise ValueError("i and j must differ")
        if i in cond or j in cond:
            raise ValueError("conditioning set must not contain i or j")
        k = len(cond)
        dof = self.n - k - 1
        if dof <= 0:
            return ResidualMoments(sxx=0.0, sxy=0.0, syy=0.0, dof=max(dof, 0), n=self.n)

        C = self.centred_cross()
        if k == 0:
            return ResidualMoments(
                sxx=float(C[i, i]), sxy=float(C[i, j]), syy=float(C[j, j]),
                dof=dof, n=self.n,
            )

        idx = np.array(cond, dtype=int)
        Czz = C[np.ix_(idx, idx)]
        if ridge > 0.0:
            Czz = Czz + ridge * np.eye(k)
        Czx = C[np.ix_(idx, [i, j])]
        Cxx = C[np.ix_([i, j], [i, j])]
        try:
            sol = np.linalg.solve(Czz, Czx)
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(Czz, Czx, rcond=None)[0]
        S = Cxx - Czx.T @ sol
        # Numerical symmetrisation; the Schur complement is exactly symmetric.
        sxx = float(max(S[0, 0], 0.0))
        syy = float(max(S[1, 1], 0.0))
        sxy = float(0.5 * (S[0, 1] + S[1, 0]))
        return ResidualMoments(sxx=sxx, sxy=sxy, syy=syy, dof=dof, n=self.n)

    def residual_block_moments(
        self,
        target: int,
        nuisance: tuple[int, ...] | list[int],
        tested: tuple[int, ...] | list[int],
        *,
        ridge: float = 0.0,
    ) -> tuple[float, np.ndarray, np.ndarray, int]:
        """Block analogue of :meth:`residual_moments`.

        Residualises the ``target`` column and the ``tested`` block on the
        ``nuisance`` block (plus intercept), returning
        ``(s_tt, s_wt, S_ww, dof)`` where ``dof = n - |nuisance| - 1``.
        """
        nuisance = tuple(nuisance)
        tested = tuple(tested)
        if target in nuisance or target in tested:
            raise ValueError("target must not appear in nuisance or tested blocks")
        if set(nuisance) & set(tested):
            raise ValueError("nuisance and tested blocks must be disjoint")
        q = len(tested)
        p = len(nuisance)
        dof = self.n - p - 1
        if dof <= 0 or q == 0:
            return 0.0, np.zeros(q), np.zeros((q, q)), max(dof, 0)

        C = self.centred_cross()
        cols = np.array((target,) + tested, dtype=int)
        block = C[np.ix_(cols, cols)]
        if p > 0:
            idx = np.array(nuisance, dtype=int)
            Czz = C[np.ix_(idx, idx)]
            if ridge > 0.0:
                Czz = Czz + ridge * np.eye(p)
            Czb = C[np.ix_(idx, cols)]
            try:
                sol = np.linalg.solve(Czz, Czb)
            except np.linalg.LinAlgError:
                sol = np.linalg.lstsq(Czz, Czb, rcond=None)[0]
            block = block - Czb.T @ sol
        block = 0.5 * (block + block.T)
        s_tt = float(max(block[0, 0], 0.0))
        s_wt = np.asarray(block[0, 1:], dtype=float)
        S_ww = np.asarray(block[1:, 1:], dtype=float)
        return s_tt, s_wt, S_ww, dof
