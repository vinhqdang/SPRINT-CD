r"""Assumption-light e-processes via sequential universal inference.

Where :mod:`sprint_cd.eprocess.safe_linear` buys sharp power from a
group-invariance argument specific to the linear-Gaussian model, the
construction here buys *generality* at some cost in power, and its validity
needs nothing beyond a correctly normalised sequential predictive density.

Construction
------------
Let :math:`\hat q_{t-1}` be any predictive density for :math:`X_t` given
:math:`(Y_t, Z_t)` that is fitted on strictly past data, and let
:math:`\mathcal{P}_0` be the null model.  Define

.. math::

    E_n = \frac{\prod_{t \le n} \hat q_{t-1}(x_t \mid y_t, z_t)}
               {\sup_{\theta \in \mathcal{P}_0} \prod_{t \le n} p_\theta(x_t \mid z_t)} .

**Proposition.** :math:`(E_n)` is an e-process under :math:`H_0`.

*Proof.* Fix the true null parameter :math:`\theta^\star` and put
:math:`M_n = \prod_{t\le n} \hat q_{t-1}(x_t\mid y_t,z_t) /
p_{\theta^\star}(x_t \mid z_t)`.  Since :math:`\hat q_{t-1}` is
:math:`\mathcal{F}_{t-1}`-measurable and integrates to one,
:math:`\mathbb{E}[M_t \mid \mathcal{F}_{t-1}] = M_{t-1}`, so :math:`M` is a
non-negative martingale with :math:`M_0 = 1`.  The denominator of
:math:`E_n` is a supremum over :math:`\mathcal{P}_0` and therefore at least
the likelihood at :math:`\theta^\star`, giving :math:`E_n \le M_n`
pointwise.  Optional stopping for non-negative martingales yields
:math:`\mathbb{E}[E_\tau] \le \mathbb{E}[M_\tau] \le 1`. :math:`\square`

The argument is indifferent to how :math:`\hat q` is fitted -- a misspecified
or badly tuned predictive costs power, never validity -- which makes this the
natural fallback when Gaussianity is doubtful, and the natural primitive for
categorical data, where no invariance argument is available.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .base import EProcessState

__all__ = ["GaussianUniversalCI", "DiscreteUniversalCI"]

_LOG_2PI = float(np.log(2.0 * np.pi))


@dataclass
class GaussianUniversalCI:
    """Sequential universal-inference e-process for a Gaussian CI hypothesis.

    Maintains its own state (unlike the safe linear e-value, which is a pure
    function of the shared Gram matrix), because the numerator is a genuinely
    prequential quantity: each observation is scored by a model fitted only on
    its predecessors.

    Parameters
    ----------
    n_cond:
        Size of the conditioning set ``S``.
    ridge:
        Ridge penalty stabilising the sequential least-squares fits while the
        design is rank-deficient.  Affects power only, never validity.
    min_var:
        Floor on the plug-in predictive variance, guarding against a
        degenerate (zero-width) predictive in the first few steps.
    """

    n_cond: int
    ridge: float = 1.0
    min_var: float = 1e-6

    n: int = field(default=0, init=False)
    log_numerator: float = field(default=0.0, init=False)
    _xtx: np.ndarray = field(default=None, init=False, repr=False)
    _xty: np.ndarray = field(default=None, init=False, repr=False)
    _yty: float = field(default=0.0, init=False, repr=False)
    _null_xtx: np.ndarray = field(default=None, init=False, repr=False)
    _null_xty: np.ndarray = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        p_alt = self.n_cond + 2   # intercept + conditioning set + tested variable
        p_null = self.n_cond + 1  # intercept + conditioning set
        self._xtx = np.zeros((p_alt, p_alt))
        self._xty = np.zeros(p_alt)
        self._null_xtx = np.zeros((p_null, p_null))
        self._null_xty = np.zeros(p_null)

    # ------------------------------------------------------------------
    def _predict(self) -> tuple[float, float]:
        """Plug-in predictive mean coefficients and variance from past data."""
        p = self._xtx.shape[0]
        A = self._xtx + self.ridge * np.eye(p)
        try:
            coef = np.linalg.solve(A, self._xty)
        except np.linalg.LinAlgError:
            coef = np.linalg.lstsq(A, self._xty, rcond=None)[0]
        rss = self._yty - 2.0 * coef @ self._xty + coef @ self._xtx @ coef
        denom = max(self.n - p, 1)
        var = max(rss / denom, self.min_var)
        return coef, var

    def update(self, x: float, y: float, z: np.ndarray | None = None) -> EProcessState:
        """Score one new observation and absorb it."""
        z = np.zeros(0) if z is None else np.asarray(z, dtype=float).ravel()
        if z.size != self.n_cond:
            raise ValueError(f"expected {self.n_cond} conditioning values, got {z.size}")
        row_alt = np.concatenate(([1.0], z, [y]))
        row_null = np.concatenate(([1.0], z))

        # --- numerator: score x_t under a model fitted on t-1 observations ---
        if self.n > 0:
            coef, var = self._predict()
            mu = float(coef @ row_alt)
            self.log_numerator += -0.5 * (_LOG_2PI + np.log(var) + (x - mu) ** 2 / var)
        else:
            # A proper but deliberately vague predictive for the first point.
            var0 = 1e4
            self.log_numerator += -0.5 * (_LOG_2PI + np.log(var0) + x * x / var0)

        # --- absorb ---
        self.n += 1
        self._xtx += np.outer(row_alt, row_alt)
        self._xty += row_alt * x
        self._yty += x * x
        self._null_xtx += np.outer(row_null, row_null)
        self._null_xty += row_null * x
        return self.state()

    def _null_max_loglik(self) -> float:
        """Maximised null log-likelihood ``sup_{alpha, sigma}``."""
        p = self._null_xtx.shape[0]
        if self.n <= p:
            return np.inf  # unbounded: no evidence can be claimed yet
        try:
            coef = np.linalg.solve(self._null_xtx, self._null_xty)
        except np.linalg.LinAlgError:
            coef = np.linalg.lstsq(self._null_xtx, self._null_xty, rcond=None)[0]
        rss = self._yty - 2.0 * coef @ self._null_xty + coef @ self._null_xtx @ coef
        sigma2 = max(rss / self.n, self.min_var)
        return -0.5 * self.n * (_LOG_2PI + np.log(sigma2) + 1.0)

    def state(self) -> EProcessState:
        null_ll = self._null_max_loglik()
        if not np.isfinite(null_ll):
            return EProcessState(log_e=0.0, n=self.n, dof=max(self.n - self.n_cond - 1, 0))
        return EProcessState(
            log_e=float(self.log_numerator - null_ll),
            n=self.n,
            dof=max(self.n - self.n_cond - 1, 0),
        )


@dataclass
class DiscreteUniversalCI:
    """Universal-inference e-process for categorical ``X indep Y | Z``.

    ``Z`` is supplied as a single stratum label, so an arbitrary conditioning
    set is handled by mapping the joint configuration to an integer code.
    The numerator uses add-``prior_count`` (Dirichlet) predictive
    probabilities from past data; the denominator is the exactly maximised
    stratified null likelihood ``prod_{x,z} (n_{xz}/n_z)^{n_{xz}}``.
    """

    n_x: int
    n_y: int
    n_z: int = 1
    prior_count: float = 0.5

    n: int = field(default=0, init=False)
    log_numerator: float = field(default=0.0, init=False)
    _joint: np.ndarray = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if min(self.n_x, self.n_y, self.n_z) < 1:
            raise ValueError("category counts must be positive")
        self._joint = np.zeros((self.n_z, self.n_y, self.n_x), dtype=float)

    def update(self, x: int, y: int, z: int = 0) -> EProcessState:
        if not (0 <= x < self.n_x and 0 <= y < self.n_y and 0 <= z < self.n_z):
            raise ValueError("category index out of range")
        cell = self._joint[z, y]
        pred = (cell[x] + self.prior_count) / (cell.sum() + self.prior_count * self.n_x)
        self.log_numerator += float(np.log(pred))
        self._joint[z, y, x] += 1.0
        self.n += 1
        return self.state()

    def _null_max_loglik(self) -> float:
        xz = self._joint.sum(axis=1)          # (n_z, n_x) counts of (x, z)
        z_tot = xz.sum(axis=1, keepdims=True)  # (n_z, 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            p = np.where(z_tot > 0, xz / np.maximum(z_tot, 1.0), 0.0)
            terms = np.where(xz > 0, xz * np.log(np.maximum(p, 1e-300)), 0.0)
        return float(terms.sum())

    def state(self) -> EProcessState:
        return EProcessState(
            log_e=float(self.log_numerator - self._null_max_loglik()),
            n=self.n,
            dof=self.n,
        )
