r"""E-ICP: anytime-valid Invariant Causal Prediction.

Invariant Causal Prediction (Peters, Buhlmann & Meinshausen, 2016) selects
causal predictors by retaining only those subsets :math:`S` under which the
conditional law of :math:`Y` given :math:`X_S` is invariant across
environments, and outputting the intersection of the retained sets.  Its
guarantee -- that the output is contained in the true parent set with
probability :math:`1 - \alpha` -- is a *fixed-sample* statement: it is void if
the analyst watches environments accumulate and stops when the answer looks
stable, which is precisely how the method invites being used when
environments arrive over time.

E-ICP replaces the invariance p-value with an **e-process**.  For each
candidate set :math:`S`, invariance of the linear model is expressed as the
null that a block of coefficients vanishes in

.. math::

    Y \;=\; \mu + X_S^\top \gamma
        \;+\; \underbrace{\textstyle\sum_e \delta_e D_e
              \;+\; \sum_e \sum_{j \in S} \theta_{ej} D_e X_j}
              _{\text{tested block: environment effects}}
        \;+\; \varepsilon ,

with :math:`D_e` the environment dummies.  The block is tested by the
right-Haar Bayes factor of :func:`sprint_cd.eprocess.safe_linear
.safe_linear_block_log_e`, which is an e-process, so Ville's inequality
applies at every stopping time.

**Guarantee.**  With :math:`w(S)` a fixed weighting of the candidate sets and
:math:`S` rejected when :math:`E_S \ge 1/(\alpha w(S))`, a union bound gives

.. math::

    P\bigl( \exists\, t :\; S^\star \text{ rejected by time } t \bigr)
        \;\le\; \alpha ,

hence :math:`\hat S_t = \bigcap_{S \text{ not rejected}} S \subseteq S^\star`
*simultaneously at all times* with probability at least :math:`1-\alpha`.
Environments may therefore be accumulated, monitored, and stopped adaptively.

Scope
-----
The tested block captures shifts in the conditional mean, including
environment-specific slopes.  A shift in the *noise variance* alone leaves the
block at zero and is not detected -- the same limitation carried by the
mean-based variants of ICP.  Li and Goeman (2024) argue that FDR sits awkwardly
with ICP's intersection output; E-ICP sidesteps that by controlling
anytime-valid family-wise error instead, which is the criterion the
intersection construction actually needs.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

from .eprocess.safe_linear import safe_linear_block_log_e
from .stats import GaussianSuffStat

__all__ = ["EICPConfig", "EICP", "icp_fixed_sample", "icp_invariance_pvalue"]


@dataclass
class EICPConfig:
    """Configuration for :class:`EICP`."""

    alpha: float = 0.05
    max_size: int = 3
    prior_scale: float = 0.4
    size_decay: float = 0.5
    warmup: int = 40
    include_interactions: bool = True

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        if self.max_size < 0:
            raise ValueError("max_size must be non-negative")


class EICP:
    """Anytime-valid invariant causal prediction over a streaming set of environments.

    Parameters
    ----------
    d:
        Number of candidate predictors.
    n_env:
        Number of environments.  Environment ``0`` acts as the reference level.
    """

    def __init__(self, d: int, n_env: int, config: EICPConfig | None = None) -> None:
        if d < 1:
            raise ValueError("need at least one predictor")
        if n_env < 2:
            raise ValueError("invariance is only testable with at least two environments")
        self.d = d
        self.n_env = n_env
        self.config = config or EICPConfig()
        self._n_dummy = n_env - 1
        self._inter = self.config.include_interactions

        # Column layout: [Y] [X_1..X_d] [D_1..D_{E-1}] [D_e * X_j]
        self._n_cols = 1 + d + self._n_dummy + (d * self._n_dummy if self._inter else 0)
        self.suffstat = GaussianSuffStat(self._n_cols)

        self._max_size = min(self.config.max_size, d)
        self._candidates = [
            S for m in range(self._max_size + 1)
            for S in itertools.combinations(range(d), m)
        ]
        raw = np.array([self.config.size_decay**m for m in range(self._max_size + 1)])
        self._size_weight = raw / raw.sum()

        self._logmax: dict[tuple[int, ...], float] = {}
        self._rejected: set[tuple[int, ...]] = set()
        self._warm_mean = np.zeros(self._n_cols)
        self._warm_scale = np.ones(self._n_cols)
        self._g = self.config.prior_scale**2
        self._warmed = False

    # ------------------------------------------------------------------
    def _dummy_col(self, e: int) -> int:
        return 1 + self.d + (e - 1)

    def _inter_col(self, e: int, j: int) -> int:
        return 1 + self.d + self._n_dummy + (e - 1) * self.d + j

    def _design(self, X: np.ndarray, y: np.ndarray, env: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).ravel()
        env = np.asarray(env, dtype=int).ravel()
        if X.shape[0] != y.size or X.shape[0] != env.size:
            raise ValueError("X, y and env must have matching lengths")
        if X.shape[1] != self.d:
            raise ValueError(f"expected {self.d} predictors")
        if env.min() < 0 or env.max() >= self.n_env:
            raise ValueError("environment labels out of range")

        n = X.shape[0]
        M = np.zeros((n, self._n_cols))
        M[:, 0] = y
        M[:, 1: 1 + self.d] = X
        for e in range(1, self.n_env):
            mask = (env == e).astype(float)
            M[:, self._dummy_col(e)] = mask
            if self._inter:
                for j in range(self.d):
                    M[:, self._inter_col(e, j)] = mask * X[:, j]
        return M

    # ------------------------------------------------------------------
    def weight(self, S: tuple[int, ...]) -> float:
        m = len(S)
        if m > self._max_size:
            return 0.0
        return float(self._size_weight[m] / math.comb(self.d, m))

    def log_threshold(self, S: tuple[int, ...]) -> float:
        w = self.weight(S)
        if w <= 0.0:
            return np.inf
        return float(-np.log(self.config.alpha * w))

    # ------------------------------------------------------------------
    def warm_up(self, X: np.ndarray, y: np.ndarray, env: np.ndarray) -> "EICP":
        """Fix standardisation constants from a prefix excluded from the e-processes."""
        M = self._design(X, y, env)
        if M.shape[0] < 2:
            raise ValueError("warm-up needs at least two rows")
        self._warm_mean = M.mean(axis=0)
        sd = M.std(axis=0, ddof=1)
        self._warm_scale = np.where(np.isfinite(sd) & (sd > 1e-12), sd, 1.0)
        self._warmed = True
        return self

    def update(self, X: np.ndarray, y: np.ndarray, env: np.ndarray) -> set[tuple[int, ...]]:
        """Absorb a batch of observations and refresh the rejection set."""
        if not self._warmed:
            raise RuntimeError("call warm_up() before update()")
        M = self._design(X, y, env)
        self.suffstat.update((M - self._warm_mean) / self._warm_scale)

        for S in self._candidates:
            if S in self._rejected:
                continue
            log_e = self._log_e(S)
            cur = max(self._logmax.get(S, -np.inf), log_e)
            self._logmax[S] = cur
            if cur >= self.log_threshold(S):
                self._rejected.add(S)
        return set(self._rejected)

    def _log_e(self, S: tuple[int, ...]) -> float:
        nuisance = tuple(j + 1 for j in S)
        tested = [self._dummy_col(e) for e in range(1, self.n_env)]
        if self._inter:
            tested += [self._inter_col(e, j) for e in range(1, self.n_env) for j in S]
        s_tt, s_wt, S_ww, dof = self.suffstat.residual_block_moments(
            0, nuisance, tuple(tested)
        )
        if dof <= len(tested):
            return 0.0
        return safe_linear_block_log_e(s_tt, s_wt, S_ww, dof, self._g)

    # ------------------------------------------------------------------
    @property
    def accepted(self) -> list[tuple[int, ...]]:
        return [S for S in self._candidates if S not in self._rejected]

    def estimate(self) -> set[int]:
        """Intersection of the non-rejected candidate sets.

        Empty either because too little evidence has accumulated (the empty set
        is still accepted) or because every candidate has been rejected, which
        signals that no subset within ``max_size`` renders ``Y`` invariant.
        """
        acc = self.accepted
        if not acc:
            return set()
        out = set(acc[0])
        for S in acc[1:]:
            out &= set(S)
            if not out:
                break
        return out

    def log_e(self, S: tuple[int, ...]) -> float:
        """Running maximum of the e-process for candidate set ``S``."""
        return float(self._logmax.get(tuple(sorted(S)), -np.inf))


def icp_invariance_pvalue(
    X: np.ndarray, y: np.ndarray, env: np.ndarray, S: tuple[int, ...]
) -> float:
    """F-test p-value for the invariance null of candidate set ``S``.

    Tests whether environment dummies and their interactions with ``X_S`` add
    anything to the regression of ``y`` on ``X_S`` -- the fixed-sample analogue
    of the e-process used by :class:`EICP`.
    """
    from scipy import stats as _stats

    X = np.atleast_2d(np.asarray(X, dtype=float))
    y = np.asarray(y, dtype=float).ravel()
    env = np.asarray(env, dtype=int).ravel()
    n = X.shape[0]
    n_env = int(env.max()) + 1

    Z = np.column_stack([np.ones(n)] + [X[:, j] for j in S])
    extra = []
    for e in range(1, n_env):
        mask = (env == e).astype(float)
        extra.append(mask)
        extra.extend(mask * X[:, j] for j in S)
    W = np.column_stack(extra) if extra else np.zeros((n, 0))
    full = np.column_stack([Z, W]) if W.shape[1] else Z

    def _rss(A):
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        r = y - A @ coef
        return float(r @ r), int(np.linalg.matrix_rank(A))

    rss0, p0 = _rss(Z)
    rss1, p1 = _rss(full)
    df1, df2 = max(p1 - p0, 1), n - p1
    if rss1 <= 0 or df2 <= 0:
        return 1.0
    F = ((rss0 - rss1) / df1) / (rss1 / df2)
    return float(_stats.f.sf(F, df1, df2))


def icp_fixed_sample(
    X: np.ndarray, y: np.ndarray, env: np.ndarray, alpha: float = 0.05, max_size: int = 3
) -> set[int]:
    """Fixed-sample ICP baseline: invariance F-test, Bonferroni-corrected.

    Provided for comparison; unlike :class:`EICP` its guarantee holds only at
    the single, pre-specified sample size.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    d = X.shape[1]
    cands = [S for m in range(min(max_size, d) + 1)
             for S in itertools.combinations(range(d), m)]
    level = alpha / len(cands)
    accepted = [S for S in cands if icp_invariance_pvalue(X, y, env, S) > level]
    if not accepted:
        return set()
    out = set(accepted[0])
    for S in accepted[1:]:
        out &= set(S)
    return out
