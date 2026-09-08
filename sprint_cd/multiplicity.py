r"""Anytime-valid multiplicity control across a causal-discovery hypothesis family.

Why a fixed, data-independent budget
------------------------------------
A constraint-based discovery algorithm decides *which* conditional-independence
hypothesis to examine next on the basis of the graph estimated so far -- that
is, on the basis of the data.  An e-process is only valid for the hypothesis
it was built for, so validity cannot be recovered after the fact by correcting
for "the tests that were run": the number and identity of those tests are
themselves random.

SPRINT-CD resolves this by budgeting over the entire *potential* family

.. math::

    \mathcal{H}_k = \bigl\{ (i, j, S) : i < j,\; S \subseteq V
        \setminus \{i, j\},\; |S| \le k \bigr\},

which is fixed before any data arrive.  Weights :math:`w(i,j,S)` summing to
one are assigned in advance, and hypothesis :math:`(i,j,S)` is rejected when
its e-process crosses :math:`1 / (\alpha\, w(i,j,S))`.  A union bound over
:math:`\mathcal{H}_k` combined with Ville's inequality then gives, for every
stopping time :math:`\tau` simultaneously,

.. math::

    P\bigl( \exists\, (i,j,S) \in \mathcal{H}_k \text{ true and rejected by }
        \tau \bigr) \;\le\; \sum_{(i,j,S)} \alpha\, w(i,j,S) \;=\; \alpha .

Crucially the bound does not reference *which* hypotheses the algorithm chose
to look at, so the adaptive, graph-dependent enumeration of conditioning sets
costs nothing extra.  Hypotheses never examined simply never cross.

Weights decay geometrically in :math:`|S|`, concentrating the budget on the
low-order tests where separating sets predominantly live.

Note on budget recycling
------------------------
Reallocating the weight of a resolved hypothesis to unresolved ones is
tempting but is *not* sound in general: the reallocation is data-dependent and
breaks the union bound.  This module therefore uses fixed weights only.  The
e-BH routine below is the sound way to trade the conservatism of a union
bound for a false-discovery-rate criterion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["HypothesisBudget", "e_bh", "e_bh_threshold_indices"]


@dataclass
class HypothesisBudget:
    """Fixed, data-independent weight allocation over ``H_k``.

    Parameters
    ----------
    d:
        Number of variables.
    max_order:
        Largest conditioning-set size ``k`` the algorithm may ever use.
    alpha:
        Total error budget spent by this family.
    order_decay:
        Geometric ratio for the mass given to successive conditioning-set
        sizes.  ``0.5`` halves the budget at each order.
    """

    d: int
    max_order: int
    alpha: float
    order_decay: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        if self.max_order < 0:
            raise ValueError("max_order must be non-negative")
        if not 0.0 < self.order_decay <= 1.0:
            raise ValueError("order_decay must lie in (0, 1]")
        k = min(self.max_order, max(self.d - 2, 0))
        raw = np.array([self.order_decay**m for m in range(k + 1)], dtype=float)
        self._order_weight = raw / raw.sum()
        self._n_pairs = max(self.d * (self.d - 1) // 2, 1)
        self._effective_order = k

    @property
    def effective_max_order(self) -> int:
        return self._effective_order

    def family_size(self) -> int:
        """Cardinality of ``H_k`` -- reported for transparency, not used in the bound."""
        total = 0
        for m in range(self._effective_order + 1):
            total += self._n_pairs * math.comb(max(self.d - 2, 0), m)
        return total

    def weight(self, order: int) -> float:
        """Weight ``w(i, j, S)`` for any hypothesis with ``|S| = order``."""
        if order < 0 or order > self._effective_order:
            return 0.0
        n_sets = math.comb(max(self.d - 2, 0), order)
        return float(self._order_weight[order] / (self._n_pairs * max(n_sets, 1)))

    def level(self, order: int) -> float:
        """Per-hypothesis level ``alpha * w`` for conditioning sets of size ``order``."""
        w = self.weight(order)
        return float(self.alpha * w) if w > 0 else 0.0

    def log_threshold(self, order: int) -> float:
        """``log(1 / (alpha * w))``: the Ville boundary on the log scale."""
        lvl = self.level(order)
        if lvl <= 0.0:
            return np.inf
        return float(-np.log(lvl))


def e_bh(evalues: np.ndarray, alpha: float) -> np.ndarray:
    """e-BH procedure: FDR control under *arbitrary* dependence.

    Given ``m`` e-values, sort them decreasingly and reject the ``k`` largest
    for the greatest ``k`` with ``E_(k) >= m / (alpha * k)``.  Validity requires
    nothing of the dependence structure between the e-values -- which matters
    here, since e-processes for overlapping conditioning sets are built from
    the same data and are strongly dependent.  Because a stopped e-process is
    still an e-value, applying e-BH at a data-dependent stopping time retains
    the guarantee.

    Returns a boolean rejection mask aligned with the input.
    """
    e = np.asarray(evalues, dtype=float)
    if e.ndim != 1:
        raise ValueError("evalues must be one-dimensional")
    m = e.size
    if m == 0:
        return np.zeros(0, dtype=bool)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")

    order = np.argsort(-e)
    sorted_e = e[order]
    ks = np.arange(1, m + 1, dtype=float)
    ok = sorted_e >= (m / (alpha * ks))
    k_star = int(np.max(np.nonzero(ok)[0]) + 1) if np.any(ok) else 0

    mask = np.zeros(m, dtype=bool)
    if k_star > 0:
        mask[order[:k_star]] = True
    return mask


def e_bh_threshold_indices(log_evalues: np.ndarray, alpha: float) -> np.ndarray:
    """``e_bh`` on the log scale, avoiding overflow for large e-values."""
    le = np.asarray(log_evalues, dtype=float)
    m = le.size
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(-le)
    sorted_le = le[order]
    ks = np.arange(1, m + 1, dtype=float)
    ok = sorted_le >= (np.log(m) - np.log(alpha) - np.log(ks))
    k_star = int(np.max(np.nonzero(ok)[0]) + 1) if np.any(ok) else 0
    mask = np.zeros(m, dtype=bool)
    if k_star > 0:
        mask[order[:k_star]] = True
    return mask
