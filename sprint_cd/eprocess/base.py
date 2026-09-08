"""Common vocabulary for e-processes and anytime-valid decisions.

An *e-process* for a null ``H0`` is a non-negative process ``(E_t)`` such that
``E[E_tau] <= 1`` for every stopping time ``tau``.  Ville's inequality then
gives the time-uniform guarantee

    P( exists t : E_t >= 1/alpha ) <= alpha,

which is what licenses continuous monitoring and data-dependent stopping.
Everything in SPRINT-CD is phrased in terms of this single primitive.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["EProcessState", "ville_threshold", "log_ville_threshold", "RunningMaxE"]


def ville_threshold(alpha: float) -> float:
    """Rejection threshold ``1/alpha`` of Ville's inequality."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    return 1.0 / alpha


def log_ville_threshold(alpha: float) -> float:
    """``log(1/alpha)``; e-values are handled on the log scale throughout."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    return -np.log(alpha)


@dataclass
class EProcessState:
    """Value of an e-process at one time point, carried on the log scale.

    Storing ``log_e`` rather than ``e`` matters: evidence accumulates
    multiplicatively and overflows double precision well before the sample
    sizes at which discovery problems become interesting.
    """

    log_e: float
    n: int
    dof: int = 0

    @property
    def e(self) -> float:
        return float(np.exp(np.clip(self.log_e, -700.0, 700.0)))

    def rejects(self, alpha: float) -> bool:
        """Whether the e-value crosses the Ville boundary at level ``alpha``."""
        return self.log_e >= log_ville_threshold(alpha)


@dataclass
class RunningMaxE:
    """Running supremum of an e-process.

    Ville's inequality bounds the probability that the *supremum* ever crosses
    ``1/alpha``, so a decision once made is never revoked: the running maximum
    is the correct statistic for anytime-valid rejection under monitoring.
    """

    log_max: float = -np.inf
    argmax_n: int = 0
    _history: list[tuple[int, float]] = field(default_factory=list, repr=False)

    def update(self, state: EProcessState, *, record: bool = False) -> "RunningMaxE":
        if state.log_e > self.log_max:
            self.log_max = float(state.log_e)
            self.argmax_n = int(state.n)
        if record:
            self._history.append((int(state.n), float(state.log_e)))
        return self

    def rejects(self, alpha: float) -> bool:
        return self.log_max >= log_ville_threshold(alpha)

    @property
    def history(self) -> list[tuple[int, float]]:
        return list(self._history)
