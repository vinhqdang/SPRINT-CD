"""CERT-FCI: the certificate-only skeleton under latent confounding.

The adjacency certificate does not assume causal sufficiency.  Nothing in the
separability null requires that all common causes are observed, so the
construction carries over to a maximal ancestral graph with d-separation
replaced by m-separation, and a certified edge then means "neither variable is
separable from the other by any observed set" -- compatible with a direct
effect, a latent common cause, or both.

**What is and is not inside the certificate-only discipline.** The skeleton is:
it grows from the empty graph and an edge appears only when the running maximum
of ``A_t(i, j)`` crosses its threshold.  The orientation layer is *not*.  The
FCI rules need separating sets, and a separating set is accepted because a test
*failed to reject* -- exactly the inference the governing rule forbids.  The
adjacency certificate compounds the problem: it is a minimum over conditioning
sets, so even when it fires it does not identify which set separated the pair,
and for an *undecided* pair there is no certified statement at all.

This module therefore does the only honest thing available: it derives sepsets
by the argmin of the per-set e-process, labels them as an uncertified input,
and reports the output as a partially oriented mixed graph rather than as a
PAG.  Orientation marks produced this way inherit no family-wise guarantee.
The certified skeleton does.

An earlier version of the experiments reported ``SprintFCI`` numbers -- the
delete-on-non-rejection comparator -- as though they came from this algorithm.
They did not, and the metrics gave it away: "lost adjacency" is not defined for
a procedure that never deletes.
"""

from __future__ import annotations

import itertools

import numpy as np

from .certcd import CERTIFIED, CertCD, CertCDConfig
from .certificates import admissible_sets
from .graph import CIRCLE, MarkedGraph
from .sprint_fci import pag_from_skeleton


class CertFCI(CertCD):
    """Certificate-only skeleton plus an uncertified FCI orientation layer.

    Inherits the adjacency layer unchanged.  Direction certificates are off by
    default: the arrowhead null of the direction certificate assumes no
    unobserved common cause of the pair, which is exactly what this setting
    gives up.
    """

    def __init__(self, d: int, config: CertCDConfig | None = None,
                 names: list[str] | None = None) -> None:
        cfg = config or CertCDConfig()
        if cfg.orient:
            cfg = CertCDConfig(**{**cfg.__dict__, "orient": False})
        super().__init__(d, cfg, names)

    # ------------------------------------------------------------------
    def certified_skeleton(self) -> MarkedGraph:
        """Certified adjacencies only, both ends circled."""
        g = MarkedGraph(d=self.d, names=self.names)
        for i, j in self.certified_edges():
            g.set_edge(i, j, CIRCLE, CIRCLE)
        return g

    def _argmin_sepsets(self) -> dict[tuple[int, int], tuple[int, ...]]:
        """Least-favourable conditioning set per *non*-certified pair.

        This is the uncertified input named in the module docstring.  For a
        pair the certificate has not decided, the set minimising the per-set
        e-process is the one the data least distinguish from independence; it
        is a plausible separating set and nothing more.  No error statement
        attaches to it.
        """
        sep: dict[tuple[int, int], tuple[int, ...]] = {}
        for i, j in itertools.combinations(range(self.d), 2):
            if self.adj_status[i, j] == CERTIFIED:
                continue
            best_v, best_S = np.inf, ()
            for S in admissible_sets(self.d, i, j, self.config.max_order):
                v = self._ci.symmetric_log_e(i, j, S).log_e
                if v < best_v:
                    best_v, best_S = v, S
            sep[(i, j)] = tuple(best_S)
        return sep

    def partially_oriented_graph(self, *, use_r4: bool = True) -> MarkedGraph:
        """Certified skeleton with FCI R0 and R1--R4 applied.

        Named for what it is.  The result is *not* a PAG in the technical
        sense: R1--R4 are sound but not complete (completeness needs R1--R10),
        and the sepsets driving R0 are uncertified.
        """
        return pag_from_skeleton(self.certified_skeleton(),
                                 self._argmin_sepsets(), use_r4=use_r4)


def run_cert_fci(data: np.ndarray, config: CertCDConfig | None = None, *,
                 batch_size: int = 100, names: list[str] | None = None):
    """Stream ``data`` through :class:`CertFCI`; returns the graph and object."""
    data = np.asarray(data, dtype=float)
    cfg = config or CertCDConfig()
    algo = CertFCI(d=data.shape[1], config=cfg, names=names)
    if data.shape[0] <= cfg.warmup:
        raise ValueError("not enough rows for the warm-up prefix")
    algo.warm_up(data[: cfg.warmup])
    rest = data[cfg.warmup:]
    for start in range(0, rest.shape[0], batch_size):
        algo.update(rest[start: start + batch_size])
    return algo.partially_oriented_graph(), algo
