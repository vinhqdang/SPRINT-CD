r"""SPRINT-FCI: anytime-valid causal discovery allowing latent confounders.

Same statistical machinery as :mod:`sprint_cd.sprint_cd` -- e-process
dependence certificates, confidence-sequence independence certificates, a
fixed budget over the potential hypothesis family -- but the output is a
**PAG** rather than a CPDAG, so causal sufficiency is not assumed.

Two deliberate restrictions keep the anytime-valid guarantee intact:

* The Possible-D-SEP refinement is run only over conditioning sets of size at
  most ``max_order``, so every hypothesis it touches already lies inside the
  budgeted family :math:`\mathcal{H}_k`.  Truncating the search costs
  informativeness (some inducing paths go undetected) but never soundness.
* Orientation uses rules R1-R4.  R5-R7 (selection bias) and R8-R10 (tail
  completeness) are omitted; the PAG returned is sound but not guaranteed
  maximally informative.

The error guarantee is inherited verbatim from SPRINT-CD: the probability that
any true adjacency is ever removed, or any true independence ever certified
dependent, at any stopping time, is at most ``alpha``.
"""

from __future__ import annotations

import itertools

import numpy as np

from .graph import ARROW, CIRCLE, NONE, MarkedGraph, fci_rules
from .sprint_cd import SprintCD, SprintCDConfig

__all__ = ["SprintFCI", "run_sprint_fci", "pag_from_skeleton"]


def pag_from_skeleton(
    skeleton: MarkedGraph, sepsets: dict[tuple[int, int], tuple[int, ...]],
    use_r4: bool = True,
) -> MarkedGraph:
    """Turn a skeleton plus separating sets into a PAG (R0 followed by R1-R4)."""
    d = skeleton.d
    g = MarkedGraph(d=d, marks=np.where(skeleton.skeleton() > 0, CIRCLE, NONE).astype(np.int8),
                    names=skeleton.names)
    np.fill_diagonal(g.marks, NONE)

    # R0: unshielded colliders.
    for k in range(d):
        for i, j in itertools.combinations(g.neighbours(k), 2):
            if g.adjacent(i, j):
                continue
            sep = sepsets.get((min(i, j), max(i, j)))
            if sep is None or k in sep:
                continue
            g.set_mark(i, k, ARROW)
            g.set_mark(j, k, ARROW)
    return fci_rules(g, sepsets, use_r4=use_r4)


class SprintFCI(SprintCD):
    """Anytime-valid FCI.  Streaming interface identical to :class:`SprintCD`."""

    def __init__(self, d: int, config: SprintCDConfig | None = None,
                 names: list[str] | None = None, use_pdsep: bool = True,
                 use_r4: bool = True) -> None:
        super().__init__(d, config=config, names=names)
        self.use_pdsep = use_pdsep
        self.use_r4 = use_r4

    # ------------------------------------------------------------------
    def possible_d_sep(self, i: int, j: int) -> list[int]:
        """Possible-D-SEP(i, j): vertices reachable from ``i`` by paths whose
        every consecutive triple is either a collider or a triangle."""
        g = self.graph
        out: set[int] = set()
        stack = [(i, k) for k in g.neighbours(i)]
        seen = set(stack)
        while stack:
            a, b = stack.pop()
            if b != j:
                out.add(b)
            for c in g.neighbours(b):
                if c == a or (b, c) in seen:
                    continue
                collider = g.has_arrow(a, b) and g.has_arrow(c, b)
                triangle = g.adjacent(a, c)
                if collider or triangle:
                    seen.add((b, c))
                    stack.append((b, c))
        out.discard(i)
        out.discard(j)
        return sorted(out)

    def _pdsep_prune(self) -> None:
        """Second adjacency pass conditioning on Possible-D-SEP subsets."""
        max_order = min(self.config.max_order, max(self.d - 2, 0))
        for i, j in list(self.graph.edges()):
            pds = [v for v in self.possible_d_sep(i, j) if v not in (i, j)]
            removed = False
            for order in range(max_order + 1):
                if len(pds) < order:
                    break
                for S in itertools.combinations(pds, order):
                    if self._independence_certificate(i, j, S):
                        self.graph.remove_edge(i, j)
                        self.sepsets[(i, j)] = S
                        removed = True
                        break
                if removed:
                    break

    # ------------------------------------------------------------------
    def pag(self) -> MarkedGraph:
        """Current estimate as a PAG."""
        if self.use_pdsep:
            self._pdsep_prune()
        return pag_from_skeleton(self.graph, self.sepsets, use_r4=self.use_r4)

    # The inherited ``cpdag`` is meaningless without causal sufficiency.
    def cpdag(self) -> MarkedGraph:  # pragma: no cover - guard against misuse
        raise NotImplementedError("SprintFCI returns a PAG; call pag() instead")

    def update(self, batch: np.ndarray, *, keep_graph: bool = False) -> MarkedGraph:
        if not self._warmed:
            raise RuntimeError("call warm_up() before update()")
        arr = np.atleast_2d(np.asarray(batch, dtype=float))
        self.suffstat.update(self._standardise(arr))
        self._sweep()
        self.history.record(self.suffstat.n, self.graph, self.resolved, keep_graph)
        return self.graph


def run_sprint_fci(
    data: np.ndarray,
    config: SprintCDConfig | None = None,
    *,
    batch_size: int = 25,
    stop_when_resolved: bool = True,
    names: list[str] | None = None,
) -> tuple[MarkedGraph, SprintFCI]:
    """Stream ``data`` through :class:`SprintFCI`, returning the final PAG."""
    data = np.asarray(data, dtype=float)
    cfg = config or SprintCDConfig()
    algo = SprintFCI(d=data.shape[1], config=cfg, names=names)
    if data.shape[0] <= cfg.warmup:
        raise ValueError("not enough rows for the warm-up prefix")
    algo.warm_up(data[: cfg.warmup])
    rest = data[cfg.warmup:]
    for start in range(0, rest.shape[0], batch_size):
        algo.update(rest[start: start + batch_size])
        if stop_when_resolved and algo.resolved:
            break
    return algo.pag(), algo
