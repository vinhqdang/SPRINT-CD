"""Partially directed graphs, CPDAGs and PAGs under one mark-based representation.

Every edge endpoint carries a *mark*: a tail ``-``, an arrowhead ``>`` or a
circle ``o``.  ``marks[i, j]`` is the mark sitting at the ``j`` end of the
edge between ``i`` and ``j``, and ``marks[i, j] == NONE`` encodes absence of
an edge.  CPDAGs use only tails and arrowheads; PAGs additionally use
circles.  Sharing one representation lets SPRINT-CD and SPRINT-FCI share the
skeleton machinery and differ only in their orientation rule sets.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

__all__ = ["NONE", "TAIL", "ARROW", "CIRCLE", "MarkedGraph", "meek_rules", "fci_rules"]

NONE = 0
TAIL = 1
ARROW = 2
CIRCLE = 3

_SYMBOL = {TAIL: "-", ARROW: ">", CIRCLE: "o"}
_SYMBOL_LEFT = {TAIL: "-", ARROW: "<", CIRCLE: "o"}


@dataclass
class MarkedGraph:
    """A graph whose edge endpoints carry tail / arrowhead / circle marks."""

    d: int
    marks: np.ndarray = field(default=None)
    names: list[str] | None = None

    def __post_init__(self) -> None:
        if self.marks is None:
            self.marks = np.zeros((self.d, self.d), dtype=np.int8)
        self.marks = np.asarray(self.marks, dtype=np.int8)
        if self.marks.shape != (self.d, self.d):
            raise ValueError("marks must be d x d")

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def complete_undirected(cls, d: int, names: list[str] | None = None) -> "MarkedGraph":
        m = np.full((d, d), TAIL, dtype=np.int8)
        np.fill_diagonal(m, NONE)
        return cls(d=d, marks=m, names=names)

    @classmethod
    def complete_circles(cls, d: int, names: list[str] | None = None) -> "MarkedGraph":
        """Starting point for FCI: every edge present with circles at both ends."""
        m = np.full((d, d), CIRCLE, dtype=np.int8)
        np.fill_diagonal(m, NONE)
        return cls(d=d, marks=m, names=names)

    @classmethod
    def from_dag(cls, adj: np.ndarray, names: list[str] | None = None) -> "MarkedGraph":
        """Build from a binary parent matrix with ``adj[i, j] = 1`` for ``i -> j``."""
        adj = np.asarray(adj)
        d = adj.shape[0]
        g = cls(d=d, names=names)
        for i, j in zip(*np.nonzero(adj)):
            g.set_edge(int(i), int(j), TAIL, ARROW)
        return g

    def copy(self) -> "MarkedGraph":
        return MarkedGraph(d=self.d, marks=self.marks.copy(), names=self.names)

    # ------------------------------------------------------------------
    # basic queries
    # ------------------------------------------------------------------
    def adjacent(self, i: int, j: int) -> bool:
        return bool(self.marks[i, j] != NONE)

    def mark_at(self, i: int, j: int) -> int:
        """Mark at the ``j`` endpoint of edge ``i - j``."""
        return int(self.marks[i, j])

    def set_edge(self, i: int, j: int, mark_at_i: int, mark_at_j: int) -> None:
        self.marks[j, i] = mark_at_i
        self.marks[i, j] = mark_at_j

    def set_mark(self, i: int, j: int, mark: int) -> None:
        """Set the mark at the ``j`` end of the edge ``i - j`` (edge must exist)."""
        if self.marks[i, j] == NONE:
            raise ValueError(f"no edge between {i} and {j}")
        self.marks[i, j] = mark

    def remove_edge(self, i: int, j: int) -> None:
        self.marks[i, j] = NONE
        self.marks[j, i] = NONE

    def neighbours(self, i: int) -> list[int]:
        return [j for j in range(self.d) if j != i and self.marks[i, j] != NONE]

    def edges(self) -> list[tuple[int, int]]:
        return [(i, j) for i, j in itertools.combinations(range(self.d), 2)
                if self.marks[i, j] != NONE]

    def is_directed(self, i: int, j: int) -> bool:
        """``i -> j``: arrowhead at ``j``, tail at ``i``."""
        return self.marks[i, j] == ARROW and self.marks[j, i] == TAIL

    def is_undirected(self, i: int, j: int) -> bool:
        return self.marks[i, j] == TAIL and self.marks[j, i] == TAIL

    def has_arrow(self, i: int, j: int) -> bool:
        """Arrowhead at the ``j`` end, whatever sits at ``i``."""
        return self.marks[i, j] == ARROW

    def skeleton(self) -> np.ndarray:
        return (self.marks != NONE).astype(int)

    # ------------------------------------------------------------------
    def edge_string(self, i: int, j: int) -> str:
        if not self.adjacent(i, j):
            return ""
        left = _SYMBOL_LEFT[int(self.marks[j, i])]
        right = _SYMBOL[int(self.marks[i, j])]
        a = self.names[i] if self.names else str(i)
        b = self.names[j] if self.names else str(j)
        return f"{a} {left}-{right} {b}"

    def __str__(self) -> str:
        es = self.edges()
        if not es:
            return "MarkedGraph(no edges)"
        return "\n".join(self.edge_string(i, j) for i, j in es)


# ----------------------------------------------------------------------
# Meek's rules (CPDAG completion)
# ----------------------------------------------------------------------
def meek_rules(g: MarkedGraph, max_iter: int = 1000) -> MarkedGraph:
    """Apply Meek's rules R1-R3 until no undirected edge can be oriented.

    R1-R3 are complete for completing a pattern to a CPDAG in the absence of
    background knowledge, and each is a purely deterministic consequence of
    the skeleton and the v-structures.  No additional error is incurred, so
    the error guarantee established for the skeleton and v-structures carries
    over to the completed CPDAG unchanged.
    """
    g = g.copy()
    for _ in range(max_iter):
        changed = False
        for a, b in itertools.permutations(range(g.d), 2):
            if not g.is_undirected(a, b):
                continue
            # R1: c -> a, a - b, c and b non-adjacent  =>  a -> b
            for c in range(g.d):
                if c in (a, b):
                    continue
                if g.is_directed(c, a) and not g.adjacent(c, b):
                    g.set_edge(a, b, TAIL, ARROW)
                    changed = True
                    break
            if changed:
                break
            # R2: a -> c -> b and a - b  =>  a -> b
            for c in range(g.d):
                if c in (a, b):
                    continue
                if g.is_directed(a, c) and g.is_directed(c, b):
                    g.set_edge(a, b, TAIL, ARROW)
                    changed = True
                    break
            if changed:
                break
            # R3: a - c -> b, a - dd -> b, c and dd non-adjacent  =>  a -> b
            cands = [c for c in range(g.d)
                     if c not in (a, b) and g.is_undirected(a, c) and g.is_directed(c, b)]
            for c, dd in itertools.combinations(cands, 2):
                if not g.adjacent(c, dd):
                    g.set_edge(a, b, TAIL, ARROW)
                    changed = True
                    break
            if changed:
                break
        if not changed:
            return g
    return g


# ----------------------------------------------------------------------
# FCI orientation rules
# ----------------------------------------------------------------------
def _is_collider_on_path(g: MarkedGraph, prev: int, mid: int, nxt: int) -> bool:
    return g.has_arrow(prev, mid) and g.has_arrow(nxt, mid)


def _discriminating_paths(g: MarkedGraph, b: int, c: int, max_len: int = 12):
    """Yield discriminating paths ``<t, ..., a, b, c>`` for ``b``.

    A path discriminates ``b`` when it spans at least three edges, ``t`` is not
    adjacent to ``c``, and every vertex *strictly between* ``t`` and ``b`` is
    both a collider on the path and a parent of ``c``.

    The search runs backwards from ``b``.  Each extension step is what
    validates the vertex it moves off: pushing ``t`` in front of ``head``
    confirms ``head`` is a collider on ``<t, head, next>``, while ``head``'s
    parenthood of ``c`` is established before it is ever placed on the path.
    A path may therefore only be emitted once its leading vertex has been
    checked -- emitting on the "``t`` not adjacent to ``c``" branch alone would
    accept paths whose first interior vertex is neither a collider nor a parent
    of ``c``, and R4 would then write orientations that are not entailed.
    """
    for a in g.neighbours(b):
        if a == c or a == b:
            continue
        if not g.is_directed(a, c):      # every interior vertex is a parent of c
            continue
        stack = [([a, b, c], {a, b, c})]
        while stack:
            path, seen = stack.pop()
            head, nxt = path[0], path[1]
            for t in g.neighbours(head):
                if t in seen:
                    continue
                if not _is_collider_on_path(g, t, head, nxt):
                    continue
                if not g.adjacent(t, c):
                    yield [t] + path     # t is the endpoint theta
                elif len(path) < max_len and g.is_directed(t, c):
                    stack.append(([t] + path, seen | {t}))


def fci_rules(
    g: MarkedGraph,
    sepsets: dict[tuple[int, int], tuple[int, ...]],
    max_iter: int = 1000,
    use_r4: bool = True,
) -> MarkedGraph:
    """Apply FCI orientation rules R1-R4 to a PAG in which R0 has been run.

    R5-R7 concern selection bias and R8-R10 add tail orientations needed for
    *completeness*; both families are omitted here.  Omitting them costs
    informativeness, never soundness: every mark this function writes is
    entailed, so the anytime-valid error guarantee is unaffected.
    """
    g = g.copy()
    for _ in range(max_iter):
        changed = False

        for a, b in itertools.permutations(range(g.d), 2):
            if not g.adjacent(a, b):
                continue
            # R1: a *-> b o-* c, a and c non-adjacent  =>  b -> c.
            # The circle of "b o-* c" sits at the *b* endpoint of edge b-c,
            # which is mark_at(c, b); reading mark_at(b, c) instead inspects
            # the far end and fires the rule against an existing arrowhead.
            if g.has_arrow(a, b):
                for c in g.neighbours(b):
                    if c == a or g.adjacent(a, c):
                        continue
                    if g.mark_at(c, b) == CIRCLE:
                        g.set_edge(b, c, TAIL, ARROW)
                        changed = True
            # R2: a *-> b -> c (or a -> b *-> c) with a *-o c  =>  a *-> c
            for c in range(g.d):
                if c in (a, b) or not g.adjacent(a, c):
                    continue
                if g.mark_at(a, c) != CIRCLE:
                    continue
                chain1 = g.has_arrow(a, b) and g.is_directed(b, c)
                chain2 = g.is_directed(a, b) and g.has_arrow(b, c)
                if chain1 or chain2:
                    g.set_mark(a, c, ARROW)
                    changed = True

        # R3: a *-> b <-* c, a *-o t o-* c, t *-o b, a and c non-adjacent
        for b in range(g.d):
            for a, c in itertools.combinations(range(g.d), 2):
                if b in (a, c) or g.adjacent(a, c):
                    continue
                if not (g.adjacent(a, b) and g.adjacent(c, b)):
                    continue
                if not (g.has_arrow(a, b) and g.has_arrow(c, b)):
                    continue
                for t in range(g.d):
                    if t in (a, b, c):
                        continue
                    if not (g.adjacent(t, a) and g.adjacent(t, c) and g.adjacent(t, b)):
                        continue
                    if g.mark_at(a, t) == CIRCLE and g.mark_at(c, t) == CIRCLE \
                            and g.mark_at(t, b) == CIRCLE:
                        g.set_mark(t, b, ARROW)
                        changed = True

        # R4: discriminating paths
        if use_r4:
            for b in range(g.d):
                for c in g.neighbours(b):
                    # "b o-* c": the circle is at the b endpoint of edge b-c.
                    if g.mark_at(c, b) != CIRCLE:
                        continue
                    for path in _discriminating_paths(g, b, c):
                        t, a = path[0], path[-3]
                        key = (min(t, c), max(t, c))
                        sep = sepsets.get(key)
                        if sep is not None and b in sep:
                            g.set_edge(b, c, TAIL, ARROW)
                        else:
                            # Orient <a, b, c> as a <-> b <-> c: arrowheads at
                            # *both* ends of each of the two edges.
                            g.set_edge(a, b, ARROW, ARROW)
                            g.set_edge(b, c, ARROW, ARROW)
                        changed = True
                        break

        if not changed:
            return g
    return g
