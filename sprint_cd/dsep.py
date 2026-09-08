"""d-separation and oracle PAG construction, for evaluating latent-variable recovery.

Ground truth for the FCI setting is the PAG of the true DAG under
marginalisation, which is obtained by running FCI's orientation rules on top
of *oracle* conditional-independence answers.  d-separation is computed via
the moralised-ancestral-graph characterisation, which is short and hard to get
subtly wrong.
"""

from __future__ import annotations

import itertools

import numpy as np

__all__ = ["ancestors", "d_separated", "oracle_skeleton_and_sepsets"]


def ancestors(adj: np.ndarray, nodes) -> set[int]:
    """All ancestors of ``nodes`` in the DAG, inclusive of ``nodes`` themselves."""
    adj = np.asarray(adj)
    out = set(int(v) for v in nodes)
    frontier = list(out)
    while frontier:
        v = frontier.pop()
        for u in np.nonzero(adj[:, v])[0]:
            u = int(u)
            if u not in out:
                out.add(u)
                frontier.append(u)
    return out


def d_separated(adj: np.ndarray, x: int, y: int, Z=()) -> bool:
    """Is ``x`` d-separated from ``y`` given ``Z`` in the DAG ``adj``?

    Uses the standard equivalence: d-separation in ``G`` is ordinary
    separation in the moral graph of the subgraph induced on
    ``An({x, y} u Z)``.
    """
    adj = np.asarray(adj)
    Z = set(int(v) for v in Z)
    if x in Z or y in Z or x == y:
        raise ValueError("x and y must be distinct and outside Z")

    A = ancestors(adj, {x, y} | Z)
    idx = sorted(A)
    pos = {v: i for i, v in enumerate(idx)}
    m = len(idx)
    M = np.zeros((m, m), dtype=bool)

    for v in idx:
        parents = [int(u) for u in np.nonzero(adj[:, v])[0] if u in A]
        for u in parents:                       # keep edges, drop directions
            M[pos[u], pos[v]] = M[pos[v], pos[u]] = True
        for u, w in itertools.combinations(parents, 2):   # moralise
            M[pos[u], pos[w]] = M[pos[w], pos[u]] = True

    blocked = {pos[v] for v in Z}
    start, target = pos[x], pos[y]
    seen = {start}
    stack = [start]
    while stack:
        v = stack.pop()
        if v == target:
            return False
        for w in np.nonzero(M[v])[0]:
            w = int(w)
            if w in seen or w in blocked:
                continue
            seen.add(w)
            stack.append(w)
    return True


def oracle_skeleton_and_sepsets(
    adj: np.ndarray, observed: list[int], max_order: int
):
    """Skeleton and separating sets over ``observed`` using d-separation as oracle.

    Conditioning sets are drawn from the observed variables only, mirroring
    what an algorithm can actually condition on when the remaining variables
    are latent.
    """
    from .graph import MarkedGraph

    obs = list(observed)
    d = len(obs)
    g = MarkedGraph.complete_undirected(d)
    sepsets: dict[tuple[int, int], tuple[int, ...]] = {}
    max_order = min(max_order, max(d - 2, 0))

    for order in range(max_order + 1):
        neigh = {v: g.neighbours(v) for v in range(d)}
        for a, b in itertools.combinations(range(d), 2):
            if not g.adjacent(a, b):
                continue
            found = False
            for base in (a, b):
                pool = [v for v in neigh[base] if v not in (a, b)]
                if len(pool) < order:
                    continue
                for S in itertools.combinations(sorted(pool), order):
                    if d_separated(adj, obs[a], obs[b], [obs[v] for v in S]):
                        g.remove_edge(a, b)
                        sepsets[(a, b)] = S
                        found = True
                        break
                if found:
                    break
    return g, sepsets
