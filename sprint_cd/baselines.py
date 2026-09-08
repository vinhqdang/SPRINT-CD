"""Fixed-sample baselines: classical PC with Fisher-z tests.

Included so that the anytime-valid procedure can be compared against the
standard practice it is meant to improve on -- in particular against the
common but invalid habit of re-running a fixed-sample test as data accumulate
and stopping when the answer looks settled.
"""

from __future__ import annotations

import itertools

import numpy as np
from scipy import stats

from .graph import ARROW, NONE, TAIL, MarkedGraph, meek_rules
from .stats import GaussianSuffStat

__all__ = ["fisher_z_pvalue", "pc_fixed_sample"]


def fisher_z_pvalue(suffstat: GaussianSuffStat, i: int, j: int, cond=()) -> float:
    """Two-sided p-value of Fisher's z transform of the sample partial correlation."""
    m = suffstat.residual_moments(i, j, cond)
    n_eff = m.n - len(tuple(cond)) - 3
    if n_eff <= 0:
        return 1.0
    r = np.clip(m.partial_corr, -0.999999, 0.999999)
    z = 0.5 * np.log1p(2 * r / (1 - r)) if abs(r) < 1 else np.inf
    stat = np.sqrt(n_eff) * abs(z)
    return float(2.0 * stats.norm.sf(stat))


def pc_fixed_sample(
    data: np.ndarray, alpha: float = 0.05, max_order: int = 3, stable: bool = True
) -> MarkedGraph:
    """Textbook PC (PC-stable) on a fixed sample, using Fisher-z tests at level ``alpha``."""
    data = np.asarray(data, dtype=float)
    d = data.shape[1]
    ss = GaussianSuffStat(d)
    ss.update(data)

    g = MarkedGraph.complete_undirected(d)
    sepsets: dict[tuple[int, int], tuple[int, ...]] = {}
    max_order = min(max_order, max(d - 2, 0))

    for order in range(max_order + 1):
        adj = {v: g.neighbours(v) for v in range(d)} if stable else None
        for i, j in list(itertools.combinations(range(d), 2)):
            if not g.adjacent(i, j):
                continue
            if not stable:
                adj = {v: g.neighbours(v) for v in range(d)}
            seen: set[tuple[int, ...]] = set()
            removed = False
            for base in (i, j):
                pool = [v for v in adj[base] if v not in (i, j)]
                if len(pool) < order:
                    continue
                for S in itertools.combinations(sorted(pool), order):
                    if S in seen:
                        continue
                    seen.add(S)
                    if fisher_z_pvalue(ss, i, j, S) > alpha:
                        g.remove_edge(i, j)
                        sepsets[(i, j)] = S
                        removed = True
                        break
                if removed:
                    break

    for k in range(d):
        for i, j in itertools.combinations(g.neighbours(k), 2):
            if g.adjacent(i, j):
                continue
            sep = sepsets.get((min(i, j), max(i, j)))
            if sep is None or k in sep:
                continue
            if g.mark_at(k, i) != ARROW and g.mark_at(k, j) != ARROW:
                g.set_edge(i, k, TAIL, ARROW)
                g.set_edge(j, k, TAIL, ARROW)
    return meek_rules(g)
