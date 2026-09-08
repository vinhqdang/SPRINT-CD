"""Random DAGs, linear-Gaussian SEMs, and structural evaluation metrics."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from .graph import ARROW, CIRCLE, NONE, TAIL, MarkedGraph, meek_rules

__all__ = [
    "LinearGaussianSEM",
    "random_dag",
    "dag_to_cpdag",
    "structural_hamming_distance",
    "skeleton_errors",
]


@dataclass
class LinearGaussianSEM:
    """Linear-Gaussian structural equation model ``X = B' X + eps``.

    ``B[i, j]`` is the coefficient of the edge ``i -> j``; the matrix is
    strictly upper triangular with respect to the causal order.
    """

    B: np.ndarray
    noise_sd: np.ndarray
    order: np.ndarray

    @property
    def d(self) -> int:
        return self.B.shape[0]

    @property
    def adjacency(self) -> np.ndarray:
        return (np.abs(self.B) > 0).astype(int)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``n`` i.i.d. observations by forward simulation in causal order."""
        d = self.d
        X = np.zeros((n, d))
        eps = rng.normal(size=(n, d)) * self.noise_sd
        for j in self.order:
            parents = np.nonzero(self.B[:, j])[0]
            X[:, j] = eps[:, j]
            if parents.size:
                X[:, j] += X[:, parents] @ self.B[parents, j]
        return X

    def true_graph(self) -> MarkedGraph:
        return MarkedGraph.from_dag(self.adjacency)

    def true_cpdag(self) -> MarkedGraph:
        return dag_to_cpdag(self.adjacency)

    def marginalise(self, hidden: list[int]) -> tuple["LinearGaussianSEM", list[int]]:
        """Return the SEM restricted to observed variables (for latent-variable tests).

        The returned object retains the full ``B``; the second element lists the
        observed indices.  Latent confounding is produced simply by sampling
        the full model and dropping the hidden columns.
        """
        observed = [v for v in range(self.d) if v not in set(hidden)]
        return self, observed


def random_dag(
    d: int,
    edge_prob: float,
    rng: np.random.Generator,
    *,
    coef_low: float = 0.4,
    coef_high: float = 1.2,
    noise_low: float = 0.8,
    noise_high: float = 1.2,
) -> LinearGaussianSEM:
    """Erdos-Renyi DAG with coefficients bounded away from zero.

    Bounding ``|coef|`` below by ``coef_low`` is what makes the
    ``delta``-strong-faithfulness premise of the SPRINT-CD guarantee
    achievable; with coefficients drawn arbitrarily close to zero no
    finite-sample method can be expected to recover the skeleton.
    """
    order = rng.permutation(d)
    B = np.zeros((d, d))
    for a, b in itertools.combinations(range(d), 2):
        i, j = order[a], order[b]
        if rng.random() < edge_prob:
            mag = rng.uniform(coef_low, coef_high)
            B[i, j] = mag * rng.choice([-1.0, 1.0])
    noise_sd = rng.uniform(noise_low, noise_high, size=d)
    return LinearGaussianSEM(B=B, noise_sd=noise_sd, order=order)


def dag_to_cpdag(adj: np.ndarray) -> MarkedGraph:
    """CPDAG (Markov equivalence class representative) of a DAG."""
    adj = np.asarray(adj)
    d = adj.shape[0]
    skel = ((adj + adj.T) > 0).astype(np.int8)
    g = MarkedGraph(d=d, marks=(skel * TAIL).astype(np.int8))
    np.fill_diagonal(g.marks, NONE)
    for k in range(d):
        parents = [i for i in range(d) if adj[i, k]]
        for i, j in itertools.combinations(parents, 2):
            if not (adj[i, j] or adj[j, i]):
                g.set_edge(i, k, TAIL, ARROW)
                g.set_edge(j, k, TAIL, ARROW)
    return meek_rules(g)


def structural_hamming_distance(est: MarkedGraph, truth: MarkedGraph) -> int:
    """Number of endpoint-mark disagreements, counted one per edge slot.

    An edge present in one graph and absent in the other counts once; an edge
    present in both but with different marks counts once.
    """
    if est.d != truth.d:
        raise ValueError("graphs must have equal size")
    shd = 0
    for i, j in itertools.combinations(range(est.d), 2):
        ea, eb = est.mark_at(j, i), est.mark_at(i, j)
        ta, tb = truth.mark_at(j, i), truth.mark_at(i, j)
        if (ea == NONE) != (ta == NONE):
            shd += 1
        elif ea != NONE and (ea != ta or eb != tb):
            shd += 1
    return shd


def skeleton_errors(est: MarkedGraph, truth: MarkedGraph) -> dict[str, int]:
    """Split skeleton disagreements into missing and extra edges.

    ``missing`` is the quantity SPRINT-CD controls uniformly in time; ``extra``
    is the price paid for that one-sided guarantee at small sample sizes.
    """
    missing = extra = 0
    for i, j in itertools.combinations(range(est.d), 2):
        e, t = est.adjacent(i, j), truth.adjacent(i, j)
        if t and not e:
            missing += 1
        elif e and not t:
            extra += 1
    return {"missing": missing, "extra": extra}
