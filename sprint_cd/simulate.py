"""Random DAGs, linear-Gaussian SEMs, and structural evaluation metrics."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from .graph import ARROW, NONE, TAIL, MarkedGraph, meek_rules

__all__ = [
    "LinearGaussianSEM",
    "random_dag",
    "dag_to_cpdag",
    "structural_hamming_distance",
    "skeleton_errors",
    "population_correlation",
    "partial_correlation",
    "strong_faithfulness_margin",
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

    def sample(self, n: int, rng: np.random.Generator,
               noise: str = "gaussian") -> np.ndarray:
        """Draw ``n`` i.i.d. observations by forward simulation in causal order.

        ``noise`` selects the (standardised) innovation law.  Non-Gaussian
        options matter for direction identifiability: under Gaussian noise the
        two orientations of an edge are observationally indistinguishable, so a
        method that certifies directions must abstain there.
        """
        d = self.d
        X = np.zeros((n, d))
        eps = _draw_noise(rng, n, d, noise) * self.noise_sd
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


def _draw_noise(rng: np.random.Generator, n: int, d: int, kind: str) -> np.ndarray:
    """Unit-variance innovations from a named family."""
    if kind == "gaussian":
        return rng.normal(size=(n, d))
    if kind == "laplace":
        return rng.laplace(size=(n, d)) / np.sqrt(2.0)
    if kind == "uniform":
        return rng.uniform(-np.sqrt(3.0), np.sqrt(3.0), size=(n, d))
    if kind == "exponential":
        return rng.exponential(size=(n, d)) - 1.0
    if kind == "t5":
        return rng.standard_t(5, size=(n, d)) / np.sqrt(5.0 / 3.0)
    if kind == "mixture":                       # bimodal, outside the fitted family
        comp = rng.integers(0, 2, size=(n, d))
        x = rng.normal(size=(n, d)) * 0.5 + np.where(comp == 0, -1.0, 1.0)
        return x / np.sqrt(1.25)
    raise ValueError(f"unknown noise family: {kind}")


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


def population_correlation(sem: "LinearGaussianSEM", observed=None) -> np.ndarray:
    """Exact correlation matrix implied by the SEM, restricted to ``observed``."""
    d = sem.d
    A = np.linalg.inv(np.eye(d) - sem.B.T)
    Sigma = A @ np.diag(sem.noise_sd**2) @ A.T
    if observed is not None:
        idx = np.array(list(observed), dtype=int)
        Sigma = Sigma[np.ix_(idx, idx)]
    scale = np.diag(1.0 / np.sqrt(np.diag(Sigma)))
    return scale @ Sigma @ scale


def partial_correlation(R: np.ndarray, i: int, j: int, cond=()) -> float:
    """Population partial correlation ``rho_{ij.S}`` from a correlation matrix."""
    cond = list(cond)
    if not cond:
        return float(R[i, j])
    idx = np.array(cond, dtype=int)
    Rzz = R[np.ix_(idx, idx)]
    Rzx = R[np.ix_(idx, [i, j])]
    S = R[np.ix_([i, j], [i, j])] - Rzx.T @ np.linalg.solve(Rzz, Rzx)
    denom = np.sqrt(max(S[0, 0], 0.0) * max(S[1, 1], 0.0))
    return 0.0 if denom <= 0 else float(S[0, 1] / denom)


def strong_faithfulness_margin(
    sem: "LinearGaussianSEM", observed=None, max_order: int = 2
) -> float:
    """Smallest ``|rho_{ij.S}|`` over true adjacencies and tested conditioning sets.

    This is the empirical counterpart of the ``delta`` in ``delta``-strong
    faithfulness: the SPRINT-CD guarantee applies to an instance only when this
    margin is at least the equivalence half-width in force.  Random graphs
    violate the condition far more often than is generally acknowledged --
    especially once latent variables are marginalised out -- so experiments
    that score the guarantee need to measure it rather than assume it.

    Returns ``inf`` when the true skeleton over ``observed`` has no edges.
    """
    from .dsep import oracle_skeleton_and_sepsets

    obs = list(range(sem.d)) if observed is None else list(observed)
    R = population_correlation(sem, obs)
    skel, _ = oracle_skeleton_and_sepsets(sem.adjacency, obs)

    margin = np.inf
    n_obs = len(obs)
    for i, j in skel.edges():
        others = [v for v in range(n_obs) if v not in (i, j)]
        for order in range(min(max_order, len(others)) + 1):
            for S in itertools.combinations(others, order):
                margin = min(margin, abs(partial_correlation(R, i, j, S)))
    return float(margin)
