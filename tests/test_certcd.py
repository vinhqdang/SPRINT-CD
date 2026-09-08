import itertools

import numpy as np
import pytest

from sprint_cd.certcd import CertCD, CertCDConfig, run_cert_cd
from sprint_cd.graph import ARROW, CIRCLE, TAIL
from sprint_cd.simulate import LinearGaussianSEM, random_dag, strong_faithfulness_margin


def _chain(d=3, coef=0.9):
    B = np.zeros((d, d))
    for i in range(d - 1):
        B[i, i + 1] = coef
    return LinearGaussianSEM(B=B, noise_sd=np.ones(d), order=np.arange(d))


def test_recovers_a_full_dag_that_a_cpdag_cannot_orient():
    """A chain has no v-structure, so PC can only ever return 0 - 1 - 2.

    With non-Gaussian noise the direction certificates orient both edges, so
    CERT-CD returns strictly more than the Markov equivalence class allows.
    """
    rng = np.random.default_rng(0)
    X = _chain().sample(4000, rng, noise="laplace")
    g, algo = run_cert_cd(X, CertCDConfig(alpha=0.05, max_order=1, warmup=50),
                          batch_size=200)
    assert g.is_directed(0, 1) and g.is_directed(1, 2)
    assert not g.adjacent(0, 2)


def test_abstains_on_orientation_under_gaussian_noise():
    """Direction is not identified under Gaussianity, so no arrowhead may appear."""
    rng = np.random.default_rng(1)
    X = _chain().sample(4000, rng, noise="gaussian")
    g, algo = run_cert_cd(X, CertCDConfig(alpha=0.05, max_order=1, warmup=50),
                          batch_size=200)
    assert algo.certified_arrows() == []
    for i, j in g.edges():
        assert g.mark_at(i, j) == CIRCLE and g.mark_at(j, i) == CIRCLE


def test_the_graph_only_ever_grows():
    """Assertions are certificates, and a certificate is never revoked."""
    rng = np.random.default_rng(2)
    sem = random_dag(5, 0.35, rng, coef_low=0.6, coef_high=1.2)
    X = sem.sample(3000, rng, noise="laplace")
    algo = CertCD(5, CertCDConfig(alpha=0.05, max_order=2, warmup=50))
    algo.warm_up(X[:50])
    seen_edges, seen_arrows = set(), set()
    for s in range(50, 3000, 250):
        algo.update(X[s:s + 250])
        e, a = set(algo.certified_edges()), set(algo.certified_arrows())
        assert seen_edges <= e          # monotone
        assert seen_arrows <= a
        seen_edges, seen_arrows = e, a


def test_starts_empty_rather_than_complete():
    rng = np.random.default_rng(3)
    X = random_dag(5, 0.35, rng).sample(500, rng, noise="laplace")
    algo = CertCD(5, CertCDConfig(alpha=0.05, max_order=2, warmup=50))
    algo.warm_up(X[:50])
    assert algo.certified_edges() == []
    assert len(algo.undecided_pairs()) == 10


def test_undecided_is_not_the_same_as_certified_absent():
    """Every pair is either certified adjacent or explicitly undecided."""
    rng = np.random.default_rng(4)
    X = random_dag(5, 0.3, rng).sample(1500, rng, noise="laplace")
    _, algo = run_cert_cd(X, CertCDConfig(alpha=0.05, max_order=2, warmup=50),
                          batch_size=250)
    both = set(algo.certified_edges()) | set(algo.undecided_pairs())
    assert both == set(itertools.combinations(range(5), 2))
    assert not (set(algo.certified_edges()) & set(algo.undecided_pairs()))


def test_no_false_edge_is_certified_under_monitoring():
    """The headline guarantee, scored over a whole monitored run."""
    rng = np.random.default_rng(5)
    alpha, reps = 0.1, 12
    violations = 0
    for _ in range(reps):
        sem = random_dag(5, 0.3, rng, coef_low=0.5, coef_high=1.2)
        adj = sem.adjacency
        X = sem.sample(2500, rng, noise="laplace")
        algo = CertCD(5, CertCDConfig(alpha=alpha, max_order=2, warmup=50,
                                      orient=False))
        algo.warm_up(X[:50])
        bad = False
        for s in range(50, 2500, 250):
            algo.update(X[s:s + 250])
            for i, j in algo.certified_edges():
                if not (adj[i, j] or adj[j, i]):
                    bad = True
        violations += bad
    assert violations / reps <= alpha + 0.15


def test_remains_valid_where_strong_faithfulness_fails():
    """The regime that breaks the delete-on-non-rejection approach.

    CERT-CD never accepts a null, so instances with tiny partial correlations
    cost it power (pairs stay undecided) rather than validity.
    """
    rng = np.random.default_rng(6)
    checked = 0
    violations = 0
    tries = 0
    while checked < 6 and tries < 120:
        tries += 1
        sem = random_dag(5, 0.35, rng, coef_low=0.5, coef_high=1.2)
        if strong_faithfulness_margin(sem, max_order=2) >= 0.15:
            continue                     # want the *violating* stratum
        checked += 1
        adj = sem.adjacency
        X = sem.sample(2000, rng, noise="laplace")
        algo = CertCD(5, CertCDConfig(alpha=0.1, max_order=2, warmup=50,
                                      orient=False))
        algo.warm_up(X[:50])
        for s in range(50, 2000, 250):
            algo.update(X[s:s + 250])
            for i, j in algo.certified_edges():
                if not (adj[i, j] or adj[j, i]):
                    violations += 1
    assert checked > 0
    assert violations == 0


def test_certified_arrows_point_the_right_way():
    rng = np.random.default_rng(7)
    sem = random_dag(4, 0.5, rng, coef_low=0.8, coef_high=1.3)
    X = sem.sample(4000, rng, noise="laplace")
    _, algo = run_cert_cd(X, CertCDConfig(alpha=0.05, max_order=2, warmup=50),
                          batch_size=250)
    for i, j in algo.certified_arrows():
        assert sem.adjacency[i, j] == 1, f"certified {i}->{j} which is not a true edge"


def test_rejects_malformed_configuration():
    with pytest.raises(ValueError):
        CertCDConfig(alpha=1.5)
    with pytest.raises(ValueError):
        CertCDConfig(adjacency_share=0.0)
    with pytest.raises(ValueError):
        CertCDConfig(direction_conditioning="sometimes")
    with pytest.raises(ValueError):
        CertCD(1)


def test_warm_up_is_required():
    algo = CertCD(3)
    with pytest.raises(RuntimeError):
        algo.update(np.zeros((10, 3)))
