import numpy as np
import pytest

from sprint_cd.dsep import oracle_skeleton_and_sepsets
from sprint_cd.graph import ARROW, CIRCLE
from sprint_cd.simulate import LinearGaussianSEM, structural_hamming_distance
from sprint_cd.sprint_cd import SprintCDConfig
from sprint_cd.sprint_fci import SprintFCI, pag_from_skeleton, run_sprint_fci


def _latent_confounder_sem():
    """0 -> 1 <- L -> 2 <- 3 with L = variable 4 held out."""
    d = 5
    B = np.zeros((d, d))
    B[0, 1] = 0.9
    B[4, 1] = 1.0
    B[4, 2] = 1.0
    B[3, 2] = 0.9
    sem = LinearGaussianSEM(B=B, noise_sd=np.ones(d), order=np.array([0, 3, 4, 1, 2]))
    return sem, [0, 1, 2, 3]


def test_recovers_the_oracle_pag_with_a_latent_confounder():
    sem, observed = _latent_confounder_sem()
    skel, seps = oracle_skeleton_and_sepsets(sem.adjacency, observed, max_order=3)
    truth = pag_from_skeleton(skel, seps)

    rng = np.random.default_rng(0)
    X = sem.sample(8000, rng)[:, observed]
    est, _ = run_sprint_fci(
        X, SprintCDConfig(alpha=0.05, max_order=3, rope=0.15, prior_scale=0.5, warmup=50),
        batch_size=100,
    )
    assert structural_hamming_distance(est, truth) == 0


def test_the_bidirected_edge_marks_the_latent_confounder():
    sem, observed = _latent_confounder_sem()
    rng = np.random.default_rng(1)
    X = sem.sample(8000, rng)[:, observed]
    est, _ = run_sprint_fci(
        X, SprintCDConfig(alpha=0.05, max_order=3, rope=0.15, prior_scale=0.5, warmup=50),
        batch_size=100,
    )
    # Variables 1 and 2 share the latent parent: arrowheads at both ends.
    assert est.mark_at(1, 2) == ARROW and est.mark_at(2, 1) == ARROW


def test_calling_cpdag_on_an_fci_run_is_refused():
    """A CPDAG presupposes causal sufficiency, which SprintFCI does not assume."""
    algo = SprintFCI(3)
    with pytest.raises(NotImplementedError):
        algo.cpdag()


def test_possible_d_sep_excludes_the_pair_itself():
    sem, observed = _latent_confounder_sem()
    rng = np.random.default_rng(2)
    X = sem.sample(2000, rng)[:, observed]
    algo = SprintFCI(4, SprintCDConfig(alpha=0.05, max_order=2, rope=0.15, warmup=50))
    algo.warm_up(X[:50])
    algo.update(X[50:])
    for i, j in algo.graph.edges():
        pds = algo.possible_d_sep(i, j)
        assert i not in pds and j not in pds


def test_no_true_adjacency_is_lost_under_monitoring():
    sem, observed = _latent_confounder_sem()
    skel, seps = oracle_skeleton_and_sepsets(sem.adjacency, observed, max_order=3)
    rng = np.random.default_rng(3)
    X = sem.sample(4000, rng)[:, observed]
    algo = SprintFCI(4, SprintCDConfig(alpha=0.05, max_order=3, rope=0.15,
                                       prior_scale=0.5, warmup=50))
    algo.warm_up(X[:50])
    for s in range(50, 4000, 100):
        algo.update(X[s:s + 100])
        for i, j in skel.edges():
            assert algo.graph.adjacent(i, j)
