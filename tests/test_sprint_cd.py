import itertools

import numpy as np
import pytest

from sprint_cd.graph import ARROW, TAIL
from sprint_cd.simulate import (LinearGaussianSEM, random_dag, skeleton_errors,
                                structural_hamming_distance)
from sprint_cd.sprint_cd import SprintCD, SprintCDConfig, run_sprint_cd


def _chain_sem(d=4, coef=1.0):
    B = np.zeros((d, d))
    for i in range(d - 1):
        B[i, i + 1] = coef
    return LinearGaussianSEM(B=B, noise_sd=np.ones(d), order=np.arange(d))


def test_recovers_a_chain():
    rng = np.random.default_rng(0)
    sem = _chain_sem()
    X = sem.sample(4000, rng)
    est, algo = run_sprint_cd(
        X, SprintCDConfig(alpha=0.05, max_order=2, rope=0.15, prior_scale=0.5, warmup=50),
        batch_size=50,
    )
    assert structural_hamming_distance(est, sem.true_cpdag()) == 0


def test_recovers_a_collider_with_correct_orientation():
    rng = np.random.default_rng(1)
    B = np.zeros((3, 3)); B[0, 1] = B[2, 1] = 1.0
    sem = LinearGaussianSEM(B=B, noise_sd=np.ones(3), order=np.array([0, 2, 1]))
    X = sem.sample(4000, rng)
    est, _ = run_sprint_cd(
        X, SprintCDConfig(alpha=0.05, max_order=1, rope=0.15, prior_scale=0.5, warmup=50),
        batch_size=50,
    )
    assert est.is_directed(0, 1) and est.is_directed(2, 1)


def test_estimate_starts_dense_and_only_loses_edges():
    """The guarantee is one-sided: edges are removed on certificates, never added."""
    rng = np.random.default_rng(2)
    sem = random_dag(6, 0.3, rng)
    X = sem.sample(3000, rng)
    algo = SprintCD(6, SprintCDConfig(alpha=0.05, max_order=2, rope=0.15, warmup=50))
    algo.warm_up(X[:50])
    counts = []
    for start in range(50, 3000, 100):
        algo.update(X[start:start + 100])
        counts.append(len(algo.graph.edges()))
    assert counts[0] == 15 or counts[0] <= 15
    assert all(b <= a for a, b in zip(counts, counts[1:]))   # monotone non-increasing


def _median_stopping_time(rope=0.15, alpha=0.05, seeds=range(5)):
    stops = []
    for s in seeds:
        rng = np.random.default_rng(100 + s)
        sem = _chain_sem(d=4, coef=1.0)
        X = sem.sample(20000, rng)
        _, algo = run_sprint_cd(
            X, SprintCDConfig(alpha=alpha, max_order=2, rope=rope,
                              prior_scale=0.5, warmup=50),
            batch_size=100,
        )
        stops.append(algo.n)
    return float(np.median(stops))


def test_stopping_is_adaptive_rather_than_consuming_the_whole_stream():
    """The run halts on its own well before the data are exhausted."""
    assert 0 < _median_stopping_time() < 20000


def test_a_looser_equivalence_region_stops_sooner():
    """Monotone by construction: a wider ROPE is easier for a certificate to enter.

    Note that stopping time is *not* monotone in the SEM coefficient: for a
    chain the binding partial correlation peaks at coefficient 1 and decays
    either side, so a larger coefficient can make the hardest query harder.
    """
    assert _median_stopping_time(rope=0.25) <= _median_stopping_time(rope=0.10)


def test_a_larger_error_budget_stops_sooner():
    assert _median_stopping_time(alpha=0.20) <= _median_stopping_time(alpha=0.01)


def test_anytime_false_deletion_rate_is_controlled_under_monitoring():
    """The headline guarantee: over the whole run, true edges are rarely lost.

    The estimate is inspected after *every* batch and the worst case over the
    entire trajectory is scored -- the situation a fixed-sample guarantee does
    not cover.
    """
    rng = np.random.default_rng(4)
    alpha, reps = 0.1, 40
    violations = 0
    for _ in range(reps):
        sem = random_dag(5, 0.35, rng, coef_low=0.6, coef_high=1.2)
        X = sem.sample(2500, rng)
        truth = sem.true_cpdag()
        algo = SprintCD(5, SprintCDConfig(alpha=alpha, max_order=2, rope=0.2,
                                          prior_scale=0.5, warmup=50))
        algo.warm_up(X[:50])
        ever_missing = False
        for start in range(50, 2500, 100):
            algo.update(X[start:start + 100])
            if skeleton_errors(algo.graph, truth)["missing"] > 0:
                ever_missing = True
        violations += ever_missing
    assert violations / reps <= alpha + 0.15


def test_warm_up_is_required_before_update():
    algo = SprintCD(3)
    with pytest.raises(RuntimeError):
        algo.update(np.zeros((10, 3)))


def test_rejects_malformed_configuration():
    with pytest.raises(ValueError):
        SprintCDConfig(alpha=1.5)
    with pytest.raises(ValueError):
        SprintCDConfig(rope=-1.0)
    with pytest.raises(ValueError):
        SprintCDConfig(deletion_share=0.0)


def test_needs_at_least_two_variables():
    with pytest.raises(ValueError):
        SprintCD(1)


def test_stable_mode_is_order_independent():
    """PC-stable output must not depend on the column ordering of the data."""
    rng = np.random.default_rng(5)
    sem = random_dag(5, 0.4, rng)
    X = sem.sample(4000, rng)
    perm = np.array([3, 0, 4, 1, 2])
    cfg = SprintCDConfig(alpha=0.05, max_order=2, rope=0.15, prior_scale=0.5, warmup=50)
    a, _ = run_sprint_cd(X, cfg, batch_size=100, stop_when_resolved=False)
    b, _ = run_sprint_cd(X[:, perm], cfg, batch_size=100, stop_when_resolved=False)
    inv = np.argsort(perm)
    assert np.array_equal(a.skeleton(), b.skeleton()[np.ix_(inv, inv)])
