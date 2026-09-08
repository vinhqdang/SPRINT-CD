"""Tests for the two certificate primitives.

The load-bearing tests are calibration ones. A certificate that fires when it
should not silently invalidates every claim CERT-CD makes, so both primitives
are checked against Ville's inequality by simulation, including in settings
that violate faithfulness outright.
"""

import itertools

import numpy as np
import pytest

from sprint_cd.certificates import (DirectionEProcess, admissible_sets,
                                    adjacency_log_e, fit_gg_regression, gg_logpdf)
from sprint_cd.eprocess.safe_linear import SafeLinearCI
from sprint_cd.stats import GaussianSuffStat


# ----------------------------------------------------------------------
# adjacency certificate
# ----------------------------------------------------------------------
def test_admissible_sets_enumerate_the_fixed_family():
    sets = list(admissible_sets(5, 0, 1, 2))
    assert () in sets
    assert len(sets) == 1 + 3 + 3          # C(3,0) + C(3,1) + C(3,2)
    assert all(0 not in S and 1 not in S for S in sets)


def _adj_log_e(X, i, j, k, prior_scale=0.5):
    ss = GaussianSuffStat(X.shape[1])
    ss.update(X)
    ci = SafeLinearCI(ss, prior_scale=prior_scale, warmup_var=np.ones(X.shape[1]))
    return adjacency_log_e(ci, i, j, k)


def test_adjacency_e_value_is_the_minimum_over_conditioning_sets():
    """A single separating set must hold the whole edge statistic down."""
    rng = np.random.default_rng(0)
    n = 3000
    z = rng.normal(size=n)
    x = 0.9 * z + rng.normal(size=n)
    y = 0.9 * z + rng.normal(size=n)          # x indep y | z, but dependent marginally
    X = np.column_stack([x, y, z, rng.normal(size=n)])
    ss = GaussianSuffStat(4)
    ss.update(X)
    ci = SafeLinearCI(ss, prior_scale=0.5, warmup_var=np.ones(4))
    marginal = ci.symmetric_log_e(0, 1, ()).log_e
    combined = adjacency_log_e(ci, 0, 1, 2)
    assert marginal > 5.0                      # strong marginal dependence ...
    assert combined < 0.0                      # ... yet the edge is not certified


def test_adjacency_certificate_fires_for_a_genuine_edge():
    rng = np.random.default_rng(1)
    n = 3000
    z = rng.normal(size=n)
    x = 0.9 * z + rng.normal(size=n)
    y = 0.9 * z + 0.9 * x + rng.normal(size=n)   # x -> y, so no set separates them
    X = np.column_stack([x, y, z, rng.normal(size=n)])
    assert _adj_log_e(X, 0, 1, 2) > np.log(20)


def test_adjacency_certificate_respects_ville_for_a_separable_pair():
    """P(sup_t A_t >= 1/alpha) <= alpha when the pair is separable."""
    rng = np.random.default_rng(2)
    reps, nmax, alpha = 200, 500, 0.1
    crossings = 0
    for _ in range(reps):
        z = rng.normal(size=nmax)
        x = 0.9 * z + rng.normal(size=nmax)
        y = 0.8 * z + rng.normal(size=nmax)      # separated by {z}
        D = np.column_stack([x, y, z, rng.normal(size=nmax)])
        ss = GaussianSuffStat(4)
        ci = SafeLinearCI(ss, prior_scale=0.5, warmup_var=np.ones(4))
        crossed = False
        for t in range(0, nmax, 25):
            ss.update(D[t:t + 25])
            if ss.n >= 10 and adjacency_log_e(ci, 0, 1, 2) >= -np.log(alpha):
                crossed = True
                break
        crossings += crossed
    assert crossings / reps <= alpha + 0.05


def test_adjacency_certificate_is_valid_without_faithfulness():
    """Validity uses only the Markov condition.

    Here two directed paths from ``x`` to ``y`` cancel exactly, so the
    distribution is unfaithful to the graph.  ``x`` and ``y`` are still
    separable (by the two mediators), so the certificate must not fire.
    """
    rng = np.random.default_rng(3)
    reps, nmax, alpha = 150, 500, 0.1
    crossings = 0
    for _ in range(reps):
        x = rng.normal(size=nmax)
        m1 = 1.0 * x + rng.normal(size=nmax)
        m2 = 1.0 * x + rng.normal(size=nmax)
        y = 1.0 * m1 - 1.0 * m2 + rng.normal(size=nmax)   # total effect cancels
        D = np.column_stack([x, y, m1, m2])
        ss = GaussianSuffStat(4)
        ci = SafeLinearCI(ss, prior_scale=0.5, warmup_var=np.ones(4))
        crossed = False
        for t in range(0, nmax, 25):
            ss.update(D[t:t + 25])
            if ss.n >= 10 and adjacency_log_e(ci, 0, 1, 2) >= -np.log(alpha):
                crossed = True
                break
        crossings += crossed
    assert crossings / reps <= alpha + 0.05


def test_adjacency_early_exit_agrees_on_the_decision():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(600, 4))
    ss = GaussianSuffStat(4)
    ss.update(X)
    ci = SafeLinearCI(ss, prior_scale=0.5, warmup_var=np.ones(4))
    thr = np.log(20)
    exact = adjacency_log_e(ci, 0, 1, 2)
    early = adjacency_log_e(ci, 0, 1, 2, stop_below=thr)
    assert (exact >= thr) == (early >= thr)


# ----------------------------------------------------------------------
# generalised Gaussian machinery
# ----------------------------------------------------------------------
def test_gg_logpdf_integrates_to_one():
    for kappa in (0.8, 1.0, 2.0, 4.0):
        grid = np.linspace(-25, 25, 400001)
        mass = np.trapezoid(np.exp(gg_logpdf(grid, 1.3, kappa)), grid)
        assert mass == pytest.approx(1.0, abs=1e-3)


def test_gg_regression_recovers_the_shape_parameter():
    rng = np.random.default_rng(5)
    n = 4000
    X = np.column_stack([np.ones(n), rng.normal(size=n)])
    y = X @ np.array([0.5, 1.5]) + rng.laplace(size=n)     # Laplace is kappa = 1
    ll, coef, sigma, kappa = fit_gg_regression(y, X)
    assert coef[1] == pytest.approx(1.5, abs=0.1)
    assert kappa <= 1.5                                    # clearly sub-Gaussian


# ----------------------------------------------------------------------
# direction certificate
# ----------------------------------------------------------------------
def _direction_run(rng, noise, n=1200, batch=200, alpha=0.05):
    """Truth is j -> i.  Returns (certified correct, certified wrong)."""
    gen = {"laplace": lambda: rng.laplace(size=n),
           "uniform": lambda: rng.uniform(-1.7, 1.7, size=n),
           "gaussian": lambda: rng.normal(size=n),
           "t3": lambda: rng.standard_t(3, size=n) / 1.7}[noise]
    xj, xi = gen(), None
    xi = 0.9 * xj + gen()
    Z = np.ones((n, 1))
    correct = DirectionEProcess(n_cond=1)
    wrong = DirectionEProcess(n_cond=1)
    for s in range(0, n, batch):
        correct.update(xj[s:s + batch], xi[s:s + batch], Z[s:s + batch])  # j -> i
        wrong.update(xi[s:s + batch], xj[s:s + batch], Z[s:s + batch])    # i -> j
    return correct.certifies(alpha), wrong.certifies(alpha)


@pytest.mark.parametrize("noise", ["laplace", "uniform", "t3"])
def test_direction_certificate_identifies_non_gaussian_direction(noise):
    rng = np.random.default_rng(6)
    hits = wrongs = 0
    for _ in range(8):
        c, w = _direction_run(rng, noise)
        hits += c
        wrongs += w
    assert hits >= 7
    assert wrongs == 0


def test_direction_certificate_abstains_under_gaussian_noise():
    """Under Gaussianity neither orientation is identified, so neither may be certified."""
    rng = np.random.default_rng(7)
    for _ in range(8):
        c, w = _direction_run(rng, "gaussian")
        assert not c and not w


def test_direction_certificate_respects_ville():
    rng = np.random.default_rng(8)
    reps, alpha = 60, 0.1
    wrongs = 0
    for _ in range(reps):
        _, w = _direction_run(rng, "laplace", n=800, alpha=alpha)
        wrongs += w
    assert wrongs / reps <= alpha + 0.05


def test_direction_numerator_and_denominator_share_an_index_set():
    """Regression test.

    If the denominator's maximised likelihood ranges over observations the
    numerator never scored, ``log E`` picks up a constant of order (extra
    observations) x (log-density) and the process certifies everything --
    including both directions at once, and directions under Gaussian noise.
    """
    rng = np.random.default_rng(9)
    n = 800
    xj = rng.normal(size=n)
    xi = 0.9 * xj + rng.normal(size=n)      # Gaussian: nothing is identifiable
    Z = np.ones((n, 1))
    a = DirectionEProcess(n_cond=1)
    b = DirectionEProcess(n_cond=1)
    for s in range(0, n, 200):
        a.update(xi[s:s + 200], xj[s:s + 200], Z[s:s + 200])
        b.update(xj[s:s + 200], xi[s:s + 200], Z[s:s + 200])
    assert not (a.certifies(0.05) and b.certifies(0.05))
    assert max(a.log_max, b.log_max) < np.log(20)


def test_direction_process_counts_observations_not_batches():
    """Regression test: the numerator must switch on once enough *rows* are seen."""
    rng = np.random.default_rng(10)
    proc = DirectionEProcess(n_cond=1, min_fit=60)
    n, batch = 600, 100
    xj = rng.laplace(size=n)
    xi = 0.9 * xj + rng.laplace(size=n)
    Z = np.ones((n, 1))
    for s in range(0, n, batch):
        proc.update(xj[s:s + batch], xi[s:s + batch], Z[s:s + batch])
    assert proc._scored > 0
    assert np.isfinite(proc.log_max)
