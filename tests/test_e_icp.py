import numpy as np
import pytest

from sprint_cd.e_icp import EICP, EICPConfig, icp_fixed_sample


def _shift_sem(n, rng, n_env=3):
    """Y <- X0, X1 ; X2 is a child of Y, so only {X0, X1} is invariant."""
    env = rng.integers(0, n_env, size=n)
    shift = np.array([0.0, 2.0, -2.0])[env]
    x0 = rng.normal(size=n) + shift
    x1 = rng.normal(size=n) + 0.5 * shift
    y = 1.2 * x0 - 0.8 * x1 + rng.normal(size=n)
    x2 = 0.9 * y + rng.normal(size=n) + shift
    return np.column_stack([x0, x1, x2]), y, env


def _run(X, y, env, cfg, batch=100):
    m = EICP(d=X.shape[1], n_env=int(env.max()) + 1, config=cfg)
    m.warm_up(X[: cfg.warmup], y[: cfg.warmup], env[: cfg.warmup])
    for s in range(cfg.warmup, X.shape[0], batch):
        m.update(X[s:s + batch], y[s:s + batch], env[s:s + batch])
    return m


def test_recovers_the_causal_parents():
    rng = np.random.default_rng(0)
    X, y, env = _shift_sem(4000, rng)
    m = _run(X, y, env, EICPConfig(alpha=0.05, max_size=3))
    assert m.estimate() == {0, 1}


def test_agrees_with_fixed_sample_icp_on_an_easy_problem():
    rng = np.random.default_rng(1)
    X, y, env = _shift_sem(4000, rng)
    m = _run(X, y, env, EICPConfig(alpha=0.05, max_size=3))
    assert m.estimate() == icp_fixed_sample(X, y, env, 0.05, 3)


def test_the_true_parent_set_is_never_rejected_under_monitoring():
    """The anytime guarantee: S* survives inspection after every batch."""
    rng = np.random.default_rng(2)
    alpha, reps = 0.1, 30
    violations = 0
    for _ in range(reps):
        X, y, env = _shift_sem(1500, rng)
        cfg = EICPConfig(alpha=alpha, max_size=3, warmup=40)
        m = EICP(d=3, n_env=3, config=cfg)
        m.warm_up(X[:40], y[:40], env[:40])
        ever = False
        for s in range(40, 1500, 100):
            m.update(X[s:s + 100], y[s:s + 100], env[s:s + 100])
            if (0, 1) in m._rejected:          # S* = {0, 1}
                ever = True
        violations += ever
    assert violations / reps <= alpha + 0.1


def test_output_is_a_subset_of_the_true_parents_throughout():
    rng = np.random.default_rng(3)
    X, y, env = _shift_sem(2000, rng)
    cfg = EICPConfig(alpha=0.05, max_size=3, warmup=40)
    m = EICP(d=3, n_env=3, config=cfg)
    m.warm_up(X[:40], y[:40], env[:40])
    for s in range(40, 2000, 100):
        m.update(X[s:s + 100], y[s:s + 100], env[s:s + 100])
        assert m.estimate() <= {0, 1}


def test_a_fully_invariant_model_yields_the_empty_set():
    """With no environment effect nothing is rejected, so the intersection is empty."""
    rng = np.random.default_rng(4)
    n = 2000
    env = rng.integers(0, 2, size=n)
    X = rng.normal(size=(n, 3))
    y = 1.0 * X[:, 0] + rng.normal(size=n)     # identical in every environment
    m = _run(X, y, env, EICPConfig(alpha=0.05, max_size=2, warmup=40))
    assert m.estimate() == set()


def test_needs_at_least_two_environments():
    with pytest.raises(ValueError):
        EICP(d=2, n_env=1)


def test_rejects_out_of_range_environment_labels():
    m = EICP(d=2, n_env=2, config=EICPConfig(warmup=10))
    with pytest.raises(ValueError):
        m.warm_up(np.zeros((10, 2)), np.zeros(10), np.full(10, 5))


def test_candidate_weights_sum_to_one():
    m = EICP(d=4, n_env=2, config=EICPConfig(max_size=2))
    assert sum(m.weight(S) for S in m._candidates) == pytest.approx(1.0)
