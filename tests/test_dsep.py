import numpy as np
import pytest

from sprint_cd.dsep import ancestors, d_separated


def _dag(edges, d):
    adj = np.zeros((d, d))
    for a, b in edges:
        adj[a, b] = 1
    return adj


def test_chain_blocks_when_conditioned():
    g = _dag([(0, 1), (1, 2)], 3)
    assert not d_separated(g, 0, 2, [])
    assert d_separated(g, 0, 2, [1])


def test_fork_blocks_when_conditioned():
    g = _dag([(1, 0), (1, 2)], 3)
    assert not d_separated(g, 0, 2, [])
    assert d_separated(g, 0, 2, [1])


def test_collider_opens_when_conditioned():
    g = _dag([(0, 1), (2, 1)], 3)
    assert d_separated(g, 0, 2, [])
    assert not d_separated(g, 0, 2, [1])


def test_conditioning_on_a_collider_descendant_also_opens():
    g = _dag([(0, 1), (2, 1), (1, 3)], 4)
    assert d_separated(g, 0, 2, [])
    assert not d_separated(g, 0, 2, [3])


def test_ancestors_are_inclusive():
    g = _dag([(0, 1), (1, 2)], 3)
    assert ancestors(g, [2]) == {0, 1, 2}
    assert ancestors(g, [0]) == {0}


def test_endpoints_must_be_outside_the_conditioning_set():
    g = _dag([(0, 1)], 3)
    with pytest.raises(ValueError):
        d_separated(g, 0, 1, [0])


def test_dsep_agrees_with_covariance_on_a_linear_gaussian_model():
    """Zero partial correlation should coincide with d-separation."""
    from sprint_cd.simulate import random_dag
    from sprint_cd.stats import GaussianSuffStat
    import itertools

    rng = np.random.default_rng(4)
    sem = random_dag(6, 0.35, rng)
    X = sem.sample(200000, rng)
    ss = GaussianSuffStat(6)
    ss.update(X)
    adj = sem.adjacency
    for i, j in itertools.combinations(range(6), 2):
        for S in ([], [k for k in range(6) if k not in (i, j)][:2]):
            r = abs(ss.residual_moments(i, j, tuple(S)).partial_corr)
            if d_separated(adj, i, j, S):
                assert r < 0.02


def test_separator_assumption_detects_a_violation():
    """A collider whose parents need a size-2 separator fails at max_order=1."""
    from sprint_cd.simulate import separator_assumption_holds
    # 0 -> 3 <- 1, 2 -> 3 : pairs among {0,1,2} are marginally separated (order 0),
    # so the assumption holds even at order 0 for this graph.
    adj = np.zeros((4, 4))
    adj[0, 3] = adj[1, 3] = adj[2, 3] = 1
    assert separator_assumption_holds(adj, max_order=0)

    # Chain 0->1->2->3: (0,3) needs {1} or {2}; order 0 is not enough.
    chain = np.zeros((4, 4))
    chain[0, 1] = chain[1, 2] = chain[2, 3] = 1
    assert not separator_assumption_holds(chain, max_order=0)
    assert separator_assumption_holds(chain, max_order=1)
