import numpy as np

from sprint_cd.npadj import NPAdjacency


def _run(X, k=1):
    d = X.shape[1]
    a = NPAdjacency(d, 0.05 / (d * (d - 1) / 2), k=k, warm=100)
    for s in range(0, len(X), 100):
        a.update(X[s:s + 100])
    return a.certified_edges()


def test_nonlinear_chain_pair_not_certified():
    rng = np.random.default_rng(0)
    n = 3000
    x = rng.normal(size=n)
    m = np.tanh(2 * x) + 0.3 * rng.laplace(size=n)
    y = np.tanh(2 * m) + 0.3 * rng.laplace(size=n)
    edges = _run(np.column_stack([x, m, y]))
    assert (0, 1) in edges and (1, 2) in edges
    assert (0, 2) not in edges


def test_symmetric_dependence_detected():
    rng = np.random.default_rng(1)
    n = 3000
    x = rng.normal(size=n)
    y = x ** 2 + 0.3 * rng.normal(size=n)
    assert (0, 1) in _run(np.column_stack([x, y]), k=0)


def test_independent_pair_not_certified():
    rng = np.random.default_rng(2)
    assert _run(rng.normal(size=(3000, 2)), k=0) == []
