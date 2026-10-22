import numpy as np

from sprint_cd.relative_fit import certify_direction


def test_nonlinear_pair_certified_correctly_and_gaussian_abstains():
    rng = np.random.default_rng(0)
    x = rng.normal(size=600)
    y = np.tanh(2 * x) + 0.3 * rng.laplace(size=600)
    r = certify_direction(x, y, seed=1)
    assert r["agree"] in (None, "x->y")
    assert r["gg"] == "x->y"
    xg = rng.normal(size=600)
    yg = 0.8 * xg + rng.normal(size=600)
    rg = certify_direction(xg, yg, seed=1)
    assert rg["agree"] is None
