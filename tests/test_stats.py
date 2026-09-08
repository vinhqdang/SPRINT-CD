import numpy as np
import pytest

from sprint_cd.stats import GaussianSuffStat


def _explicit_residual_moments(X, i, j, cond):
    n = X.shape[0]
    Z = np.column_stack([np.ones(n)] + [X[:, c] for c in cond])
    P = Z @ np.linalg.pinv(Z)
    rx = X[:, i] - P @ X[:, i]
    ry = X[:, j] - P @ X[:, j]
    return rx @ rx, rx @ ry, ry @ ry


@pytest.mark.parametrize("cond", [(), (2,), (2, 3), (2, 3, 4)])
def test_schur_complement_matches_explicit_regression(cond):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 6))
    X[:, 0] += 0.7 * X[:, 2] - 0.3 * X[:, 3]
    X[:, 1] += 0.5 * X[:, 2] + 0.4 * X[:, 4]
    ss = GaussianSuffStat(6)
    ss.update(X)
    m = ss.residual_moments(0, 1, cond)
    sxx, sxy, syy = _explicit_residual_moments(X, 0, 1, cond)
    assert m.sxx == pytest.approx(sxx, rel=1e-9)
    assert m.sxy == pytest.approx(sxy, rel=1e-9)
    assert m.syy == pytest.approx(syy, rel=1e-9)
    assert m.dof == 400 - len(cond) - 1


def test_batched_updates_match_single_pass():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 4))
    a = GaussianSuffStat(4)
    a.update(X)
    b = GaussianSuffStat(4)
    for start in range(0, 300, 37):
        b.update(X[start:start + 37])
    assert b.n == a.n
    np.testing.assert_allclose(b.centred_cross(), a.centred_cross(), rtol=1e-10)


def test_cache_invalidated_on_update():
    rng = np.random.default_rng(2)
    ss = GaussianSuffStat(3)
    ss.update(rng.normal(size=(50, 3)))
    first = ss.centred_cross().copy()
    ss.update(rng.normal(size=(50, 3)))
    assert not np.allclose(first, ss.centred_cross())


def test_conditioning_set_must_exclude_endpoints():
    ss = GaussianSuffStat(4)
    ss.update(np.random.default_rng(3).normal(size=(50, 4)))
    with pytest.raises(ValueError):
        ss.residual_moments(0, 1, (0,))
    with pytest.raises(ValueError):
        ss.residual_moments(0, 0, ())


def test_insufficient_data_returns_zero_dof():
    ss = GaussianSuffStat(4)
    ss.update(np.random.default_rng(4).normal(size=(2, 4)))
    m = ss.residual_moments(0, 1, (2, 3))
    assert m.dof == 0


def test_block_moments_agree_with_scalar_case():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(300, 5))
    ss = GaussianSuffStat(5)
    ss.update(X)
    scalar = ss.residual_moments(0, 1, (2, 3))
    s_tt, s_wt, S_ww, dof = ss.residual_block_moments(0, (2, 3), (1,))
    assert s_tt == pytest.approx(scalar.sxx, rel=1e-9)
    assert s_wt[0] == pytest.approx(scalar.sxy, rel=1e-9)
    assert S_ww[0, 0] == pytest.approx(scalar.syy, rel=1e-9)
    assert dof == scalar.dof
