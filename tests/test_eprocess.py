"""Tests for the e-process constructions.

The substantive tests here are the *calibration* ones: an e-process that is
not anytime-valid invalidates every guarantee the package claims, so the
Ville bound is checked by simulation rather than assumed.
"""

import numpy as np
import pytest

from sprint_cd.eprocess import (
    DiscreteUniversalCI,
    GaussianUniversalCI,
    coefficient_confidence_sequence,
    safe_linear_block_log_e,
    safe_linear_log_e,
)
from sprint_cd.stats import GaussianSuffStat, ResidualMoments


# ----------------------------------------------------------------------
# closed forms
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "sxx,sxy,syy,dof,n",
    [(520.0, -18.7, 439.0, 498, 500), (1000.0, 300.0, 900.0, 95, 100),
     (50.0, 2.0, 60.0, 20, 25)],
)
def test_confidence_sequence_inverts_the_e_value(sxx, sxy, syy, dof, n):
    """The interval must be exactly ``{b : E_n(b) < 1/alpha}``."""
    m = ResidualMoments(sxx, sxy, syy, dof, n)
    g, alpha = 0.4**2, 0.05
    lo, hi = coefficient_confidence_sequence(m, g, alpha)
    grid = np.linspace(lo - 0.5, hi + 0.5, 40001)
    inside = grid[[safe_linear_log_e(m, g, beta_null=b) < -np.log(alpha) for b in grid]]
    assert lo == pytest.approx(inside.min(), abs=2e-4)
    assert hi == pytest.approx(inside.max(), abs=2e-4)


def test_e_value_is_below_one_under_no_signal():
    m = ResidualMoments(sxx=500.0, sxy=0.0, syy=500.0, dof=498, n=500)
    assert safe_linear_log_e(m, 0.4**2) < 0.0


def test_e_value_increases_with_evidence():
    vals = [safe_linear_log_e(ResidualMoments(500.0, s, 500.0, 498, 500), 0.4**2)
            for s in (0.0, 50.0, 150.0, 300.0)]
    assert all(b > a for a, b in zip(vals, vals[1:]))


def test_block_e_value_matches_scalar_for_one_column():
    m = ResidualMoments(sxx=500.0, sxy=120.0, syy=480.0, dof=498, n=500)
    g = 0.4**2
    scalar = safe_linear_log_e(m, g)
    block = safe_linear_block_log_e(m.sxx, np.array([m.sxy]),
                                    np.array([[m.syy]]), m.dof, g)
    assert block == pytest.approx(scalar, rel=1e-10)


def test_confidence_sequence_is_uninformative_with_little_data():
    m = ResidualMoments(sxx=3.0, sxy=0.1, syy=3.0, dof=2, n=5)
    lo, hi = coefficient_confidence_sequence(m, 0.4**2, 0.05)
    assert not np.isfinite(lo) and not np.isfinite(hi)


def test_confidence_sequence_shrinks_with_sample_size():
    widths = []
    for n in (100, 400, 1600, 6400):
        m = ResidualMoments(sxx=float(n), sxy=0.0, syy=float(n), dof=n - 1, n=n)
        lo, hi = coefficient_confidence_sequence(m, 0.4**2, 0.05)
        widths.append(hi - lo)
    assert all(b < a for a, b in zip(widths, widths[1:]))


def test_prior_scale_must_be_positive():
    m = ResidualMoments(500.0, 10.0, 500.0, 498, 500)
    with pytest.raises(ValueError):
        safe_linear_log_e(m, 0.0)


# ----------------------------------------------------------------------
# anytime validity
# ----------------------------------------------------------------------
def test_safe_linear_respects_ville_under_continuous_monitoring():
    """P(sup_t E_t >= 1/alpha) <= alpha, monitoring after every observation."""
    rng = np.random.default_rng(7)
    reps, nmax, g, alpha = 400, 400, 0.4**2, 0.1
    crossings = 0
    for _ in range(reps):
        Z = rng.normal(size=(nmax, 2))
        X = 0.9 * Z[:, 0] - 0.5 * Z[:, 1] + rng.normal(size=nmax)
        Y = 0.7 * Z[:, 0] + 0.6 * Z[:, 1] + rng.normal(size=nmax)
        D = np.column_stack([X, Y, Z])
        ss = GaussianSuffStat(4)
        crossed = False
        for t in range(nmax):
            ss.update(D[t])
            if ss.n >= 6 and safe_linear_log_e(
                ss.residual_moments(0, 1, (2, 3)), g
            ) >= -np.log(alpha):
                crossed = True
                break
        crossings += crossed
    # Generous allowance for Monte-Carlo error; a broken construction
    # overshoots this by a wide margin rather than marginally.
    assert crossings / reps <= alpha + 0.05


def test_safe_linear_has_power_against_a_real_dependence():
    rng = np.random.default_rng(8)
    detected = 0
    for _ in range(30):
        n = 400
        Z = rng.normal(size=(n, 2))
        Y = 0.7 * Z[:, 0] + rng.normal(size=n)
        X = 0.9 * Z[:, 0] + 0.6 * Y + rng.normal(size=n)
        ss = GaussianSuffStat(4)
        ss.update(np.column_stack([X, Y, Z]))
        detected += safe_linear_log_e(ss.residual_moments(0, 1, (2, 3)), 0.4**2) >= np.log(20)
    assert detected >= 28


def test_universal_inference_respects_ville():
    rng = np.random.default_rng(9)
    reps, nmax, alpha = 300, 300, 0.1
    crossings = 0
    for _ in range(reps):
        proc = GaussianUniversalCI(n_cond=1)
        crossed = False
        for _ in range(nmax):
            z = rng.normal(size=1)
            x = 0.8 * z[0] + rng.normal()
            y = 0.6 * z[0] + rng.normal()
            if proc.update(x, y, z).log_e >= -np.log(alpha):
                crossed = True
                break
        crossings += crossed
    assert crossings / reps <= alpha + 0.05


def test_universal_inference_detects_dependence():
    rng = np.random.default_rng(10)
    proc = GaussianUniversalCI(n_cond=1)
    log_e = 0.0
    for _ in range(1500):
        z = rng.normal(size=1)
        y = 0.6 * z[0] + rng.normal()
        x = 0.8 * z[0] + 1.0 * y + rng.normal()
        log_e = proc.update(x, y, z).log_e
    assert log_e > np.log(100)


def test_discrete_universal_respects_ville():
    rng = np.random.default_rng(11)
    reps, nmax, alpha = 300, 300, 0.1
    crossings = 0
    for _ in range(reps):
        proc = DiscreteUniversalCI(n_x=2, n_y=2, n_z=2)
        crossed = False
        for _ in range(nmax):
            z = int(rng.integers(2))
            p = 0.3 + 0.4 * z
            x = int(rng.random() < p)      # depends on z only
            y = int(rng.random() < p)      # depends on z only  => X indep Y | Z
            if proc.update(x, y, z).log_e >= -np.log(alpha):
                crossed = True
                break
        crossings += crossed
    assert crossings / reps <= alpha + 0.05


def test_discrete_universal_detects_dependence():
    rng = np.random.default_rng(12)
    proc = DiscreteUniversalCI(n_x=2, n_y=2, n_z=1)
    log_e = 0.0
    for _ in range(2000):
        y = int(rng.integers(2))
        x = y if rng.random() < 0.85 else 1 - y
        log_e = proc.update(x, y, 0).log_e
    assert log_e > np.log(100)
