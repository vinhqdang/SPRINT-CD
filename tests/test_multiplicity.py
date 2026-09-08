import math

import numpy as np
import pytest

from sprint_cd.multiplicity import (HypothesisBudget, e_bh, e_bh_threshold_indices,
                                    e_holm)


def test_weights_over_the_whole_family_sum_to_one():
    for d, k in [(5, 2), (8, 3), (10, 1), (4, 0)]:
        b = HypothesisBudget(d=d, max_order=k, alpha=0.05)
        total = sum(
            b.weight(m) * (d * (d - 1) // 2) * math.comb(d - 2, m)
            for m in range(b.effective_max_order + 1)
        )
        assert total == pytest.approx(1.0)


def test_budget_favours_low_order_conditioning_sets():
    b = HypothesisBudget(d=8, max_order=3, alpha=0.05)
    levels = [b.level(m) for m in range(4)]
    assert all(a > c for a, c in zip(levels, levels[1:]))


def test_levels_never_exceed_the_total_budget():
    b = HypothesisBudget(d=8, max_order=3, alpha=0.05)
    assert all(b.level(m) <= 0.05 for m in range(4))


def test_out_of_range_order_gets_no_budget():
    b = HypothesisBudget(d=6, max_order=2, alpha=0.05)
    assert b.weight(5) == 0.0
    assert not np.isfinite(b.log_threshold(5))


def test_e_bh_rejects_only_large_e_values():
    e = np.array([100.0, 50.0, 2.0, 1.0, 0.5])
    mask = e_bh(e, 0.05)
    assert mask.tolist() == [True, True, False, False, False]


def test_e_bh_rejects_nothing_when_all_e_values_are_small():
    assert not e_bh(np.ones(10), 0.05).any()


def test_log_scale_e_bh_agrees_and_survives_overflow():
    e = np.array([1e3, 1e2, 5.0, 1.0])
    np.testing.assert_array_equal(e_bh(e, 0.1), e_bh_threshold_indices(np.log(e), 0.1))
    huge = np.array([900.0, 800.0, 1.0])       # exp() would overflow
    assert e_bh_threshold_indices(huge, 0.05)[:2].all()


def test_e_bh_controls_fdr_under_the_global_null():
    """All nulls true: e-values have mean <= 1, so rejections must be rare."""
    rng = np.random.default_rng(0)
    false_discoveries = 0
    reps = 400
    for _ in range(reps):
        e = rng.exponential(size=20)      # mean-one e-values
        false_discoveries += e_bh(e, 0.1).any()
    assert false_discoveries / reps <= 0.1 + 0.05


# ----------------------------------------------------------------------
# e-Holm
# ----------------------------------------------------------------------
def test_e_holm_matches_the_closure_definition_by_brute_force():
    """Check the O(n) threshold against the defining closure rule."""
    import itertools
    rng = np.random.default_rng(0)
    alpha = 0.1
    for _ in range(200):
        n = int(rng.integers(2, 7))
        e = rng.exponential(scale=8.0, size=n)
        mask = e_holm(e, alpha)
        for i in range(n):
            brute = all(
                e[list(I)].sum() >= len(I) / alpha
                for k in range(1, n + 1)
                for I in itertools.combinations(range(n), k) if i in I
            )
            assert bool(mask[i]) == brute


def test_e_holm_is_never_weaker_than_the_union_bound():
    rng = np.random.default_rng(1)
    alpha = 0.05
    for _ in range(300):
        n = int(rng.integers(2, 12))
        e = rng.exponential(scale=20.0, size=n)
        bonferroni = e >= n / alpha
        assert np.all(e_holm(e, alpha) >= bonferroni)


def test_e_holm_is_strictly_stronger_when_others_carry_evidence():
    """Every e-value above 1/alpha leaves J*, lowering the threshold."""
    alpha = 0.05
    e = np.array([25.0, 1e6, 1e6, 1e6])       # 25 < 4/alpha = 80, so Bonferroni fails
    assert not (e >= len(e) / alpha)[0]
    assert e_holm(e, alpha)[0]


def test_e_holm_controls_fwer_under_the_global_null_with_dependent_e_values():
    rng = np.random.default_rng(2)
    alpha, reps = 0.1, 2000
    false_rejections = 0
    for _ in range(reps):
        # Strongly dependent mean-one e-values sharing a common factor.
        common = rng.exponential()
        e = common * rng.exponential(size=8)
        e = e / np.mean(e) * np.mean(rng.exponential(size=8))
        false_rejections += e_holm(e, alpha).any()
    assert false_rejections / reps <= alpha + 0.05


def test_e_holm_rejects_nothing_when_all_e_values_are_small():
    assert not e_holm(np.full(10, 0.5), 0.05).any()


def test_e_holm_validates_inputs():
    with pytest.raises(ValueError):
        e_holm(np.ones((2, 2)), 0.05)
    with pytest.raises(ValueError):
        e_holm(np.ones(3), 1.5)
