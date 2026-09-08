r"""Certificates: the two primitives of certificate-only causal discovery.

The design principle is a single rule:

    **A feature of the graph may be asserted only by REJECTING a null that is
    true whenever the feature is absent.  Nothing is ever asserted by failing
    to reject.**

Constraint-based discovery violates this rule at its core.  PC removes an edge
when a conditional-independence test *fails to reject*, so absence of evidence
is silently converted into evidence of absence; the equivalence-region repair
in :mod:`sprint_cd.sprint_cd` restores a guarantee but only by assuming
``delta``-strong faithfulness, which our own experiments show fails in roughly
a third of random DAGs and four fifths of random latent-variable instances.

This module supplies the two certificates that let a discovery algorithm obey
the rule instead.

Adjacency certificate
---------------------
For a pair ``(i, j)`` the relevant null is *composite*:

.. math::

    H^{\mathrm{sep}}_{ij} \;:\; \exists\, S \in \mathcal{S}_k \ \text{ such that }\
        X_i \perp X_j \mid X_S ,

i.e. "the pair is separable".  This is true whenever ``i`` and ``j`` are
non-adjacent, so rejecting it certifies adjacency.  Since the null is a union,
the minimum of the per-set e-processes is an e-process for it (Vovk and Wang,
2021; Ramdas and Wang, 2025):

.. math::

    A_t(i,j) \;=\; \min_{S \in \mathcal{S}_k} E^{(S)}_t(i,j).

*Proof.* If the null holds, some ``S*`` separates the pair, so
:math:`A_t \le E^{(S^*)}_t` pointwise; ``E^{(S*)}`` is an e-process, so for
every stopping time :math:`\tau`, :math:`\mathbb{E}[A_\tau] \le
\mathbb{E}[E^{(S^*)}_\tau] \le 1`. :math:`\square`

Two things follow that matter more than the construction itself.

* **No faithfulness assumption enters validity.**  If ``i`` and ``j`` are
  non-adjacent in the true DAG then ``pa(i)`` or ``pa(j)`` d-separates them, so
  the null is true and the bound applies.  Faithfulness is needed only for
  *power* -- specifically adjacency-faithfulness (Ramsey, Spirtes and Zhang,
  2006), which is what makes every ``E^{(S)}`` grow for a genuinely adjacent
  pair.  No strong faithfulness and no equivalence region appear anywhere.

  Stated exactly, validity needs three things: the Markov condition, a
  separating set inside ``S_k``, and a *valid per-set e-process* ``E^{(S)}``.
  The third is not free: the default primitive is the linear-Gaussian safe
  test, so that model is assumed there.  Substituting a nonparametric
  sequential CI test (SKCI, He and Sutherland, ICML 2026) removes it, and the
  argument above is indifferent to which primitive is used.
* **Multiplicity is over pairs, not over (pair, conditioning set) triples.**
  One hypothesis is tested per pair, so the union bound runs over
  ``C(d, 2)`` hypotheses rather than ``C(d,2) * |S_k|``.

*Prior art.*  Aggregating conditional-independence tests over conditioning sets
into an edge-level statistic is not new: PC-p (Strobl, Spirtes and
Visweswaran, 2019) upper-bounds an edge's p-value by the **maximum** p-value
over conditioning sets, of which the minimum-e-value above is the exact
e-value analogue, and it uses that to control FDR at a fixed sample size.  What
is new here is that the aggregated object is an *e-process*, so the certificate
holds at every stopping time under continuous monitoring, and that the
resulting algorithm grows rather than prunes.

Direction certificate
---------------------
Orientation error *is* controllable within the constraint-based framework:
PC-p (Strobl, Spirtes and Visweswaran, arXiv:1607.03975) formulates
edge-specific hypothesis tests for unshielded colliders and for Meek-rule
orientations, and controls their FDR with Benjamini-Yekutieli.  What no
constraint-based orientation test can do is orient an edge lying in no
v-structure: those tests are confined to the Markov equivalence class, and a
chain is unorientable in principle.  Conversely the functional-model methods
that do escape the equivalence class -- LiNGAM and its descendants -- decide
directions by comparing scores, with no error control on the decision.

The certificate below sits in the gap: error-controlled orientation *beyond*
the Markov equivalence class, and time-uniform rather than fixed-sample.

Under a linear non-Gaussian acyclic model the two orientations of a pair induce
different joint densities, and exactly one of them is correct (Shimizu et al.,
2006; Darmois-Skitovich).  So "the direction is ``j -> i``" is a genuine null
that is true when the arrowhead ``i -> j`` is absent, and rejecting it
certifies ``i -> j``.  We test it by sequential universal inference:

.. math::

    \log E_t \;=\; \sum_{s \le t} \log D^{i \to j}_{\hat\theta_{s-1}}(o_s)
                 \;-\; \sup_{\theta} \sum_{s \le t} \log D^{j \to i}_{\theta}(o_s),

with the numerator's parameters fitted on strictly past data.  Dominating the
denominator by the likelihood at the true null parameter makes ``E`` an
e-process exactly as in :mod:`sprint_cd.eprocess.universal`.

Under Gaussian noise the two factorisations are observationally
indistinguishable, the ratio does not grow, and the certificate is simply never
issued -- the procedure abstains precisely where the direction is not
identified, rather than guessing.

**Assumption.**  Validity of this certificate requires the noise to lie in the
fitted family (a generalised Gaussian: unimodal, symmetric, exponential
tails).  Universal inference needs the denominator's supremum to dominate the
likelihood at the true null parameter, and outside the family that is not
guaranteed -- so this is the certificate's main theoretical risk, and unlike
the adjacency certificate's assumptions it is genuinely parametric.

Empirically it is more robust than that suggests.  Against noise breaking each
family property in turn -- Student-t (polynomial rather than exponential
tails), shifted exponential (asymmetric), and a two-component mixture
(bimodal) -- the reversed direction was certified in 0 of 12 runs each, with
full power retained, at ``alpha = 0.05``.  These are locked in as tests in
``tests/test_certificates.py`` rather than left as an informal claim.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import gammaln

from .eprocess.safe_linear import SafeLinearCI

__all__ = [
    "admissible_sets",
    "adjacency_log_e",
    "DirectionEProcess",
    "gg_logpdf",
    "fit_gg_regression",
]


# ----------------------------------------------------------------------
# Adjacency certificate
# ----------------------------------------------------------------------
def admissible_sets(d: int, i: int, j: int, max_order: int):
    """All conditioning sets of size at most ``max_order``, excluding ``i`` and ``j``.

    The family is fixed before any data arrive, which is what allows the
    multiplicity bound to ignore which sets the algorithm actually inspects.
    """
    others = [v for v in range(d) if v not in (i, j)]
    cap = min(max_order, len(others))
    for order in range(cap + 1):
        yield from itertools.combinations(others, order)


def adjacency_log_e(
    ci: SafeLinearCI, i: int, j: int, max_order: int, *, stop_below: float | None = None
) -> float:
    """``log A_t(i, j)``: the minimum log e-value over all admissible sets.

    ``stop_below`` allows an early exit once the minimum is known to be under a
    decision threshold; the value returned is then a valid upper bound on the
    decision but not the exact minimum, which is all a certificate needs.
    """
    best = np.inf
    for S in admissible_sets(ci.suffstat.d, i, j, max_order):
        v = ci.symmetric_log_e(i, j, S).log_e
        if v < best:
            best = v
            if stop_below is not None and best < stop_below:
                return float(best)
    return float(best if np.isfinite(best) else -np.inf)


# ----------------------------------------------------------------------
# Direction certificate
# ----------------------------------------------------------------------
# Coarse bracketing grid. It deliberately includes the two shapes that arise
# most often -- 1.0 (Laplace) and 2.0 (Gaussian) -- because a grid that omits
# the true shape makes the "supremum" in the universal-inference denominator
# smaller than the likelihood at the true parameter, which is exactly the
# inequality the e-process argument needs. Omitting 1.0 inflated log E by up to
# 2.4 nats on a correctly specified Laplace null, in 16% of replications.
# The grid only brackets; ``fit_gg_regression`` then refines continuously.
_KAPPA_GRID = np.array([0.6, 0.8, 1.0, 1.3, 1.7, 2.0, 2.6, 3.5, 5.0])
_KAPPA_BOUNDS = (0.35, 12.0)
_EPS = 1e-9


def gg_logpdf(r: np.ndarray, sigma: float, kappa: float) -> np.ndarray:
    """Log-density of the generalised Gaussian (exponential power) family.

    ``kappa = 2`` is Gaussian, ``kappa = 1`` Laplace, ``kappa -> inf`` uniform.
    """
    return (np.log(kappa) - np.log(2.0) - np.log(sigma) - gammaln(1.0 / kappa)
            - np.abs(r / sigma) ** kappa)


def _profile(resid: np.ndarray, kappa: float) -> tuple[float, float]:
    """Profile log-likelihood at fixed shape, with the scale maximised out."""
    n = resid.size
    s = float(np.sum(np.abs(resid) ** kappa))
    if s <= 0.0 or n == 0:
        return -np.inf, _EPS
    sigma = (kappa * s / n) ** (1.0 / kappa)
    ll = n * (np.log(kappa) - np.log(2.0) - np.log(sigma) - gammaln(1.0 / kappa)) - n / kappa
    return float(ll), float(sigma)


def _irls(y: np.ndarray, X: np.ndarray, kappa: float, coef0: np.ndarray,
          iters: int = 25) -> np.ndarray:
    """Iteratively reweighted least squares for the ``|r|^kappa`` loss."""
    coef = coef0.copy()
    for _ in range(iters):
        r = y - X @ coef
        # A residual of exactly zero gives an infinite weight for kappa < 2;
        # clipping keeps the reweighting finite without changing the optimum.
        with np.errstate(divide="ignore", invalid="ignore"):
            w = np.abs(r) ** (kappa - 2.0)
        w = np.clip(np.nan_to_num(w, nan=1.0, posinf=1e12), 1e-12, 1e12)
        XW = X * w[:, None]
        A = X.T @ XW
        b = XW.T @ y
        try:
            new = np.linalg.solve(A + 1e-10 * np.eye(A.shape[0]), b)
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(new)):
            break
        if np.max(np.abs(new - coef)) < 1e-9:
            coef = new
            break
        coef = new
    return coef


def fit_gg_regression(y: np.ndarray, X: np.ndarray):
    """Maximise the generalised-Gaussian regression likelihood over coefficients,
    scale and shape.  Returns ``(loglik, coef, sigma, kappa)``.

    Universal inference needs a genuine supremum over the null family: the
    argument dominates ``E_t`` by a martingale only if the denominator is at
    least the likelihood at the true parameter.  A maximum over a fixed shape
    grid does not provide that when the true shape falls between grid points,
    so the grid is used only to bracket and the shape is then refined
    continuously by bounded Brent search on the profile likelihood.
    """
    coef0, *_ = np.linalg.lstsq(X, y, rcond=None)

    def profile_at(k: float):
        k = float(np.clip(k, *_KAPPA_BOUNDS))
        coef = coef0 if abs(k - 2.0) < 1e-9 else _irls(y, X, k, coef0)
        ll, sig = _profile(y - X @ coef, k)
        return ll, coef, sig, k

    best = max((profile_at(k) for k in _KAPPA_GRID), key=lambda t: t[0])

    # Refine within the bracket around the best grid point.
    lo = max(_KAPPA_BOUNDS[0], best[3] / 1.8)
    hi = min(_KAPPA_BOUNDS[1], best[3] * 1.8)
    try:
        res = minimize_scalar(lambda k: -profile_at(k)[0], bounds=(lo, hi),
                              method="bounded", options={"xatol": 1e-3})
        if res.success:
            cand = profile_at(float(res.x))
            if cand[0] > best[0]:
                best = cand
    except Exception:            # refinement is an optimisation, never a
        pass                     # correctness dependency; the grid value stands
    return best


@dataclass
class _Factorisation:
    """Fitted ``x_a = a'Z + u ;  x_b = beta x_a + g'Z + e`` factorisation."""

    loglik: float
    ca: np.ndarray
    s1: float
    k1: float
    cb: np.ndarray
    s2: float
    k2: float

    def score(self, xa: np.ndarray, xb: np.ndarray, Z: np.ndarray) -> np.ndarray:
        Xb = np.column_stack([xa, Z])
        return (gg_logpdf(xa - Z @ self.ca, self.s1, self.k1)
                + gg_logpdf(xb - Xb @ self.cb, self.s2, self.k2))


def _fit_factorisation(xa: np.ndarray, xb: np.ndarray, Z: np.ndarray) -> _Factorisation:
    ll1, ca, s1, k1 = fit_gg_regression(xa, Z)
    ll2, cb, s2, k2 = fit_gg_regression(xb, np.column_stack([xa, Z]))
    return _Factorisation(ll1 + ll2, ca, s1, k1, cb, s2, k2)


@dataclass
class DirectionEProcess:
    """E-process for the null "the direction of the pair is ``j -> i``".

    Crossing ``1/alpha`` certifies the arrowhead ``i -> j``.  Instantiated with
    a *frozen* conditioning set, so the hypothesis it tests never changes once
    accumulation has begun.

    Observations scored by the numerator and observations entering the
    denominator's maximised null likelihood must be the same set.  Letting the
    denominator range over extra observations adds their log-density to
    ``log E`` with no matching numerator term, which inflates the process by a
    constant of order (extra observations) and makes it certify every
    direction, including under Gaussian noise where none is identifiable.
    """

    n_cond: int
    min_fit: int = 60

    log_numerator: float = field(default=0.0, init=False)
    log_max: float = field(default=-np.inf, init=False)
    _xi: list = field(default_factory=list, init=False, repr=False)
    _xj: list = field(default_factory=list, init=False, repr=False)
    _z: list = field(default_factory=list, init=False, repr=False)
    _scored: int = field(default=0, init=False)
    _stored: int = field(default=0, init=False)

    def update(self, cause: np.ndarray, effect: np.ndarray, Z: np.ndarray) -> float:
        """Absorb a batch and return the running maximum of ``log E``.

        The argument order fixes the meaning: this process tests the null
        ``effect -> cause`` and crossing certifies the arrowhead
        ``cause -> effect``.  Naming the arguments after their role -- rather
        than after the pair's index order -- keeps callers from silently
        certifying the reverse arrow.
        """
        xi = np.asarray(cause, dtype=float).ravel()
        xj = np.asarray(effect, dtype=float).ravel()
        Z = np.atleast_2d(np.asarray(Z, dtype=float))
        if Z.shape[0] != xi.size:
            Z = Z.T

        # Count observations, not batches: len(self._xi) is the number of
        # stored batches, so comparing it against min_fit leaves the numerator
        # switched off forever on any realistic batch schedule.
        if self._stored >= self.min_fit:
            past_i = np.concatenate(self._xi)
            past_j = np.concatenate(self._xj)
            past_z = np.vstack(self._z)
            # Numerator: alternative (i -> j) factorisation fitted on past data only.
            alt = _fit_factorisation(past_i, past_j, past_z)
            self.log_numerator += float(np.sum(alt.score(xi, xj, Z)))
            self._scored += xi.size

        self._xi.append(xi)
        self._xj.append(xj)
        self._z.append(Z)
        self._stored += xi.size

        if self._scored >= self.min_fit:
            # Denominator: null (j -> i) likelihood maximised over the SCORED window.
            all_i = np.concatenate(self._xi)
            all_j = np.concatenate(self._xj)
            all_z = np.vstack(self._z)
            start = all_i.size - self._scored
            null = _fit_factorisation(all_j[start:], all_i[start:], all_z[start:])
            self.log_max = max(self.log_max, self.log_numerator - null.loglik)
        return self.log_max

    def certifies(self, alpha: float) -> bool:
        return self.log_max >= -np.log(alpha)
