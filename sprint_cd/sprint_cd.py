r"""SPRINT-CD: anytime-valid constraint-based causal discovery.

The algorithm is a streaming PC-stable in which

* every conditional-independence query is answered by an **e-process**
  (:mod:`sprint_cd.eprocess.safe_linear`) rather than a fixed-sample p-value;
* edges are removed only on presentation of an **independence certificate** --
  an anytime-valid confidence sequence for the relevant coefficient that has
  shrunk inside a region of practical equivalence;
* both kinds of decision are charged against a **fixed budget** over the entire
  potential hypothesis family (:mod:`sprint_cd.multiplicity`).

Guarantee
---------
Let :math:`\mathcal{G}^\star` be the true DAG and let ``tau`` be *any* stopping
time -- including one chosen by looking at the output.  Under

1. a linear-Gaussian SEM and causal sufficiency,
2. :math:`\delta`-strong faithfulness: :math:`|\beta_{ij\cdot S}| \ge \delta`
   for every adjacent pair :math:`(i,j)` and every :math:`S` with
   :math:`|S| \le k`,

the estimate :math:`\hat{\mathcal{G}}_\tau` satisfies

.. math::

    P\Bigl( \exists\, \tau :\;
        \underbrace{\text{a true edge is absent from }\hat{\mathcal{G}}_\tau}
                   _{\text{false deletion}}
        \;\text{or}\;
        \underbrace{\text{a true independence is certified dependent}}
                   _{\text{false dependence}}
    \Bigr) \;\le\; \alpha .

*Proof sketch.* A false deletion of a truly adjacent pair requires some
confidence sequence for :math:`\beta_{ij\cdot S}` to have excluded its true
value (which has modulus at least :math:`\delta`, while the certificate
requires the interval to sit inside :math:`(-\delta, \delta)`), an event of
probability at most :math:`\alpha_{\mathrm{del}} w(i,j,S)` uniformly in time by
Ville.  A false dependence claim requires an e-process under its own null to
cross :math:`1/(\alpha_{\mathrm{dep}} w(i,j,S))`, likewise at most
:math:`\alpha_{\mathrm{dep}} w(i,j,S)` uniformly in time.  Summing over the
fixed family :math:`\mathcal{H}_k` and over the two budgets gives
:math:`\alpha_{\mathrm{del}} + \alpha_{\mathrm{dep}} = \alpha`.  On the
complement of that event, all retained edges are correct supersets, all
recorded separating sets are genuine, hence every v-structure is correctly
oriented; Meek's rules are deterministic and introduce no further error.
:math:`\square`

The guarantee is deliberately **one-sided on the skeleton**: at small ``t``
nothing has been certified and the output is a dense superset of the truth.
What is controlled uniformly in time is the *removal* of true edges and the
*assertion* of false dependence -- exactly the errors that are irreversible
in a streaming setting.

A terminological note
---------------------
"Anytime FCI" (Spirtes, 2001) refers to a *computational* property -- the
outer loop over conditioning-set sizes may be interrupted and still return a
sound graph.  "Anytime-valid" here is an *inferential* property: time-uniform
Type-I error control at data-dependent stopping times.  The two are unrelated.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from .eprocess.safe_linear import SafeLinearCI
from .graph import ARROW, TAIL, MarkedGraph, meek_rules
from .multiplicity import HypothesisBudget
from .stats import GaussianSuffStat

__all__ = ["SprintCDConfig", "SprintCD", "DiscoveryHistory"]


@dataclass
class SprintCDConfig:
    """Configuration for :class:`SprintCD`.

    Parameters
    ----------
    alpha:
        Total anytime-valid error budget, split between deletion and
        dependence decisions by ``deletion_share``.
    max_order:
        Largest conditioning-set size considered.
    rope:
        Half-width :math:`\\delta` of the region of practical equivalence, on
        the **partial-correlation** scale.  An edge is removed only once a
        confidence sequence fits entirely inside the ROPE.  This is the same
        :math:`\\delta` appearing in the strong-faithfulness assumption: the
        method cannot, and does not claim to, distinguish an exactly-zero
        association from an arbitrarily small non-zero one.

        The confidence sequence lives on the coefficient scale, where
        :math:`\\beta_{ij \\cdot S} = \\rho_{ij \\cdot S}\\,
        \\sigma_{i|S} / \\sigma_{j|S}`, so the ROPE is widened by that ratio
        before use.  The ratio is estimated from the **warm-up prefix** and is
        therefore a fixed constant, leaving the anytime-valid guarantee intact.
        Doing this matters: a raw coefficient-scale ROPE is not scale-free
        across conditioning sets, and on strongly connected graphs it inflates
        the sample size needed to delete an edge by exactly that ratio.
    prior_scale:
        Prior standard deviation of the tested coefficient; sets the effect
        size the e-process is tuned to detect.
    deletion_share:
        Fraction of ``alpha`` allocated to independence certificates.
    warmup:
        Number of initial observations used **only** to set prior scales and
        standardisation constants.  They are excluded from every e-process, so
        the resulting hyper-parameters are legitimately data-independent.
    order_decay:
        Geometric decay of the budget across conditioning-set sizes.
    stable:
        Use PC-stable adjacency snapshots (order-independent output).
    """

    alpha: float = 0.05
    max_order: int = 3
    rope: float = 0.1
    prior_scale: float = 0.4
    deletion_share: float = 0.5
    warmup: int = 50
    order_decay: float = 0.5
    stable: bool = True
    ridge: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        if not 0.0 < self.deletion_share < 1.0:
            raise ValueError("deletion_share must lie in (0, 1)")
        if self.rope <= 0.0:
            raise ValueError("rope must be positive")
        if self.warmup < 2:
            raise ValueError("warmup must be at least 2")


@dataclass
class DiscoveryHistory:
    """Per-sweep record of the evolving estimate."""

    n: list[int] = field(default_factory=list)
    n_edges: list[int] = field(default_factory=list)
    resolved: list[bool] = field(default_factory=list)
    graphs: list[MarkedGraph] = field(default_factory=list)

    def record(self, n: int, graph: MarkedGraph, resolved: bool, keep_graph: bool) -> None:
        self.n.append(int(n))
        self.n_edges.append(len(graph.edges()))
        self.resolved.append(bool(resolved))
        if keep_graph:
            self.graphs.append(graph.copy())


class SprintCD:
    """Streaming, anytime-valid causal discovery over a fixed variable set.

    Typical use is incremental::

        algo = SprintCD(d=8, config=SprintCDConfig(alpha=0.05))
        algo.warm_up(first_50_rows)
        for batch in stream:
            algo.update(batch)
            if algo.resolved:
                break
        cpdag = algo.cpdag()

    The estimate may be inspected after *every* batch without invalidating the
    guarantee -- that is the entire point of the construction.
    """

    def __init__(self, d: int, config: SprintCDConfig | None = None,
                 names: list[str] | None = None) -> None:
        if d < 2:
            raise ValueError("need at least two variables")
        self.d = d
        self.config = config or SprintCDConfig()
        self.names = names
        self.suffstat = GaussianSuffStat(d)

        eff_order = min(self.config.max_order, max(d - 2, 0))
        self.budget_del = HypothesisBudget(
            d=d, max_order=eff_order,
            alpha=self.config.alpha * self.config.deletion_share,
            order_decay=self.config.order_decay,
        )
        self.budget_dep = HypothesisBudget(
            d=d, max_order=eff_order,
            alpha=self.config.alpha * (1.0 - self.config.deletion_share),
            order_decay=self.config.order_decay,
        )

        self.graph = MarkedGraph.complete_undirected(d, names=names)
        self.sepsets: dict[tuple[int, int], tuple[int, ...]] = {}
        # Running maxima of dependence e-processes: Ville bounds the supremum,
        # so a crossing once achieved is never revoked.
        self._dep_logmax: dict[tuple[int, int, tuple[int, ...]], float] = {}
        self._confirmed: set[tuple[int, int]] = set()
        self._ci: SafeLinearCI | None = None
        self._warm_suffstat = GaussianSuffStat(d)
        self._rope_cache: dict[tuple[int, int, tuple[int, ...]], float] = {}
        self._warm_mean = np.zeros(d)
        self._warm_scale = np.ones(d)
        self._warmed = False
        self.history = DiscoveryHistory()

    # ------------------------------------------------------------------
    # set-up
    # ------------------------------------------------------------------
    def warm_up(self, batch: np.ndarray) -> "SprintCD":
        """Consume the warm-up prefix used to fix scales and prior hyper-parameters.

        These observations are deliberately *not* fed to the sufficient
        statistics: keeping them out is what makes ``g`` and the
        standardisation constants measurable with respect to data external to
        every e-process.
        """
        arr = np.atleast_2d(np.asarray(batch, dtype=float))
        if arr.shape[1] != self.d:
            raise ValueError(f"expected {self.d} columns")
        if arr.shape[0] < 2:
            raise ValueError("warm-up needs at least two rows")
        self._warm_mean = arr.mean(axis=0)
        sd = arr.std(axis=0, ddof=1)
        self._warm_scale = np.where(np.isfinite(sd) & (sd > 1e-12), sd, 1.0)
        self._ci = SafeLinearCI(
            self.suffstat,
            prior_scale=self.config.prior_scale,
            warmup_var=np.ones(self.d),  # stream is standardised below
            ridge=self.config.ridge,
        )
        # Retained separately from the streaming statistics: these rows are
        # never fed to an e-process, which is what lets quantities derived
        # from them serve as fixed constants.
        self._warm_suffstat = GaussianSuffStat(self.d)
        self._warm_suffstat.update((arr - self._warm_mean) / self._warm_scale)
        self._warmed = True
        return self

    def _standardise(self, arr: np.ndarray) -> np.ndarray:
        return (arr - self._warm_mean) / self._warm_scale

    # ------------------------------------------------------------------
    # streaming update
    # ------------------------------------------------------------------
    def update(self, batch: np.ndarray, *, keep_graph: bool = False) -> MarkedGraph:
        """Absorb a batch, re-run the discovery sweep, return the current estimate."""
        if not self._warmed:
            raise RuntimeError("call warm_up() before update()")
        arr = np.atleast_2d(np.asarray(batch, dtype=float))
        if arr.shape[1] != self.d:
            raise ValueError(f"expected {self.d} columns")
        self.suffstat.update(self._standardise(arr))
        self._sweep()
        self.history.record(self.suffstat.n, self.graph, self.resolved, keep_graph)
        return self.cpdag()

    # ------------------------------------------------------------------
    # certificates
    # ------------------------------------------------------------------
    def _rope_for(self, i: int, j: int, cond: tuple[int, ...]) -> float:
        """ROPE half-width on the coefficient scale for the pair ``(i, j)`` given ``S``.

        Converts the partial-correlation-scale ``config.rope`` using the
        warm-up estimate of ``sigma_{i|S} / sigma_{j|S}``.  Depends only on the
        warm-up prefix, so it is a constant as far as every e-process is
        concerned.
        """
        key = (i, j, cond)
        cached = self._rope_cache.get(key)
        if cached is not None:
            return cached
        ratio = 1.0
        if self._warm_suffstat.n > len(cond) + 2:
            m = self._warm_suffstat.residual_moments(i, j, cond)
            if m.sxx > 0.0 and m.syy > 0.0:
                ratio = float(np.sqrt(m.sxx / m.syy))
        delta = self.config.rope * ratio
        self._rope_cache[key] = delta
        return delta

    def _independence_certificate(
        self, i: int, j: int, cond: tuple[int, ...]
    ) -> bool:
        """Does the confidence sequence for ``beta_{ij.S}`` fit inside the ROPE?"""
        level = self.budget_del.level(len(cond))
        if level <= 0.0:
            return False
        lo, hi = self._ci.confidence_sequence(i, j, cond, alpha=level)
        if not np.isfinite(lo) or not np.isfinite(hi):
            return False
        delta = self._rope_for(i, j, cond)
        return bool(lo > -delta and hi < delta)

    def _dependence_certificate(self, i: int, j: int, cond: tuple[int, ...]) -> bool:
        """Has the e-process for ``X_i indep X_j | X_S`` crossed its Ville boundary?"""
        key = (i, j, cond)
        state = self._ci.symmetric_log_e(i, j, cond)
        prev = self._dep_logmax.get(key, -np.inf)
        cur = max(prev, state.log_e)
        self._dep_logmax[key] = cur
        return cur >= self.budget_dep.log_threshold(len(cond))

    # ------------------------------------------------------------------
    # discovery sweep
    # ------------------------------------------------------------------
    def _candidate_sets(self, i: int, j: int, order: int, adj: dict[int, list[int]]):
        """Conditioning sets of the given order drawn from the PC neighbourhoods.

        Restricting to graph neighbourhoods is a purely computational saving:
        the error budget is allocated over the full family, so examining fewer
        hypotheses can only make the procedure more conservative.
        """
        seen: set[tuple[int, ...]] = set()
        for base in (i, j):
            pool = [v for v in adj[base] if v != i and v != j]
            if len(pool) < order:
                continue
            for S in itertools.combinations(sorted(pool), order):
                if S not in seen:
                    seen.add(S)
                    yield S

    def _sweep(self) -> None:
        cfg = self.config
        max_order = min(cfg.max_order, max(self.d - 2, 0))

        for order in range(max_order + 1):
            adj = {v: self.graph.neighbours(v) for v in range(self.d)} if cfg.stable else None
            for i, j in list(itertools.combinations(range(self.d), 2)):
                if not self.graph.adjacent(i, j):
                    continue
                if not cfg.stable:
                    adj = {v: self.graph.neighbours(v) for v in range(self.d)}
                for S in self._candidate_sets(i, j, order, adj):
                    if self._independence_certificate(i, j, S):
                        self.graph.remove_edge(i, j)
                        self.sepsets[(i, j)] = S
                        self._confirmed.discard((i, j))
                        break

        # Dependence certificates for the surviving edges.
        adj = {v: self.graph.neighbours(v) for v in range(self.d)}
        for i, j in self.graph.edges():
            all_crossed = True
            for order in range(max_order + 1):
                for S in self._candidate_sets(i, j, order, adj):
                    if not self._dependence_certificate(i, j, S):
                        all_crossed = False
            if all_crossed:
                self._confirmed.add((i, j))
            else:
                self._confirmed.discard((i, j))

    # ------------------------------------------------------------------
    # orientation
    # ------------------------------------------------------------------
    def cpdag(self) -> MarkedGraph:
        """Current estimate, oriented into a CPDAG."""
        g = MarkedGraph(d=self.d, marks=(self.graph.skeleton() * TAIL).astype(np.int8),
                        names=self.names)
        np.fill_diagonal(g.marks, 0)

        # R0: v-structures.  i *-> k <-* j whenever i, j are non-adjacent and
        # the certified separating set for (i, j) excludes k.
        for k in range(self.d):
            nbrs = g.neighbours(k)
            for i, j in itertools.combinations(nbrs, 2):
                if g.adjacent(i, j):
                    continue
                sep = self.sepsets.get((min(i, j), max(i, j)))
                if sep is None or k in sep:
                    continue
                if g.mark_at(k, i) != ARROW and g.mark_at(k, j) != ARROW:
                    g.set_edge(i, k, TAIL, ARROW)
                    g.set_edge(j, k, TAIL, ARROW)
        return meek_rules(g)

    # ------------------------------------------------------------------
    @property
    def resolved(self) -> bool:
        """Every pair is either separated or has a full set of dependence certificates.

        This is the natural stopping rule: no further data can change the
        skeleton through any conditioning set the algorithm examines.  Because
        the guarantee is anytime-valid, stopping here -- a thoroughly
        data-dependent decision -- costs nothing.
        """
        for i, j in itertools.combinations(range(self.d), 2):
            if self.graph.adjacent(i, j) and (i, j) not in self._confirmed:
                return False
        return True

    @property
    def n(self) -> int:
        return self.suffstat.n


def run_sprint_cd(
    data: np.ndarray,
    config: SprintCDConfig | None = None,
    *,
    batch_size: int = 25,
    stop_when_resolved: bool = True,
    names: list[str] | None = None,
) -> tuple[MarkedGraph, SprintCD]:
    """Convenience driver: stream ``data`` through :class:`SprintCD` in batches.

    Returns the final CPDAG and the fitted object (for its history and the
    sample size at which it stopped).
    """
    data = np.asarray(data, dtype=float)
    cfg = config or SprintCDConfig()
    algo = SprintCD(d=data.shape[1], config=cfg, names=names)
    if data.shape[0] <= cfg.warmup:
        raise ValueError("not enough rows for the warm-up prefix")
    algo.warm_up(data[: cfg.warmup])

    rest = data[cfg.warmup:]
    for start in range(0, rest.shape[0], batch_size):
        algo.update(rest[start: start + batch_size])
        if stop_when_resolved and algo.resolved:
            break
    return algo.cpdag(), algo
