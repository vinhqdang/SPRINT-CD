r"""CERT-CD: certificate-only causal discovery.

One rule governs the algorithm:

    **Assert a feature of the graph only by REJECTING a null that is true
    whenever that feature is absent.  Never assert anything by failing to
    reject.**

Every constraint-based algorithm since PC breaks this rule at its first step:
it deletes an edge when a conditional-independence test *fails to reject*.
That converts absence of evidence into evidence of absence, and it is why
finite-sample guarantees for those methods need faithfulness -- or, for the
equivalence-region repair in :mod:`sprint_cd.sprint_cd`, ``delta``-strong
faithfulness, which our own experiments find holds in only about two thirds of
random DAGs and a fifth of random latent-variable instances.

CERT-CD obeys the rule instead, and consequently inverts the shape of the
algorithm:

===================  ==========================  ==========================
                     PC / SPRINT-CD              CERT-CD
===================  ==========================  ==========================
starts from          the complete graph          the empty graph
edges are            removed on weak evidence    added on strong evidence
graph over time      shrinks                     grows
controlled error     missing edges (SPRINT-CD)   false edges and arrowheads
needs faithfulness   yes, for validity           only for power
orientation error    uncontrolled                certified
undecided pairs      forced to a decision        reported as undecided
===================  ==========================  ==========================

Two certificates (:mod:`sprint_cd.certificates`) do the work.

* **Adjacency.**  ``A_t(i,j) = min_S E^{(S)}_t`` is an e-process for the
  composite null "the pair is separable by some conditioning set".  Crossing
  certifies an edge.  No faithfulness assumption is used -- only the Markov
  condition, a separating set inside the family, and a valid per-set
  e-process (linear-Gaussian for the default primitive).
* **Direction.**  A sequential universal-inference e-process against the null
  "the direction is ``j -> i``" under a linear non-Gaussian model.  Crossing
  certifies the arrowhead ``i -> j``.  Valid under that model; it abstains
  entirely under Gaussian noise, where no direction is identified.

Guarantee
---------
For **any** stopping time ``tau``, with ``alpha = alpha_A + alpha_D``:

.. math::

    P\Bigl( \exists\, \tau :\ \text{some certified edge is absent from } G^\star
        \ \text{ or some certified arrowhead is wrong} \Bigr) \;\le\; \alpha .

*Proof.* Each certificate is an e-process for a null that holds whenever the
asserted feature is absent, so by Ville each is falsely issued with probability
at most its own level, uniformly in time.  The families -- ``C(d,2)`` adjacency
nulls and ``d(d-1)`` direction nulls -- are fixed before any data arrive, so a
union bound over them gives ``alpha_A + alpha_D``.  Which certificates the
algorithm chooses to evaluate never enters the bound. :math:`\square`

No faithfulness, no strong faithfulness and no equivalence region appear
anywhere in that argument.

What the output does *not* say
------------------------------
The absence of an edge in a CERT-CD graph means **undecided**, not "certified
absent".  This is the honest reading of what the data support and is the price
of the rule: early in a run the graph is empty, and it fills in as evidence
arrives.  A pair may remain undecided forever if adjacency-faithfulness fails
for it -- which is exactly the situation in which every other constraint-based
method silently returns a confident wrong answer.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from .certificates import DirectionEProcess, adjacency_log_e
from .eprocess.safe_linear import SafeLinearCI
from .graph import ARROW, CIRCLE, MarkedGraph, TAIL
from .stats import GaussianSuffStat

__all__ = ["CertCDConfig", "CertCD", "run_cert_cd"]

UNDECIDED = 0
CERTIFIED = 1


@dataclass
class CertCDConfig:
    """Configuration for :class:`CertCD`.

    Parameters
    ----------
    alpha:
        Total anytime-valid budget, split between the adjacency and direction
        certificate families by ``adjacency_share``.
    max_order:
        Largest conditioning-set size in the adjacency family ``S_k``.  The
        adjacency guarantee requires every non-adjacent pair to have *some*
        separating set of size at most this; ``k`` at least the maximum
        in-degree suffices, since ``pa(i)`` always separates.
    prior_scale:
        Prior scale of the tested coefficient in the per-set e-processes.
    adjacency_share:
        Fraction of ``alpha`` given to adjacency certificates.
    warmup:
        Observations used only to fix standardisation constants; excluded from
        every e-process.
    orient:
        Whether to run direction certificates at all.  Turn off for Gaussian
        data, where no direction is identifiable and the certificates would
        simply never fire.
    direction_conditioning:
        ``"neighbours"`` freezes the conditioning set for a pair to its
        certified neighbours at the moment its adjacency is certified;
        ``"none"`` uses an intercept only.  Freezing matters: a conditioning
        set that kept changing would change the hypothesis under test, and
        Ville's inequality bounds a fixed one.
    """

    alpha: float = 0.05
    max_order: int = 2
    prior_scale: float = 0.5
    adjacency_share: float = 0.5
    warmup: int = 50
    orient: bool = True
    direction_conditioning: str = "neighbours"
    direction_min_fit: int = 80

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        if not 0.0 < self.adjacency_share <= 1.0:
            raise ValueError("adjacency_share must lie in (0, 1]")
        if self.max_order < 0:
            raise ValueError("max_order must be non-negative")
        if self.direction_conditioning not in ("neighbours", "none"):
            raise ValueError("direction_conditioning must be 'neighbours' or 'none'")


class CertCD:
    """Certificate-only, anytime-valid causal discovery.

    The estimate may be inspected after every batch and the run stopped on any
    data-dependent rule; both certificate families are anytime-valid.
    """

    def __init__(self, d: int, config: CertCDConfig | None = None,
                 names: list[str] | None = None) -> None:
        if d < 2:
            raise ValueError("need at least two variables")
        self.d = d
        self.config = config or CertCDConfig()
        self.names = names
        self.suffstat = GaussianSuffStat(d)

        n_pairs = d * (d - 1) // 2
        self.alpha_adj = self.config.alpha * self.config.adjacency_share / n_pairs
        n_dir = max(d * (d - 1), 1)
        self.alpha_dir = (self.config.alpha * (1.0 - self.config.adjacency_share)
                          / n_dir)

        self.adj_status = np.zeros((d, d), dtype=np.int8)
        self._adj_logmax: dict[tuple[int, int], float] = {}
        self._dir: dict[tuple[int, int], DirectionEProcess] = {}
        self._dir_cond: dict[tuple[int, int], tuple[int, ...]] = {}
        self._dir_certified: set[tuple[int, int]] = set()

        self._ci: SafeLinearCI | None = None
        self._warm_mean = np.zeros(d)
        self._warm_scale = np.ones(d)
        self._warmed = False

    # ------------------------------------------------------------------
    def warm_up(self, batch: np.ndarray) -> "CertCD":
        """Fix standardisation constants from a prefix excluded from every e-process."""
        arr = np.atleast_2d(np.asarray(batch, dtype=float))
        if arr.shape[1] != self.d:
            raise ValueError(f"expected {self.d} columns")
        if arr.shape[0] < 2:
            raise ValueError("warm-up needs at least two rows")
        self._warm_mean = arr.mean(axis=0)
        sd = arr.std(axis=0, ddof=1)
        self._warm_scale = np.where(np.isfinite(sd) & (sd > 1e-12), sd, 1.0)
        self._ci = SafeLinearCI(self.suffstat, prior_scale=self.config.prior_scale,
                                warmup_var=np.ones(self.d))
        self._warmed = True
        return self

    def _standardise(self, arr: np.ndarray) -> np.ndarray:
        return (arr - self._warm_mean) / self._warm_scale

    # ------------------------------------------------------------------
    def update(self, batch: np.ndarray) -> MarkedGraph:
        """Absorb a batch, refresh both certificate families, return the estimate."""
        if not self._warmed:
            raise RuntimeError("call warm_up() before update()")
        arr = np.atleast_2d(np.asarray(batch, dtype=float))
        if arr.shape[1] != self.d:
            raise ValueError(f"expected {self.d} columns")
        std = self._standardise(arr)
        self.suffstat.update(std)

        self._update_adjacency()
        if self.config.orient:
            self._update_direction(std)
        return self.graph()

    # ------------------------------------------------------------------
    def _update_adjacency(self) -> None:
        thr = -np.log(self.alpha_adj)
        for i, j in itertools.combinations(range(self.d), 2):
            if self.adj_status[i, j] == CERTIFIED:
                continue
            cur = adjacency_log_e(self._ci, i, j, self.config.max_order,
                                  stop_below=thr)
            prev = self._adj_logmax.get((i, j), -np.inf)
            best = max(prev, cur)
            self._adj_logmax[(i, j)] = best
            if best >= thr:
                # Certified adjacent: freeze the conditioning set and open the
                # two direction e-processes from this point onwards.
                self.adj_status[i, j] = self.adj_status[j, i] = CERTIFIED
                if self.config.orient:
                    self._open_direction(i, j)

    def _open_direction(self, i: int, j: int) -> None:
        if self.config.direction_conditioning == "neighbours":
            cond = tuple(v for v in range(self.d)
                         if v not in (i, j)
                         and (self.adj_status[i, v] == CERTIFIED
                              or self.adj_status[j, v] == CERTIFIED))
        else:
            cond = ()
        for a, b in ((i, j), (j, i)):
            self._dir_cond[(a, b)] = cond
            self._dir[(a, b)] = DirectionEProcess(
                n_cond=len(cond) + 1, min_fit=self.config.direction_min_fit)

    def _update_direction(self, std: np.ndarray) -> None:
        thr = -np.log(self.alpha_dir)
        n = std.shape[0]
        for (a, b), proc in self._dir.items():
            if (a, b) in self._dir_certified:
                continue
            cond = self._dir_cond[(a, b)]
            Z = np.column_stack([np.ones(n)] + [std[:, c] for c in cond])
            # Tests the null "b -> a"; crossing certifies the arrowhead a -> b,
            # so the candidate cause a is passed first.
            if proc.update(std[:, a], std[:, b], Z) >= thr:
                self._dir_certified.add((a, b))

    # ------------------------------------------------------------------
    def graph(self) -> MarkedGraph:
        """Certified estimate.

        Contains **only** certified edges.  An absent edge means *undecided*,
        not certified absent.  A circle endpoint means the edge is certified
        but its direction is not.
        """
        g = MarkedGraph(d=self.d, names=self.names)
        for i, j in itertools.combinations(range(self.d), 2):
            if self.adj_status[i, j] != CERTIFIED:
                continue
            fwd = (i, j) in self._dir_certified      # certified i -> j
            bwd = (j, i) in self._dir_certified      # certified j -> i
            if fwd and not bwd:
                g.set_edge(i, j, TAIL, ARROW)
            elif bwd and not fwd:
                g.set_edge(i, j, ARROW, TAIL)
            else:
                # Undetermined, or (pathologically) both certified: report
                # circles rather than inventing an orientation.
                g.set_edge(i, j, CIRCLE, CIRCLE)
        return g

    # ------------------------------------------------------------------
    def certified_edges(self) -> list[tuple[int, int]]:
        return [(i, j) for i, j in itertools.combinations(range(self.d), 2)
                if self.adj_status[i, j] == CERTIFIED]

    def undecided_pairs(self) -> list[tuple[int, int]]:
        return [(i, j) for i, j in itertools.combinations(range(self.d), 2)
                if self.adj_status[i, j] != CERTIFIED]

    def certified_arrows(self) -> list[tuple[int, int]]:
        return sorted(self._dir_certified)

    @property
    def n(self) -> int:
        return self.suffstat.n


def run_cert_cd(data: np.ndarray, config: CertCDConfig | None = None, *,
                batch_size: int = 100, names: list[str] | None = None):
    """Stream ``data`` through :class:`CertCD`; returns the graph and the object."""
    data = np.asarray(data, dtype=float)
    cfg = config or CertCDConfig()
    algo = CertCD(d=data.shape[1], config=cfg, names=names)
    if data.shape[0] <= cfg.warmup:
        raise ValueError("not enough rows for the warm-up prefix")
    algo.warm_up(data[: cfg.warmup])
    rest = data[cfg.warmup:]
    for start in range(0, rest.shape[0], batch_size):
        algo.update(rest[start: start + batch_size])
    return algo.graph(), algo
