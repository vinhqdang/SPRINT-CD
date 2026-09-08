# CERT-CD: certificate-only causal discovery

## The rule

> **Assert a feature of the graph only by REJECTING a null that is true
> whenever that feature is absent. Never assert anything by failing to reject.**

Constraint-based causal discovery breaks this rule at its first step. PC starts
from the complete graph and *removes* an edge when a conditional-independence
test fails to reject. Absence of evidence becomes evidence of absence, which is
why finite-sample correctness for these methods needs faithfulness. The
equivalence-region repair in `sprint_cd/sprint_cd.py` restores a genuine
anytime-valid guarantee, but only by assuming `delta`-strong faithfulness —
and our own measurements put that premise at roughly 65% of random 6-variable
DAGs and 19% of 7-variable graphs with one latent variable marginalised out.
Where it fails, that method deletes true edges in about three quarters of runs.

CERT-CD obeys the rule instead. The consequence is not a tweak but a different
shape of algorithm:

| | PC / SPRINT-CD | CERT-CD |
|---|---|---|
| starts from | complete graph | empty graph |
| edges are | removed on weak evidence | added on strong evidence |
| graph over time | shrinks | grows |
| controlled error | missing edges | false edges **and** wrong arrowheads |
| faithfulness | needed for validity | needed only for power |
| orientation error | uncontrolled | certified |
| undecided pairs | forced to a decision | reported as undecided |

---

## Certificate 1 — adjacency

For a pair `(i, j)` the relevant null is composite:

```
H_sep(i,j):  there exists S in S_k  such that  X_i ⟂ X_j | X_S
```

"The pair is separable." It is true whenever `i` and `j` are non-adjacent, so
rejecting it certifies an edge. Since the null is a *union*, the minimum of the
per-set e-processes is an e-process for it:

```
A_t(i,j) = min over S in S_k of E^(S)_t(i,j)
```

**Proposition.** `A_t` is an e-process for `H_sep(i,j)`.

*Proof.* If the null holds, some `S*` separates the pair, so `A_t ≤ E^(S*)_t`
pointwise. `E^(S*)` is an e-process, so for every stopping time `tau`,
`E[A_tau] ≤ E[E^(S*)_tau] ≤ 1`. ∎

Two consequences matter more than the construction:

**Validity needs only the Markov condition.** If `i` and `j` are non-adjacent
in the true DAG then `pa(i)` or `pa(j)` d-separates them, so the null is true
and Ville applies. Faithfulness never enters. What faithfulness buys is
*power*: specifically adjacency-faithfulness (Ramsey, Spirtes and Zhang, 2006),
which is what makes every `E^(S)` grow for a genuinely adjacent pair. An
instance where it fails leaves pairs undecided; it does not produce wrong
edges. The one structural requirement is that `k` be large enough for some
separating set to fit in `S_k` — `k ≥ max in-degree` suffices, since `pa(i)`
always separates.

**Multiplicity is over pairs, not triples.** One hypothesis is tested per pair,
so the union bound runs over `C(d,2)` hypotheses instead of
`C(d,2) × |S_k|`. At `d = 6, k = 2` that is a 24× larger per-hypothesis level
than SPRINT-CD's triple-indexed budget.

---

## Certificate 2 — direction

This is the part with no counterpart in the literature. PC-p and its
successors control *edge* errors; orientation is deterministic post-processing
carried out with no error control at all. Yet the direction is usually the
causal claim of interest.

Under a linear non-Gaussian acyclic model the two orientations of a pair induce
different joint densities and exactly one is correct (Shimizu et al., 2006; via
Darmois–Skitovich). So "the direction is `j -> i`" is a genuine null, true
precisely when the arrowhead `i -> j` is absent. Rejecting it certifies
`i -> j`. We test it by sequential universal inference (Wasserman, Ramdas and
Balakrishnan, 2020):

```
log E_t = Σ_s log D^(i→j)_θ̂(s-1) (o_s)  −  sup_θ Σ_s log D^(j→i)_θ (o_s)
```

with the numerator's parameters fitted on strictly past data. Dominating the
denominator by the likelihood at the true null parameter makes `E` an
e-process, so crossing `1/alpha` certifies the arrowhead at any stopping time.

Both factorisations are fitted with generalised-Gaussian noise (shape `kappa`
spanning Laplace through Gaussian to near-uniform), each as two regressions:

```
D^(j→i):   x_j = a'z + u ,   x_i = β x_j + g'z + e
D^(i→j):   x_i = b'z + v ,   x_j = δ x_i + h'z + w
```

**Under Gaussian noise the two are observationally indistinguishable, the ratio
does not grow, and no certificate is issued.** The procedure abstains exactly
where the direction is not identified rather than guessing — verified in
`tests/test_certcd.py`.

Two implementation constraints are load-bearing, and both were bugs before they
were features:

* The numerator's scored observations and the denominator's maximised
  likelihood must range over the **same** index set. A denominator covering
  extra observations subtracts their log-density with no matching numerator
  term, adding a constant of order (extra observations) to `log E`. In testing
  this certified *both* directions of every pair at once, including under
  Gaussian noise.
* The conditioning set is **frozen** when a pair's adjacency is certified, and
  the direction e-process accumulates only from that point. A conditioning set
  that kept changing would change the hypothesis under test, and Ville's
  inequality bounds a fixed one.

**Assumption.** Validity here requires the noise to lie in the fitted family.
That is a real parametric assumption, unlike the adjacency certificate's, and
is checked empirically against noise inside and outside the family.

---

## Guarantee

For **any** stopping time `tau`, with `alpha = alpha_A + alpha_D`:

```
P( ∃ tau :  some certified edge is absent from G*,
            or some certified arrowhead is wrong )   ≤   alpha
```

*Proof.* Each certificate is an e-process for a null that holds whenever the
asserted feature is absent, so by Ville each is falsely issued with probability
at most its own level, uniformly in time. The families — `C(d,2)` adjacency
nulls and `d(d-1)` direction nulls — are fixed before any data arrive, so a
union bound gives `alpha_A + alpha_D`. Which certificates the algorithm chooses
to evaluate never enters the bound. ∎

No faithfulness, no strong faithfulness, no equivalence region.

### What the output does not say

The absence of an edge means **undecided**, not "certified absent". Early in a
run the graph is empty and fills in as evidence arrives. A pair may stay
undecided forever if adjacency-faithfulness fails for it — which is exactly the
situation where every other constraint-based method returns a confident wrong
answer. This is the honest cost of the rule, not a limitation to be hidden:
CERT-CD cannot tell you an edge is absent, and does not pretend to.

---

## What is new here, and what is not

Being precise about this matters, because most of the primitives are standard.

**Not new.**

* The minimum of e-values is an e-value for a union null — standard
  (Vovk and Wang, 2021; Ramdas and Wang, 2025).
* Aggregating CI tests over conditioning sets into an edge-level statistic —
  **PC-p** (Strobl, Spirtes and Visweswaran, 2019) upper-bounds an edge's
  p-value by the **maximum** p-value over conditioning sets, of which
  `min_S E^(S)` is the exact e-value analogue. PC-p uses it for fixed-sample
  FDR control.
* Adjacency as the object being tested — **DAT** (Amin and Wilson, 2024)
  replaces the exponential set of tests with a differentiable relaxation
  solved by neural networks.
* Asymmetric edge-error control — **ε-CUT** (Shaska and Mitra, 2025) minimises
  one edge-error type subject to a tolerance on the other, with finite-sample
  false-positive guarantees.
* Residual-independence as a direction criterion — DirectLiNGAM (Shimizu
  et al., 2011).
* Universal inference; safe testing; sequential CI tests (He and Sutherland,
  ICML 2026, is the current SOTA single-hypothesis sequential CI test and is a
  drop-in replacement for `E^(S)` in nonparametric settings).

**New, as far as a targeted search of the 2024–2026 literature found.**

1. **A certified orientation.** No constraint-based method controls orientation
   error at all, and no functional-causal-discovery method gives a
   time-uniform certificate for a direction. The direction e-process does, and
   it abstains under Gaussianity rather than guessing.
2. **The certificate-only design rule and the algorithm it forces** — a
   monotone-growing graph in which every edge and every arrowhead is backed by
   an e-process, undecided pairs are explicit, and the whole output is
   anytime-valid under one budget.
3. **Making the edge-level aggregate an e-process rather than a p-value**,
   which converts PC-p's fixed-sample FDR statement into a time-uniform
   guarantee, and the accompanying observation that the resulting validity
   argument needs only the Markov condition, with multiplicity over pairs
   rather than (pair, conditioning set) triples.

If a referee locates an existing anytime-valid orientation certificate, claim 1
collapses and the contribution reduces to 2 and 3. The related-work section
should be written to survive that.

**Citations above were identified by web search and have not been verified
against publisher records.** Author lists, venues, years and numbering need
checking before submission.

---

## Empirical summary

From `experiments/exp6_certcd.py`, `d = 5`, Laplace noise, `alpha = 0.1`,
monitored throughout the run:

| stratum | CERT-CD false edge | CERT-CD wrong arrow | SPRINT-CD lost edge | PC lost edge |
|---|---|---|---|---|
| `delta`-strong faithfulness holds (n=40) | 0.000 | 0.000 | 0.000 | 0.000 |
| `delta`-strong faithfulness fails (n=20) | 0.050 | 0.000 | 0.600 | 0.800 |

The second row is the whole argument. Where the premise fails, CERT-CD stays
inside its budget of 0.1 (95% CI [0.009, 0.236]) because it never accepts a
null; SPRINT-CD loses a true edge in 60% of runs and PC in 80%, because both
decide absence by failing to reject. The premise itself held in 77% of the 87
DAGs generated here.

On premise-satisfying instances at `n = 3000`: all true edges certified, 96% of
them also oriented — against a **CPDAG orientation ceiling of 33%**, the
fraction a Markov equivalence class can orient at all on these graphs. That gap
is the point of the direction certificate: a chain has no v-structure and is
unorientable in principle by any constraint-based method.
