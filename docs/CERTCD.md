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

**No faithfulness assumption enters validity.** If `i` and `j` are non-adjacent
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

The literature splits cleanly here, and the gap is between the halves rather
than in either one.

*Constraint-based orientation tests have error control but cannot leave the
equivalence class.* PC-p formulates edge-specific hypothesis tests for
unshielded colliders and for Meek-rule orientations and controls their FDR
(Benjamini–Yekutieli). But an edge lying in no v-structure is unorientable in
principle by any such test: a chain is not orientable from conditional
independence alone.

*Functional methods leave the equivalence class but have no error control.*
LiNGAM, DirectLiNGAM and pairwise-LiNGAM pick a direction by comparing an
independence measure or a likelihood-ratio score; Ruiz, Madrid Padilla and Zhou
(arXiv:2202.01748) prove population-level identifiability of the topological
ordering from likelihood-ratio scores on residuals, with no finite-sample error
statement; Prakash, Xia and Erosheva (arXiv:2406.07787) add an explicit
*inconclusive* outcome to a bivariate direction test, with asymptotic
guarantees.

The certificate below occupies the gap: **error-controlled orientation beyond
the Markov equivalence class**, and time-uniform rather than fixed-sample.

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

## Related work, and what survives as new

Five of the closest papers were read in full. **One earlier claim did not
survive:** I had written that nothing in the constraint-based literature
certifies an orientation. PC-p does.

| paper | what it does | guarantee | beyond MEC? |
|---|---|---|---|
| Strobl, Spirtes & Visweswaran, *Estimating and Controlling the False Discovery Rate of the PC Algorithm Using Edge-Specific P-Values* (arXiv:1607.03975v2) | edge p-value = **max p over CI tests**; hypothesis tests for skeleton, unshielded colliders **and Meek-rule orientations** | fixed-sample FDR (Benjamini–Yekutieli); needs a **zero-Type-II-error assumption** unless *all* possible CI tests are run | no |
| Uehara, *Iterative Causal Discovery: Per-Edge Impossibility Certificates…* (arXiv:2605.27477v1) | per-edge `resolved_*` / `impossible_*` codes; tiers abstain when a precondition test rejects; expert-query protocol | **provenance**, not error: records *which theorem* licensed a direction. FDR on the skeleton only; Theorem 1 bounds expert interactions under an ideal oracle | via functional tiers |
| Prakash, Xia & Erosheva, *A Diagnostic Tool for Functional Causal Discovery* (arXiv:2406.07787v2) | test-based bivariate direction with four outcomes incl. **inconclusive**; CDDR diagnostic vs sample size | asymptotic consistency / normality | yes (LiNGAM) |
| Ruiz, Madrid Padilla & Zhou, *Sequentially learning the topological ordering…* (arXiv:2202.01748v2) | greedy ordering by **likelihood-ratio scores on residuals** — "sequential" in the sorting sense | **population-level identifiability only**; needs all noises from one scale-location family | yes |
| Ding & Zhang, *GaussDetect-LiNGAM* (preprint, 4 Dec 2025) | proves forward-noise Gaussianity ⟺ reverse-regression residual independence; kernel tests replace Gaussianity tests | none on the direction decision | yes |
| Amin & Wilson, *Scalable and Flexible Causal Discovery with an Efficient Test for Adjacency* (DAT; ICML 2024, PMLR 235) | adjacency as the tested object; replaces the exponential set of CI tests with a "provably equivalent" relaxed problem solved by two neural networks | **none** — the equivalence is between the exponential test and its relaxation, not an error bound; relies on faithfulness | n/a (skeleton) |
| Shaska & Mitra, *Causal Link Discovery with Unequal Edge Error Tolerance* (ε-CUT; arXiv:2507.21570v1) | Neyman–Pearson framing: minimise one edge-error type subject to a tolerance on the other; per-edge regression test with a finite-sample threshold | **finite-sample false-positive rate.** Thm 2: for every fixed *n*, P(false edge) ≤ ε, under an LSEM and with **no faithfulness needed**. Pointwise in *n*, not uniform over *n* | n/a (skeleton) |

The last of these is **support rather than competition**: it is the theorem
explaining why the direction certificate must abstain under Gaussian noise.

### Not new

* Minimum of e-values as an e-value for a union null (Vovk & Wang).
* Aggregating CI tests over conditioning sets into an edge-level statistic —
  PC-p's max-p is the exact p-value analogue of `min_S E^(S)`.
* Adjacency as the object being tested (DAT, Amin & Wilson).
* **One-sided, faithfulness-free, finite-sample control of false edges** —
  this is exactly ε-CUT's Theorem 2, and it was reached by the same route I
  use: a false-positive guarantee does not need faithfulness, only a correct
  null. Its guarantee is pointwise in *n*.
* Residual independence as a direction criterion (DirectLiNGAM).
* **Abstention and three-valued output** — Uehara's `impossible_*` codes,
  Prakash et al.'s *inconclusive* outcome, and FCI's undirected edges all
  decline to commit. This was overclaimed in an earlier draft.
* Universal inference; safe testing; sequential CI tests.

### New

1. **Error-controlled orientation beyond the Markov equivalence class.** The
   table's last column is the point: every row with error control is confined
   to the MEC, and every row that escapes the MEC has no error control on the
   direction. The direction certificate has both — 96% of edges oriented
   against a 33% CPDAG ceiling, with a time-uniform bound on wrong arrowheads.
2. **Time-uniformity.** Every comparable guarantee above is fixed-sample,
   asymptotic, population-level, or provenance-only. None survives optional
   stopping or continuous monitoring.
3. **Time-uniform edge control.** ε-CUT already gives one-sided,
   faithfulness-free, finite-sample false-edge control — *at each fixed n*.
   That is not the same as bounding `P(∃n : a false edge is ever certified)`,
   which is what a monitored or adaptively stopped run needs; Experiment 1
   measures precisely this gap for a fixed-sample CI test (0.056 at a fixed
   sample size, 0.346 under monitoring). The adjacency certificate is uniform
   over stopping times. Secondarily, minimising over the complete fixed family
   `S_k` performs all the CI tests PC-p's max-p bound requires, so PC-p's
   zero-Type-II-error assumption (needed for `N(A) ⊆ N̂(A)`) is not; and
   multiplicity runs over pairs rather than (pair, conditioning set) triples.
4. **Never asserting absence.** PC, FCI, PC-p, and Uehara's Stage-1 skeleton
   all assert non-adjacency. CERT-CD never does; an absent edge is undecided.

### Residual risk

Claim 1 is the load-bearing one. It would collapse if a method exists that
orients outside the equivalence class *and* bounds the probability of a wrong
arrowhead. Nothing among the seven papers read does: the two that control edge
errors rigorously (PC-p, ε-CUT) say nothing about arrowheads outside the
equivalence class, and the ones that orient outside it (LiNGAM family, Ruiz
et al., Prakash et al., Uehara) bound no error.

Claim 3 is now the *narrowest* of the four, since ε-CUT holds the same
one-sided, faithfulness-free ground at fixed sample sizes. Everything it adds
rests on the fixed-*n* versus uniform-over-*n* distinction, which is real but
should be argued explicitly rather than assumed obvious.

Still unread and potentially relevant: Csillag et al. (Prediction-Powered
E-Values, ICML 2025), which puts e-values inside PC but is fixed-sample; and
the multiplicity literature on e-value FWER (Hartog & Lei, 2025), which could
sharpen the union bound.

Bibliographic details in the table are verified against the PDFs. Citations
elsewhere in this document came from web search and are **not** yet verified.

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
