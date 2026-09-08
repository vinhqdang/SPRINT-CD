# CERT-CD — certificate-only causal discovery

Causal discovery in which **every edge and every arrowhead is backed by a
certificate**, valid at any data-dependent stopping time, and pairs the data
cannot resolve are reported as undecided rather than guessed.

Prepared for the *Japanese Journal of Statistics and Data Science* special
issue on [Recent Advances in Causal Inference, Causal Discovery, and
Applications](https://link.springer.com/journal/42081/updates/27837542).

---

## The rule

> **Assert a feature of the graph only by REJECTING a null that is true
> whenever that feature is absent. Never assert anything by failing to reject.**

Every constraint-based algorithm since PC breaks this rule at its first step:
it *removes* an edge when a conditional-independence test fails to reject, so
absence of evidence becomes evidence of absence. That is why their
finite-sample guarantees need faithfulness. Our own measurements put
`delta`-strong faithfulness at 65% of random 6-variable DAGs and 19% of
7-variable graphs with one latent variable — and where it fails, the
delete-on-non-rejection approach discards true edges in about three quarters of
runs.

Obeying the rule inverts the algorithm:

| | PC / SPRINT-CD | **CERT-CD** |
|---|---|---|
| starts from | complete graph | empty graph |
| edges are | removed on weak evidence | added on strong evidence |
| graph over time | shrinks | grows |
| controlled error | missing edges | false edges **and wrong arrowheads** |
| faithfulness | needed for validity | needed only for power |
| orientation error | uncontrolled | **certified** |
| undecided pairs | forced to a decision | reported as undecided |

---

## Two certificates

**Adjacency.** For a pair, the null "there exists a conditioning set that
separates them" is true whenever the pair is non-adjacent, so rejecting it
certifies an edge. It is a *union* null, and the minimum of the per-set
e-processes is an e-process for it:

```
A_t(i,j) = min over S of E^(S)_t(i,j)
```

because `A_t ≤ E^(S*)_t` for the true separator `S*`. **Validity needs only the
Markov condition** — `pa(i)` always separates a non-adjacent pair.
Faithfulness buys only power. Multiplicity runs over `C(d,2)` pairs, not over
(pair, conditioning set) triples.

**Direction.** Constraint-based orientation error *can* be controlled — PC-p
tests colliders and Meek-rule orientations under FDR — but only inside the
Markov equivalence class, so an edge in no v-structure stays unorientable.
Functional methods (LiNGAM and descendants) escape that class but decide
directions by comparing scores, with no error control. Under a linear
non-Gaussian model the two orientations induce different joint densities and
exactly one is correct, so "the direction is `j -> i`" is a genuine null.
A sequential universal-inference e-process against it certifies `i -> j`:

```
log E_t = Σ log D^(i→j)_θ̂(past) (o_s)  −  sup_θ Σ log D^(j→i)_θ (o_s)
```

Under Gaussian noise the two factorisations are indistinguishable, the ratio
does not grow, and **no certificate is issued** — the method abstains exactly
where direction is unidentified.

**Guarantee.** For any stopping time, with `alpha = alpha_A + alpha_D`:

```
P( ∃ tau :  a certified edge is absent from G*,
            or a certified arrowhead is wrong )   ≤   alpha
```

No faithfulness, no strong faithfulness, no equivalence region.

**What it does not say.** An absent edge means *undecided*, not "certified
absent". CERT-CD cannot tell you an edge is missing, and does not pretend to.

---

## Results

`d = 5`, Laplace noise, `alpha = 0.1`, estimate monitored throughout the run:

| stratum | CERT-CD false edge | CERT-CD wrong arrow | SPRINT-CD lost edge | PC lost edge |
|---|---|---|---|---|
| `delta`-strong faithfulness **holds** (n=40) | 0.000 | 0.000 | 0.000 | 0.000 |
| `delta`-strong faithfulness **fails** (n=20) | **0.050** | **0.000** | 0.600 | 0.800 |

Where the premise fails — the regime that breaks delete-on-non-rejection —
CERT-CD stays inside its budget of 0.1 (95% CI [0.009, 0.236]) while SPRINT-CD
loses a true edge in 60% of runs and PC in 80%. Near-unfaithful instances cost
CERT-CD *power* — the affected pairs stay undecided — rather than *validity*.

On premise-satisfying instances every true edge is certified and **96% are also
oriented, against a CPDAG orientation ceiling of 33%** (the fraction a Markov
equivalence class can orient at all). A chain has no v-structure and is
unorientable in principle by any constraint-based method; direction
certificates are not bound by that ceiling.

---

## Quick start

```bash
pip install -e ".[experiments,dev]"
```

```python
import numpy as np
from sprint_cd import CertCD, CertCDConfig, random_dag

rng = np.random.default_rng(0)
sem = random_dag(5, edge_prob=0.35, rng=rng)
data = sem.sample(4000, rng, noise="laplace")     # non-Gaussian: direction identified

algo = CertCD(d=5, config=CertCDConfig(alpha=0.05))
algo.warm_up(data[:50])
for start in range(50, len(data), 250):
    algo.update(data[start:start + 250])           # inspect after every batch: free

print(algo.graph())                 # only certified edges; circles = direction undecided
print("certified edges :", algo.certified_edges())
print("certified arrows:", algo.certified_arrows())
print("undecided pairs :", algo.undecided_pairs())   # NOT 'certified absent'
```

---

## What is here

| module | contents |
|---|---|
| `sprint_cd/certificates.py` | The two certificate primitives: the adjacency e-process `min_S E^(S)`, and the direction e-process via sequential universal inference |
| `sprint_cd/certcd.py` | **CERT-CD** — the certificate-only algorithm |
| `sprint_cd/eprocess/safe_linear.py` | Exact right-Haar Bayes-factor e-process for linear-Gaussian partial correlation, plus its inverted confidence sequence |
| `sprint_cd/eprocess/universal.py` | Sequential universal-inference e-processes (Gaussian, categorical) |
| `sprint_cd/sprint_cd.py` | **SPRINT-CD** — anytime-valid PC (delete-on-certificate). The baseline CERT-CD is measured against |
| `sprint_cd/sprint_fci.py` | SPRINT-FCI — the same under latent confounding, returning a PAG |
| `sprint_cd/e_icp.py` | E-ICP — anytime-valid Invariant Causal Prediction |
| `docs/CERTCD.md` | **CERT-CD method notes, including an explicit account of what is and is not new** |
| `sprint_cd/lingam.py` | DirectLiNGAM (pairwise likelihood ratios, causal ordering) — the orientation baseline |
| `docs/METHOD.md` | SPRINT-CD method notes |
| `paper/jjsds/` | The manuscript, in Springer's `sn-jnl` class |

---

## Honest positioning

Most primitives here are standard, and `docs/CERTCD.md` says so in detail:
min-of-e-values for union nulls (Vovk and Wang), aggregating CI tests over
conditioning sets into an edge statistic (**PC-p** uses the max *p*-value, of
which `min_S E^(S)` is the e-value analogue), adjacency as a test target
(**DAT**), residual independence for direction (DirectLiNGAM), universal
inference, safe testing — plus two that cost me claims outright:
**one-sided faithfulness-free false-edge control at finite samples** is exactly
ε-CUT's Theorem 2, and **sequential, anytime-valid e-values inside PC** is
already done by **Csillag et al.**

Nine of the closest papers have now been read in full, and two of my earlier
claims did not survive: **PC-p already controls orientation error**, via
FDR-controlled hypothesis tests for colliders and Meek-rule orientations. What
survives, stated narrowly:

1. **Error-controlled orientation *beyond* the Markov equivalence class.**
   Constraint-based orientation tests have error control but are confined to
   the equivalence class; functional methods (LiNGAM, DirectLiNGAM,
   pairwise-LiNGAM, the sequential-ordering score of Ruiz et al.) escape it but
   have no error control on the direction decision. The direction certificate
   has both — 96% of edges oriented against a 33% CPDAG ceiling, with a
   time-uniform bound on wrong arrowheads.
2. **A proved, not assumed, whole-graph guarantee.** Csillag et al. already
   put *sequential, anytime-valid* e-values inside PC — so anytime-validity per
   CI test is not new. But their graph-level validity is an explicit
   assumption (their Assumption 2.5), with the multiple-comparison concern
   noted and unresolved. The fixed-family union bound here proves it, and
   covers orientation as well as edges.
3. **Time-uniform edge control.** ε-CUT already gives one-sided,
   faithfulness-free false-edge control — *at each fixed n*. Bounding
   `P(∃n : a false edge is ever certified)` is a different statement, and the
   one a monitored run needs; Experiment 1 measures the gap for a fixed-sample
   CI test (0.056 at fixed *n*, 0.346 monitored). Secondarily, minimising over
   the complete fixed family runs all the CI tests PC-p's max-p bound
   requires, so its zero-Type-II-error assumption is not needed.
4. **Never asserting absence.** PC, FCI, PC-p and Uehara's Stage-1 skeleton all
   assert non-adjacency. CERT-CD does not.

Abstention itself is **not** new, and neither is faithfulness-free one-sided
edge control — see `docs/CERTCD.md` for the full accounting, including a
residual-risk note on what would still collapse claim 1.

Bibliographic details for the nine papers read in full are verified; the
remaining citations were identified by web search and **have not been checked
against publisher records**.

---

## Experiments

```bash
python experiments/run_all.py            # full run
python experiments/run_all.py --quick    # smoke run, writes to results/quick/
```

| experiment | question |
|---|---|
| 1 | What does monitoring cost a fixed-sample CI test, and does an e-process repair it? |
| 2 | Is the whole *graph* protected across a monitored run? |
| 3 | Does adaptive stopping save data, and what does `delta` control? |
| 4 | Does the guarantee survive latent confounders, and how often does its premise hold? |
| 5 | Does the ICP subset guarantee survive optional continuation? |
| 6 | **CERT-CD vs delete-on-non-rejection: validity when faithfulness fails, and certified orientation** |
| 7 | Certified orientation against DirectLiNGAM, under Laplace and under Gaussian noise |
| 8 | Is the per-set primitive calibrated under noise laws outside its model? |
| 9 | What do the certificates do when their assumptions fail? |
| 10 | **Real data: the Sachs et al. (2005) protein-signalling sample** |

Findings for 1–5 (the SPRINT-CD line of work) are in `docs/METHOD.md`; the
headline there is that naive monitoring inflates Type-I error 7× (0.346 vs a
nominal 0.05) while the e-process holds at 0.024 — with a real power cost that
is documented rather than hidden.

## Tests

```bash
python -m pytest
```

109 tests. The load-bearing ones are calibration checks: every e-process is
verified against Ville's inequality by simulation, the adjacency certificate is
checked in a distribution where two directed paths cancel exactly (faithfulness
violated outright), and the direction certificate is checked to abstain under
Gaussian noise.

## The paper

The manuscript is in `paper/jjsds/` (Springer `sn-jnl` class, 45 pp.):

```bash
cd paper/jjsds
pdflatex certcd_jjsds && bibtex certcd_jjsds && pdflatex certcd_jjsds && pdflatex certcd_jjsds
```

`paper/certcd.tex` is the earlier short version and is superseded.

## Citation

Dang, Q.-V. *Certificate-only causal discovery: anytime-valid adjacency and
orientation.* Working paper, British University Vietnam, 2026.

## Licence

MIT — see [LICENSE](LICENSE).
