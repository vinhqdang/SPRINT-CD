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

**Direction.** Nothing in the constraint-based literature controls orientation
error; orientation is deterministic post-processing. Under a linear
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
| `docs/METHOD.md` | SPRINT-CD method notes |

---

## Honest positioning

Most primitives here are standard, and `docs/CERTCD.md` says so in detail:
min-of-e-values for union nulls (Vovk and Wang), aggregating CI tests over
conditioning sets into an edge statistic (**PC-p** uses the max *p*-value, of
which `min_S E^(S)` is the e-value analogue), adjacency as a test target (DAT),
asymmetric edge-error control (ε-CUT), residual independence for direction
(DirectLiNGAM), universal inference, safe testing.

What appears to be new is: **(1)** a certified, anytime-valid *orientation* —
no constraint-based method controls orientation error and no functional method
gives a time-uniform direction certificate; **(2)** the certificate-only rule
and the growing, abstaining algorithm it forces; **(3)** turning the
edge-level aggregate into an e-process, which makes PC-p's fixed-sample FDR
statement time-uniform and reduces its validity requirement to the Markov
condition alone.

Citations were identified by web search and **have not been verified against
publisher records**.

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

Findings for 1–5 (the SPRINT-CD line of work) are in `docs/METHOD.md`; the
headline there is that naive monitoring inflates Type-I error 7× (0.346 vs a
nominal 0.05) while the e-process holds at 0.024 — with a real power cost that
is documented rather than hidden.

## Tests

```bash
python -m pytest
```

99 tests. The load-bearing ones are calibration checks: every e-process is
verified against Ville's inequality by simulation, the adjacency certificate is
checked in a distribution where two directed paths cancel exactly (faithfulness
violated outright), and the direction certificate is checked to abstain under
Gaussian noise.

## Citation

Dang, Q.-V. *Certificate-only causal discovery.* Working paper, British
University Vietnam, 2026.

## Licence

MIT — see [LICENSE](LICENSE).
