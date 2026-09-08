# SPRINT-CD

**S**equential **P**C/FCI with a**N**ytime-valid **T**ests — constraint-based
causal discovery whose recovered graph carries time-uniform error control at
**any** data-dependent stopping time.

Prepared for the *Japanese Journal of Statistics and Data Science* special
issue on [Recent Advances in Causal Inference, Causal Discovery, and
Applications](https://link.springer.com/journal/42081/updates/27837542).

---

## The problem

A conditional-independence test is valid at the sample size it was computed
for. In a streaming setting the analyst recomputes it as data arrive and stops
when the answer looks settled — and that stopping rule voids the guarantee.
On the very queries a discovery algorithm issues, monitoring a null
partial-correlation test at nominal `alpha = 0.05` produces a Type-I error
rate of about **34%**.

SPRINT-CD replaces every conditional-independence query with an **e-process**,
so Ville's inequality bounds the error uniformly in time. The estimate may be
inspected after every batch, and the run stopped whenever the analyst likes,
without invalidating anything.

| procedure at `alpha = 0.05` | Type-I error under monitoring |
|---|---|
| Fisher-z, monitored | 0.346 |
| Fisher-z, fixed `n` | 0.056 |
| **SPRINT-CD e-process** | **0.024** |

2000 replications, monitored every 10 observations up to `n = 1000`.

**This is not free.** Against a fixed-sample Fisher-z test at the same `n` —
one that may be evaluated only once — the e-process detects less:

| true `beta` | 0.04 | 0.06 | 0.08 | 0.10 | 0.15 |
|---|---|---|---|---|---|
| Fisher-z, fixed `n` | 0.21 | 0.47 | 0.72 | 0.91 | 1.00 |
| SPRINT-CD e-process | 0.05 | 0.16 | 0.32 | 0.58 | 0.96 |

The gap closes as the effect grows, and it is the price of a guarantee that
survives monitoring and optional stopping. If the sample size can genuinely be
fixed in advance and the analysis run once, a fixed-sample test is the more
powerful choice and should be preferred.

---

## What is here

| module | contents |
|---|---|
| `sprint_cd/eprocess/safe_linear.py` | Exact right-Haar Bayes-factor e-process for linear-Gaussian partial correlation (scalar and block), plus the closed-form confidence sequence obtained by inverting it |
| `sprint_cd/eprocess/universal.py` | Sequential universal-inference e-processes (Gaussian and categorical) — assumption-light alternatives |
| `sprint_cd/multiplicity.py` | Fixed-family budgeting and e-BH (valid under arbitrary dependence) |
| `sprint_cd/sprint_cd.py` | **SPRINT-CD**: streaming, anytime-valid PC returning a CPDAG |
| `sprint_cd/sprint_fci.py` | **SPRINT-FCI**: the same under latent confounding, returning a PAG |
| `sprint_cd/e_icp.py` | **E-ICP**: anytime-valid Invariant Causal Prediction |
| `sprint_cd/dsep.py` | d-separation and oracle PAG construction, for evaluation |
| `docs/METHOD.md` | Constructions, proofs, and an explicit account of the assumptions |

---

## Quick start

```bash
pip install -e ".[experiments,dev]"
```

```python
import numpy as np
from sprint_cd import SprintCD, SprintCDConfig, random_dag

rng = np.random.default_rng(0)
sem = random_dag(6, edge_prob=0.3, rng=rng)
data = sem.sample(5000, rng)

algo = SprintCD(d=6, config=SprintCDConfig(alpha=0.05, rope=0.15))
algo.warm_up(data[:50])                 # sets prior scales; excluded from every e-process

for start in range(50, len(data), 100):
    algo.update(data[start:start + 100])
    if algo.resolved:                   # a data-dependent stopping rule — and that is fine
        break

print(algo.cpdag())
print(f"stopped at n = {algo.n}")
```

Latent confounders and streaming environments:

```python
from sprint_cd import run_sprint_fci, EICP, EICPConfig

pag, algo = run_sprint_fci(observed_data)      # PAG; bi-directed edges flag latents

model = EICP(d=3, n_env=3, config=EICPConfig(alpha=0.05))
model.warm_up(X[:40], y[:40], env[:40])
model.update(X[40:], y[40:], env[40:])
model.estimate()                                # subset of the true parents, at any time
```

---

## How it works

Three ideas, developed in `docs/METHOD.md`.

**1. An e-value that is a function of the Gram matrix.** For a linear-Gaussian
stream, the right-Haar Bayes factor for `X_i indep X_j | X_S` depends on the
data only through Schur complements of the running cross-product matrix. A
single `O(d^2)` sufficient statistic therefore serves the whole hypothesis
family; nothing is stored per hypothesis.

**2. A confidence sequence to *accept* independence.** E-processes accumulate
evidence *against* a null, so they cannot remove an edge on their own.
Inverting the e-value gives a closed-form anytime-valid interval for the
coefficient, and an edge is removed once that interval fits inside a region of
practical equivalence. The region is specified on the partial-correlation
scale — a raw coefficient-scale region is not scale-free across conditioning
sets, and using one makes a *stronger* SEM coefficient slow deletion down.

**3. A budget fixed before the data arrive.** Which hypothesis the algorithm
examines next depends on the graph so far, hence on the data. Budgeting over
the whole *potential* family `{(i,j,S) : |S| <= k}` makes the union bound
independent of what was actually examined, so adaptive enumeration of
conditioning sets costs nothing.

---

## Guarantee, and what it excludes

For **any** stopping time `tau`, under a linear-Gaussian SEM and
`delta`-strong faithfulness:

```
P( exists tau :  a true edge is absent from G_tau
                 or a true independence is certified dependent )  <=  alpha
```

The guarantee is **one-sided on the skeleton**. Early in a run nothing has
been certified and the output is a dense superset of the truth; what is
controlled uniformly in time is the *removal* of true edges, the error that is
irreversible in streaming. Extra edges are the price.

`delta`-strong faithfulness is a genuine restriction, not a formality. At
`delta = 0.15` it holds in about 65% of random 6-variable DAGs, and in only
19% once a latent variable is marginalised out — the median smallest partial
correlation over true adjacencies there is about 0.03. Removing an edge whose
association genuinely lies inside the equivalence region is correct behaviour
by construction, not a calibration failure, so Experiments 2 and 4 **stratify
instances by whether the premise holds** and report both strata rather than
pooling them into a misleading average.

---

## Experiments

```bash
python experiments/run_all.py            # full run
python experiments/run_all.py --quick    # fast smoke run
```

| experiment | question |
|---|---|
| 1 — Type-I calibration | What does monitoring cost a fixed-sample test, and does the e-process repair it without losing power? |
| 2 — whole-graph error | Is the *graph* protected across a monitored run, and what are the extra edges that buys? |
| 3 — adaptive stopping | Does stopping on a data-dependent rule save data, and what does `delta` control? |
| 4 — latent confounders | Does the guarantee survive Possible-D-SEP and PAG orientation, and how often does its premise hold? |
| 5 — E-ICP | Does the ICP subset guarantee survive optional continuation? |

Headline findings, in addition to the calibration table above:

* **Whole-graph control.** Over a monitored run on instances satisfying the
  premise, SPRINT-CD ever loses a true edge with probability 0.008 against a
  budget of 0.1 (n = 120); PC re-run at each sample size does so with
  probability 0.133 on the *same* instances. By `n = 4000` SPRINT-CD averages
  0.01 missing and 0.01 extra edges, against PC's 0.00 missing and 0.57 extra.
  Where the premise fails, SPRINT-CD loses an edge in 75% of runs — outside
  what the theorem covers, and reported as such.
* **Adaptive stopping.** SPRINT-CD halts at a median `n` of 1900 with mean SHD
  0.83 (SE 0.22). Fixed-sample PC plateaus at 0.90 and does not reach that
  accuracy anywhere up to `n = 20000`.
* **Latent confounders.** Where the premise holds, no true adjacency was lost
  in 80 runs (budget 0.1; 95% CI [0.000, 0.046]), and SHD to the oracle PAG
  falls from 12.7 to 0.35 by `n = 4000`. The bi-directed edge marking the
  latent was recovered in 7 of 7 cases.
* **E-ICP.** Monitoring multiplies ICP's rejection rate for the true parent set
  by about 12× (0.007 → 0.080); E-ICP's stays at 0.000 across 300 runs while
  recovering every true parent.

---

## Tests

```bash
python -m pytest
```

74 tests. The substantive ones are calibration checks: all three e-process
constructions are verified against Ville's inequality by simulation rather
than assumed valid, and the FCI orientation rules are checked against
hand-derived PAGs.

---

## Citation

Dang, Q.-V. *Anytime-valid constraint-based causal discovery.* Working paper,
British University Vietnam, 2026.

## Licence

MIT — see [LICENSE](LICENSE).
