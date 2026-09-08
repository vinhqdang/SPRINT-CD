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
| Fisher-z, monitored | 0.337 |
| Fisher-z, fixed `n` | 0.070 |
| **SPRINT-CD e-process** | **0.030** |

Power is not sacrificed: detection rate 0.987 against a moderate alternative.

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

`delta`-strong faithfulness is a genuine restriction, not a formality. In
random 7-variable graphs with one latent variable marginalised out, only about
a quarter of instances satisfy it at `delta = 0.15`. The experiments measure
this rather than assume it, and stratify results accordingly.

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

* **Whole-graph control.** Over a monitored run, SPRINT-CD ever loses a true
  edge with probability 0.04 against a budget of 0.1; PC re-run at each sample
  size does so with probability 0.36. The cost is extra edges early, which
  decay to zero as data accumulate.
* **Adaptive stopping.** SPRINT-CD halts at a median `n` of about 2100, at an
  accuracy fixed-sample PC does not reach until roughly twice that much data.
* **Latent confounders.** Where `delta`-strong faithfulness holds, no true
  adjacency was lost in 20 runs (budget 0.1); where it fails, 80% of runs lose
  one — as the theory says they may.
* **E-ICP.** Monitoring multiplies ICP's rejection rate for the true parent set
  by about 4×; E-ICP's stays at zero while recovering every true parent.

---

## Tests

```bash
python -m pytest
```

71 tests. The substantive ones are calibration checks: all three e-process
constructions are verified against Ville's inequality by simulation rather
than assumed valid, and the FCI orientation rules are checked against
hand-derived PAGs.

---

## Citation

Dang, Q.-V. *Anytime-valid constraint-based causal discovery.* Working paper,
British University Vietnam, 2026.

## Licence

MIT — see [LICENSE](LICENSE).
