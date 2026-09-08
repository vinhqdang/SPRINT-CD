"""Experiment 4: recovery under latent confounding.

Random DAGs are generated over ``d`` variables and a subset is hidden, so the
observed distribution carries confounding that no causally-sufficient method
can represent.  Ground truth is the oracle PAG obtained by running the same
orientation rules on exact d-separation answers, which is the sharpest object
any constraint-based algorithm could hope to return.

Two things are measured: how the estimate approaches that target as data
accumulate, and whether the anytime guarantee survives the extra machinery
(Possible-D-SEP pruning and the PAG orientation rules) that FCI layers on top
of the skeleton search.
"""

from __future__ import annotations

import argparse
import sys, pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from common import PALETTE, RESULTS, save_json, setup_matplotlib, wilson_interval

from sprint_cd.dsep import oracle_skeleton_and_sepsets
from sprint_cd.graph import ARROW
from sprint_cd.simulate import (random_dag, strong_faithfulness_margin,
                                structural_hamming_distance)
from sprint_cd.sprint_cd import SprintCDConfig
from sprint_cd.sprint_fci import SprintFCI, pag_from_skeleton


def _has_bidirected(g):
    return any(g.mark_at(i, j) == ARROW and g.mark_at(j, i) == ARROW
               for i, j in g.edges())


def run(reps, d, n_hidden, edge_prob, nmax, batch, alpha, rope, seed):
    """Evaluate on instances stratified by whether the guarantee's premise holds.

    The strong-faithfulness margin is a population quantity, so it can be
    computed before any data are simulated.  That makes it cheap to
    rejection-sample a well-powered sample of premise-satisfying instances --
    the only ones the theorem speaks about -- while still recording how rare
    they are and keeping a contrast group that violates the premise.
    """
    rng = np.random.default_rng(seed)
    warmup = 50
    grid = list(range(batch, nmax + 1, batch))
    target_ok = reps
    target_bad = max(reps // 2, 5)

    shd = []
    ever_missing_ok = ever_missing_bad = 0
    n_ok = n_bad = 0
    generated = 0
    n_generated_ok = 0
    margins = []
    truth_bidirected = found_bidirected = 0

    attempts = 0
    while (n_ok < target_ok or n_bad < target_bad) and attempts < reps * 400:
        attempts += 1
        sem = random_dag(d, edge_prob, rng, coef_low=0.6, coef_high=1.2)
        hidden = list(rng.choice(d, size=n_hidden, replace=False))
        observed = [v for v in range(d) if v not in hidden]
        # Require the latent to actually confound; otherwise this is not an
        # FCI problem and the instance says nothing about latent handling.
        if not any(sum(sem.adjacency[h, o] for o in observed) >= 2 for h in hidden):
            continue

        margin = strong_faithfulness_margin(sem, observed, max_order=2)
        generated += 1
        margins.append(margin)
        premise_holds = margin >= rope
        n_generated_ok += premise_holds

        if premise_holds and n_ok >= target_ok:
            continue
        if not premise_holds and n_bad >= target_bad:
            continue

        skel, seps = oracle_skeleton_and_sepsets(sem.adjacency, observed)
        truth = pag_from_skeleton(skel, seps)

        X = sem.sample(nmax + warmup, rng)[:, observed]
        algo = SprintFCI(len(observed),
                         SprintCDConfig(alpha=alpha, max_order=2, rope=rope,
                                        prior_scale=0.5, warmup=warmup))
        algo.warm_up(X[:warmup])
        stream = X[warmup:]

        bad = False
        row = []
        for n in grid:
            algo.update(stream[n - batch:n])
            row.append(structural_hamming_distance(algo.pag(), truth))
            for i, j in skel.edges():
                if not algo.graph.adjacent(i, j):
                    bad = True

        if premise_holds:
            n_ok += 1
            ever_missing_ok += bad
            shd.append(row)          # SHD curve reported on premise-holding runs
            if _has_bidirected(truth):
                truth_bidirected += 1
                found_bidirected += _has_bidirected(algo.pag())
        else:
            n_bad += 1
            ever_missing_bad += bad

    shd = np.array(shd) if shd else np.zeros((1, len(grid)))
    return {
        "grid": grid,
        "shd_mean": shd.mean(axis=0),
        "shd_se": shd.std(axis=0, ddof=1) / np.sqrt(max(shd.shape[0], 2)),
        "ever_missing_ok": ever_missing_ok, "n_ok": n_ok,
        "ever_missing_bad": ever_missing_bad, "n_bad": n_bad,
        "margins": margins,
        "premise_rate": n_generated_ok / max(generated, 1),
        "generated": generated,
        "reps": n_ok + n_bad,
        "truth_bidirected": truth_bidirected,
        "found_bidirected": found_bidirected,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=80)
    ap.add_argument("--d", type=int, default=7)
    ap.add_argument("--n-hidden", type=int, default=1)
    ap.add_argument("--edge-prob", type=float, default=0.35)
    ap.add_argument("--nmax", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=250)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--rope", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.reps, args.nmax = 20, 2000

    res = run(args.reps, args.d, args.n_hidden, args.edge_prob, args.nmax,
              args.batch, args.alpha, args.rope, args.seed)

    print(f"\nExperiment 4 -- latent confounding "
          f"({res['reps']} DAGs, d={args.d}, {args.n_hidden} hidden, alpha={args.alpha})\n")

    margins = np.array(res["margins"])
    print(f"delta-strong faithfulness at delta={args.rope}: holds in "
          f"{res['premise_rate']:.2f} of {res['generated']} randomly generated "
          f"instances.")
    print("  (Instances below are rejection-sampled into the two strata so that "
          "each is\n  adequately powered; the rate above is the unconditional one.)")
    print(f"  median margin min|rho_ij.S| over true adjacencies: "
          f"{np.median(margins):.3f}   10th pct: {np.percentile(margins, 10):.3f}")
    print("  Marginalising a latent variable out routinely drives some true "
          "adjacency's\n  partial correlation below any usable delta, which is a "
          "property of the\n  instance, not of the procedure.\n")

    print("P(any oracle adjacency ever absent, at any monitoring point):")
    if res["n_ok"]:
        lo, hi = wilson_interval(res["ever_missing_ok"], res["n_ok"])
        print(f"  premise holds     {res['ever_missing_ok'] / res['n_ok']:.3f}   "
              f"95% CI [{lo:.3f}, {hi:.3f}]   budget = {args.alpha}   "
              f"(n={res['n_ok']})  <-- the guarantee applies here")
    if res["n_bad"]:
        lo2, hi2 = wilson_interval(res["ever_missing_bad"], res["n_bad"])
        print(f"  premise violated  {res['ever_missing_bad'] / res['n_bad']:.3f}   "
              f"95% CI [{lo2:.3f}, {hi2:.3f}]   no guarantee     "
              f"(n={res['n_bad']})")
    print(f"\nSHD to the oracle PAG: {res['shd_mean'][0]:.2f} at n={res['grid'][0]}  ->  "
          f"{res['shd_mean'][-1]:.2f} at n={res['grid'][-1]}")
    if res["truth_bidirected"]:
        print(f"\nLatent signature: a bi-directed edge is present in "
              f"{res['truth_bidirected']} oracle PAGs; SPRINT-FCI recovers one in "
              f"{res['found_bidirected']} of them "
              f"({res['found_bidirected'] / res['truth_bidirected']:.2f}).")
    print()

    save_json("exp4_latent_confounders", {"config": vars(args), **{
        k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in res.items()}})

    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    ax = axes[0]
    g = res["grid"]
    ax.plot(g, res["shd_mean"], "-o", ms=3.5, color=PALETTE["sprint"], label="SPRINT-FCI")
    ax.fill_between(g, res["shd_mean"] - 2 * res["shd_se"],
                    res["shd_mean"] + 2 * res["shd_se"],
                    color=PALETTE["sprint"], alpha=0.18)
    ax.axhline(0, color=PALETTE["oracle"], ls="--", lw=1.2, label="oracle PAG")
    ax.set_xlabel("sample size $n$")
    ax.set_ylabel("SHD to oracle PAG")  # premise-satisfying instances
    ax.set_title(f"Recovery with {args.n_hidden} latent variable(s)")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.hist(margins, bins=25, color=PALETTE["fixed"], alpha=0.85)
    ax.axvline(args.rope, color=PALETTE["naive"], lw=1.6,
               label=rf"$\delta={args.rope}$")
    ax.set_xlabel(r"instance margin $\min_{ij,S}|\rho_{ij\cdot S}|$")
    ax.set_ylabel("count")
    ax.set_title("Is the premise of the guarantee met?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = RESULTS / "exp4_latent_confounders.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
