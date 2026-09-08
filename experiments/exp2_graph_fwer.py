"""Experiment 2: whole-graph error control while the estimate is monitored.

Lifting a single-test guarantee to a whole graph is the substantive step in
SPRINT-CD, so this experiment scores the *graph*, not individual queries, and
scores it over the entire trajectory rather than at one sample size.

The comparison with textbook PC is deliberately structural rather than a
like-for-like contest.  PC removes an edge when it *fails to reject*
independence, so at small sample sizes it discards true edges purely for want
of power, and no choice of ``alpha`` repairs that.  SPRINT-CD removes an edge
only on positive evidence that the association lies inside the equivalence
region, which is why its missing-edge count can be controlled uniformly in
time.  The price is the mirror image: its estimate begins dense and thins out,
so it carries extra edges early.  Both sides of that trade are plotted.
"""

from __future__ import annotations

import argparse
import sys, pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from common import PALETTE, RESULTS, save_json, setup_matplotlib, wilson_interval

from sprint_cd.baselines import pc_fixed_sample
from sprint_cd.simulate import random_dag, skeleton_errors
from sprint_cd.sprint_cd import SprintCD, SprintCDConfig


def run(reps, d, edge_prob, nmax, batch, alpha, rope, seed):
    rng = np.random.default_rng(seed)
    grid = list(range(batch, nmax + 1, batch))

    miss_sprint = np.zeros((reps, len(grid)))
    extra_sprint = np.zeros((reps, len(grid)))
    miss_pc = np.zeros((reps, len(grid)))
    extra_pc = np.zeros((reps, len(grid)))
    ever_missing_sprint = 0
    ever_missing_pc = 0

    warmup = 50
    for r in range(reps):
        sem = random_dag(d, edge_prob, rng, coef_low=0.5, coef_high=1.2)
        truth = sem.true_cpdag()
        X = sem.sample(nmax + warmup, rng)

        algo = SprintCD(d, SprintCDConfig(alpha=alpha, max_order=2, rope=rope,
                                          prior_scale=0.5, warmup=warmup))
        algo.warm_up(X[:warmup])
        stream = X[warmup:]

        bad_s = bad_p = False
        for gi, n in enumerate(grid):
            algo.update(stream[n - batch:n])
            es = skeleton_errors(algo.graph, truth)
            miss_sprint[r, gi], extra_sprint[r, gi] = es["missing"], es["extra"]
            bad_s |= es["missing"] > 0

            pc = pc_fixed_sample(stream[:n], alpha=alpha, max_order=2)
            ep = skeleton_errors(pc, truth)
            miss_pc[r, gi], extra_pc[r, gi] = ep["missing"], ep["extra"]
            bad_p |= ep["missing"] > 0

        ever_missing_sprint += bad_s
        ever_missing_pc += bad_p

    return {
        "grid": grid,
        "miss_sprint": miss_sprint.mean(axis=0), "extra_sprint": extra_sprint.mean(axis=0),
        "miss_pc": miss_pc.mean(axis=0), "extra_pc": extra_pc.mean(axis=0),
        "ever_missing_sprint": ever_missing_sprint, "ever_missing_pc": ever_missing_pc,
        "reps": reps,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=120)
    ap.add_argument("--d", type=int, default=6)
    ap.add_argument("--edge-prob", type=float, default=0.3)
    ap.add_argument("--nmax", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--rope", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.reps, args.nmax = 25, 1000

    res = run(args.reps, args.d, args.edge_prob, args.nmax, args.batch,
              args.alpha, args.rope, args.seed)

    lo_s, hi_s = wilson_interval(res["ever_missing_sprint"], res["reps"])
    lo_p, hi_p = wilson_interval(res["ever_missing_pc"], res["reps"])
    rate_s = res["ever_missing_sprint"] / res["reps"]
    rate_p = res["ever_missing_pc"] / res["reps"]

    print(f"\nExperiment 2 -- whole-graph error over a monitored run "
          f"({res['reps']} random DAGs, d={args.d}, alpha={args.alpha})\n")
    print("P(any true edge ever absent from the estimate, at any monitoring point):")
    print(f"  SPRINT-CD          {rate_s:.3f}   95% CI [{lo_s:.3f}, {hi_s:.3f}]"
          f"   budget = {args.alpha}")
    print(f"  PC re-run at each n {rate_p:.3f}   95% CI [{lo_p:.3f}, {hi_p:.3f}]"
          f"   (no such guarantee)")
    print(f"\nFinal-sample-size averages at n={res['grid'][-1]}:")
    print(f"  SPRINT-CD   missing {res['miss_sprint'][-1]:.2f}   extra {res['extra_sprint'][-1]:.2f}")
    print(f"  PC          missing {res['miss_pc'][-1]:.2f}   extra {res['extra_pc'][-1]:.2f}\n")

    save_json("exp2_graph_fwer", {"config": vars(args), **{
        k: (v if not isinstance(v, np.ndarray) else v.tolist()) for k, v in res.items()}})

    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharex=True)
    g = res["grid"]
    axes[0].plot(g, res["miss_sprint"], "-o", ms=3, color=PALETTE["sprint"], label="SPRINT-CD")
    axes[0].plot(g, res["miss_pc"], "-s", ms=3, color=PALETTE["naive"], label="PC at each $n$")
    axes[0].set_ylabel("missing true edges (mean)")
    axes[0].set_title("Edges wrongly removed\n(controlled uniformly in time)")
    axes[0].legend(fontsize=8)

    axes[1].plot(g, res["extra_sprint"], "-o", ms=3, color=PALETTE["sprint"], label="SPRINT-CD")
    axes[1].plot(g, res["extra_pc"], "-s", ms=3, color=PALETTE["naive"], label="PC at each $n$")
    axes[1].set_ylabel("extra edges (mean)")
    axes[1].set_title("Edges not yet removed\n(the price of the one-sided guarantee)")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xlabel("sample size $n$")
    fig.tight_layout()
    out = RESULTS / "exp2_graph_fwer.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
