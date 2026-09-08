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
from common import PALETTE, figure_path, save_json, setup_matplotlib, wilson_interval

from sprint_cd.baselines import pc_fixed_sample
from sprint_cd.simulate import (random_dag, skeleton_errors,
                                strong_faithfulness_margin)
from sprint_cd.sprint_cd import SprintCD, SprintCDConfig


def run(reps, d, edge_prob, nmax, batch, alpha, rope, seed):
    """Score the graph over a monitored run, stratified by the guarantee's premise.

    The theorem bounds false deletions only for instances satisfying
    ``delta``-strong faithfulness.  A true edge whose partial correlation lies
    inside the equivalence region is *supposed* to be removed, so pooling both
    kinds of instance measures how often random graphs are near-unfaithful, not
    whether the procedure is calibrated.  The margin is a population quantity,
    so instances can be sorted into strata before any data are simulated.
    """
    rng = np.random.default_rng(seed)
    grid = list(range(batch, nmax + 1, batch))
    warmup = 50
    target_ok, target_bad = reps, max(reps // 3, 5)

    miss_sprint, extra_sprint = [], []
    miss_pc, extra_pc = [], []
    ever_missing_ok = ever_missing_bad = 0
    ever_missing_pc = 0
    n_ok = n_bad = 0
    generated = n_generated_ok = 0
    margins = []

    attempts = 0
    while (n_ok < target_ok or n_bad < target_bad) and attempts < reps * 200:
        attempts += 1
        sem = random_dag(d, edge_prob, rng, coef_low=0.5, coef_high=1.2)
        margin = strong_faithfulness_margin(sem, max_order=2)
        generated += 1
        margins.append(margin)
        premise_holds = margin >= rope
        n_generated_ok += premise_holds
        if premise_holds and n_ok >= target_ok:
            continue
        if not premise_holds and n_bad >= target_bad:
            continue

        truth = sem.true_cpdag()
        X = sem.sample(nmax + warmup, rng)
        algo = SprintCD(d, SprintCDConfig(alpha=alpha, max_order=2, rope=rope,
                                          prior_scale=0.5, warmup=warmup))
        algo.warm_up(X[:warmup])
        stream = X[warmup:]

        bad_s = bad_p = False
        ms, xs, mp, xp = [], [], [], []
        for n in grid:
            algo.update(stream[n - batch:n])
            es = skeleton_errors(algo.graph, truth)
            ms.append(es["missing"]); xs.append(es["extra"])
            bad_s |= es["missing"] > 0

            pc = pc_fixed_sample(stream[:n], alpha=alpha, max_order=2)
            ep = skeleton_errors(pc, truth)
            mp.append(ep["missing"]); xp.append(ep["extra"])
            bad_p |= ep["missing"] > 0

        if premise_holds:
            n_ok += 1
            ever_missing_ok += bad_s
            ever_missing_pc += bad_p
            miss_sprint.append(ms); extra_sprint.append(xs)
            miss_pc.append(mp); extra_pc.append(xp)
        else:
            n_bad += 1
            ever_missing_bad += bad_s

    to_mean = lambda a: np.array(a).mean(axis=0) if a else np.zeros(len(grid))
    return {
        "grid": grid,
        "miss_sprint": to_mean(miss_sprint), "extra_sprint": to_mean(extra_sprint),
        "miss_pc": to_mean(miss_pc), "extra_pc": to_mean(extra_pc),
        "ever_missing_ok": ever_missing_ok, "n_ok": n_ok,
        "ever_missing_bad": ever_missing_bad, "n_bad": n_bad,
        "ever_missing_pc": ever_missing_pc,
        "premise_rate": n_generated_ok / max(generated, 1),
        "generated": generated,
        "margins": margins,
        "reps": n_ok,
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

    print(f"\nExperiment 2 -- whole-graph error over a monitored run "
          f"(d={args.d}, alpha={args.alpha}, delta={args.rope})\n")
    print(f"delta-strong faithfulness holds in {res['premise_rate']:.2f} of "
          f"{res['generated']} randomly generated DAGs.")
    print("  (Instances are rejection-sampled into strata so each is powered; "
          "the rate above\n  is the unconditional one.)\n")

    print("P(any true edge ever absent from the estimate, at any monitoring point):")
    lo_s, hi_s = wilson_interval(res["ever_missing_ok"], max(res["n_ok"], 1))
    print(f"  SPRINT-CD, premise holds     {res['ever_missing_ok'] / max(res['n_ok'], 1):.3f}"
          f"   95% CI [{lo_s:.3f}, {hi_s:.3f}]   budget = {args.alpha}   "
          f"(n={res['n_ok']})")
    lo_p, hi_p = wilson_interval(res["ever_missing_pc"], max(res["n_ok"], 1))
    print(f"  PC at each n, same instances {res['ever_missing_pc'] / max(res['n_ok'], 1):.3f}"
          f"   95% CI [{lo_p:.3f}, {hi_p:.3f}]   no such guarantee")
    if res["n_bad"]:
        lo_b, hi_b = wilson_interval(res["ever_missing_bad"], res["n_bad"])
        print(f"  SPRINT-CD, premise violated  {res['ever_missing_bad'] / res['n_bad']:.3f}"
              f"   95% CI [{lo_b:.3f}, {hi_b:.3f}]   outside the theorem  "
              f"(n={res['n_bad']})")
    print(f"\nFinal-sample-size averages at n={res['grid'][-1]} "
          f"(premise-satisfying instances):")
    print(f"  SPRINT-CD   missing {res['miss_sprint'][-1]:.2f}   "
          f"extra {res['extra_sprint'][-1]:.2f}")
    print(f"  PC          missing {res['miss_pc'][-1]:.2f}   "
          f"extra {res['extra_pc'][-1]:.2f}\n")

    save_json("exp2_graph_fwer", quick=args.quick, payload={"config": vars(args), **{
        k: (v if not isinstance(v, np.ndarray) else v.tolist()) for k, v in res.items()}})

    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharex=True)
    g = res["grid"]
    axes[0].plot(g, res["miss_sprint"], "-o", ms=3, color=PALETTE["sprint"], label="SPRINT-CD")
    axes[0].plot(g, res["miss_pc"], "-s", ms=3, color=PALETTE["naive"], label="PC at each $n$")
    axes[0].set_ylabel("missing true edges (mean)")
    axes[0].set_title("Edges wrongly removed\n(premise-satisfying instances)")
    axes[0].legend(fontsize=8)

    axes[1].plot(g, res["extra_sprint"], "-o", ms=3, color=PALETTE["sprint"], label="SPRINT-CD")
    axes[1].plot(g, res["extra_pc"], "-s", ms=3, color=PALETTE["naive"], label="PC at each $n$")
    axes[1].set_ylabel("extra edges (mean)")
    axes[1].set_title("Edges not yet removed\n(the price of the one-sided guarantee)")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xlabel("sample size $n$")
    fig.tight_layout()
    out = figure_path("exp2_graph_fwer", quick=args.quick)
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
