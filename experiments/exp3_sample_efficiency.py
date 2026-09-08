"""Experiment 3: adaptive stopping, and what the equivalence region buys.

Two questions.

**Does stopping adaptively save data?**  SPRINT-CD halts of its own accord
once every pair is either separated or certified dependent.  Because the
guarantee is anytime-valid, that data-dependent stopping rule is free.  The
comparison is against fixed-sample PC, which must have its sample size chosen
in advance: we ask what fixed ``n`` PC needs before its accuracy matches what
SPRINT-CD achieved at its own stopping time.

**What does the equivalence region control?**  The half-width ``delta`` is the
one genuinely consequential tuning knob.  A loose region lets edges be removed
quickly but violates the ``delta``-strong-faithfulness premise for any true
edge whose partial correlation falls below it -- and the guarantee simply does
not cover those edges.  The sweep below shows that failure mode appearing
exactly where the theory says it should, rather than hiding it.
"""

from __future__ import annotations

import argparse
import sys, pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from common import PALETTE, RESULTS, save_json, setup_matplotlib

from sprint_cd.baselines import pc_fixed_sample
from sprint_cd.simulate import random_dag, skeleton_errors, structural_hamming_distance
from sprint_cd.sprint_cd import SprintCD, SprintCDConfig


def efficiency(reps, d, edge_prob, nmax, batch, alpha, rope, seed):
    """Stopping times of SPRINT-CD against the SHD curve of fixed-sample PC."""
    rng = np.random.default_rng(seed)
    warmup = 50
    pc_grid = [n for n in (250, 500, 1000, 2000, 4000, 8000) if n <= nmax]

    stops, shd_at_stop, censored = [], [], 0
    pc_shd = {n: [] for n in pc_grid}

    for _ in range(reps):
        sem = random_dag(d, edge_prob, rng, coef_low=0.5, coef_high=1.2)
        truth = sem.true_cpdag()
        X = sem.sample(nmax + warmup, rng)
        algo = SprintCD(d, SprintCDConfig(alpha=alpha, max_order=2, rope=rope,
                                          prior_scale=0.5, warmup=warmup))
        algo.warm_up(X[:warmup])
        stream = X[warmup:]
        stopped = None
        for start in range(0, nmax, batch):
            algo.update(stream[start:start + batch])
            if algo.resolved:
                stopped = algo.n
                break
        if stopped is None:
            censored += 1          # never resolved within nmax; see the report
        stops.append(stopped if stopped is not None else nmax)
        shd_at_stop.append(structural_hamming_distance(algo.cpdag(), truth))
        for n in pc_grid:
            pc_shd[n].append(structural_hamming_distance(
                pc_fixed_sample(stream[:n], alpha=alpha, max_order=2), truth))

    return {
        "stops": stops,
        "censored": censored,
        "shd_at_stop": shd_at_stop,
        "shd_at_stop_se": float(np.std(shd_at_stop, ddof=1) / np.sqrt(len(shd_at_stop))),
        "pc_grid": pc_grid,
        "pc_shd_mean": [float(np.mean(pc_shd[n])) for n in pc_grid],
        "pc_shd_se": [float(np.std(pc_shd[n], ddof=1) / np.sqrt(len(pc_shd[n])))
                      for n in pc_grid],
    }


def rope_sweep(reps, d, edge_prob, n, alpha, ropes, seed):
    rng = np.random.default_rng(seed + 1)
    warmup = 50
    out = {}
    for rope in ropes:
        miss, extra = [], []
        r = np.random.default_rng(seed + 1)   # identical DAGs across settings
        for _ in range(reps):
            sem = random_dag(d, edge_prob, r, coef_low=0.5, coef_high=1.2)
            truth = sem.true_cpdag()
            X = sem.sample(n + warmup, r)
            algo = SprintCD(d, SprintCDConfig(alpha=alpha, max_order=2, rope=rope,
                                              prior_scale=0.5, warmup=warmup))
            algo.warm_up(X[:warmup])
            algo.update(X[warmup:])
            e = skeleton_errors(algo.graph, truth)
            miss.append(e["missing"]); extra.append(e["extra"])
        out[rope] = {"missing": float(np.mean(miss)), "extra": float(np.mean(extra))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=60)
    ap.add_argument("--d", type=int, default=6)
    ap.add_argument("--edge-prob", type=float, default=0.3)
    ap.add_argument("--nmax", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--rope", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        # nmax stays large: capping it censors the stopping time and would make
        # SPRINT-CD look worse than it is by scoring it mid-run.
        args.reps = 12

    eff = efficiency(args.reps, args.d, args.edge_prob, args.nmax, args.batch,
                     args.alpha, args.rope, args.seed)
    ropes = [0.05, 0.10, 0.15, 0.20, 0.30]
    sweep_n = 4000
    sweep = rope_sweep(max(args.reps // 2, 10), args.d, args.edge_prob,
                       sweep_n, args.alpha, ropes, args.seed)

    stops = np.array(eff["stops"])
    mean_shd = float(np.mean(eff["shd_at_stop"]))
    matched = [n for n, s in zip(eff["pc_grid"], eff["pc_shd_mean"]) if s <= mean_shd]
    matched_n = matched[0] if matched else None

    print(f"\nExperiment 3 -- adaptive stopping ({args.reps} random DAGs, d={args.d})\n")
    print(f"SPRINT-CD stopping time: median {np.median(stops):.0f}, "
          f"IQR [{np.percentile(stops, 25):.0f}, {np.percentile(stops, 75):.0f}], "
          f"max {stops.max():.0f}   "
          f"(runs never resolving within n<={args.nmax}: {eff['censored']}/{args.reps})")
    print(f"SHD at its own stopping time: {mean_shd:.2f} "
          f"(SE {eff['shd_at_stop_se']:.2f})")
    print("\nFixed-sample PC, mean SHD by sample size:")
    for n, sv, se in zip(eff["pc_grid"], eff["pc_shd_mean"], eff["pc_shd_se"]):
        print(f"    n = {n:>5}   SHD = {sv:.2f} (SE {se:.2f})")
    if matched_n is not None:
        ratio = matched_n / max(np.median(stops), 1)
        print(f"\nPC first matches that accuracy at n = {matched_n}, against a "
              f"median stop of {np.median(stops):.0f} ({ratio:.1f}x the data).")
    else:
        print(f"\nPC never reaches SHD {mean_shd:.2f} within n <= {args.nmax}, "
              f"while SPRINT-CD attains it at a median stop of "
              f"{np.median(stops):.0f}.")
    if eff["censored"]:
        print("NOTE: censored runs are scored mid-run and understate accuracy; "
              "raise --nmax.")

    print(f"\nEquivalence region sweep at n={sweep_n}:")
    print(f"  {'delta':>6} | {'missing edges':>14} | {'extra edges':>12}")
    print("  " + "-" * 38)
    for rope in ropes:
        print(f"  {rope:>6} | {sweep[rope]['missing']:>14.2f} | {sweep[rope]['extra']:>12.2f}")
    print("\nA larger delta removes edges sooner but voids the guarantee for any true\n"
          "edge whose partial correlation falls below it -- visible as rising "
          "missing-edge counts.\n")

    save_json("exp3_sample_efficiency",
              {"config": vars(args), "efficiency": eff,
               "rope_sweep": {str(k): v for k, v in sweep.items()}})

    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    ax = axes[0]
    ax.hist(stops, bins=15, color=PALETTE["sprint"], alpha=0.85)
    ax.axvline(np.median(stops), color=PALETTE["accent"], lw=1.6,
               label=f"median = {np.median(stops):.0f}")
    if matched_n is not None:
        ax.axvline(matched_n, color=PALETTE["fixed"], ls="--", lw=1.4,
                   label=f"$n$ PC needs = {matched_n}")
    ax.set_xlabel("sample size at which SPRINT-CD stopped")
    ax.set_ylabel("count")
    ax.set_title("Adaptive stopping time")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(ropes, [sweep[r]["missing"] for r in ropes], "-o", ms=4,
            color=PALETTE["naive"], label="missing true edges")
    ax.plot(ropes, [sweep[r]["extra"] for r in ropes], "-s", ms=4,
            color=PALETTE["sprint"], label="extra edges")
    ax.set_xlabel(r"equivalence half-width $\delta$")
    ax.set_ylabel("mean count")
    ax.set_title(rf"Effect of $\delta$ at $n={sweep_n}$")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = RESULTS / "exp3_sample_efficiency.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
