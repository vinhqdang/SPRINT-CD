"""Experiment 5: invariant causal prediction under optional continuation.

Environments arrive over time and the analyst watches the selected predictor
set settle down.  ICP's guarantee -- that the output is a subset of the true
causal parents -- is a fixed-sample statement, and monitoring voids it in
exactly the way monitoring voids any p-value.  E-ICP restores it.

Scored quantities:

* ``P(S* ever rejected)``, the event that breaks the subset guarantee.  For
  E-ICP this is bounded by ``alpha`` uniformly in time; for Bonferroni ICP it
  is bounded only at a single pre-specified sample size.
* whether the reported set stays inside the true parent set throughout;
* how much of the true parent set is eventually recovered, since a guarantee
  that is satisfied by always returning the empty set would be worthless.
"""

from __future__ import annotations

import argparse
import itertools
import sys, pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from common import PALETTE, figure_path, save_json, setup_matplotlib, wilson_interval

from sprint_cd.e_icp import EICP, EICPConfig, icp_invariance_pvalue

TRUE_PARENTS = (0, 1)


def simulate(n, rng, n_env=3, shift=2.0):
    """``Y <- X0, X1``; ``X2`` is a child of ``Y``, so only ``{X0, X1}`` is invariant."""
    env = rng.integers(0, n_env, size=n)
    sh = np.linspace(-shift, shift, n_env)[env]
    x0 = rng.normal(size=n) + sh
    x1 = rng.normal(size=n) + 0.5 * sh
    y = 1.2 * x0 - 0.8 * x1 + rng.normal(size=n)
    x2 = 0.9 * y + rng.normal(size=n) + sh
    return np.column_stack([x0, x1, x2]), y, env


def run(reps, nmax, batch, alpha, max_size, seed):
    rng = np.random.default_rng(seed)
    warmup = 40
    d, n_env = 3, 3
    cands = [S for m in range(max_size + 1) for S in itertools.combinations(range(d), m)]
    bonf_level = alpha / len(cands)

    eicp_reject = 0
    icp_reject = 0
    icp_reject_fixed = 0
    eicp_subset_ok = 0
    eicp_final, icp_final = [], []
    grid = list(range(batch, nmax + 1, batch))

    for _ in range(reps):
        X, y, env = simulate(nmax + warmup, rng, n_env=n_env)
        cfg = EICPConfig(alpha=alpha, max_size=max_size, warmup=warmup)
        model = EICP(d=d, n_env=n_env, config=cfg)
        model.warm_up(X[:warmup], y[:warmup], env[:warmup])
        Xs, ys, es = X[warmup:], y[warmup:], env[warmup:]

        e_bad = i_bad = False
        subset_ok = True
        last_p = 1.0
        for n in grid:
            model.update(Xs[n - batch:n], ys[n - batch:n], es[n - batch:n])
            if TRUE_PARENTS in model._rejected:
                e_bad = True
            if not model.estimate() <= set(TRUE_PARENTS):
                subset_ok = False
            # Bonferroni ICP, evaluated at the same monitoring points.
            last_p = icp_invariance_pvalue(Xs[:n], ys[:n], es[:n], TRUE_PARENTS)
            if last_p <= bonf_level:
                i_bad = True

        eicp_reject += e_bad
        icp_reject += i_bad
        icp_reject_fixed += last_p <= bonf_level   # final sample size only
        eicp_subset_ok += subset_ok
        eicp_final.append(len(model.estimate() & set(TRUE_PARENTS)))
        from sprint_cd.e_icp import icp_fixed_sample
        icp_final.append(len(icp_fixed_sample(Xs, ys, es, alpha, max_size)
                             & set(TRUE_PARENTS)))

    return {
        "reps": reps, "grid": grid,
        "eicp_reject": eicp_reject, "icp_reject": icp_reject,
        "icp_reject_fixed": icp_reject_fixed, "bonf_level": bonf_level,
        "eicp_subset_ok": eicp_subset_ok,
        "eicp_recovered": float(np.mean(eicp_final)),
        "icp_recovered": float(np.mean(icp_final)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--nmax", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--max-size", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.reps, args.nmax = 60, 1000

    r = run(args.reps, args.nmax, args.batch, args.alpha, args.max_size, args.seed)
    lo_e, hi_e = wilson_interval(r["eicp_reject"], r["reps"])
    lo_i, hi_i = wilson_interval(r["icp_reject"], r["reps"])

    print(f"\nExperiment 5 -- ICP under continuous monitoring "
          f"({r['reps']} runs, alpha={args.alpha})\n")
    print("P(true parent set S* rejected at any monitoring point) "
          "-- this breaks the guarantee:")
    print(f"  E-ICP (e-process)      {r['eicp_reject'] / r['reps']:.3f}   "
          f"95% CI [{lo_e:.3f}, {hi_e:.3f}]   budget = {args.alpha}")
    print(f"  ICP (Bonferroni), monitored  {r['icp_reject'] / r['reps']:.3f}   "
          f"95% CI [{lo_i:.3f}, {hi_i:.3f}]   valid only at a fixed n")
    print(f"  ICP (Bonferroni), fixed n    {r['icp_reject_fixed'] / r['reps']:.3f}   "
          f"                       reference: its own per-set level is "
          f"{r['bonf_level']:.4f}")
    infl = (r["icp_reject"] / max(r["icp_reject_fixed"], 1))
    print(f"\n  Monitoring multiplies ICP's rejection rate for S* by about "
          f"{infl:.1f}x relative to\n  evaluating once at the final sample size. "
          f"Bonferroni over {len(TRUE_PARENTS) and 8} candidate sets is\n"
          f"  conservative enough that the inflated rate may still sit under "
          f"alpha -- but nothing\n  bounds it there, whereas Ville's inequality "
          f"bounds E-ICP's.")
    print(f"\nE-ICP output stayed inside S* throughout: "
          f"{r['eicp_subset_ok']}/{r['reps']}")
    print(f"True parents recovered at the end (out of {len(TRUE_PARENTS)}): "
          f"E-ICP {r['eicp_recovered']:.2f}, fixed-sample ICP {r['icp_recovered']:.2f}")
    print("  Recovery is reported because the subset guarantee alone is "
          "satisfied trivially\n  by returning nothing.\n")

    save_json("exp5_eicp", quick=args.quick, payload={"config": vars(args), **r})

    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    ax = axes[0]
    vals = [r["icp_reject_fixed"] / r["reps"], r["icp_reject"] / r["reps"],
            r["eicp_reject"] / r["reps"]]
    cols = [PALETTE["fixed"], PALETTE["naive"], PALETTE["sprint"]]
    ax.bar(["ICP\n(fixed $n$)", "ICP\n(monitored)", "E-ICP"], vals,
           color=cols, width=0.6)
    ax.axhline(args.alpha, color="k", ls="--", lw=1.1, label=rf"$\alpha={args.alpha}$")
    ax.set_ylabel(r"P($S^\star$ ever rejected)")
    ax.set_title("Breaking the subset guarantee")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.bar(["E-ICP", "ICP\n(fixed $n$)"],
           [r["eicp_recovered"], r["icp_recovered"]],
           color=[PALETTE["sprint"], PALETTE["fixed"]], width=0.55)
    ax.axhline(len(TRUE_PARENTS), color=PALETTE["oracle"], ls="--", lw=1.2,
               label="all true parents")
    ax.set_ylabel("true parents recovered")
    ax.set_title("Power is retained")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = figure_path("exp5_eicp", quick=args.quick)
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
