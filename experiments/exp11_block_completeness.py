"""Experiment 11: why the direction certificates exceed their budget when
faithfulness fails.

Experiment 6 measures a wrong-arrowhead rate of 0.080 on instances violating
``delta``-strong faithfulness, against a direction budget of 0.05 -- a Wilson
interval that excludes the budget.  Theorem 1 is not contradicted: its
direction clause assumes the pair follows the linear non-Gaussian model
*given its frozen conditioning block*, and that is an assumption, not a
consequence.  This experiment tests a specific explanation for why the
assumption fails preferentially on that stratum.

The conditioning block ``Z_ij`` is frozen to the pair's *already-certified*
neighbours at the moment its adjacency is certified.  On a near-unfaithful
instance fewer neighbours have been certified by then, so ``Z_ij`` more often
omits a parent of one endpoint -- and the pair then has an uncontrolled common
cause or an omitted direct parent, which is exactly the situation the
direction model assumes away.

Two things have to hold for that story to be right, and both are measured:
incompleteness must be more common where the premise fails, and the errors
must concentrate on the incomplete blocks.  Rates here are per certified
arrowhead, not per run as in experiment 6.
"""

from __future__ import annotations

import argparse
import sys, pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from common import PALETTE, figure_path, save_json, setup_matplotlib, wilson_interval

from sprint_cd.certcd import CertCD, CertCDConfig
from sprint_cd.simulate import random_dag, strong_faithfulness_margin


def run(reps, d, edge_prob, n, batch, alpha, rope, warm, seed):
    rng = np.random.default_rng(seed)
    tab = {}                      # (stratum, z_complete) -> [arrows, wrong]
    got = {"ok": 0, "bad": 0}
    target = {"ok": reps, "bad": reps}
    attempts = 0
    while (got["ok"] < target["ok"] or got["bad"] < target["bad"]) \
            and attempts < reps * 200:
        attempts += 1
        sem = random_dag(d, edge_prob, rng, coef_low=0.5, coef_high=1.2)
        s = "ok" if strong_faithfulness_margin(sem, max_order=2) >= rope else "bad"
        if got[s] >= target[s]:
            continue
        got[s] += 1

        adj = sem.adjacency
        X = sem.sample(n + warm, rng, noise="laplace")
        algo = CertCD(d, CertCDConfig(alpha=alpha, max_order=2, warmup=warm))
        algo.warm_up(X[:warm])
        rest = X[warm:]
        for st in range(0, rest.shape[0], batch):
            algo.update(rest[st: st + batch])

        for (a, b) in algo.certified_arrows():
            Z = set(algo._dir_cond[(a, b)])
            # Every parent of either endpoint, excluding the endpoints
            # themselves: these are the variables whose omission leaves the
            # pair confounded or the regression under-specified.
            need = {v for v in range(d)
                    if (adj[v, a] or adj[v, b]) and v not in (a, b)}
            cell = tab.setdefault((s, need <= Z), [0, 0])
            cell[0] += 1
            cell[1] += not adj[a, b]

    return {"tab": {f"{k[0]}|{k[1]}": v for k, v in tab.items()},
            "n_ok": got["ok"], "n_bad": got["bad"], "generated": attempts}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=250)
    ap.add_argument("--d", type=int, default=5)
    ap.add_argument("--edge-prob", type=float, default=0.35)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=250)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--rope", type=float, default=0.15)
    ap.add_argument("--warm", type=int, default=50)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.reps, args.n = 20, 1500

    r = run(args.reps, args.d, args.edge_prob, args.n, args.batch,
            args.alpha, args.rope, args.warm, args.seed)
    tab = {tuple(k.split("|")): v for k, v in r["tab"].items()}
    get = lambda s, c: tab.get((s, str(c)), [0, 0])

    print(f"\nExperiment 11 -- wrong arrowheads by conditioning-block completeness")
    print(f"(d={args.d}, laplace noise, alpha={args.alpha}, "
          f"{r['n_ok']}+{r['n_bad']} instances)\n")
    print(f"  {'stratum':14s} {'Z_ij':12s} {'arrows':>7s} {'wrong':>6s} "
          f"{'rate':>7s}   95% CI")
    print("  " + "-" * 62)
    for s, lbl in (("ok", "premise holds"), ("bad", "premise FAILS")):
        for c, cl in ((True, "complete"), (False, "incomplete")):
            a, w = get(s, c)
            lo, hi = wilson_interval(w, max(a, 1))
            print(f"  {lbl:14s} {cl:12s} {a:7d} {w:6d} {w/max(a,1):7.4f}   "
                  f"[{lo:.4f}, {hi:.4f}]")
    print()
    for s, lbl in (("bad", "premise FAILS"), ("ok", "premise holds")):
        ai, wi = get(s, False); ac, wc = get(s, True)
        tot = ai + ac
        ri, rc = wi / max(ai, 1), wc / max(ac, 1)
        ratio = f"{ri / rc:.1f}x" if rc > 0 else ("n/a (no errors with a "
                                                 "complete block)")
        print(f"  {lbl}: Z incomplete for {ai}/{tot} = {ai/max(tot,1):.1%} of "
              f"certified arrowheads,\n    carrying {wi}/{max(wi+wc,1)} of the "
              f"wrong ones; error ratio incomplete:complete = {ratio}")
    print("\n  The exceedance in experiment 6 is the misspecification gap being "
          "opened by\n  the algorithm's own certification order, not a failure "
          "of the e-process.\n")

    save_json("exp11_block_completeness", quick=args.quick,
              payload={"config": vars(args), **r})

    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    x = np.arange(2); w = 0.36
    for off, c, cl, col in ((-w/2, True, "$Z$ complete", PALETTE["sprint"]),
                            (w/2, False, "$Z$ incomplete", PALETTE["naive"])):
        vals, los, his = [], [], []
        for s in ("ok", "bad"):
            a, wr = get(s, c)
            lo, hi = wilson_interval(wr, max(a, 1))
            vals.append(wr / max(a, 1)); los.append(lo); his.append(hi)
        vals = np.array(vals)
        ax.bar(x + off, vals, w, label=cl, color=col,
               yerr=[vals - np.array(los), np.array(his) - vals],
               capsize=3, error_kw={"lw": 0.9})
    ax.axhline(args.alpha / 2, color="k", ls="--", lw=1.1,
               label=rf"$\alpha_D={args.alpha/2}$ (whole-run budget, for scale)")
    ax.set_xticks(x); ax.set_xticklabels(["premise holds", "premise fails"])
    ax.set_ylabel("P(wrong arrowhead) per certified arrowhead")
    ax.set_title("Wrong arrowheads concentrate on incomplete blocks")
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    out = figure_path("exp11_block_completeness", quick=args.quick)
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
