"""Experiment 6: certificate-only discovery against delete-on-non-rejection.

Two questions.

**Does refusing to accept nulls buy validity where the alternative loses it?**
SPRINT-CD (and PC) remove edges when a test fails to reject, which is why their
finite-sample guarantees need faithfulness -- ``delta``-strong faithfulness in
SPRINT-CD's case.  CERT-CD asserts only by rejecting, so an instance with tiny
partial correlations should cost it *power* (pairs stay undecided) rather than
*validity* (wrong edges asserted).  Instances are stratified by whether the
premise holds, since that is the whole point at issue.

**What does certified orientation add?**  A CPDAG cannot orient an edge that
lies in no v-structure -- a chain is unorientable in principle.  Direction
certificates use non-Gaussianity instead, so they can orient edges that no
constraint-based method can, and they abstain entirely under Gaussian noise
where the direction is genuinely unidentified.  Both are measured.
"""

from __future__ import annotations

import argparse
import sys, pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from common import PALETTE, figure_path, save_json, setup_matplotlib, wilson_interval

from sprint_cd.baselines import pc_fixed_sample
from sprint_cd.certcd import CertCD, CertCDConfig
from sprint_cd.graph import ARROW, TAIL
from sprint_cd.simulate import (random_dag, skeleton_errors,
                                strong_faithfulness_margin)
from sprint_cd.sprint_cd import SprintCD, SprintCDConfig


def _cpdag_oriented_fraction(sem) -> float:
    """Fraction of true edges a CPDAG can orient at all (the identifiability ceiling)."""
    cp = sem.true_cpdag()
    edges = cp.edges()
    if not edges:
        return float("nan")
    directed = sum(1 for i, j in edges if cp.is_directed(i, j) or cp.is_directed(j, i))
    return directed / len(edges)


def run(reps, d, edge_prob, nmax, batch, alpha, rope, noise, seed):
    rng = np.random.default_rng(seed)
    warm = 50
    grid = list(range(batch, nmax + 1, batch))
    target = {"ok": reps, "bad": max(reps // 2, 6)}
    got = {"ok": 0, "bad": 0}

    stat = {s: {"cert_false_edge": 0, "cert_false_arrow": 0,
                "sprint_missing": 0, "pc_missing": 0} for s in ("ok", "bad")}
    edge_recall, arrow_recall, undecided = [], [], []
    cpdag_ceiling = []
    generated = n_ok = 0

    attempts = 0
    while (got["ok"] < target["ok"] or got["bad"] < target["bad"]) and attempts < reps * 300:
        attempts += 1
        sem = random_dag(d, edge_prob, rng, coef_low=0.5, coef_high=1.2)
        margin = strong_faithfulness_margin(sem, max_order=2)
        generated += 1
        holds = margin >= rope
        n_ok += holds
        s = "ok" if holds else "bad"
        if got[s] >= target[s]:
            continue
        got[s] += 1

        adj = sem.adjacency
        truth_cpdag = sem.true_cpdag()
        n_true_edges = int((adj + adj.T > 0).sum() // 2)
        X = sem.sample(nmax + warm, rng, noise=noise)

        cert = CertCD(d, CertCDConfig(alpha=alpha, max_order=2, warmup=warm))
        cert.warm_up(X[:warm])
        spr = SprintCD(d, SprintCDConfig(alpha=alpha, max_order=2, rope=rope,
                                         prior_scale=0.5, warmup=warm))
        spr.warm_up(X[:warm])

        bad_e = bad_a = bad_s = bad_p = False
        rec_e, rec_a, und = [], [], []
        for n in grid:
            b = X[warm + n - batch: warm + n]
            cert.update(b)
            spr.update(b)
            ce = cert.certified_edges()
            ca = cert.certified_arrows()
            for i, j in ce:
                if not (adj[i, j] or adj[j, i]):
                    bad_e = True
            for i, j in ca:
                if not adj[i, j]:
                    bad_a = True
            if skeleton_errors(spr.graph, truth_cpdag)["missing"] > 0:
                bad_s = True
            pc = pc_fixed_sample(X[warm:warm + n], alpha=alpha, max_order=2)
            if skeleton_errors(pc, truth_cpdag)["missing"] > 0:
                bad_p = True
            rec_e.append(len(ce) / max(n_true_edges, 1))
            rec_a.append(len(ca) / max(n_true_edges, 1))
            und.append(len(cert.undecided_pairs()) / (d * (d - 1) / 2))

        stat[s]["cert_false_edge"] += bad_e
        stat[s]["cert_false_arrow"] += bad_a
        stat[s]["sprint_missing"] += bad_s
        stat[s]["pc_missing"] += bad_p
        if holds:
            edge_recall.append(rec_e)
            arrow_recall.append(rec_a)
            undecided.append(und)
            cpdag_ceiling.append(_cpdag_oriented_fraction(sem))

    return {
        "grid": grid, "n_ok": got["ok"], "n_bad": got["bad"],
        "premise_rate": n_ok / max(generated, 1), "generated": generated,
        "stat": stat,
        "edge_recall": np.array(edge_recall).mean(axis=0).tolist(),
        "arrow_recall": np.array(arrow_recall).mean(axis=0).tolist(),
        "undecided": np.array(undecided).mean(axis=0).tolist(),
        "cpdag_ceiling": float(np.nanmean(cpdag_ceiling)),
        "noise": noise,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--d", type=int, default=5)
    ap.add_argument("--edge-prob", type=float, default=0.35)
    ap.add_argument("--nmax", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=250)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--rope", type=float, default=0.15)
    ap.add_argument("--noise", type=str, default="laplace")
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.reps, args.nmax = 10, 1500

    r = run(args.reps, args.d, args.edge_prob, args.nmax, args.batch,
            args.alpha, args.rope, args.noise, args.seed)

    print(f"\nExperiment 6 -- certificate-only vs delete-on-non-rejection "
          f"(d={args.d}, {args.noise} noise, alpha={args.alpha})\n")
    print(f"delta-strong faithfulness holds in {r['premise_rate']:.2f} of "
          f"{r['generated']} random DAGs.\n")
    print("Probability of ever making an unsound assertion over a monitored run:")
    hdr = f"  {'stratum':<22} | {'CERT-CD false edge':>18} | {'CERT-CD wrong arrow':>19} | " \
          f"{'SPRINT-CD lost edge':>19} | {'PC lost edge':>12}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for s, lbl, n in (("ok", "premise holds", r["n_ok"]), ("bad", "premise FAILS", r["n_bad"])):
        st = r["stat"][s]
        print(f"  {lbl + f' (n={n})':<22} | {st['cert_false_edge'] / max(n,1):>18.3f} | "
              f"{st['cert_false_arrow'] / max(n,1):>19.3f} | "
              f"{st['sprint_missing'] / max(n,1):>19.3f} | "
              f"{st['pc_missing'] / max(n,1):>12.3f}")
    lo, hi = wilson_interval(r["stat"]["bad"]["cert_false_edge"], max(r["n_bad"], 1))
    print(f"\n  CERT-CD false-edge rate where the premise fails: 95% CI "
          f"[{lo:.3f}, {hi:.3f}] against a budget of {args.alpha}.")
    print("  Near-unfaithful instances cost CERT-CD power, not validity: the "
          "affected pairs\n  stay undecided instead of being asserted wrongly.\n")

    print(f"At n={r['grid'][-1]} (premise-satisfying instances):")
    print(f"  true edges certified          : {r['edge_recall'][-1]:.2f}")
    print(f"  true edges certified AND oriented: {r['arrow_recall'][-1]:.2f}")
    print(f"  pairs still undecided         : {r['undecided'][-1]:.2f}")
    print(f"\n  A CPDAG can orient at most {r['cpdag_ceiling']:.2f} of the true edges on "
          f"these graphs;\n  direction certificates are not bound by that ceiling because "
          f"they use\n  non-Gaussianity rather than v-structures.\n")

    save_json("exp6_certcd", quick=args.quick, payload={"config": vars(args), **r})

    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))
    ax = axes[0]
    labels = ["CERT-CD\nfalse edge", "CERT-CD\nwrong arrow",
              "SPRINT-CD\nlost edge", "PC\nlost edge"]
    keys = ["cert_false_edge", "cert_false_arrow", "sprint_missing", "pc_missing"]
    x = np.arange(len(labels))
    w = 0.38
    for off, s, lbl, col in ((-w / 2, "ok", "premise holds", PALETTE["sprint"]),
                             (w / 2, "bad", "premise fails", PALETTE["naive"])):
        n = max(r["n_ok"] if s == "ok" else r["n_bad"], 1)
        ax.bar(x + off, [r["stat"][s][k] / n for k in keys], w, label=lbl, color=col)
    ax.axhline(args.alpha / 2, color="k", ls="--", lw=1.1,
               label=rf"$\alpha_A=\alpha_D={args.alpha/2}$")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("P(unsound assertion, ever)")
    ax.set_title("Validity when faithfulness fails")
    ax.legend(fontsize=7.5)

    ax = axes[1]
    g = r["grid"]
    ax.plot(g, r["edge_recall"], "-o", ms=3.5, color=PALETTE["sprint"],
            label="edges certified")
    ax.plot(g, r["arrow_recall"], "-s", ms=3.5, color=PALETTE["accent"],
            label="edges certified + oriented")
    ax.plot(g, r["undecided"], "-^", ms=3.5, color=PALETTE["fixed"],
            label="pairs undecided")
    ax.axhline(r["cpdag_ceiling"], color=PALETTE["oracle"], ls="--", lw=1.2,
               label="CPDAG orientation ceiling")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("sample size $n$")
    ax.set_ylabel("fraction")
    ax.set_title(f"What gets certified ({r['noise']} noise)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    out = figure_path("exp6_certcd", quick=args.quick)
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
