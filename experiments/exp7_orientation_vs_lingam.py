"""Experiment 7: certified orientation against DirectLiNGAM.

Added in response to review: comparing certified orientation against a CPDAG
orientation ceiling is not a fair contest, because the certificate is allowed
to use non-Gaussianity and a CPDAG is not. The honest comparator is a
functional-model method using the *same* assumption.

Both methods are handed the **true skeleton**, so the comparison isolates
orientation. DirectLiNGAM assigns a direction to every edge and never abstains;
the direction certificate commits only where an e-process crosses. The quantity
of interest is therefore not the orientation rate but the arrowhead error at
that rate, and how each behaves under Gaussian noise, where no direction is
identified.
"""

from __future__ import annotations

import argparse
import sys, pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from common import PALETTE, figure_path, save_json, setup_matplotlib, wilson_interval

from sprint_cd.certcd import CertCD, CertCDConfig
from sprint_cd.lingam import direct_lingam_orient
from sprint_cd.simulate import random_dag


def run(reps, d, edge_prob, n, batch, alpha, noise, seed):
    rng = np.random.default_rng(seed)
    warm = 50
    out = {
        "cert": {"oriented": 0, "wrong": 0, "runs_with_wrong": 0},
        "lingam": {"oriented": 0, "wrong": 0, "runs_with_wrong": 0},
        "edges": 0, "reps": reps,
    }

    for _ in range(reps):
        sem = random_dag(d, edge_prob, rng, coef_low=0.5, coef_high=1.2)
        adj = sem.adjacency
        skel = [(i, j) for i in range(d) for j in range(i + 1, d)
                if adj[i, j] or adj[j, i]]
        if not skel:
            continue
        X = sem.sample(n + warm, rng, noise=noise)
        out["edges"] += len(skel)

        # --- DirectLiNGAM: orients every edge, no abstention, no error control ---
        lin = direct_lingam_orient(X[warm:], skel)
        bad_l = 0
        for (i, j), (c, e) in lin.items():
            out["lingam"]["oriented"] += 1
            if not adj[c, e]:
                out["lingam"]["wrong"] += 1
                bad_l += 1
        out["lingam"]["runs_with_wrong"] += bad_l > 0

        # --- CERT-CD direction certificates on the same skeleton ---
        algo = CertCD(d, CertCDConfig(alpha=alpha, max_order=2, warmup=warm))
        algo.warm_up(X[:warm])
        for s in range(0, n, batch):
            algo.update(X[warm + s: warm + s + batch])
        bad_c = 0
        certified = set(algo.certified_arrows())
        for (i, j) in skel:
            fwd, bwd = (i, j) in certified, (j, i) in certified
            if fwd ^ bwd:
                out["cert"]["oriented"] += 1
                c, e = (i, j) if fwd else (j, i)
                if not adj[c, e]:
                    out["cert"]["wrong"] += 1
                    bad_c += 1
        out["cert"]["runs_with_wrong"] += bad_c > 0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--d", type=int, default=5)
    ap.add_argument("--edge-prob", type=float, default=0.35)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=250)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.reps, args.n = 20, 1500

    results = {}
    for noise in ("laplace", "gaussian"):
        results[noise] = run(args.reps, args.d, args.edge_prob, args.n,
                             args.batch, args.alpha, noise, args.seed)

    print(f"\nExperiment 7 -- certified orientation vs DirectLiNGAM "
          f"(d={args.d}, n={args.n}, alpha={args.alpha}, {args.reps} DAGs)\n")
    print("Both methods receive the TRUE skeleton; only orientation differs.\n")
    hdr = f"  {'noise':10s} {'method':14s} {'oriented/edges':>16s} {'wrong arrows':>14s} " \
          f"{'arrowhead err':>14s} {'P(any wrong run)':>17s}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for noise, r in results.items():
        for key, lbl in (("lingam", "DirectLiNGAM"), ("cert", "CERT-CD")):
            m = r[key]
            rate = m["oriented"] / max(r["edges"], 1)
            err = m["wrong"] / max(m["oriented"], 1)
            pw = m["runs_with_wrong"] / max(r["reps"], 1)
            lo, hi = wilson_interval(m["runs_with_wrong"], r["reps"])
            print(f"  {noise:10s} {lbl:14s} {m['oriented']:>7d}/{r['edges']:<8d} "
                  f"{m['wrong']:>14d} {err:>14.3f} {pw:>10.3f} [{lo:.3f},{hi:.3f}]")
    print(f"\n  Under Gaussian noise no direction is identified. DirectLiNGAM still "
          f"orients every\n  edge, at chance; the certificate abstains. That is the "
          f"distinction the CPDAG-ceiling\n  comparison in the earlier draft obscured.\n")

    save_json("exp7_orientation_vs_lingam", quick=args.quick,
              payload={"config": vars(args), "results": results})

    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))
    for ax, noise in zip(axes, ("laplace", "gaussian")):
        r = results[noise]
        labels = ["DirectLiNGAM", "CERT-CD"]
        rates = [r[k]["oriented"] / max(r["edges"], 1) for k in ("lingam", "cert")]
        errs = [r[k]["wrong"] / max(r[k]["oriented"], 1) for k in ("lingam", "cert")]
        x = np.arange(2); w = 0.35
        ax.bar(x - w / 2, rates, w, label="edges oriented", color=PALETTE["fixed"])
        ax.bar(x + w / 2, errs, w, label="arrowhead error", color=PALETTE["naive"])
        ax.axhline(args.alpha, color="k", ls="--", lw=1.0, label=rf"$\alpha={args.alpha}$")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0, 1.05); ax.set_ylabel("fraction")
        ax.set_title(f"{noise} noise")
        ax.legend(fontsize=7)
    fig.tight_layout()
    out = figure_path("exp7_orientation_vs_lingam", quick=args.quick)
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
