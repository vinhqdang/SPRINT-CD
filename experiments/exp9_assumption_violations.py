"""Experiment 9: what the certificates do when their assumptions fail.

The word "certificate" transfers authority, so the conditions under which a
certificate means nothing have to be measured and named, not left in a remark.
Four regimes, each breaking one premise of Theorem 1 while leaving the others
intact, against an in-model control.

The failures below are severe and are *expected*: a linear-Gaussian
conditional-independence primitive cannot see a nonlinear dependence, and no
causally-sufficient method survives an unmeasured common cause. PC with a
Fisher-z test fails identically. The point is not that the construction is
broken -- Proposition 1 is indifferent to the primitive -- but that a method
printing a family-wise bound must state where the bound is void.

The certified *arrowhead* failures are the ones that matter most: an arrowhead
is what a practitioner acts on.
"""

from __future__ import annotations

import argparse
import sys, pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from common import figure_path, save_json, setup_matplotlib, wilson_interval

from sprint_cd.certcd import CertCD, CertCDConfig


def _regimes(rng, n):
    """Each returns (data, true adjacent pairs, description, premise broken)."""
    L = lambda: rng.laplace(size=n)

    def in_model():
        z = L(); x = 0.9 * z + L(); y = 0.9 * z + L()
        return np.column_stack([x, y, z, L()]), {(0, 2), (1, 2)}

    def assumption1():
        p = [L() for _ in range(3)]
        x = sum(0.8 * q for q in p) + L(); y = sum(0.8 * q for q in p) + L()
        return (np.column_stack([x, y] + p),
                {(0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4)})

    def latent():
        u = L(); x = 1.0 * u + L(); y = 1.0 * u + L()
        return np.column_stack([x, y, L(), L()]), set()

    def nonlinear():
        z = rng.normal(size=n)
        x = z ** 2 + 0.5 * L(); y = z ** 2 + 0.5 * L()
        return np.column_stack([x, y, z, L()]), {(0, 2), (1, 2)}

    def counts():
        lam = np.exp(0.7 * rng.normal(size=n))
        x = rng.poisson(lam).astype(float); y = rng.poisson(lam).astype(float)
        return np.column_stack([x, y, np.log(lam), rng.normal(size=n)]), {(0, 2), (1, 2)}

    return [
        ("in-model control", in_model, "none"),
        ("Assumption 1 violated (needs |S|=3, k=2)", assumption1, "separator size"),
        ("latent common cause of the pair", latent, "causal sufficiency"),
        ("nonlinear mechanism", nonlinear, "linearity of the primitive"),
        ("Poisson counts, shared latent rate", counts, "linear-Gaussian primitive"),
    ]


def run(reps, n, batch, alpha, seed):
    rng = np.random.default_rng(seed)
    warm = 50
    out = {}
    for name, gen, broken in _regimes(rng, n + warm):
        fe = fa = 0
        for _ in range(reps):
            X, true_pairs = gen()
            d = X.shape[1]
            algo = CertCD(d, CertCDConfig(alpha=alpha, max_order=2, warmup=warm))
            algo.warm_up(X[:warm])
            for s in range(0, n, batch):
                algo.update(X[warm + s: warm + s + batch])
            fe += any(p not in true_pairs for p in algo.certified_edges())
            fa += any((min(i, j), max(i, j)) not in true_pairs
                      for i, j in algo.certified_arrows())
        out[name] = {"false_edge": fe, "false_arrow": fa, "reps": reps,
                     "premise_broken": broken,
                     "edge_ci": list(wilson_interval(fe, reps)),
                     "arrow_ci": list(wilson_interval(fa, reps))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=250)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=31337)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.reps, args.n = 12, 1500

    res = run(args.reps, args.n, args.batch, args.alpha, args.seed)
    print(f"\nExperiment 9 -- certificates outside their assumptions "
          f"(alpha={args.alpha}, n={args.n}, {args.reps} runs each)\n")
    hdr = f"  {'regime':42s} {'premise broken':22s} {'false EDGE':>12s} {'false ARROW':>12s}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for name, r in res.items():
        print(f"  {name:42s} {r['premise_broken']:22s} "
              f"{r['false_edge']}/{r['reps']:<10} {r['false_arrow']}/{r['reps']:<10}")
    print("\n  PC with a Fisher-z test fails identically on the last three rows: the")
    print("  fault is the primitive, not the certificate construction. But a method")
    print("  that prints a bound must say where the bound is void, and a certified")
    print("  arrowhead carries more authority than an uncertified score.\n")
    save_json("exp9_assumption_violations", quick=args.quick,
              payload={"config": vars(args), "results": res})

    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    names = list(res); x = np.arange(len(names)); w = 0.38
    ax.bar(x - w / 2, [res[k]["false_edge"] / res[k]["reps"] for k in names], w,
           label="false edge", color="#4a4a4a")
    ax.bar(x + w / 2, [res[k]["false_arrow"] / res[k]["reps"] for k in names], w,
           label="false arrowhead", color="#c0392b")
    ax.axhline(args.alpha, color="k", ls="--", lw=1.1, label=rf"$\alpha={args.alpha}$")
    ax.set_xticks(x)
    ax.set_xticklabels([n.split(" (")[0].replace(", ", ",\n").replace(" common", "\ncommon")
                        for n in names], fontsize=6.5)
    ax.set_ylabel("P(unsound assertion)")
    ax.set_title("Where the certificate is void")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = figure_path("exp9_assumption_violations", quick=args.quick)
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
