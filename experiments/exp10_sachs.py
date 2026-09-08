"""Experiment 10: the protein-signalling data of Sachs et al. (2005).

The point of this experiment is not to win a structure-recovery benchmark.  It
is to show what the certificate-only rule does on data it was not designed
around: a real, non-Gaussian, moderately sized sample whose generating
mechanism is not a linear SEM and whose "ground truth" is a published
scientific consensus rather than a simulation parameter.

We use the observational condition only (853 cells, 11 phosphoproteins), which
is the hardest and most honest setting for a purely observational method: the
interventional conditions that make the Sachs network identifiable are
withheld.  Measurements are log-transformed, as in the original analysis, since
the raw fluorescence intensities are strongly right-skewed.

What we report is the three-valued output.  A method that deletes on
non-rejection would return a graph over all 55 pairs with every pair called
present or absent; CERT-CD returns certified edges, certified arrowheads, and
an explicit undecided set, and the interesting quantity is how the certified
part compares with the consensus network -- not how the whole graph does.

The reference network is the 17-arc consensus DAG of Sachs et al. (2005) as
distributed with bnlearn.  It is a scientific consensus assembled from
interventional experiments and prior literature, not ground truth, and several
of its arcs are contested; we treat disagreement as informative rather than as
error.
"""

from __future__ import annotations

import argparse
import gzip
import io
import pathlib
import sys
import urllib.request

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from common import PALETTE, figure_path, save_json, setup_matplotlib

from sprint_cd.baselines import pc_fixed_sample
from sprint_cd.certcd import CertCD, CertCDConfig
from sprint_cd.graph import ARROW
from sprint_cd.lingam import direct_lingam_order

# The continuous observational sample, as redistributed with the bnlearn book
# code.  Cached locally; ``data/`` is not tracked, so the download is part of
# reproducing the experiment rather than of the repository.
SACHS_URL = "https://www.bnlearn.com/book-crc/code/sachs.data.txt.gz"
CACHE = pathlib.Path(__file__).resolve().parent.parent / "data" / "sachs.txt"

# Sachs et al. (2005), Figure 3A, as encoded in bnlearn's reference DAG.
CONSENSUS = [
    ("PKC", "PKA"), ("PKC", "Raf"), ("PKC", "Mek"), ("PKC", "Jnk"), ("PKC", "P38"),
    ("PKA", "Raf"), ("PKA", "Mek"), ("PKA", "Erk"), ("PKA", "Akt"),
    ("PKA", "Jnk"), ("PKA", "P38"),
    ("Raf", "Mek"), ("Mek", "Erk"), ("Erk", "Akt"),
    ("Plcg", "PIP2"), ("Plcg", "PIP3"), ("PIP3", "PIP2"),
]


def load_sachs() -> tuple[np.ndarray, list[str]]:
    if not CACHE.exists():
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(SACHS_URL, timeout=120) as fh:
            raw = gzip.decompress(fh.read()).decode("utf-8")
        CACHE.write_text(raw)
    lines = CACHE.read_text().strip().splitlines()
    names = lines[0].split()
    X = np.array([[float(v) for v in ln.split()] for ln in lines[1:]])
    return X, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--max-order", type=int, default=2)
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    X_raw, names = load_sachs()
    # Log-transform, as in the original analysis: the raw intensities span
    # three orders of magnitude and are strongly right-skewed.
    X = np.log(np.clip(X_raw, 1e-3, None))
    n, d = X.shape
    idx = {nm: i for i, nm in enumerate(names)}
    truth_arcs = {(idx[a], idx[b]) for a, b in CONSENSUS}
    truth_adj = {frozenset(e) for e in truth_arcs}

    # The row order in the file is not a time order; permute once with a fixed
    # seed so that "streaming" does not accidentally track a collection order.
    rng = np.random.default_rng(args.seed)
    X = X[rng.permutation(n)]

    cfg = CertCDConfig(alpha=args.alpha, max_order=args.max_order,
                       warmup=args.warmup)
    algo = CertCD(d, cfg, names=names)
    algo.warm_up(X[:args.warmup])
    rest = X[args.warmup:]
    grid, n_edges, n_arrows, n_undec = [], [], [], []
    for s in range(0, rest.shape[0], args.batch):
        algo.update(rest[s: s + args.batch])
        grid.append(algo.n)
        n_edges.append(len(algo.certified_edges()))
        n_arrows.append(len(algo.certified_arrows()))
        n_undec.append(len(algo.undecided_pairs()))

    edges = algo.certified_edges()
    arrows = algo.certified_arrows()
    undecided = algo.undecided_pairs()

    in_consensus = [e for e in edges if frozenset(e) in truth_adj]
    outside = [e for e in edges if frozenset(e) not in truth_adj]
    arrows_ok = [a for a in arrows if a in truth_arcs]
    arrows_rev = [a for a in arrows if (a[1], a[0]) in truth_arcs]
    arrows_new = [a for a in arrows
                  if a not in truth_arcs and (a[1], a[0]) not in truth_arcs]

    # Cross-check the arrowheads against DirectLiNGAM, which shares the linear
    # non-Gaussian model but not the inferential machinery.  If the two
    # disagree, the certificate is suspect; if they agree, a disagreement with
    # the consensus is a statement about the model, not about the procedure.
    Xs = (X - X.mean(0)) / X.std(0)
    order = direct_lingam_order(Xs)
    pos = {v: k for k, v in enumerate(order)}
    lingam_agrees = [a for a in arrows if pos[a[0]] < pos[a[1]]]

    pc = pc_fixed_sample(X, alpha=args.alpha, max_order=args.max_order)
    pc_edges = pc.edges()
    pc_in = [e for e in pc_edges if frozenset(e) in truth_adj]

    nm = lambda i: names[i]
    print(f"\nSachs et al. (2005) observational sample: n={n}, d={d}\n")
    print(f"CERT-CD at alpha={args.alpha}, k={args.max_order}, "
          f"streaming in batches of {args.batch}:\n")
    print(f"  certified edges       : {len(edges)} of "
          f"{d*(d-1)//2} pairs")
    print(f"    in the consensus skeleton : {len(in_consensus)} of "
          f"{len(truth_adj)} consensus edges")
    print(f"    outside it                : {len(outside)}")
    print(f"  certified arrowheads  : {len(arrows)}")
    print(f"    agreeing with a consensus arc : {len(arrows_ok)}")
    print(f"    reversed w.r.t. one           : {len(arrows_rev)}")
    print(f"    on a non-consensus edge       : {len(arrows_new)}")
    print(f"    agreed by DirectLiNGAM's order: {len(lingam_agrees)} of "
          f"{len(arrows)}")
    print(f"  undecided pairs       : {len(undecided)} "
          f"({len(undecided)/(d*(d-1)/2):.2f} of all pairs)")
    print(f"\n  PC at the same alpha and k returns {len(pc_edges)} edges, "
          f"{len(pc_in)} of them\n  in the consensus skeleton, and calls the "
          f"remaining {d*(d-1)//2 - len(pc_edges)} pairs absent.\n")

    print("  certified edges, in the consensus skeleton:")
    for i, j in in_consensus:
        d_ = [f"{nm(a)}->{nm(b)}" for a, b in arrows if {a, b} == {i, j}]
        print(f"    {nm(i)} -- {nm(j)}" + (f"   [oriented {d_[0]}]" if d_ else ""))
    print("  certified edges, outside it:")
    for i, j in outside:
        d_ = [f"{nm(a)}->{nm(b)}" for a, b in arrows if {a, b} == {i, j}]
        print(f"    {nm(i)} -- {nm(j)}" + (f"   [oriented {d_[0]}]" if d_ else ""))

    payload = {
        "config": vars(args), "n": int(n), "d": int(d), "names": names,
        "grid": grid, "n_edges": n_edges, "n_arrows": n_arrows,
        "n_undecided": n_undec,
        "certified_edges": [[nm(i), nm(j)] for i, j in edges],
        "certified_arrows": [[nm(i), nm(j)] for i, j in arrows],
        "in_consensus": [[nm(i), nm(j)] for i, j in in_consensus],
        "outside_consensus": [[nm(i), nm(j)] for i, j in outside],
        "arrows_agreeing": [[nm(i), nm(j)] for i, j in arrows_ok],
        "arrows_reversed": [[nm(i), nm(j)] for i, j in arrows_rev],
        "arrows_novel": [[nm(i), nm(j)] for i, j in arrows_new],
        "lingam_order": [nm(i) for i in order],
        "arrows_agreed_by_lingam": [[nm(i), nm(j)] for i, j in lingam_agrees],
        "n_undecided_final": len(undecided),
        "n_consensus_edges": len(truth_adj),
        "pc_edges": [[nm(i), nm(j)] for i, j in pc_edges],
        "pc_in_consensus": len(pc_in),
    }
    save_json("exp10_sachs", quick=args.quick, payload=payload)

    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(grid, n_edges, "-o", ms=3, color=PALETTE["sprint"],
            label="certified edges")
    ax.plot(grid, n_arrows, "-s", ms=3, color=PALETTE["accent"],
            label="certified arrowheads")
    ax.plot(grid, n_undec, "-^", ms=3, color=PALETTE["fixed"],
            label="undecided pairs")
    ax.axhline(len(truth_adj), color=PALETTE["oracle"], ls="--", lw=1.1,
               label="consensus edges (17)")
    ax.set_xlabel("observations processed $n$")
    ax.set_ylabel("count")
    ax.set_title("Sachs et al. (2005), observational condition")
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    out = figure_path("exp10_sachs", quick=args.quick)
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
