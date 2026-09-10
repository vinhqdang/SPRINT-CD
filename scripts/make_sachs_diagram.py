"""A worked-example diagram of CERT-CD's output on the Sachs data.

Every other figure in this paper reports an aggregate statistic. This one
does not: it draws the eleven proteins, the seventeen-arc published
consensus, and CERT-CD's six certified edges and four certified arrowheads
on the same layout, so a reader can see the concrete example the numbers in
Section 4.10 summarise -- which edges were certified, which of those were
also oriented, and how each certified arrowhead sits relative to the
consensus arc on the same pair.

Reads results/exp10_sachs.json (already produced by experiments/exp10_sachs.py)
and the hard-coded 17-arc consensus network; does not re-run anything.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from experiments.common import PALETTE, setup_matplotlib  # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results" / "exp10_sachs.json"
OUT = pathlib.Path(__file__).resolve().parent.parent / "results" / "sachs_network_demo.png"

# Sachs et al. (2005), Figure 3A, as encoded in bnlearn's reference DAG --
# identical to experiments/exp10_sachs.py's CONSENSUS constant.
CONSENSUS = [
    ("PKC", "PKA"), ("PKC", "Raf"), ("PKC", "Mek"), ("PKC", "Jnk"), ("PKC", "P38"),
    ("PKA", "Raf"), ("PKA", "Mek"), ("PKA", "Erk"), ("PKA", "Akt"),
    ("PKA", "Jnk"), ("PKA", "P38"),
    ("Raf", "Mek"), ("Mek", "Erk"), ("Erk", "Akt"),
    ("Plcg", "PIP2"), ("Plcg", "PIP3"), ("PIP3", "PIP2"),
]

# A manual layout approximating the standard MAPK/PI3K pathway diagram,
# grouped by biological role rather than a force-directed layout, so the
# cascade and the two regulatory hubs are legible.
POS = {
    "Plcg": (0.0, 3.0), "PIP3": (0.0, 2.0), "PIP2": (1.0, 2.0),
    "PKC": (2.0, 3.6), "PKA": (4.3, 3.6),
    "Raf": (1.0, 1.0), "Mek": (2.2, 1.0), "Erk": (3.4, 1.0), "Akt": (4.6, 1.0),
    "P38": (3.4, 0.0), "Jnk": (4.6, 0.0),
}


def draw_arrow(ax, a, b, color, lw, reversed_vs_consensus=None, undirected=False,
               z=2, style="-"):
    xa, ya = POS[a]
    xb, yb = POS[b]
    if undirected:
        ax.plot([xa, xb], [ya, yb], color=color, lw=lw, zorder=z, linestyle=style,
                solid_capstyle="round")
        return
    ax.annotate("", xy=(xb, yb), xytext=(xa, ya), zorder=z,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                 shrinkA=14, shrinkB=14, mutation_scale=14,
                                 linestyle=style))


def main():
    r = json.loads(RESULTS.read_text())
    certified_edges = {frozenset(e) for e in r["certified_edges"]}
    certified_arrows = {(a, b) for a, b in r["certified_arrows"]}
    consensus_dir = {frozenset(e): e for e in CONSENSUS}

    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(7.2, 5.0))

    for a, b in CONSENSUS:
        draw_arrow(ax, a, b, color="#b8b8b8", lw=1.1, z=1)

    for edge in certified_edges:
        a, b = tuple(edge)
        oriented = None
        for x, y in ((a, b), (b, a)):
            if (x, y) in certified_arrows:
                oriented = (x, y)
        if oriented is None:
            draw_arrow(ax, a, b, color=PALETTE["sprint"], lw=3.2, undirected=True, z=3)
        else:
            cx, cy = consensus_dir.get(edge, oriented)
            reversed_ = oriented != (cx, cy)
            color = PALETTE["naive"] if reversed_ else PALETTE["oracle"]
            draw_arrow(ax, oriented[0], oriented[1], color=color, lw=3.2, z=3)

    for name, (x, y) in POS.items():
        ax.scatter([x], [y], s=520, color="white", edgecolor="black", zorder=4, lw=1.2)
        ax.annotate(name, (x, y), ha="center", va="center", zorder=5, fontsize=8.2)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color="#b8b8b8", lw=1.5, label="consensus arc (17), not certified"),
        Line2D([0], [0], color=PALETTE["naive"], lw=3, label="certified arrowhead, reversed vs. consensus (4)"),
        Line2D([0], [0], color=PALETTE["sprint"], lw=3, label="certified edge, direction undecided (2)"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.22),
              ncol=1, fontsize=8, frameon=False)
    ax.set_title("CERT-CD on the Sachs network: adjacency right, orientation wrong",
                 fontsize=10.5)
    ax.set_xlim(-0.8, 5.4)
    ax.set_ylim(-0.9, 4.3)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
