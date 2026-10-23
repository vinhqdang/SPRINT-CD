"""Evidence trajectories of the adjacency certificate on one random DAG.

Shows, for one 6-node DAG with Laplace errors, the running maximum of
log10 A_t(i,j) for every pair against the threshold, so that true edges
(whose evidence rises) can be seen next to non-adjacent pairs (whose evidence
stays below the threshold).  Seed fixed; the DAG is the first draw, not chosen.
"""
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from common import setup_matplotlib  # noqa: E402
from sprint_cd.certcd import CertCD, CertCDConfig  # noqa: E402
from sprint_cd.simulate import random_dag  # noqa: E402

D, N, BATCH, WARM = 6, 3200, 50, 50


def main():
    import itertools
    import matplotlib.pyplot as plt
    setup_matplotlib()
    sem = random_dag(D, 0.4, np.random.default_rng(7), coef_low=0.5, coef_high=1.2)
    X = sem.sample(N, np.random.default_rng(8), noise="laplace")
    X = (X - X[:100].mean(0)) / X[:100].std(0, ddof=1)
    algo = CertCD(D, CertCDConfig(alpha=0.1, max_order=2, warmup=WARM, orient=False))
    algo.warm_up(X[:WARM])
    ns, traj = [], {p: [] for p in itertools.combinations(range(D), 2)}
    for s in range(WARM, N, BATCH):
        algo.update(X[s: s + BATCH])
        ns.append(s + BATCH)
        for p in traj:
            traj[p].append(algo._adj_logmax.get(p, -np.inf) / np.log(10))
    thr = -np.log10(algo.alpha_adj)
    adj = sem.adjacency
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for p, v in traj.items():
        edge = bool(adj[p[0], p[1]] or adj[p[1], p[0]])
        ax.plot(ns, np.maximum(v, -3), color="C3" if edge else "0.55",
                lw=1.4 if edge else 0.9, alpha=0.95 if edge else 0.7)
    ax.axhline(thr, color="k", ls="--", lw=0.9)
    ax.text(N * 0.62, thr + 0.12, "threshold", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("observations")
    ax.set_ylabel(r"running max of $\log_{10}A_t(i,j)$")
    ax.plot([], [], color="C3", label="true edge")
    ax.plot([], [], color="0.55", label="non-adjacent pair")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    out = ROOT / "results" / "trace_demo.png"
    fig.savefig(out, dpi=180)
    print("wrote", out, "edges", int(sum(adj[p] or adj[p[::-1]] for p in traj)),
          "certified", algo.certified_edges())


if __name__ == "__main__":
    main()
