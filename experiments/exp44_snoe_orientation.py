"""Experiment 44: SNOE-style edge orientation (arXiv 2506.05590, see sprint_cd/snoe_lite.py).

Written and committed before running.  Every true edge of the reference network is oriented with the
SNOE-lite rule given the true skeleton (so only the orientation step is evaluated, which is what SNOE
adds on top of a skeleton learner), using the other true neighbours of the two endpoints as parents.
Sets: Sachs (17 arcs, log scale), Causal Chamber i.i.d. (2 recordings, 2000 shuffled rows) and the
untouched walk recordings (first of each type, 2000 shuffled rows), as in exp38.  For comparison, the
RECI/IGCI numbers come from exp38 on the same pairs (marginal rule, no parents).  Note: the neighbours
come from the reference graph, which favours SNOE-lite; the rule is evaluated as an orientation oracle.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json, pathlib, sys, warnings
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from sprint_cd.snoe_lite import snoe_delta
import exp20_causal_chamber as e20, exp27_chamber_fresh as e27, exp10_sachs as e10


def nbrs(d, edges):
    N = {k: set() for k in range(d)}
    for a, b in edges: N[a].add(b); N[b].add(a)
    return N


def run(X, edges, tag, rows, rng):
    N = nbrs(X.shape[1], edges); out = []
    for (i, j) in edges:
        idx = rng.permutation(len(X))[:rows]
        d = snoe_delta(X[idx], i, j, N[i], N[j])
        out.append({"set": tag, "edge": [int(i), int(j)], "delta": d})
    return out


def main():
    warnings.filterwarnings("ignore")
    recs = []
    X, names = e10.load_sachs(); X = np.log(np.maximum(X, 1e-3))
    ed = [(names.index(a), names.index(b)) for a, b in e10.CONSENSUS]
    recs += run(X, ed, "sachs", 853, np.random.default_rng(20262000))
    for di in range(len(e20.DATASETS)):
        X, var, edges = e20.load(*e20.DATASETS[di])
        recs += run(X, list(edges), f"chamber_iid{di}", 2000, np.random.default_rng(20262001 + di))
    sp = e27.specs(); seen = set()
    for k, s in enumerate(sp):
        key = (s[0], s[3]) if s[0] != "lt_walks_v1" else (s[0], s[1])
        if key in seen: continue
        seen.add(key)
        X, var, edges = e27.load(s)
        recs += run(X, list(edges), f"walk{k}", 2000, np.random.default_rng(20262100 + k))
    (HERE.parent / "results" / "exp44_snoe_orientation.json").write_text(json.dumps({"records": recs}))


if __name__ == "__main__":
    main()
