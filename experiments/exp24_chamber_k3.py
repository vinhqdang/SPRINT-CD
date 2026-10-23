"""Experiment 24: Causal Chambers with conditioning-set order k = 3.

Post-hoc, disclosed.  exp23 (k = 2) left false edges in the light tunnel whose
separators have more than two members (e.g. ir_1 and ir_2 are separated by the
five parents of ir_1).  The theory requires k to be at least the size of a
separator of every non-adjacent pair, so k is raised to 3 (the chamber graphs
have in-degree up to 5; we did not raise k further for cost).  Arms: the
nonparametric certificate and the linear Bayes-factor certificate, both with
k = 3, alpha_A = .05, orientation off, on the same subsamples as exp20/exp23.
Scoring against the exp23 reference set is done in scripts/analyse_exp24.py.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json, pathlib, sys, warnings
from multiprocessing import Pool
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from exp20_causal_chamber import DATASETS, GRID, NMAX, SEED, load
from sprint_cd.certcd import CertCD, CertCDConfig
from sprint_cd.npadj import NPAdjacency

BATCH, WARM, K = 100, 100, 3


def one(job):
    di, r = job
    warnings.filterwarnings("ignore")
    X, var, edges = load(*DATASETS[di])
    rng = np.random.default_rng([SEED, di, r])
    X = X[rng.permutation(len(X))[:NMAX]]
    Xs = (X - X[:100].mean(0)) / (X[:100].std(0, ddof=1) + 1e-12)
    d = Xs.shape[1]
    out = {"ds": di, "r": r}
    b = NPAdjacency(d, 0.05 / (d * (d - 1) / 2), k=K, warm=WARM)
    cur, res = 0, []
    for n in GRID:
        for s in range(cur, n, BATCH):
            b.update(Xs[s:min(s + BATCH, n)])
        cur = n
        res.append([list(p) for p in b.certified_edges()])
    out["nonparam_k3"] = res
    a = CertCD(d, CertCDConfig(alpha=0.1, max_order=K, warmup=50, orient=False))
    a.warm_up(Xs[:50]); cur, res = 50, []
    for n in GRID:
        for s in range(cur, n, BATCH):
            a.update(Xs[s:min(s + BATCH, n)])
        cur = n
        res.append([list(p) for p in a.certified_edges()])
    out["linear_k3"] = res
    return out


def main(reps=10):
    part = HERE.parent / "results" / "exp24_chamber_k3.partial.json"
    recs = json.loads(part.read_text())["records"] if part.exists() else []
    done = {(r["ds"], r["r"]) for r in recs}
    jobs = [(di, r) for di in range(len(DATASETS)) for r in range(reps) if (di, r) not in done]
    with Pool(4) as p:
        for rec in p.imap_unordered(one, jobs, chunksize=1):
            recs.append(rec)
            part.write_text(json.dumps({"records": recs}))
            print(len(recs), flush=True)
    (HERE.parent / "results" / "exp24_chamber_k3.json").write_text(json.dumps({"records": recs, "grid": GRID, "k": K}))
    part.unlink()


if __name__ == "__main__":
    main()
