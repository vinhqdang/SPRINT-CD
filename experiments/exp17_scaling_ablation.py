"""Experiment 17: scaling with graph size and ablation of CERT-CD.

Written and committed before it was run.  Both suites use Laplace errors and
the same monitoring criterion as exp13: a DAG counts as a failure if the
estimate at ANY sample size on the grid contains an edge absent from the truth.

suite 'scaling':  d in {4, 8, 12, 16}, expected degree about 1.75, R DAGs per
                  size, grid n = 100..3200.  Methods: CERT-CD (alpha = 0.1,
                  k = 2), PC (alpha = .05, .01), GES-BIC, DirectLiNGAM.
suite 'ablation': d = 8, edge probability .25, R DAGs.  Variants of CERT-CD:
                  max_order k in {0,1,2,3}, no orientation, prior scale
                  {0.25, 0.5, 1.0}, total budget alpha in {0.05, 0.1, 0.2}.

Usage:  python experiments/exp17_scaling_ablation.py --suite scaling --R 60
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import itertools
import json
import pathlib
import signal
import sys
import time
import warnings
from multiprocessing import Pool

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from sprint_cd.certcd import CertCD, CertCDConfig
from sprint_cd.simulate import random_dag

BASE_SEED = 20261024
GRID = [100, 200, 400, 800, 1600, 3200]
WARM, STD_ROWS, BATCH = 50, 100, 100
COEF_LOW, COEF_HIGH = 0.5, 1.2
LINGAM_THRESH = 0.05
TIMEOUT_S = 180
SIZES = [4, 8, 12, 16]

ABLATION = {
    "default": dict(alpha=0.1, max_order=2, prior_scale=0.5),
    "k=0": dict(alpha=0.1, max_order=0, prior_scale=0.5),
    "k=1": dict(alpha=0.1, max_order=1, prior_scale=0.5),
    "k=3": dict(alpha=0.1, max_order=3, prior_scale=0.5),
    "no orientation": dict(alpha=0.1, max_order=2, prior_scale=0.5, orient=False),
    "prior 0.25": dict(alpha=0.1, max_order=2, prior_scale=0.25),
    "prior 1.0": dict(alpha=0.1, max_order=2, prior_scale=1.0),
    "alpha 0.05": dict(alpha=0.05, max_order=2, prior_scale=0.5),
    "alpha 0.2": dict(alpha=0.2, max_order=2, prior_scale=0.5),
}


class _Timeout(Exception):
    pass


def _guarded(fn, seconds):
    def h(*_):
        raise _Timeout()
    signal.signal(signal.SIGALRM, h)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return fn(), None
    except _Timeout:
        return None, "timeout"
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"[:200]
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def _adj(M):
    M = np.asarray(M)
    return [[i, j] for i, j in itertools.combinations(range(M.shape[0]), 2)
            if M[i, j] != 0 or M[j, i] != 0]


def _pc(X, a):
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.cit import fisherz
    return _adj(pc(X, a, indep_test=fisherz, show_progress=False).G.graph)


def _ges(X):
    from causallearn.search.ScoreBased.GES import ges
    return _adj(ges(X, score_func="local_score_BIC")["G"].graph)


def _lingam(X):
    from causallearn.search.FCMBased.lingam import DirectLiNGAM
    m = DirectLiNGAM()
    m.fit(X)
    B = np.asarray(m.adjacency_matrix_)
    d = X.shape[1]
    return [[i, j] for i, j in itertools.combinations(range(d), 2)
            if abs(B[i, j]) > LINGAM_THRESH or abs(B[j, i]) > LINGAM_THRESH]


def _stream(Xs, d, cfg):
    algo = CertCD(d, CertCDConfig(warmup=WARM, **cfg))
    algo.warm_up(Xs[:WARM])
    cur, edges, und = WARM, [], []
    for n in GRID:
        for s in range(cur, n, BATCH):
            algo.update(Xs[s: min(s + BATCH, n)])
        cur = n
        edges.append([list(p) for p in algo.certified_edges()])
        und.append(len(algo.undecided_pairs()))
    return edges, und


def _data(d, ep, r, tag):
    sem = random_dag(d, ep, np.random.default_rng([BASE_SEED, tag, d, r, 0]),
                     coef_low=COEF_LOW, coef_high=COEF_HIGH)
    X = sem.sample(GRID[-1], np.random.default_rng([BASE_SEED, tag, d, r, 1]),
                   noise="laplace")
    Xs = (X - X[:STD_ROWS].mean(0)) / X[:STD_ROWS].std(0, ddof=1)
    truth = [[i, j] for i, j in itertools.combinations(range(d), 2)
             if sem.adjacency[i, j] or sem.adjacency[j, i]]
    return Xs, truth


def run_scaling(job):
    d, r = job
    warnings.filterwarnings("ignore")
    ep = min(0.5, 1.75 / (d - 1))
    Xs, truth = _data(d, ep, r, 1)
    out = {}
    t = time.time()
    v, e = _guarded(lambda: _stream(Xs, d, dict(alpha=0.1, max_order=2)), 3600)
    out["CERT-CD"] = {"edges": v[0] if v else None, "undecided": v[1] if v else None,
                      "err": e, "sec": time.time() - t}
    for name, fn in (("PC a=0.05", lambda X: _pc(X, 0.05)),
                     ("PC a=0.01", lambda X: _pc(X, 0.01)),
                     ("GES-BIC", _ges), ("DirectLiNGAM", _lingam)):
        res, errs, t = [], [], time.time()
        for n in GRID:
            v, e = _guarded(lambda: fn(Xs[:n]), TIMEOUT_S)
            res.append(v)
            errs.append(e)
        out[name] = {"edges": res, "err": errs, "sec": time.time() - t}
    return {"d": d, "r": r, "truth": truth, "methods": out}


def run_ablation(job):
    d, r = 8, job
    warnings.filterwarnings("ignore")
    Xs, truth = _data(d, 0.25, r, 2)
    out = {}
    for name, cfg in ABLATION.items():
        t = time.time()
        v, e = _guarded(lambda: _stream(Xs, d, cfg), 3600)
        out[name] = {"edges": v[0] if v else None, "undecided": v[1] if v else None,
                     "err": e, "sec": time.time() - t}
    return {"d": d, "r": r, "truth": truth, "methods": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["scaling", "ablation"], required=True)
    ap.add_argument("--R", type=int, default=60)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    t0 = time.time()
    jobs = ([(d, r) for d in SIZES for r in range(args.R)] if args.suite == "scaling"
            else list(range(args.R)))
    fn = run_scaling if args.suite == "scaling" else run_ablation
    path = HERE.parent / "results" / f"exp17_{args.suite}.json"
    part = path.with_suffix(".partial.json")
    recs = json.loads(part.read_text())["records"] if part.exists() else []
    done = {(r["d"], r["r"]) if args.suite == "scaling" else r["r"] for r in recs}
    jobs = [j for j in jobs if j not in done]
    with Pool(args.workers) as pool:
        for i, rec in enumerate(pool.imap_unordered(fn, jobs, chunksize=1)):
            recs.append(rec)
            part.write_text(json.dumps({"records": recs}))
            if (i + 1) % 5 == 0:
                print(f"{len(recs)} done, {(time.time() - t0) / 60:.1f} min", flush=True)
    out = {"config": {"suite": args.suite, "R": args.R, "grid": GRID,
                      "alpha_cert": 0.1, "base_seed": BASE_SEED,
                      "elapsed_s": time.time() - t0},
           "records": sorted(recs, key=lambda r: (r["d"], r["r"]))}
    path.write_text(json.dumps(out))
    part.unlink()
    print("wrote", path)


if __name__ == "__main__":
    main()
