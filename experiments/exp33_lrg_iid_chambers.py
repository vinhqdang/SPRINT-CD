"""Experiment 33: frozen LRG and BTAD on the two Causal Chamber recordings with i.i.d. random actuators.

Written and committed before running.  The wind-tunnel random-intake-load recording and the
light-tunnel uniform-input recording of exp20 (actuators are redrawn independently at random,
so rows are close to i.i.d.), the same 10 subsamples of 4000 rows per recording as exp20 (same
seeds), the frozen LRG model (results/lrg_model.joblib) and BTAD with its frozen settings.
The estimates of the other methods at n = 4000 are in results/exp20_causal_chamber.json and
results/exp23_chamber_nonparam.json (shuffled rows of the same subsamples).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json, pathlib, sys, warnings
from multiprocessing import Pool
import numpy as np
import joblib

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from sprint_cd.btad import btad_skeleton
from sprint_cd.learned import pair_features
import exp20_causal_chamber as e20


def one(job):
    di, r = job
    warnings.filterwarnings("ignore")
    X, var, edges = e20.load(*e20.DATASETS[di])
    rng = np.random.default_rng([e20.SEED, di, r])
    X = X[rng.permutation(len(X))[:e20.NMAX]]
    Z = (X - X[:100].mean(0)) / (X[:100].std(0, ddof=1) + 1e-12)
    pairs, F = pair_features(Z)
    mdl = joblib.load(HERE.parent / "results" / "lrg_model.joblib")
    p = mdl["model"].predict_proba(F)[:, 1]
    return {"ds": di, "r": r, "pairs": [list(map(int, q)) for q in pairs], "prob": p.tolist(),
            "LRG": [list(map(int, q)) for q, pr in zip(pairs, p) if pr >= mdl["threshold"]],
            "BTAD": btad_skeleton(Z, K=60, alpha=0.2, max_blanket=8)}


def main():
    jobs = [(di, r) for di in range(len(e20.DATASETS)) for r in range(10)]
    for di in range(len(e20.DATASETS)):
        e20.load(*e20.DATASETS[di])
    with Pool(4) as p:
        recs = p.map(one, jobs, chunksize=1)
    (HERE.parent / "results" / "exp33_lrg_iid_chambers.json").write_text(json.dumps({"records": recs}))


if __name__ == "__main__":
    main()
