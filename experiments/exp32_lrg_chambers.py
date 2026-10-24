"""Experiment 32: the frozen LRG model on the untouched, time-ordered Causal Chamber recordings.

Written and committed before running.  The model and threshold (results/lrg_model.joblib,
trained only on simulated data, exp31) are frozen.  Rows and recordings are those of exp30
(first 4000 rows in time order of the same 19 recordings), so that the stored estimates of the
other methods in results/exp30_btad_chambers.json are on identical data.  Output: LRG's
predicted skeleton per recording at the frozen threshold (and the probabilities).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json, pathlib, sys, warnings
from multiprocessing import Pool
import numpy as np
import joblib

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from sprint_cd.learned import pair_features
import exp27_chamber_fresh as e27

NROWS = 4000


def one(idx):
    warnings.filterwarnings("ignore")
    spec = e27.specs()[idx]
    X, var, edges = e27.load(spec)
    XA = X[:NROWS]
    Z = (XA - XA[:100].mean(0)) / (XA[:100].std(0, ddof=1) + 1e-12)
    pairs, F = pair_features(Z)
    mdl = joblib.load(HERE.parent / "results" / "lrg_model.joblib")
    p = mdl["model"].predict_proba(F)[:, 1]
    return {"spec": list(spec), "pairs": [list(map(int, q)) for q in pairs], "prob": p.tolist(),
            "threshold": mdl["threshold"], "edges": [list(map(int, q)) for q, pr in zip(pairs, p) if pr >= mdl["threshold"]]}


def main():
    n = len(e27.specs())
    with Pool(4) as pool:
        recs = pool.map(one, range(n), chunksize=1)
    (HERE.parent / "results" / "exp32_lrg_chambers.json").write_text(json.dumps({"records": recs}))


if __name__ == "__main__":
    main()
