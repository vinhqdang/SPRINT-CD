"""Experiment 60: TRA on raw arrays for the all-recordings walk pairs (exp58) and the three new collections (exp53).

Written and committed before running.  TRA's smoothing-spline fit is ill-posed on arrays with the 1e-6 range jitter that
exp53 and exp58 gave it, but works on the raw arrays (exp42 used raw chamber arrays).  TRA (kappa 1, c_beta 4, 5 folds, at
most 1000 rows) is therefore run on the raw arrays of the 486 walk pairs of exp58 and the 134 pairs of exp53; a pair on which
it raises an error has no value.  Stored: per-pair scores.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json, pathlib, sys, warnings
from multiprocessing import Pool
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
R = HERE.parent / "results"
from sprint_cd.pair_2026 import tra_delta
import exp53_more_chamber_pairs as e53
import exp58_walks_all as e58


def job(p):
    warnings.filterwarnings("ignore")
    try: s = tra_delta(p["x"], p["y"], seed=0)
    except Exception: s = None
    return {"set": p["set"], "id": p["id"], "score": s}


def main():
    pairs = [{"set": "walks_all", "id": p["rec"] + "/" + p["id"], "x": p["x"], "y": p["y"]} for p in e58.arrays()] + [{"set": p["set"], "id": p["id"], "x": p["x"], "y": p["y"]} for p in e53.extract()]
    with Pool(4) as pool: recs = pool.map(job, pairs, chunksize=4)
    (R / "exp60_tra_raw.json").write_text(json.dumps({"records": recs}))
    for s in sorted({r["set"] for r in recs}):
        v = [r["score"] for r in recs if r["set"] == s and r["score"] is not None]
        print(s, len(v), "of", sum(1 for r in recs if r["set"] == s), "acc", round(float(np.mean([0.5 if a == 0 else float(a > 0) for a in v])), 3) if v else None)


if __name__ == "__main__":
    main()
