"""Experiment 40: LOCI on the Tuebingen pairs with up to 2000 rows per pair.

Written and committed before running.  The referee-style audit noted that exp38 capped each pair
at 1000 rows and used 5000 steps; here the cap is 2000 rows (the exp14 preprocessing) with the
authors' default 5000 steps, to check how much the Tuebingen accuracy of LOCI depends on the
row cap.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json, pathlib, sys, warnings
from multiprocessing import Pool
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from sprint_cd.loci_wrapper import loci_score


def one(a):
    warnings.filterwarnings("ignore")
    pid, x, y, w = a
    from exp14_tuebingen_relfit import MAX_N, SEED, dequantise
    rng = np.random.default_rng(SEED + int(pid))
    if len(x) > MAX_N:
        idx = np.sort(rng.choice(len(x), MAX_N, replace=False)); x, y = x[idx], y[idx]
    x, y = dequantise(x, rng), dequantise(y, rng)
    try: s = loci_score(x, y)
    except Exception: s = None
    return {"id": pid, "weight": w, "n": len(x), "score": s}


def main():
    from exp12_tuebingen_cep import load_pairs
    jobs = list(load_pairs())
    part = HERE.parent / "results" / "exp40_loci_tuebingen_full.partial.json"
    recs = json.loads(part.read_text())["records"] if part.exists() else []
    with Pool(4) as p:
        for s in range(len(recs), len(jobs), 20):
            recs.extend(p.map(one, jobs[s:s + 20], chunksize=1))
            part.write_text(json.dumps({"records": recs})); print(len(recs), flush=True)
    (HERE.parent / "results" / "exp40_loci_tuebingen_full.json").write_text(json.dumps({"records": recs}))
    part.unlink()


if __name__ == "__main__":
    main()
