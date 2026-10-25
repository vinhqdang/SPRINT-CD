"""Experiment 55: reference implementations from the cdt package, and LOCI/ANM-GP/PDM on the three new collections.

Written and committed before running.
(A) cdt (Causal Discovery Toolbox 0.6.0, library defaults) IGCI, RECI, ANM (Gaussian-process regression with an HSIC
    test), CDS and BivariateFit are applied to every pair of the seven collections (same arrays as exp49/53; T\"ubingen
    dequantised, at most 2000 rows; the new collections with the 1e-6 range jitter on tied values).  Sign convention:
    a positive score means x -> y; exact zeros count as half.  Purpose: add rules and check our IGCI/RECI/ANM
    re-implementations against an independent implementation (agreement of the decisions on T\"ubingen).
(B) LOCI (authors' code, <= 1000 rows), the Gaussian-process ANM of causal-learn (400 rows) and the PDM classifier
    of exp35 on the Bernoulli, changepoint and Malus collections (the rules of exp38/35, same settings).
Stored: per-pair scores.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json, pathlib, sys, warnings
from multiprocessing import Pool
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
R = HERE.parent / "results"
import exp49_explain as e49
import exp53_more_chamber_pairs as e53


def jit(v, seed=1):
    rng = np.random.default_rng(seed)
    return v + 1e-6 * np.ptp(v) * rng.standard_normal(len(v)) if len(np.unique(v)) < len(v) else v


def job_cdt(p):
    warnings.filterwarnings("ignore")
    from cdt.causality.pairwise import IGCI, ANM, RECI, CDS, BivariateFit
    x, y = jit(p["x"]), jit(p["y"], 2); out = {}
    for name, M in (("cdt-IGCI", IGCI), ("cdt-RECI", RECI), ("cdt-CDS", CDS), ("cdt-ANM", ANM), ("cdt-BivariateFit", BivariateFit)):
        try:
            xs = x if len(x) <= 2000 else x[:2000]; ys = y if len(y) <= 2000 else y[:2000]
            out[name] = float(M().predict_proba((xs, ys)))
        except Exception: out[name] = None
    return {"set": p["set"], "id": p["id"], "m": out}


def job_new(p):
    warnings.filterwarnings("ignore")
    import exp38_pairs_modern as e38
    import joblib
    from sprint_cd.pairmodel import pair_features
    x, y = jit(p["x"]), jit(p["y"], 2)
    m = e38._methods(x, y, 1)
    try:
        mdl = joblib.load(R / "pdm_model.joblib"); m["PDM"] = float(mdl.predict_proba(pair_features(x, y, seed=1).reshape(1, -1))[0, 1]) - 0.5
    except Exception: m["PDM"] = None
    return {"set": p["set"], "id": p["id"], "m": m}


def main():
    pairs = []
    for s, pid, x, y in e49.gather(): pairs.append({"set": s, "id": pid, "x": x, "y": y})
    new = e53.extract(); pairs += [{"set": p["set"], "id": p["id"], "x": p["x"], "y": p["y"]} for p in new]
    part = R / "exp55_cdt_loci_new.partial.json"
    recs = json.loads(part.read_text())["records"] if part.exists() else []
    jobs = [("cdt", p) for p in pairs] + [("new", {"set": p["set"], "id": p["id"], "x": p["x"], "y": p["y"]}) for p in new]
    todo = jobs[len(recs):]
    with Pool(4) as pool:
        for s in range(0, len(todo), 40):
            chunk = todo[s:s + 40]
            recs.extend(pool.map(_call, chunk, chunksize=2)); part.write_text(json.dumps({"records": recs})); print(len(recs), "/", len(jobs), flush=True)
    (R / "exp55_cdt_loci_new.json").write_text(json.dumps({"records": recs})); part.unlink()


def _call(a):
    return job_cdt(a[1]) if a[0] == "cdt" else job_new(a[1])


if __name__ == "__main__":
    main()
