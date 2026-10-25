"""Experiment 58: pair rules on every untouched Causal Chambers recording (not only the first of each type).

Written and committed before running.  Exp38/42/46 used the first recording of each of five types (122 pairs).  Here every
recording of exp27's list (wind-tunnel random-walk actuators, the pressure-control recording hatch_0, the light-tunnel walks)
contributes every directed edge of its documented graph as a pair (cause first), with 2000 random rows
(default_rng([20263300, recording index, i, j])).  Rules (settings of the earlier experiments): IGCI, RECI, ANM-HSIC, pairwise
LiNGAM (scored), compression (faithful), TRA, RESIT, AVICI, cdt IGCI/RECI/CDS.  The cluster is the recording; the analysis
reports accuracy per rule with a cluster bootstrap over recordings and the between-recording sd of the per-recording accuracy.
Stored: per-pair scores.  No setting is tuned.
Tie handling (revision after a check of exp53/55): all rules see the raw arrays, as in exp35/38 for the chambers; only TRA, whose spline fit is ill-posed on tied values, receives a copy with 1e-6 range jitter on tied values.  (A first start of this script jittered every rule, which reverses IGCI on tied data; it was stopped before finishing and nothing from it is used.)
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json, pathlib, sys, warnings
from multiprocessing import Pool
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
R = HERE.parent / "results"
from sprint_cd.pair_baselines import scored
from sprint_cd.pair_2026 import sorted_compression_faithful, tra_delta
from sprint_cd.sota_wrappers import resit_pair, avici_pairs
import exp27_chamber_fresh as e27


def arrays():
    out = []
    for si, spec in enumerate(e27.specs()):
        X, var, edges = e27.load(spec)
        for (i, j) in edges:
            idx = np.random.default_rng([20263300, si, i, j]).permutation(len(X))[:2000]
            out.append({"rec": f"{spec[0]}/{spec[1]}", "id": f"{spec[1]}/{var[i]}-{var[j]}", "x": X[idx, i], "y": X[idx, j]})
    return out


def jit(v, seed=1):
    rng = np.random.default_rng(seed)
    return v + 1e-6 * np.ptp(v) * rng.standard_normal(len(v)) if len(np.unique(v)) < len(v) else v


def job(p):
    warnings.filterwarnings("ignore")
    from cdt.causality.pairwise import IGCI, RECI, CDS
    x, y = p["x"], p["y"]; xj, yj = jit(x), jit(y, 2)
    sc = scored(x, y, 1); m = {k: (float(v[1]) if v[0] == "x->y" else -float(v[1])) for k, v in sc.items()}
    for name, fn in (("Compression", lambda: sorted_compression_faithful(p["x"], p["y"])), ("TRA", lambda: tra_delta(xj, yj, seed=0)), ("RESIT", lambda: resit_pair(x, y)),
                     ("cdt-IGCI", lambda: float(IGCI().predict_proba((x, y)))), ("cdt-RECI", lambda: float(RECI().predict_proba((x, y)))), ("cdt-CDS", lambda: float(CDS().predict_proba((x, y))))):
        try: m[name] = fn()
        except Exception: m[name] = None
    return {"rec": p["rec"], "id": p["id"], "m": m}


def main():
    pairs = arrays(); print(len(pairs), "pairs", len({p["rec"] for p in pairs}), "recordings", flush=True)
    part = R / "exp58_walks_all.partial.json"
    recs = json.loads(part.read_text())["records"] if part.exists() else []
    todo = pairs[len(recs):]
    with Pool(4) as pool:
        for s in range(0, len(todo), 80):
            recs.extend(pool.map(job, todo[s:s + 80], chunksize=4)); part.write_text(json.dumps({"records": recs})); print(len(recs), "/", len(pairs), flush=True)
    av = avici_pairs([(p["x"], p["y"]) for p in pairs])
    for r, v in zip(recs, av): r["m"]["AVICI"] = v
    (R / "exp58_walks_all.json").write_text(json.dumps({"records": recs})); part.unlink()


if __name__ == "__main__":
    main()
