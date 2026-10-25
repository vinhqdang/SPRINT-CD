"""Experiment 59: tie handling as a free parameter, and raw-array versions for the three new collections.

Written and committed before running.  Exp53 and exp55 gave every rule a copy of the arrays with 1e-6 range jitter on tied
values, whereas exp35/38 had given the chamber pairs to the rules raw (ties kept).  A check showed that this matters
for IGCI: with the jitter the tied values of a discrete variable become tiny gaps and the spacing entropy of that variable
collapses.  This experiment evaluates the four rules of sprint_cd.pair_baselines.scored (IGCI, RECI, ANM-HSIC, pairwise
LiNGAM) and the cdt IGCI/RECI/CDS under two conditions on all seven collections: RAW (ties kept) and JITTER (uniform noise of
1e-6 times the range on tied values, seed 1 for x, 2 for y), and, on the three new collections only, RESIT (both
conditions), the faithful compression (raw) and LOCI and the Gaussian-process ANM (raw).  The raw condition is the
protocol of exp35/38 for the chambers; Tuebingen arrays are the dequantised ones of exp14 in both conditions (they have
no ties).  Stored: per-pair scores for both conditions.  No rule is tuned.
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
from sprint_cd.pair_2026 import sorted_compression_faithful
from sprint_cd.sota_wrappers import resit_pair
import exp49_explain as e49
import exp53_more_chamber_pairs as e53

NEW = ("bernoulli", "changepoints", "malus")


def jit(v, seed):
    rng = np.random.default_rng(seed)
    return v + 1e-6 * np.ptp(v) * rng.standard_normal(len(v)) if len(np.unique(v)) < len(v) else v


def job(p):
    warnings.filterwarnings("ignore")
    from cdt.causality.pairwise import IGCI, RECI, CDS
    out = {}
    for cond, (x, y) in (("raw", (p["x"], p["y"])), ("jitter", (jit(p["x"], 1), jit(p["y"], 2)))):
        try:
            sc = scored(x, y, 1)
            for k, v in sc.items(): out[f"{k}|{cond}"] = float(v[1]) if v[0] == "x->y" else -float(v[1])
        except Exception: pass
        for name, M in (("cdt-IGCI", IGCI), ("cdt-RECI", RECI), ("cdt-CDS", CDS)):
            try: out[f"{name}|{cond}"] = float(M().predict_proba((x, y)))
            except Exception: out[f"{name}|{cond}"] = None
        if p["set"] in NEW:
            try: out[f"RESIT|{cond}"] = resit_pair(x, y)
            except Exception: out[f"RESIT|{cond}"] = None
    if p["set"] in NEW:
        try: out["Compression|raw"] = sorted_compression_faithful(p["x"], p["y"])
        except Exception: out["Compression|raw"] = None
        import exp38_pairs_modern as e38
        m = e38._methods(p["x"], p["y"], 1)
        out["LOCI|raw"], out["ANM-GP|raw"] = m.get("LOCI"), m.get("ANM")
    return {"set": p["set"], "id": p["id"], "m": out}


def main():
    pairs = [{"set": s, "id": pid, "x": x, "y": y} for s, pid, x, y in e49.gather()] + [{"set": p["set"], "id": p["id"], "x": p["x"], "y": p["y"]} for p in e53.extract()]
    part = R / "exp59_ties.partial.json"
    recs = json.loads(part.read_text())["records"] if part.exists() else []
    todo = pairs[len(recs):]
    with Pool(4) as pool:
        for s in range(0, len(todo), 40):
            recs.extend(pool.map(job, todo[s:s + 40], chunksize=2)); part.write_text(json.dumps({"records": recs})); print(len(recs), "/", len(pairs), flush=True)
    (R / "exp59_ties.json").write_text(json.dumps({"records": recs})); part.unlink()


if __name__ == "__main__":
    main()
