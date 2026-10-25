"""Experiment 61: how much does the evaluation protocol move each rule on Tuebingen?

Written and committed before running.  Configurations: row cap in {250, 500, 1000, 2000, all rows} (a random subsample of
the cap, seeds 0-4; for 'all' only the 5 seeds of the rule's own randomness) x dequantisation {yes, no} x seed {0,...,4}
= 50 configurations.  Rules: IGCI (uniform reference), RECI, ANM-HSIC, pairwise LiNGAM (sprint_cd.pair_baselines.scored),
compression (faithful), and cdt IGCI (Gaussian reference), cdt RECI and cdt CDS.  For each rule and configuration the
unweighted and the official-weight accuracy on the 102 pairs.  Reported per rule: min, max, mean and sd over the 50
configurations.  No rule is tuned; the seed changes the subsample and the rule's own randomness.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json, pathlib, sys, warnings, itertools
from multiprocessing import Pool
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
R = HERE.parent / "results"
from sprint_cd.pair_baselines import scored
from sprint_cd.pair_2026 import sorted_compression_faithful as scf
from exp14_tuebingen_relfit import dequantise

CAPS = [250, 500, 1000, 2000, None]


def load():
    from exp12_tuebingen_cep import load_pairs
    return [(pid, np.asarray(x, float), np.asarray(y, float), w) for pid, x, y, w in load_pairs()]


def config_job(args):
    warnings.filterwarnings("ignore")
    from cdt.causality.pairwise import IGCI, RECI, CDS
    cap, deq, seed = args
    pairs = load(); out = {m: [] for m in ("IGCI", "RECI", "ANM-HSIC", "LiNGAM-pair", "Compression", "cdt-IGCI", "cdt-RECI", "cdt-CDS")}; w = []
    for pid, x, y, wt in pairs:
        rng = np.random.default_rng([20263400, seed, int(pid)])
        if cap is not None and len(x) > cap:
            idx = np.sort(rng.choice(len(x), cap, replace=False)); x, y = x[idx], y[idx]
        if deq: x, y = dequantise(x, rng), dequantise(y, rng)
        sc = scored(x, y, seed)
        for k in ("IGCI", "RECI", "ANM-HSIC", "LiNGAM-pair"):
            lab, mag = sc[k][0], sc[k][1]; out[k].append(1.0 if lab == "x->y" else 0.0)
        s = scf(x, y); out["Compression"].append(0.5 if s == 0 else float(s > 0))
        for name, M in (("cdt-IGCI", IGCI), ("cdt-RECI", RECI), ("cdt-CDS", CDS)):
            try: v = float(M().predict_proba((x, y))); out[name].append(0.5 if v == 0 else float(v > 0))
            except Exception: out[name].append(np.nan)
        w.append(wt)
    w = np.array(w)
    res = {m: {"unweighted": float(np.nanmean(v)), "weighted": float(np.nansum(np.array(v) * w) / w[~np.isnan(v)].sum())} for m, v in out.items()}
    return {"cap": cap, "deq": deq, "seed": seed, "res": res}


def main():
    cfgs = list(itertools.product(CAPS, (True, False), range(5)))
    with Pool(4) as p: recs = p.map(config_job, cfgs, chunksize=1)
    (R / "exp61_protocol_multiverse.json").write_text(json.dumps({"records": recs}))
    for m in recs[0]["res"]:
        u = np.array([r["res"][m]["unweighted"] for r in recs]); w = np.array([r["res"][m]["weighted"] for r in recs])
        print(f"{m:12s} unweighted {u.min():.3f}-{u.max():.3f} (mean {u.mean():.3f}, sd {u.std():.3f}) weighted {w.min():.3f}-{w.max():.3f} (mean {w.mean():.3f}, sd {w.std():.3f})")


if __name__ == "__main__":
    main()
