"""Experiment 35: the frozen pair-direction model (PDM) on real and held-out pair sets.

Written and committed before running.  Model: results/pdm_model.joblib (exp34/exp34b, trained on
simulated pairs only).  Pair sets (the rule is fixed in advance):
 (a) Tuebingen: all 102 scalar pairs, official weights, exp14 preprocessing (tie dequantisation,
     at most 2000 rows);
 (b) Causal Chambers, i.i.d. recordings: every directed edge among varying variables of the
     two exp20 recordings, 2 random subsamples of 2000 rows (as exp21);
 (c) Causal Chambers, walk recordings: every directed edge of the first recording (in sorted
     order) of each configuration of exp27 (wind-tunnel walks, pressure control, three light-tunnel
     walk experiments), 2000 randomly drawn rows;
 (d) held-out simulated pair families of exp18 (9 families x 30 pairs, n = 800).
For each pair we store the PDM probability of x->y and the direction and confidence of IGCI,
RECI, ANM-HSIC and the pairwise LiNGAM rule.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json, pathlib, sys, warnings
from multiprocessing import Pool
import numpy as np
import joblib

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from sprint_cd.pair_baselines import scored
from sprint_cd.pairmodel import pair_features
import exp18_pairs_heldout as e18
import exp20_causal_chamber as e20
import exp21_chamber_pairs as e21
import exp27_chamber_fresh as e27

_MDL = None


def _prob(x, y, seed):
    global _MDL
    if _MDL is None:
        _MDL = joblib.load(HERE.parent / "results" / "pdm_model.joblib")
    return float(_MDL.predict_proba(pair_features(x, y, seed=seed).reshape(1, -1))[0, 1])


def _rec(set_, id_, truth, x, y, seed, w=1.0):
    return {"set": set_, "id": id_, "truth": truth, "weight": w, "pdm": _prob(x, y, seed), "scores": scored(x, y, seed)}


def tueb(a):
    pid, x, y, w = a
    warnings.filterwarnings("ignore")
    from exp14_tuebingen_relfit import MAX_N, SEED, dequantise
    rng = np.random.default_rng(SEED + int(pid))
    if len(x) > MAX_N:
        idx = np.sort(rng.choice(len(x), MAX_N, replace=False))
        x, y = x[idx], y[idx]
    x, y = dequantise(x, rng), dequantise(y, rng)
    return _rec("tuebingen", pid, "x->y", x, y, int(pid), w)


def chamber_iid(a):
    warnings.filterwarnings("ignore")
    di, i, j, seed = a
    X, var, edges = e20.load(*e20.DATASETS[di])
    rng = np.random.default_rng([seed, di, i, j])
    idx = rng.permutation(len(X))[:2000]
    x, y = X[idx, i], X[idx, j]
    flip = rng.random() < 0.5
    if flip:
        x, y = y, x
    return _rec("chamber_iid", f"{di}/{var[i]}-{var[j]}/{seed}", "y->x" if flip else "x->y", x, y, seed)


def chamber_walk(a):
    warnings.filterwarnings("ignore")
    si, i, j = a
    spec = e27.specs()[si]
    X, var, edges = e27.load(spec)
    rng = np.random.default_rng([20261500, si, i, j])
    idx = rng.permutation(len(X))[:2000]
    x, y = X[idx, i], X[idx, j]
    flip = rng.random() < 0.5
    if flip:
        x, y = y, x
    return _rec("chamber_walk", f"{spec[1]}/{var[i]}-{var[j]}", "y->x" if flip else "x->y", x, y, si)


def heldout(a):
    warnings.filterwarnings("ignore")
    fam, n, seed = a
    rng = np.random.default_rng(seed)
    x, y = e18.gen(fam, n, rng)
    flip = rng.random() < 0.5
    if flip:
        x, y = y, x
    return _rec("heldout", f"{fam}/{seed}", "y->x" if flip else "x->y", x, y, seed)


def main():
    from exp12_tuebingen_cep import load_pairs
    for di in range(len(e20.DATASETS)):
        e20.load(*e20.DATASETS[di])
    sp = e27.specs()
    firsts = {}
    for k, s in enumerate(sp):
        firsts.setdefault((s[0], s[3]) if s[0] != "lt_walks_v1" else (s[0], s[1]), k)
    jobs_c = []
    for di in range(len(e20.DATASETS)):
        _, var, edges = e20.load(*e20.DATASETS[di])
        jobs_c += [(di, i, j, s) for (i, j) in edges for s in e21.SEEDS]
    jobs_w = []
    for k in firsts.values():
        _, var, edges = e27.load(sp[k])
        jobs_w += [(k, i, j) for (i, j) in edges]
    jobs_h = [(f, 800, 1000 * k + 7) for f in e18.FAMILIES for k in range(30)]
    with Pool(4) as p:
        r = p.map(tueb, list(load_pairs()), chunksize=2)
        r += p.map(chamber_iid, jobs_c, chunksize=2)
        r += p.map(chamber_walk, jobs_w, chunksize=2)
        r += p.map(heldout, jobs_h, chunksize=4)
    (HERE.parent / "results" / "exp35_pdm_eval.json").write_text(json.dumps({"records": r}))


if __name__ == "__main__":
    main()
