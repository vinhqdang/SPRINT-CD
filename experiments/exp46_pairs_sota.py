"""Experiment 46: published pair rules with public code (AVICI pretrained, RESIT) on the exp38 pair sets.

Written and committed before running.  AVICI (scm-v0, two-variable input, score = P(x->y) - P(y->x), at most 1000
rows) and RESIT (lingam, random-forest regression depth 4, alpha 0.01, at most 1000 rows).  PNL (causal-learn) was
tried and dropped: 880 s per pair on 400 rows.  The text below is the exp42 description.

Written and committed before running.  Sorted-conditional bz2 compression (arXiv 2606.23767; on the
rows as given "SCraw" and on dequantised rows "SC") and Topological Residual Asymmetry (arXiv
2602.00427; kappa=1, c_beta=4, K=5 fixed a priori, rows capped at 1000) are scored on the exp38 sets.
The text below is the exp38 description, kept for the shared protocol.

Written and committed before running.  LOCI (Immer et al., ICML 2023; neural network with
the HSIC independence test, 5000 steps, authors' code fetched at run time) and ANM with a
Gaussian-process regression (causal-learn; direction with the larger independence p-value)
are run on the pair sets of exp35 under the same preprocessing and the same rows: Tuebingen
(102), Causal Chambers i.i.d. (94 pair-runs), Causal Chambers walks (122), each pair capped at
1000 rows (400 for ANM).  Sachs pairs (new set): the 17 directed edges of the consensus
network on the log-transformed observational data (853 cells), analysed with every pair rule
(IGCI, RECI, ANM-HSIC, LiNGAM, PDM, LOCI, ANM).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json, pathlib, sys, warnings
from multiprocessing import Pool
import numpy as np
import joblib

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from sprint_cd.sota_wrappers import resit_pair, avici_pairs
from sprint_cd.pair_baselines import scored
from sprint_cd.pairmodel import pair_features
import exp35_pdm_eval as e35
import exp20_causal_chamber as e20
import exp21_chamber_pairs as e21
import exp27_chamber_fresh as e27
import exp10_sachs as e10

LOCI_N, ANM_N = 1000, 400


def _anm(x, y):
    from causallearn.search.FCMBased.ANM.ANM import ANM
    rng = np.random.default_rng(0)
    if len(x) > ANM_N:
        idx = rng.choice(len(x), ANM_N, replace=False); x, y = x[idx], y[idx]
    p1, p2 = ANM().cause_or_effect(((x - x.mean()) / x.std()).reshape(-1, 1), ((y - y.mean()) / y.std()).reshape(-1, 1))
    return float(p1 - p2)


def _methods(x, y, seed, xr=None, yr=None):
    try: r = resit_pair(x, y)
    except Exception: r = None
    return {"RESIT": r, "_xy": [x.tolist(), y.tolist()]}


def tueb(a):
    warnings.filterwarnings("ignore")
    pid, x, y, w = a
    from exp14_tuebingen_relfit import MAX_N, SEED, dequantise
    rng = np.random.default_rng(SEED + int(pid))
    if len(x) > MAX_N:
        idx = np.sort(rng.choice(len(x), MAX_N, replace=False)); x, y = x[idx], y[idx]
    xr, yr = x.copy(), y.copy()
    x, y = dequantise(x, rng), dequantise(y, rng)
    return {"set": "tuebingen", "id": pid, "truth": "x->y", "weight": w, "m": _methods(x, y, int(pid), xr, yr)}


def chamber_iid(a):
    warnings.filterwarnings("ignore")
    di, i, j, seed = a
    X, var, edges = e20.load(*e20.DATASETS[di])
    rng = np.random.default_rng([seed, di, i, j]); idx = rng.permutation(len(X))[:2000]
    x, y = X[idx, i], X[idx, j]; flip = rng.random() < 0.5
    if flip: x, y = y, x
    return {"set": "chamber_iid", "id": f"{di}/{var[i]}-{var[j]}/{seed}", "truth": "y->x" if flip else "x->y", "m": _methods(x, y, seed)}


def chamber_walk(a):
    warnings.filterwarnings("ignore")
    si, i, j = a
    spec = e27.specs()[si]; X, var, edges = e27.load(spec)
    rng = np.random.default_rng([20261500, si, i, j]); idx = rng.permutation(len(X))[:2000]
    x, y = X[idx, i], X[idx, j]; flip = rng.random() < 0.5
    if flip: x, y = y, x
    return {"set": "chamber_walk", "id": f"{spec[1]}/{var[i]}-{var[j]}", "truth": "y->x" if flip else "x->y", "m": _methods(x, y, si)}


def sachs_pair(a):
    warnings.filterwarnings("ignore")
    ca, ef = a
    X, names = e10.load_sachs(); X = np.log(np.maximum(X, 1e-3))
    i, j = names.index(ca), names.index(ef)
    rng = np.random.default_rng([20261700, i, j]); flip = rng.random() < 0.5
    x, y = X[:, i], X[:, j]
    if flip: x, y = y, x
    return {"set": "sachs", "id": f"{ca}->{ef}", "truth": "y->x" if flip else "x->y",
            "m": _methods(x, y, 1, x, y)}


def main():
    from exp12_tuebingen_cep import load_pairs
    for di in range(len(e20.DATASETS)): e20.load(*e20.DATASETS[di])
    sp = e27.specs(); firsts = {}
    for k, s in enumerate(sp):
        firsts.setdefault((s[0], s[3]) if s[0] != "lt_walks_v1" else (s[0], s[1]), k)
    jobs = [("t", a) for a in list(load_pairs())]
    for di in range(len(e20.DATASETS)):
        _, var, edges = e20.load(*e20.DATASETS[di]); jobs += [("c", (di, i, j, s)) for (i, j) in edges for s in e21.SEEDS]
    for k in firsts.values():
        _, var, edges = e27.load(sp[k]); jobs += [("w", (k, i, j)) for (i, j) in edges]
    jobs += [("s", p) for p in e10.CONSENSUS]
    fn = {"t": tueb, "c": chamber_iid, "w": chamber_walk, "s": sachs_pair}
    part = HERE.parent / "results" / "exp46_pairs_sota.partial.json"
    recs = json.loads(part.read_text())["records"] if part.exists() else []
    todo = jobs[len(recs):]
    with Pool(2) as p:
        for s in range(0, len(todo), 40):
            recs.extend(p.map(_call, [(fn[k].__name__, a) for k, a in todo[s:s + 40]], chunksize=2))
            part.write_text(json.dumps({"records": recs})); print(len(recs), "/", len(jobs), flush=True)
    sc = avici_pairs([(np.array(r["m"]["_xy"][0]), np.array(r["m"]["_xy"][1])) for r in recs])
    for r, v in zip(recs, sc):
        r["m"]["AVICI"] = v; r["m"].pop("_xy")
    (HERE.parent / "results" / "exp46_pairs_sota.json").write_text(json.dumps({"records": recs}))
    part.unlink()


def _call(a):
    name, args = a
    return globals()[name](args)


if __name__ == "__main__":
    main()
