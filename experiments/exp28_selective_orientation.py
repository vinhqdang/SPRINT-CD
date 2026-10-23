"""Experiment 28: abstaining comparison of the pair rules at matched coverage.

Written and committed before running.  For each pair set (Tuebingen, 102 pairs
under the exp14 preprocessing; the held-out simulated families of exp18 with the
same seeds; the Causal Chamber edge pairs of exp21 with the same subsamples) this
stores, for each of IGCI, RECI, ANM-HSIC and the pairwise LiNGAM rule, the direction
and a confidence score.  scripts/analyse_exp28.py ranks pairs by confidence and
reports the accuracy of the k most confident pairs, with k equal to the number of
pairs on which the agreement certificate (or either class) commits, so the rules
are compared at the same coverage.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json, pathlib, sys, warnings
from multiprocessing import Pool
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from sprint_cd.pair_baselines import scored
import exp18_pairs_heldout as e18
import exp21_chamber_pairs as e21
from exp20_causal_chamber import DATASETS, load as load20


def tueb(a):
    pid, x, y, w = a
    from exp14_tuebingen_relfit import MAX_N, SEED, dequantise
    rng = np.random.default_rng(SEED + int(pid))
    if len(x) > MAX_N:
        idx = np.sort(rng.choice(len(x), MAX_N, replace=False))
        x, y = x[idx], y[idx]
    x, y = dequantise(x, rng), dequantise(y, rng)
    return {"set": "tuebingen", "id": pid, "weight": w, "truth": "x->y", "n": len(x), "scores": scored(x, y)}


def sim(a):
    fam, n, seed = a
    rng = np.random.default_rng(seed)
    x, y = e18.gen(fam, n, rng)
    flip = rng.random() < 0.5
    if flip:
        x, y = y, x
    return {"set": "heldout", "id": f"{fam}/{seed}", "truth": "y->x" if flip else "x->y", "scores": scored(x, y, seed)}


def chamber(a):
    di, i, j, seed = a
    X, var, edges = load20(*DATASETS[di])
    rng = np.random.default_rng([seed, di, i, j])
    idx = rng.permutation(len(X))[:e21.N]
    x, y = X[idx, i], X[idx, j]
    flip = rng.random() < 0.5
    if flip:
        x, y = y, x
    return {"set": "chamber", "id": f"{di}/{var[i]}-{var[j]}/{seed}", "truth": "y->x" if flip else "x->y",
            "scores": scored(x, y, seed)}


def main():
    warnings.filterwarnings("ignore")
    from exp12_tuebingen_cep import load_pairs
    for di in range(len(DATASETS)):
        load20(*DATASETS[di])
    jobs_c = []
    for di in range(len(DATASETS)):
        _, var, edges = load20(*DATASETS[di])
        jobs_c += [(di, i, j, s) for (i, j) in edges for s in e21.SEEDS]
    jobs_s = [(f, 800, 1000 * k + 7) for f in e18.FAMILIES for k in range(30)]
    with Pool(4) as p:
        r = p.map(tueb, list(load_pairs()), chunksize=2)
        r += p.map(sim, jobs_s, chunksize=4)
        r += p.map(chamber, jobs_c, chunksize=2)
    (HERE.parent / "results" / "exp28_selective_orientation.json").write_text(json.dumps({"records": r}))


if __name__ == "__main__":
    main()
