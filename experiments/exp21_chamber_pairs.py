"""Experiment 21: direction of the known edges of the Causal Chambers.

Fixed before running.  Pairs: every directed edge of the chamber ground-truth
graph between two variables that vary (>= 20 distinct values) in the
experiments of exp20 (wind tunnel random_loads_intake, light tunnel
uniform_reference).  Each pair is analysed bivariately and marginally (other
variables ignored) on a random subsample of 2000 rows, for 2 seeds, with the
relative-fit certificate (alpha = .05) and the four forced-choice rules of
sprint_cd.pair_baselines.  The cause is the first variable of the edge.
Reported: agreement certificate decided/correct, each class, forced accuracies.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json, pathlib, sys, warnings
from multiprocessing import Pool
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from exp20_causal_chamber import DATASETS, load
from sprint_cd.pair_baselines import anm_hsic, igci, lingam_pair, reci
from sprint_cd.relative_fit import certify_direction

BASE = {"IGCI": igci, "RECI": reci, "ANM-HSIC": anm_hsic, "LiNGAM-pair": lingam_pair}
SEEDS = [20261201, 20261202]
N = 2000


def one(job):
    di, i, j, seed = job
    warnings.filterwarnings("ignore")
    X, var, edges = load(*DATASETS[di])
    rng = np.random.default_rng([seed, di, i, j])
    idx = rng.permutation(len(X))[:N]
    x, y = X[idx, i], X[idx, j]
    flip = rng.random() < 0.5
    if flip:
        x, y = y, x
    truth = "y->x" if flip else "x->y"
    r = certify_direction(x, y, seed=seed + 31 * i + j)
    return {"ds": di, "pair": [var[i], var[j]], "seed": seed, "truth": truth, "relfit": r,
            "base": {k: f(x, y) for k, f in BASE.items()}}


def main():
    for di in range(len(DATASETS)):
        load(*DATASETS[di])
    jobs = []
    for di in range(len(DATASETS)):
        _, var, edges = load(*DATASETS[di])
        jobs += [(di, i, j, s) for (i, j) in edges for s in SEEDS]
    part = HERE.parent / "results" / "exp21_chamber_pairs.partial.json"
    recs = json.loads(part.read_text())["records"] if part.exists() else []
    with Pool(4) as p:
        for s in range(len(recs), len(jobs), 40):
            recs.extend(p.map(one, jobs[s:s + 40], chunksize=2))
            part.write_text(json.dumps({"records": recs}))
            print(len(recs), "/", len(jobs), flush=True)
    (HERE.parent / "results" / "exp21_chamber_pairs.json").write_text(json.dumps({"records": recs}))
    part.unlink()


if __name__ == "__main__":
    main()
