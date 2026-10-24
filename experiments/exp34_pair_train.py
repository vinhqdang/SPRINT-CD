"""Experiment 34: train the learned pair-direction model (PDM) on simulated pairs.

Training pairs come from sprint_cd.pairsim (random cause distributions, 12 mechanism families,
6 noise laws, additive / multiplicative / location-scale / post-nonlinear structure, random
signal-to-noise ratio, quantisation, measurement noise, serial dependence; n from 100 to 1500).
Features: sprint_cd.pairmodel.  Each pair is flipped at random.  Split: 40000 train, 5000
validation, 5000 simulated test (disjoint seeds).  No Tuebingen or Chamber data are used.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import pathlib, sys, warnings
from multiprocessing import Pool
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from sprint_cd.pairmodel import pair_features
from sprint_cd.pairsim import gen_pair


def one(seed):
    warnings.filterwarnings("ignore")
    x, y = gen_pair(seed)
    rng = np.random.default_rng([7, seed])
    flip = rng.random() < 0.5
    if flip:
        x, y = y, x
    return pair_features(x, y, seed=seed), 0 if flip else 1


def main(n_train=40000, n_val=5000, n_test=5000):
    seeds = list(range(n_train + n_val + n_test))
    with Pool(4) as p:
        res = p.map(one, seeds, chunksize=50)
    np.savez_compressed(HERE.parent / "results" / "exp34_pair_features.npz",
                        F=np.vstack([f for f, y in res]), y=np.array([y for f, y in res]),
                        n_train=n_train, n_val=n_val, n_test=n_test)


if __name__ == "__main__":
    main()
