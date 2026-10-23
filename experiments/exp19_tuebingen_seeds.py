"""Experiment 19: sensitivity of the Tuebingen result to the random seed.

Written and committed before running.  exp14 fixed one seed that controls the
row permutation, the subsample of long pairs and the dequantisation of ties.
Here the identical procedure is repeated for 5 further seeds; the headline
count (12 decided, 11 correct for the agreement certificate) is a single draw,
and this experiment reports its spread.  Truth is taken from the benchmark
(x is the cause in every pair as returned by load_pairs).
"""
from __future__ import annotations

import json
import pathlib
import sys
from multiprocessing import Pool

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from exp12_tuebingen_cep import load_pairs  # noqa: E402
from exp14_tuebingen_relfit import MAX_N, dequantise  # noqa: E402
from sprint_cd.relative_fit import certify_direction  # noqa: E402

SEEDS = [20261100 + 17 * k for k in range(5)]


def one(a):
    pid, x, y, w, seed = a
    rng = np.random.default_rng(seed + int(pid))
    if len(x) > MAX_N:
        idx = np.sort(rng.choice(len(x), MAX_N, replace=False))
        x, y = x[idx], y[idx]
    x, y = dequantise(x, rng), dequantise(y, rng)
    r = certify_direction(x, y, alpha=0.05, seed=seed + int(pid))
    return {"seed": seed, "pair": pid, "weight": w, "r": r}



def resumable_map(pool, fn, jobs, part_path, chunk=40):
    """Map ``fn`` over ``jobs`` in chunks, checkpointing to ``part_path`` so a restart resumes."""
    import json as _j
    done = _j.loads(part_path.read_text())["records"] if part_path.exists() else []
    for s in range(len(done), len(jobs), chunk):
        done.extend(pool.map(fn, jobs[s:s + chunk], chunksize=2))
        part_path.write_text(_j.dumps({"records": done}))
        print(f"{len(done)}/{len(jobs)}", flush=True)
    return done


def main():
    pairs = list(load_pairs())
    jobs = [(p[0], p[1], p[2], p[3], s) for s in SEEDS for p in pairs]
    with Pool(4) as pool:
        recs = resumable_map(pool, one, jobs, HERE.parent / 'results' / 'exp19_tuebingen_seeds.partial.json', chunk=80)
    out = {"seeds": SEEDS, "records": recs}
    (HERE.parent / "results" / "exp19_tuebingen_seeds.json").write_text(json.dumps(out))
    pp = HERE.parent / "results" / "exp19_tuebingen_seeds.partial.json"
    if pp.exists():
        pp.unlink()
    print("wrote results/exp19_tuebingen_seeds.json")


if __name__ == "__main__":
    main()
