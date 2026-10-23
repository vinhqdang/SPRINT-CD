"""Experiment 18: held-out simulated pair families, sample-size sweep, baselines.

Written and committed before running.  None of the families below was used while
designing the relative-fit certificate (those are the nine of exp16).  For each
family we report the commit rate and error rate of the agreement certificate and
of each class, and the forced accuracy of four published rules (IGCI, RECI,
ANM-HSIC, pairwise LiNGAM) on the same pairs.  A second part sweeps n on three
families.  The Tuebingen pairs are evaluated with the same four baselines under
the exp14 preprocessing (suite 'tuebingen').
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from multiprocessing import Pool

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from sprint_cd.pair_baselines import anm_hsic, igci, lingam_pair, reci  # noqa: E402
from sprint_cd.relative_fit import certify_direction  # noqa: E402

BASELINES = {"IGCI": igci, "RECI": reci, "ANM-HSIC": anm_hsic, "LiNGAM-pair": lingam_pair}
FAMILIES = ["h_cubic", "h_exp", "h_logistic", "h_lin_t3", "h_lin_skew",
            "h_mult", "h_het_cos", "h_pnl", "h_mixcause"]
SWEEP_FAMILIES = ["h_cubic", "h_lin_skew", "h_het_cos"]
SWEEP_N = [200, 400, 1600]


def gen(fam, n, rng):
    if fam == "h_cubic":
        x = rng.normal(size=n); y = 0.5 * x ** 3 + 0.5 * rng.laplace(size=n)
    elif fam == "h_exp":
        x = rng.uniform(-2, 2, n); y = np.exp(x / 1.5) + 0.3 * rng.normal(size=n)
    elif fam == "h_logistic":
        x = rng.normal(size=n); y = 1 / (1 + np.exp(-3 * x)) + 0.1 * rng.standard_t(5, n)
    elif fam == "h_lin_t3":
        x = rng.standard_t(3, n); y = 0.8 * x + rng.standard_t(3, n)
    elif fam == "h_lin_skew":
        x = rng.exponential(size=n) - 1; y = 0.8 * x + rng.exponential(size=n) - 1
    elif fam == "h_mult":
        x = rng.uniform(1, 3, n); y = x ** 1.5 * np.exp(0.3 * rng.normal(size=n))
    elif fam == "h_het_cos":
        x = rng.normal(size=n); y = np.cos(x) + (0.2 + 0.3 * np.abs(x)) * rng.normal(size=n)
    elif fam == "h_pnl":
        x = rng.normal(size=n); y = np.exp(0.5 * (x + 0.5 * rng.normal(size=n)))
    elif fam == "h_mixcause":
        z = rng.random(n) < 0.5
        x = np.where(z, rng.normal(-1.5, 0.6, n), rng.normal(1.5, 0.6, n))
        y = 0.8 * x + 0.5 * rng.normal(size=n)
    else:
        raise ValueError(fam)
    return x, y


def sim_one(a):
    fam, n, seed = a
    rng = np.random.default_rng(seed)
    x, y = gen(fam, n, rng)
    flip = rng.random() < 0.5
    if flip:
        x, y = y, x
    truth = "y->x" if flip else "x->y"
    r = certify_direction(x, y, seed=seed)
    base = {k: f(x, y) for k, f in BASELINES.items()}
    return {"fam": fam, "n": n, "truth": truth, "relfit": r, "base": base}


def tueb_one(a):
    pid, x, y, w = a
    from exp14_tuebingen_relfit import MAX_N, SEED, dequantise
    rng = np.random.default_rng(SEED + int(pid))
    if len(x) > MAX_N:
        idx = np.sort(rng.choice(len(x), MAX_N, replace=False))
        x, y = x[idx], y[idx]
    x, y = dequantise(x, rng), dequantise(y, rng)
    return {"pair": pid, "n": len(x), "weight": w,
            "base": {k: f(x, y) for k, f in BASELINES.items()}}



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
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["sim", "sweep", "tuebingen"], required=True)
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    with Pool(a.workers) as pool:
        if a.suite == "sim":
            jobs = [(f, a.n, 1000 * k + 7) for f in FAMILIES for k in range(a.reps)]
            recs = resumable_map(pool, sim_one, jobs, HERE.parent / 'results' / f'exp18_{a.suite}.partial.json')
        elif a.suite == "sweep":
            jobs = [(f, n, 1000 * k + 7) for f in SWEEP_FAMILIES for n in SWEEP_N
                    for k in range(a.reps)]
            recs = resumable_map(pool, sim_one, jobs, HERE.parent / 'results' / f'exp18_{a.suite}.partial.json')
        else:
            from exp12_tuebingen_cep import load_pairs
            recs = resumable_map(pool, tueb_one, list(load_pairs()), HERE.parent / 'results' / 'exp18_tuebingen.partial.json')
    out = {"config": vars(a), "records": recs}
    path = HERE.parent / "results" / f"exp18_{a.suite}.json"
    path.write_text(json.dumps(out))
    part = path.with_suffix(".partial.json")
    if part.exists():
        part.unlink()
    print("wrote", path)


if __name__ == "__main__":
    main()
