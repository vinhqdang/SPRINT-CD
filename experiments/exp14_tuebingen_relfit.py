"""Experiment 14: prequential relative-fit direction certificate on the Tuebingen pairs.

PROTOCOL (frozen before the first run; see the commit that introduces this file):
  pairs     all 102 pairs with scalar cause and effect (pairs 52-55, 71, 105 excluded
            as multivariate), official pairmeta weights, no exclusions after seeing results
  preproc   dequantise tied values (U(-D/2, D/2), D = smallest positive gap between
            distinct values), random permutation, subsample to at most 2000 points,
            standardise with the first 100 points
  method    sprint_cd.relative_fit.certify_direction, alpha = 0.05, seed fixed
  primary   the `agree` certificate (both model classes certify the same direction)
  reported  gg-only, ggh-only, forced sign accuracy, weighted and unweighted
No hyper-parameter was tuned on these data; design choices were made on simulated
pairs only.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import pathlib
from multiprocessing import Pool

import numpy as np
from scipy import stats

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import save_json  # noqa: E402
from exp12_tuebingen_cep import load_pairs  # noqa: E402
from sprint_cd.relative_fit import certify_direction  # noqa: E402

MAX_N = 2000
SEED = 20261023


def dequantise(v, rng):
    u = np.unique(v)
    if len(u) < len(v):
        gaps = np.diff(u)
        gaps = gaps[gaps > 0]
        if len(gaps):
            v = v + rng.uniform(-0.5, 0.5, size=len(v)) * gaps.min()
    return v


def one(args):
    pid, x, y, w = args
    rng = np.random.default_rng(SEED + int(pid))
    if len(x) > MAX_N:
        idx = np.sort(rng.choice(len(x), MAX_N, replace=False))
        x, y = x[idx], y[idx]
    x, y = dequantise(x, rng), dequantise(y, rng)
    r = certify_direction(x, y, alpha=0.05, seed=SEED + int(pid))
    return pid, len(x), w, r


def cp(k, n, a=0.05):
    if n == 0:
        return (0.0, 1.0)
    lo = stats.beta.ppf(a / 2, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(1 - a / 2, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    pairs = [(pid, x, y, w) for pid, x, y, w in load_pairs()]
    with Pool(args.workers) as p:
        res = p.map(one, pairs, chunksize=1)
    res = [r for r in res if r[3] is not None]
    wt = sum(r[2] for r in res)
    out = {"n_pairs": len(res), "weights_total": wt, "seed": SEED, "max_n": MAX_N,
           "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                 text=True).stdout.strip(), "kinds": {}}
    for kind in ("agree", "gg", "ggh"):
        dec = [r for r in res if r[3][kind] is not None]
        cor = [r for r in dec if r[3][kind] == "x->y"]
        k, n = len(cor), len(dec)
        wdec = sum(r[2] for r in dec)
        out["kinds"][kind] = {
            "decided": n, "correct": k, "coverage": n / len(res),
            "weighted_coverage": wdec / wt,
            "acc": k / n if n else None, "acc_cp95": cp(k, n),
            "weighted_acc": sum(r[2] for r in cor) / wdec if wdec else None,
            "binom_p_gt_half": float(stats.binomtest(k, n, 0.5, alternative="greater").pvalue) if n else None,
        }
    for f in ("forced_gg", "forced_ggh"):
        c = [r for r in res if r[3][f] == "x->y"]
        out["kinds"][f] = {"acc": len(c) / len(res), "weighted_acc": sum(r[2] for r in c) / wt}
    out["rows"] = [{"pair": r[0], "n": r[1], "weight": r[2], **r[3]} for r in res]
    save_json("exp14_tuebingen_relfit", out)
    for k, v in out["kinds"].items():
        print(k, v)


if __name__ == "__main__":
    main()
