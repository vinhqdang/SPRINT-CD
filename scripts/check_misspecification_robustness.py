"""Reproduction script for Remark 6's misspecification-robustness claim.

Remark 6 (paper/jjsds/part3.tex) reports that under three noise laws each
breaking a different property of the fitted generalised-Gaussian family
(Student-t3: polynomial tails; shifted exponential: asymmetric; a
two-component mixture: bimodal), the reversed direction was never wrongly
certified in 12 runs at a single coefficient of 0.9. Review found two
problems with that check, neither previously reproducible from a script:

1. 0 of 12 has a Wilson upper limit of 0.247 -- it cannot rule out a true
   error rate five times the nominal alpha.
2. Coefficient 0.9 is an easy design: the causal signal is strong relative to
   the noise, so the correct direction's evidence is overwhelming regardless
   of the noise law. The check says nothing about weaker signals, where
   misspecification has more room to matter.

This script reruns the original design at higher replication for a tighter
Wilson bound, and adds a coefficient sweep (0.9, 0.5, 0.3) to probe whether
the error rate rises as the signal weakens.
"""

from __future__ import annotations

import argparse
import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from sprint_cd.certificates import DirectionEProcess  # noqa: E402

NOISE_LAWS = ["student_t3", "shifted_exp", "mixture"]


def wilson_interval(wrong: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return 0.0, 1.0
    phat = wrong / total
    denom = 1 + z * z / total
    centre = phat + z * z / (2 * total)
    half = z * np.sqrt(phat * (1 - phat) / total + z * z / (4 * total * total))
    return (centre - half) / denom, (centre + half) / denom


def draw_noise(law: str, size: int, rng: np.random.Generator) -> np.ndarray:
    if law == "student_t3":
        x = rng.standard_t(3, size=size)
        return x / x.std()
    if law == "shifted_exp":
        x = rng.exponential(1.0, size=size)
        x = x - x.mean()
        return x / x.std()
    if law == "mixture":
        comp = rng.uniform(size=size) < 0.5
        x = np.where(comp, rng.normal(-2.0, 1.0, size), rng.normal(2.0, 1.0, size))
        return (x - x.mean()) / x.std()
    raise ValueError(law)


def one_run(law: str, coef: float, n: int, batch: int, alpha: float,
            min_fit: int, rng: np.random.Generator) -> bool:
    """Returns True iff the reversed direction (y -> x) was wrongly certified."""
    x = draw_noise(law, n, rng)
    y = coef * x + draw_noise(law, n, rng)
    thr = -np.log(alpha / 2.0)
    reversed_proc = DirectionEProcess(n_cond=1, min_fit=min_fit)  # tests x->y; crossing certifies y->x
    for s in range(0, n, batch):
        xb, yb = x[s:s + batch], y[s:s + batch]
        Z = np.ones((xb.size, 1))
        if reversed_proc.update(yb, xb, Z) >= thr:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--min-fit", type=int, default=60)
    ap.add_argument("--coefs", type=float, nargs="+", default=[0.9, 0.5, 0.3])
    ap.add_argument("--seed", type=int, default=20260909)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    print(f"\nMisspecification robustness: false-reversal rate over {args.reps} "
          f"runs per cell\n")
    print(f"  {'noise law':13s} {'coef':>5s} {'wrong':>6s} {'rate':>7s}   95% CI")
    print("  " + "-" * 52)
    worst = 0.0
    for law in NOISE_LAWS:
        for coef in args.coefs:
            wrong = sum(one_run(law, coef, args.n, args.batch, args.alpha,
                                 args.min_fit, rng)
                        for _ in range(args.reps))
            lo, hi = wilson_interval(wrong, args.reps)
            worst = max(worst, hi)
            print(f"  {law:13s} {coef:5.1f} {wrong:6d} {wrong/args.reps:7.4f}   "
                  f"[{lo:.4f}, {hi:.4f}]")
    print(f"\nWorst Wilson upper bound across all cells: {worst:.4f} "
          f"(nominal alpha/2 = {args.alpha/2:.4f})")


if __name__ == "__main__":
    main()
