"""Does the fitted null likelihood actually attain its supremum over Theta?

Universal inference bounds ``E_t`` by a martingale only if the denominator's
maximised null likelihood is at least the likelihood at the true parameter.
When the fit maximises over a strict subset of Theta that excludes the truth,
the subtracted term falls *below* the true-parameter likelihood, ``E_t`` is
inflated by the shortfall, and nothing in the output signals it.

Run this whenever the fitting procedure changes.  It simulates from a
correctly specified null at a known theta-star and reports how often, and by
how much, the fitted "supremum" falls short.

The sweep covers the whole of ``_KAPPA_BOUNDS``.  Checking one shape is not
enough and is actively misleading if that shape is a grid node: an earlier
version of this code was verified at kappa = 1 immediately after adding 1.0 to
the shape grid, so the check could not distinguish a working refinement from a
lucky grid hit.  It reported zero shortfall while failing at 122 of 200
replications at kappa = 0.5.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from sprint_cd.certificates import (_KAPPA_BOUNDS, fit_gg_regression,  # noqa: E402
                                    gg_logpdf)


def gg_sample(kappa: float, size: int, rng: np.random.Generator) -> np.ndarray:
    """Draw from the generalised Gaussian with unit scale: |x|^kappa ~ Gamma(1/kappa)."""
    g = rng.gamma(1.0 / kappa, 1.0, size)
    return np.sign(rng.uniform(-1.0, 1.0, size)) * g ** (1.0 / kappa)


def _one_shape(args):
    k, reps, n, seed = args
    rng = np.random.default_rng(seed)
    bad, worst = 0, 0.0
    for _ in range(reps):
        X = np.column_stack([np.ones(n), rng.normal(size=n)])
        beta = np.array([0.3, 0.8])
        sigma = 1.0
        y = X @ beta + sigma * gg_sample(k, n, rng)
        ll_fit = fit_gg_regression(y, X)[0]
        ll_true = gg_logpdf(y - X @ beta, sigma, k).sum()
        short = ll_true - ll_fit
        if short > 1e-8:
            bad += 1
            worst = max(worst, short)
    return (float(k), bad, reps, worst)


def sweep_parallel(shapes, reps: int, n: int, seed: int, workers: int):
    jobs = [(float(k), reps, n, seed + i) for i, k in enumerate(shapes)]
    if workers <= 1:
        return [_one_shape(j) for j in jobs]
    with mp.Pool(workers) as pool:
        return pool.map(_one_shape, jobs)


def sweep(shapes, reps: int, n: int, seed: int):
    rng = np.random.default_rng(seed)
    rows = []
    for k in shapes:
        bad, worst = 0, 0.0
        for _ in range(reps):
            X = np.column_stack([np.ones(n), rng.normal(size=n)])
            beta = np.array([0.3, 0.8])
            sigma = 1.0
            y = X @ beta + sigma * gg_sample(k, n, rng)
            ll_fit = fit_gg_regression(y, X)[0]
            ll_true = gg_logpdf(y - X @ beta, sigma, k).sum()
            short = ll_true - ll_fit      # > 0 means the "supremum" fell short
            if short > 1e-8:
                bad += 1
                worst = max(worst, short)
        rows.append((float(k), bad, reps, worst))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--points", type=int, default=12)
    ap.add_argument("--tol-nats", type=float, default=0.02,
                    help="shortfall below this is treated as numerical "
                         "optimiser noise (IRLS/Brent tolerance), not a "
                         "supremum-attainment failure; the original defect "
                         "this script exists to catch was 2.4 nats")
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--workers", type=int, default=1,
                    help="parallelise the sweep across shapes")
    args = ap.parse_args()

    lo, hi = _KAPPA_BOUNDS
    # Deliberately include points that are NOT grid nodes -- an off-node sweep
    # is the only kind that can distinguish refinement from a lucky hit.
    shapes = np.exp(np.linspace(np.log(lo * 1.03), np.log(hi * 0.97), args.points))

    print(f"\nSupremum-attainment check over Theta = [{lo}, {hi}]")
    print(f"{args.reps} replications per shape, n = {args.n}, "
          f"{args.workers} worker(s)\n")
    print(f"  {'true kappa':>10} {'shortfall':>12} {'worst nats':>11} "
          f"{'E_t inflation':>14}")
    print("  " + "-" * 52)
    rows = sweep_parallel(shapes, args.reps, args.n, args.seed, args.workers)
    rows.sort(key=lambda r: r[0])
    failed = 0
    for k, bad, reps, worst in rows:
        infl = np.exp(min(worst, 700.0))
        flag = ""
        if worst > args.tol_nats:
            failed += 1
            flag = "  <-- SHORTFALL"
        print(f"  {k:>10.3f} {f'{bad}/{reps}':>12} {worst:>11.3f} "
              f"{infl:>14.4g}{flag}")

    if failed:
        print(f"\nFAIL: the fitted supremum falls below the true-parameter "
              f"likelihood at {failed} of {len(rows)} shapes.")
        print("The direction certificate is NOT anytime-valid over this Theta.")
        return 1
    print("\nPASS: no shortfall anywhere in Theta. The denominator attains its "
          "supremum\nto within numerical tolerance across the sweep.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
