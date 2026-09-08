"""Experiment 1: what continuous monitoring does to a fixed-sample test.

Data are generated under the null ``X indep Y | Z`` and the analyst inspects
the evidence repeatedly as observations accumulate, stopping as soon as the
result looks significant -- the behaviour a streaming setting invites.

Three procedures are compared at matched nominal levels:

``naive Fisher-z``
    Reject if the fixed-sample p-value falls below ``alpha`` at *any*
    monitoring point.  This is the standard practice whose error rate the
    e-process construction is meant to repair.
``fixed-sample Fisher-z``
    Reject on the p-value at the final sample size only.  Correctly
    calibrated, but it cannot stop early.
``SPRINT-CD e-process``
    Reject if the e-process ever crosses ``1/alpha``.  Anytime-valid by
    Ville's inequality.

The point of the experiment is not that the naive procedure is invalid --
that is well known -- but to quantify the gap on exactly the conditional
independence queries a discovery algorithm issues, and to confirm that the
e-process closes it without collapsing power (checked separately below).
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy import stats

from common import (PALETTE, RESULTS, save_json, setup_matplotlib,
                    wilson_interval, residual_moments_at, running_cross_products)

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from sprint_cd.eprocess.safe_linear import safe_linear_log_e
from sprint_cd.stats import ResidualMoments


def simulate_null(n: int, rng: np.random.Generator) -> np.ndarray:
    """X and Y share two common causes but are conditionally independent."""
    Z = rng.normal(size=(n, 2))
    X = 0.9 * Z[:, 0] - 0.5 * Z[:, 1] + rng.normal(size=n)
    Y = 0.7 * Z[:, 0] + 0.6 * Z[:, 1] + rng.normal(size=n)
    return np.column_stack([X, Y, Z])


def simulate_alt(n: int, rng: np.random.Generator, beta: float) -> np.ndarray:
    Z = rng.normal(size=(n, 2))
    Y = 0.7 * Z[:, 0] + 0.6 * Z[:, 1] + rng.normal(size=n)
    X = 0.9 * Z[:, 0] - 0.5 * Z[:, 1] + beta * Y + rng.normal(size=n)
    return np.column_stack([X, Y, Z])


def _fisher_z_p(sxx, sxy, syy, dof, n, k):
    n_eff = n - k - 3
    if n_eff <= 0 or sxx <= 0 or syy <= 0:
        return 1.0
    r = float(np.clip(sxy / np.sqrt(sxx * syy), -0.999999, 0.999999))
    z = np.arctanh(r)
    return float(2.0 * stats.norm.sf(np.sqrt(n_eff) * abs(z)))


def run(reps: int, nmax: int, alphas, g: float, seed: int, beta_alt: float):
    rng = np.random.default_rng(seed)
    grid = np.arange(19, nmax, 10)          # monitoring points, shared by all methods

    out = {a: {"naive": 0, "fixed": 0, "eproc": 0} for a in alphas}
    power = {a: {"naive": 0, "fixed": 0, "eproc": 0} for a in alphas}
    stop_n = {a: [] for a in alphas}

    for rep in range(reps):
        for mode, tally in (("null", out), ("alt", power)):
            D = (simulate_null(nmax, rng) if mode == "null"
                 else simulate_alt(nmax, rng, beta_alt))
            csum, ccross = running_cross_products(D)

            min_p = 1.0
            max_loge = -np.inf
            first_cross = {a: None for a in alphas}
            for t in grid:
                sxx, sxy, syy, dof = residual_moments_at(csum, ccross, t, 0, 1, (2, 3))
                if dof <= 0:
                    continue
                p = _fisher_z_p(sxx, sxy, syy, dof, t + 1, 2)
                min_p = min(min_p, p)
                le = safe_linear_log_e(
                    ResidualMoments(sxx, sxy, syy, dof, t + 1), g)
                max_loge = max(max_loge, le)
                for a in alphas:
                    if first_cross[a] is None and max_loge >= -np.log(a):
                        first_cross[a] = t + 1
            final_p = p

            for a in alphas:
                tally[a]["naive"] += min_p <= a
                tally[a]["fixed"] += final_p <= a
                tally[a]["eproc"] += max_loge >= -np.log(a)
                if mode == "alt" and first_cross[a] is not None:
                    stop_n[a].append(first_cross[a])

    return out, power, stop_n, reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--nmax", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--beta-alt", type=float, default=0.25)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.reps, args.nmax = 300, 500

    alphas = [0.01, 0.05, 0.10, 0.20]
    g = 0.4**2
    null, alt, stop_n, reps = run(args.reps, args.nmax, alphas, g,
                                  args.seed, args.beta_alt)

    rows = []
    for a in alphas:
        row = {"alpha": a, "reps": reps}
        for m in ("naive", "fixed", "eproc"):
            k = null[a][m]
            lo, hi = wilson_interval(k, reps)
            row[f"type1_{m}"] = k / reps
            row[f"type1_{m}_ci"] = [lo, hi]
            row[f"power_{m}"] = alt[a][m] / reps
        row["median_stop_eproc"] = (float(np.median(stop_n[a])) if stop_n[a] else None)
        rows.append(row)

    print(f"\nExperiment 1 -- continuous monitoring of a null CI query "
          f"({reps} reps, n<={args.nmax})\n")
    print(f"{'alpha':>6} | {'naive Fisher-z':>16} | {'fixed-sample':>14} | "
          f"{'SPRINT e-process':>17} | {'power (e-proc)':>14}")
    print("-" * 82)
    for r in rows:
        print(f"{r['alpha']:>6} | {r['type1_naive']:>16.4f} | {r['type1_fixed']:>14.4f} | "
              f"{r['type1_eproc']:>17.4f} | {r['power_eproc']:>14.3f}")
    print("\nType-I error must not exceed alpha.  The naive column is the cost of "
          "peeking;\nthe e-process column is the same peeking made valid.\n")

    save_json("exp1_type1_calibration", {"config": vars(args), "rows": rows})

    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    ax = axes[0]
    x = np.arange(len(alphas))
    w = 0.26
    for off, key, lbl, col in ((-w, "naive", "Fisher-z, monitored", PALETTE["naive"]),
                               (0.0, "fixed", "Fisher-z, fixed n", PALETTE["fixed"]),
                               (w, "eproc", "SPRINT-CD e-process", PALETTE["sprint"])):
        vals = [null[a][key] / reps for a in alphas]
        errs = np.array([[v - wilson_interval(null[a][key], reps)[0] for v, a in zip(vals, alphas)],
                         [wilson_interval(null[a][key], reps)[1] - v for v, a in zip(vals, alphas)]])
        ax.bar(x + off, vals, w, label=lbl, color=col, yerr=errs, capsize=2,
               error_kw={"lw": 0.8})
    ax.plot(x, alphas, "k--", lw=1.1, label="nominal level", zorder=5)
    ax.set_xticks(x); ax.set_xticklabels([str(a) for a in alphas])
    ax.set_xlabel(r"nominal level $\alpha$"); ax.set_ylabel("Type-I error rate")
    ax.set_title("Null holds: error under continuous monitoring")
    ax.legend(fontsize=7.5)

    ax = axes[1]
    for key, lbl, col in (("naive", "Fisher-z, monitored", PALETTE["naive"]),
                          ("fixed", "Fisher-z, fixed n", PALETTE["fixed"]),
                          ("eproc", "SPRINT-CD e-process", PALETTE["sprint"])):
        ax.plot(alphas, [alt[a][key] / reps for a in alphas], "o-", color=col,
                label=lbl, ms=4)
    ax.set_xlabel(r"nominal level $\alpha$"); ax.set_ylabel("detection rate")
    ax.set_title(rf"Alternative $\beta={args.beta_alt}$: power")
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    out_png = RESULTS / "exp1_type1_calibration.png"
    fig.savefig(out_png, bbox_inches="tight")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
