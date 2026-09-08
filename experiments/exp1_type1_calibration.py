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

from common import (PALETTE, figure_path, save_json, setup_matplotlib,
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


def run_null(reps: int, nmax: int, alphas, g: float, seed: int):
    """Type-I error under continuous monitoring, for each nominal level."""
    rng = np.random.default_rng(seed)
    grid = np.arange(19, nmax, 10)        # monitoring points, shared by all methods
    tally = {a: {"naive": 0, "fixed": 0, "eproc": 0} for a in alphas}

    for _ in range(reps):
        D = simulate_null(nmax, rng)
        csum, ccross = running_cross_products(D)
        min_p, max_loge, final_p = 1.0, -np.inf, 1.0
        for t in grid:
            sxx, sxy, syy, dof = residual_moments_at(csum, ccross, t, 0, 1, (2, 3))
            if dof <= 0:
                continue
            final_p = _fisher_z_p(sxx, sxy, syy, dof, t + 1, 2)
            min_p = min(min_p, final_p)
            max_loge = max(max_loge, safe_linear_log_e(
                ResidualMoments(sxx, sxy, syy, dof, t + 1), g))
        for a in alphas:
            tally[a]["naive"] += min_p <= a
            tally[a]["fixed"] += final_p <= a
            tally[a]["eproc"] += max_loge >= -np.log(a)
    return tally


def run_power(reps: int, nmax: int, alpha: float, g: float, seed: int, betas):
    """Detection rate against effect size, at a single nominal level.

    Reported as a curve rather than at one alternative.  At any effect the
    tests all detect easily every method sits at 1.0 and the comparison says
    nothing; what the e-process gives up relative to a correctly calibrated
    fixed-sample test is visible only where power is actually changing.
    """
    rng = np.random.default_rng(seed + 1)
    grid = np.arange(19, nmax, 10)
    out = {b: {"fixed": 0, "eproc": 0, "stop": []} for b in betas}

    for beta in betas:
        for _ in range(reps):
            D = simulate_alt(nmax, rng, beta)
            csum, ccross = running_cross_products(D)
            max_loge, final_p, first = -np.inf, 1.0, None
            for t in grid:
                sxx, sxy, syy, dof = residual_moments_at(csum, ccross, t, 0, 1, (2, 3))
                if dof <= 0:
                    continue
                final_p = _fisher_z_p(sxx, sxy, syy, dof, t + 1, 2)
                max_loge = max(max_loge, safe_linear_log_e(
                    ResidualMoments(sxx, sxy, syy, dof, t + 1), g))
                if first is None and max_loge >= -np.log(alpha):
                    first = t + 1
            out[beta]["fixed"] += final_p <= alpha
            out[beta]["eproc"] += max_loge >= -np.log(alpha)
            if first is not None:
                out[beta]["stop"].append(first)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--nmax", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.reps, args.nmax = 300, 500

    alphas = [0.01, 0.05, 0.10, 0.20]
    betas = [0.0, 0.04, 0.06, 0.08, 0.10, 0.15]
    g = 0.4**2
    reps = args.reps
    n_pow = max(reps // 4, 100)

    null = run_null(reps, args.nmax, alphas, g, args.seed)
    power = run_power(n_pow, args.nmax, 0.05, g, args.seed, betas)

    rows = []
    for a in alphas:
        row = {"alpha": a, "reps": reps}
        for m in ("naive", "fixed", "eproc"):
            k = null[a][m]
            lo, hi = wilson_interval(k, reps)
            row[f"type1_{m}"] = k / reps
            row[f"type1_{m}_ci"] = [lo, hi]
        rows.append(row)

    print(f"\nExperiment 1 -- continuous monitoring of a null CI query "
          f"({reps} reps, n<={args.nmax})\n")
    print(f"{'alpha':>6} | {'naive Fisher-z':>16} | {'fixed-sample':>14} | "
          f"{'SPRINT e-process':>17}")
    print("-" * 64)
    for r in rows:
        print(f"{r['alpha']:>6} | {r['type1_naive']:>16.4f} | "
              f"{r['type1_fixed']:>14.4f} | {r['type1_eproc']:>17.4f}")
    print("\nType-I error must not exceed alpha.  The naive column is the cost of "
          "peeking;\nthe e-process column is the same peeking made valid.\n")

    print(f"Power at alpha=0.05 ({n_pow} reps per effect size, n<={args.nmax}):")
    print(f"  {'beta':>6} | {'fixed-sample':>13} | {'e-process':>10} | "
          f"{'median stop':>12}")
    print("  " + "-" * 50)
    for b in betas:
        st = power[b]["stop"]
        med = f"{np.median(st):.0f}" if st else "-"
        print(f"  {b:>6} | {power[b]['fixed'] / n_pow:>13.3f} | "
              f"{power[b]['eproc'] / n_pow:>10.3f} | {med:>12}")
    print("\nThe e-process gives up a little power relative to a fixed-sample test\n"
          "that cannot be monitored, and returns a stopping time in exchange.\n")

    save_json("exp1_type1_calibration", quick=args.quick, payload=
              {"config": vars(args), "rows": rows,
               "power": {str(b): {"fixed": power[b]["fixed"],
                                  "eproc": power[b]["eproc"], "reps": n_pow,
                                  "median_stop": (float(np.median(power[b]["stop"]))
                                                  if power[b]["stop"] else None)}
                         for b in betas}})

    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    ax = axes[0]
    x = np.arange(len(alphas))
    w = 0.26
    for off, key, lbl, col in ((-w, "naive", "Fisher-z, monitored", PALETTE["naive"]),
                               (0.0, "fixed", "Fisher-z, fixed $n$", PALETTE["fixed"]),
                               (w, "eproc", "SPRINT-CD e-process", PALETTE["sprint"])):
        vals = [null[a][key] / reps for a in alphas]
        errs = np.array([
            [v - wilson_interval(null[a][key], reps)[0] for v, a in zip(vals, alphas)],
            [wilson_interval(null[a][key], reps)[1] - v for v, a in zip(vals, alphas)]])
        ax.bar(x + off, vals, w, label=lbl, color=col, yerr=errs, capsize=2,
               error_kw={"lw": 0.8})
    ax.plot(x, alphas, "k--", lw=1.1, label="nominal level", zorder=5)
    ax.set_xticks(x); ax.set_xticklabels([str(a) for a in alphas])
    ax.set_xlabel(r"nominal level $\alpha$"); ax.set_ylabel("Type-I error rate")
    ax.set_ylim(0, None)
    ax.set_title("Null holds: error under continuous monitoring")
    ax.legend(fontsize=7.5)

    ax = axes[1]
    ax.plot(betas, [power[b]["fixed"] / n_pow for b in betas], "s--",
            color=PALETTE["fixed"], label="Fisher-z, fixed $n$", ms=4)
    ax.plot(betas, [power[b]["eproc"] / n_pow for b in betas], "o-",
            color=PALETTE["sprint"], label="SPRINT-CD e-process", ms=4)
    ax.axhline(0.05, color="k", ls=":", lw=0.9)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel(r"true coefficient $\beta$")
    ax.set_ylabel(r"detection rate at $\alpha=0.05$")
    ax.set_title("Power against effect size")
    ax.legend(fontsize=7.5, loc="lower right")

    fig.tight_layout()
    out_png = figure_path("exp1_type1_calibration", quick=args.quick)
    fig.savefig(out_png, bbox_inches="tight")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
