"""Experiment 13: family-wise error of a monitored discovery run.

Question.  An analyst watches a discovery run as data accumulate and keeps any
edge that is ever reported.  How often is some false edge ever reported?

Pre-specified design (frozen before any results were seen):

* R random DAGs, d=8, edge_prob=0.25, coefficients uniform on +-[0.5, 1.2].
* Two noise conditions on the same DAGs: Laplace (main) and Gaussian (negative
  control for the non-Gaussian orientation, relevant to adjacency only).
* All methods see the same data.  Columns are standardised with the mean and
  sd of the first 100 rows, for every method.
* Monitoring grid n in {100, ..., 6400}.  At each grid point a method reports a
  skeleton.  An edge is "ever reported" if it appears at any grid point.
* CERT-CD (alpha=0.1, max_order=2, warmup=50) is streamed once per DAG, with
  batches of at most 100 rows aligned to the grid.  A certified edge counts as
  reported as soon as it is certified.
* PC (Fisher-z, causal-learn) at alpha=0.05 and alpha=0.01, GES (BIC) and
  DirectLiNGAM (default adaptive-lasso adjacency, edge if |B|>0.05) are each
  rerun from scratch on the first n rows at each grid point.

Notes on the implementation:

* For every method the grid point n uses the first n rows.  CERT-CD uses the
  first 50 of them only to fix its internal standardisation constants, as in
  its own warm-up, and absorbs rows 50..n-1.
* A failed or timed-out call contributes no output at that grid point.  It is
  counted and reported.  A DAG enters a method's metrics if at least one of
  its grid calls succeeded.
* Results are written to JSON after every chunk of DAGs (with
  ``complete: false`` until the run finishes), so a partial run is never lost.
"""

from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import itertools
import signal
import sys
import pathlib
import time
import warnings
from collections import Counter
from multiprocessing import Pool

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(HERE))
from common import PALETTE, figure_path, save_json, setup_matplotlib, wilson_interval

from sprint_cd.certcd import CertCD, CertCDConfig
from sprint_cd.simulate import random_dag

BASE_SEED = 20261022
D = 8
EDGE_PROB = 0.25
COEF_LOW, COEF_HIGH = 0.5, 1.2
NMAX = 6400
GRID = [100, 200, 400, 800, 1600, 3200, 6400]
WARM = 50
STD_ROWS = 100
BATCH = 100
CERT_ALPHA = 0.1
CONDITIONS = ("laplace", "gaussian")
METHODS = ("CERT-CD", "PC a=0.05", "PC a=0.01", "GES-BIC", "DirectLiNGAM")
TIMEOUT_S = 120
CERT_TIMEOUT_S = 1800
LINGAM_THRESH = 0.05
N_PAIRS = D * (D - 1) // 2


class _Timeout(Exception):
    pass


def _on_alarm(signum, frame):
    raise _Timeout()


def _guarded(fn, seconds):
    """Run ``fn`` under a wall-clock limit; return ``(value, error_string)``."""
    signal.signal(signal.SIGALRM, _on_alarm)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return fn(), None
    except _Timeout:
        return None, "timeout"
    except Exception as exc:  # a baseline crash is recorded, not hidden
        return None, f"{type(exc).__name__}: {exc}"[:200]
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def _adj_pairs(M) -> list[list[int]]:
    M = np.asarray(M)
    return [[i, j] for i, j in itertools.combinations(range(M.shape[0]), 2)
            if M[i, j] != 0 or M[j, i] != 0]


def _pc_edges(X, alpha):
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.cit import fisherz
    G = pc(X, alpha, indep_test=fisherz, show_progress=False)
    return _adj_pairs(G.G.graph)


def _ges_edges(X):
    from causallearn.search.ScoreBased.GES import ges
    Record = ges(X, score_func="local_score_BIC")
    return _adj_pairs(Record["G"].graph)


def _lingam_edges(X):
    from causallearn.search.FCMBased.lingam import DirectLiNGAM
    model = DirectLiNGAM()
    model.fit(X)
    B = np.asarray(model.adjacency_matrix_)
    return [[i, j] for i, j in itertools.combinations(range(D), 2)
            if abs(B[i, j]) > LINGAM_THRESH or abs(B[j, i]) > LINGAM_THRESH]


def _certcd_stream(Xs):
    """Stream the standardised data once through CERT-CD, reading out at each grid point."""
    algo = CertCD(D, CertCDConfig(alpha=CERT_ALPHA, max_order=2, warmup=WARM))
    algo.warm_up(Xs[:WARM])
    cur = WARM
    edges, undecided = [], []
    for n in GRID:
        for s in range(cur, n, BATCH):
            algo.update(Xs[s: min(s + BATCH, n)])
        cur = n
        edges.append([list(p) for p in algo.certified_edges()])
        undecided.append(len(algo.undecided_pairs()))
    return edges, undecided


def run_dag(r: int) -> dict:
    """Everything for one DAG: both noise conditions, all methods, all grid points."""
    warnings.filterwarnings("ignore")
    sem = random_dag(D, EDGE_PROB, np.random.default_rng([BASE_SEED, r, 0]),
                     coef_low=COEF_LOW, coef_high=COEF_HIGH)
    adj = sem.adjacency
    truth = [[i, j] for i, j in itertools.combinations(range(D), 2)
             if adj[i, j] or adj[j, i]]
    rec = {"r": r, "truth": truth, "conds": {}}

    for ci, cond in enumerate(CONDITIONS):
        X = sem.sample(NMAX, np.random.default_rng([BASE_SEED, r, 1 + ci]), noise=cond)
        mu = X[:STD_ROWS].mean(axis=0)
        sd = X[:STD_ROWS].std(axis=0, ddof=1)
        Xs = (X - mu) / sd

        out = {m: [None] * len(GRID) for m in METHODS}
        errs = {m: [None] * len(GRID) for m in METHODS}
        undecided = [None] * len(GRID)

        stream, err = _guarded(lambda: _certcd_stream(Xs), CERT_TIMEOUT_S)
        if stream is None:
            for gi in range(len(GRID)):
                errs["CERT-CD"][gi] = err
        else:
            out["CERT-CD"] = stream[0]
            undecided = stream[1]

        for gi, n in enumerate(GRID):
            Xn = Xs[:n]
            calls = {
                "PC a=0.05": lambda: _pc_edges(Xn, 0.05),
                "PC a=0.01": lambda: _pc_edges(Xn, 0.01),
                "GES-BIC": lambda: _ges_edges(Xn),
                "DirectLiNGAM": lambda: _lingam_edges(Xn),
            }
            for m, fn in calls.items():
                val, e = _guarded(fn, TIMEOUT_S)
                out[m][gi] = val
                errs[m][gi] = e

        rec["conds"][cond] = {"edges": out, "errors": errs, "undecided": undecided}
    return rec


def _pairs(lst):
    return {tuple(p) for p in lst}


def summarise(records: list[dict], cond: str) -> dict:
    """Per-method metrics for one noise condition.  Uses only DAGs with successful output."""
    summ = {}
    G = len(GRID)
    for m in METHODS:
        ever_k = 0
        included = 0
        distinct_false = []
        ever_up_k = np.zeros(G, dtype=int)     # DAGs with a false edge ever reported up to grid i
        ever_up_n = np.zeros(G, dtype=int)     # DAGs with a successful call up to grid i
        at_k = np.zeros(G, dtype=int)          # DAGs with a false edge in the estimate at grid i
        at_n = np.zeros(G, dtype=int)
        tp = fp = fn = 0
        final_ok = 0
        failed_calls = 0
        fail_reasons = Counter()
        undecided_final = []
        for rec in records:
            c = rec["conds"][cond]
            T = _pairs(rec["truth"])
            outs = c["edges"][m]
            errs = c["errors"][m]
            for e in errs:
                if e is not None:
                    failed_calls += 1
                    fail_reasons[e.split(":")[0]] += 1
            if all(o is None for o in outs):
                continue
            included += 1
            ever = set()
            for gi in range(G):
                if outs[gi] is None:
                    continue
                F = _pairs(outs[gi]) - T
                at_n[gi] += 1
                if F:
                    at_k[gi] += 1
                ever |= F
            # cumulative indicators (monotone in grid index)
            running_false = False
            running_seen = False
            for gi in range(G):
                if outs[gi] is not None:
                    running_seen = True
                    E = _pairs(outs[gi])
                    running_false = running_false or bool(E - T)
                if running_seen:
                    ever_up_n[gi] += 1
                    ever_up_k[gi] += int(running_false)
            if ever:
                ever_k += 1
            distinct_false.append(len(ever))
            if outs[-1] is not None:
                final_ok += 1
                E = _pairs(outs[-1])
                tp += len(E & T)
                fp += len(E - T)
                fn += len(T - E)
                if m == "CERT-CD" and c["undecided"][-1] is not None:
                    undecided_final.append(c["undecided"][-1] / N_PAIRS)
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        rec_ = tp / (tp + fn) if (tp + fn) else float("nan")
        f1 = 2 * prec * rec_ / (prec + rec_) if (prec and rec_) else float("nan")
        lo, hi = wilson_interval(ever_k, included)
        summ[m] = {
            "included_dags": included,
            "p_ever_false": ever_k / max(included, 1),
            "p_ever_false_ci": [lo, hi],
            "mean_distinct_false_ever": float(np.mean(distinct_false)) if distinct_false else float("nan"),
            "at_final_n": {"n": GRID[-1], "dags": final_ok,
                           "precision": prec, "recall": rec_, "f1": f1,
                           "tp": int(tp), "fp": int(fp), "fn": int(fn)},
            "p_false_in_estimate_by_n": {
                str(GRID[gi]): [int(at_k[gi]), int(at_n[gi])] for gi in range(G)},
            "p_ever_false_by_n": {
                str(GRID[gi]): [int(ever_up_k[gi]), int(ever_up_n[gi])] for gi in range(G)},
            "failed_calls": failed_calls,
            "failure_reasons": dict(fail_reasons),
        }
        if m == "CERT-CD":
            summ[m]["undecided_fraction_final"] = (
                float(np.mean(undecided_final)) if undecided_final else float("nan"))
    return summ


def _fmt(x, nd=3):
    return "  nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def print_tables(summaries: dict, n_dags: int):
    for cond in CONDITIONS:
        s = summaries[cond]
        print(f"\n=== Noise: {cond}  (R={n_dags} DAGs, d={D}, edge_prob={EDGE_PROB}) ===")
        hdr = (f"{'method':<13}{'incl':>5}  {'P(ever false edge) [95% CI]':<31}"
               f"{'mean #false ever':>17}{'prec@6400':>10}{'recall@6400':>12}"
               f"{'F1@6400':>9}{'failed':>8}")
        print(hdr)
        print("-" * len(hdr))
        for m in METHODS:
            r = s[m]
            lo, hi = r["p_ever_false_ci"]
            p = f"{r['p_ever_false']:.3f} [{lo:.3f}, {hi:.3f}]"
            f = r["at_final_n"]
            print(f"{m:<13}{r['included_dags']:>5}  {p:<31}"
                  f"{_fmt(r['mean_distinct_false_ever'], 2):>17}"
                  f"{_fmt(f['precision']):>10}{_fmt(f['recall']):>12}"
                  f"{_fmt(f['f1']):>9}{r['failed_calls']:>8}")
        print(f"\nCERT-CD pairs undecided at n={GRID[-1]} (fraction of {N_PAIRS}): "
              f"{_fmt(s['CERT-CD']['undecided_fraction_final'])}")
        print("\nFraction of DAGs whose estimate at n contains >=1 false edge:")
        cols = "".join(f"{n:>8}" for n in GRID)
        print(f"{'method':<13}{cols}")
        for m in METHODS:
            vals = []
            for n in GRID:
                k, nn = s[m]["p_false_in_estimate_by_n"][str(n)]
                vals.append(f"{(k / nn if nn else float('nan')):>8.3f}")
            print(f"{m:<13}{''.join(vals)}")
        print("\nP(ever false edge by n), cumulative over the grid:")
        print(f"{'method':<13}{cols}")
        for m in METHODS:
            vals = []
            for n in GRID:
                k, nn = s[m]["p_ever_false_by_n"][str(n)]
                vals.append(f"{(k / nn if nn else float('nan')):>8.3f}")
            print(f"{m:<13}{''.join(vals)}")
        print("\nFailed calls by reason: " + "; ".join(
            f"{m}: {s[m]['failure_reasons'] or 'none'}" for m in METHODS))


def _nan_to_none(o):
    if isinstance(o, dict):
        return {k: _nan_to_none(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_nan_to_none(v) for v in o]
    if isinstance(o, float) and np.isnan(o):
        return None
    return o


def _payload(records, args, complete, elapsed):
    summaries = {c: summarise(records, c) for c in CONDITIONS} if records else {}
    return _nan_to_none({
        "config": {
            "R_requested": args.R, "R_done": len(records), "d": D, "edge_prob": EDGE_PROB,
            "coef_range": [COEF_LOW, COEF_HIGH], "nmax": NMAX, "grid": GRID,
            "standardise_rows": STD_ROWS, "certcd": {"alpha": CERT_ALPHA, "max_order": 2,
                                                     "warmup": WARM, "batch": BATCH},
            "pc_alphas": [0.05, 0.01], "lingam_threshold": LINGAM_THRESH,
            "base_seed": BASE_SEED, "conditions": list(CONDITIONS), "methods": list(METHODS),
            "timeout_s": TIMEOUT_S, "certcd_timeout_s": CERT_TIMEOUT_S,
        },
        "complete": complete,
        "elapsed_s": elapsed,
        "summary": summaries,
        "raw": records,
    })


def make_figure(summaries: dict):
    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True)
    styles = {
        "CERT-CD": (PALETTE["sprint"], "-o"),
        "PC a=0.05": (PALETTE["naive"], "-s"),
        "PC a=0.01": (PALETTE["fixed"], "--s"),
        "GES-BIC": (PALETTE["oracle"], "-^"),
        "DirectLiNGAM": (PALETTE["accent"], "-D"),
    }
    x = np.arange(len(GRID))
    for ax, cond, title in zip(axes, CONDITIONS,
                               ("Laplace noise (non-Gaussian)", "Gaussian noise (control)")):
        s = summaries[cond]
        for m in METHODS:
            col, sty = styles[m]
            ks = [s[m]["p_ever_false_by_n"][str(n)] for n in GRID]
            p = [k / max(nn, 1) for k, nn in ks]
            ci = np.array([wilson_interval(k, max(nn, 1)) for k, nn in ks])
            yerr = np.vstack([np.array(p) - ci[:, 0], ci[:, 1] - np.array(p)])
            ax.errorbar(x, p, yerr=yerr, fmt=sty, ms=3.5, lw=1.3, capsize=2,
                        color=col, label=m)
        ax.set_xticks(x)
        ax.set_xticklabels([str(n) for n in GRID], rotation=0, fontsize=7.5)
        ax.set_xlabel("monitoring sample size $n$")
        ax.set_title(title, fontsize=9.5)
        ax.set_ylim(-0.03, 1.03)
        ax.axhline(0.1, color="k", ls=":", lw=1)
    axes[0].set_ylabel("P(a false edge has been reported by $n$)")
    axes[1].legend(fontsize=7.2, loc="upper left")
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=int, default=150)
    ap.add_argument("--chunk", type=int, default=20)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-save", action="store_true",
                    help="smoke tests only: print tables, write nothing")
    args = ap.parse_args()

    t0 = time.time()
    records: list[dict] = []
    print(f"Experiment 13: monitored discovery FWER, R={args.R}, workers={args.workers}")
    with Pool(args.workers) as pool:
        for start in range(0, args.R, args.chunk):
            stop = min(start + args.chunk, args.R)
            records.extend(pool.map(run_dag, range(start, stop), chunksize=1))
            elapsed = time.time() - t0
            print(f"  chunk done: DAGs {start}..{stop - 1}  elapsed {elapsed / 60:.1f} min",
                  flush=True)
            if not args.no_save and stop < args.R:
                save_json("exp13_monitoring_fwer",
                          _payload(records, args, complete=False, elapsed=elapsed))

    records.sort(key=lambda r: r["r"])
    elapsed = time.time() - t0
    payload = _payload(records, args, complete=True, elapsed=elapsed)
    print_tables(payload["summary"], len(records))
    print(f"\nTotal wall time {elapsed / 60:.1f} min")

    if not args.no_save:
        save_json("exp13_monitoring_fwer", payload)
        fig = make_figure(payload["summary"])
        out = figure_path("exp13_monitoring_fwer")
        fig.savefig(out, bbox_inches="tight")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
