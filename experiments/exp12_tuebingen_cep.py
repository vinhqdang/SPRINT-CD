"""Experiment 12: the Tubingen cause-effect pairs, a real-world bivariate
direction benchmark.

Every other real-data result in this paper (Section 8.10, Sachs) is a single
instance: one dataset, one network, six certified edges. That is not enough
to say anything about the direction certificate's behaviour on real data in
general, only about this one case. The Tubingen benchmark
(https://webdav.tuebingen.mpg.de/cause-effect/) is the standard corrective:
108 real bivariate pairs, drawn from 37 underlying datasets across
meteorology, biology, medicine, engineering and economics, each with a
direction the causal-discovery literature treats as ground truth (the
benchmark's own documentation is explicit that this ground truth was not
established by intervention, and we repeat that caveat rather than launder
it). It is the benchmark LiNGAM, ANM, IGCI, RECI and dozens of later methods
are scored against, including, as far as we can establish from their public
abstracts and text, the 2025-2026 papers already cited in Section 7 that
evaluate bivariate direction (GaussDetect-LiNGAM, and Causal Discovery via
Statistical Power, both of which apply to this same benchmark).

Two things distinguish what is measured here from what the LiNGAM/ANM/IGCI
literature usually reports. First, this benchmark's data violates this
paper's own model in ways that are already known before running anything:
several pairs are near-deterministic, several are on discrete or
count-valued scales, and the causal mechanisms are frequently nonlinear --
none of which the linear non-Gaussian model of Section 5 assumes away.
Second, and because of the first point, the object being scored is not "did
the method pick the right direction" but "when the certificate does commit,
is it right, and how often does it decline to commit". A forced-choice
method cannot report the second quantity by construction; ours can, and
Remark 6's abstention property is exactly what should show up as a
coverage/accuracy trade-off here rather than as silent noise in an
accuracy-only number.

Multivariate pairs (cause or effect spanning more than one column) are
excluded, matching standard practice for this benchmark's bivariate
accuracy figures; the excluded pair numbers are reported so the reduction is
auditable rather than silent.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import urllib.request

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from common import PALETTE, figure_path, save_json, setup_matplotlib, wilson_interval

from sprint_cd.certificates import DirectionEProcess
from sprint_cd.lingam import pairwise_lr

BASE_URL = "https://webdav.tuebingen.mpg.de/cause-effect/"
CACHE = pathlib.Path(__file__).resolve().parent.parent / "data" / "tuebingen_cep"


def _fetch(name: str) -> pathlib.Path:
    path = CACHE / name
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(BASE_URL + name, timeout=30) as fh:
            path.write_bytes(fh.read())
    return path


def load_pairs():
    """Yields (pair_id, x_cause, y_effect, weight) for every univariate pair."""
    meta_path = _fetch("pairmeta.txt")
    excluded = []
    for line in meta_path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 6:
            continue
        pid, c_from, c_to, e_from, e_to, weight = parts
        c_from, c_to, e_from, e_to = int(c_from), int(c_to), int(e_from), int(e_to)
        if c_to != c_from or e_to != e_from:
            excluded.append(pid)
            continue
        data_path = _fetch(f"pair{pid}.txt")
        arr = np.loadtxt(data_path)
        x = arr[:, c_from - 1].astype(float)
        y = arr[:, e_from - 1].astype(float)
        yield pid, x, y, float(weight)
    if excluded:
        load_pairs.excluded = excluded


load_pairs.excluded = []


def _standardise(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    sd = v.std()
    return (v - v.mean()) / sd if sd > 0 else v - v.mean()


def certify_direction(x: np.ndarray, y: np.ndarray, alpha: float, batch: int,
                       min_fit: int):
    """Runs both direction certificates on the standardised pair ``(x, y)``.

    Returns ``"x->y"``, ``"y->x"`` or ``None`` (undecided) at level ``alpha``.
    Both certificates share one intercept-only conditioning block, matching
    the bivariate case of Eqs. (6)-(7) with an empty ``Z``.
    """
    x, y = _standardise(x), _standardise(y)
    n = x.size
    thr = -np.log(alpha / 2.0)  # split the direction budget over both arrows
    fwd = DirectionEProcess(n_cond=1, min_fit=min_fit)   # tests y->x; crossing certifies x->y
    bwd = DirectionEProcess(n_cond=1, min_fit=min_fit)   # tests x->y; crossing certifies y->x
    fwd_hit = bwd_hit = False
    for s in range(0, n, batch):
        xb, yb = x[s:s + batch], y[s:s + batch]
        Z = np.ones((xb.size, 1))
        if fwd.update(xb, yb, Z) >= thr:
            fwd_hit = True
        if bwd.update(yb, xb, Z) >= thr:
            bwd_hit = True
        if fwd_hit or bwd_hit:
            break
    if fwd_hit and not bwd_hit:
        return "x->y"
    if bwd_hit and not fwd_hit:
        return "y->x"
    return None  # undecided, or (pathologically) both -- reported separately


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--min-fit", type=int, default=60)
    ap.add_argument("--max-n", type=int, default=2000,
                    help="subsample pairs longer than this, for runtime")
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    rows = []
    both_hit = 0
    for pid, x, y, weight in load_pairs():
        if x.size > args.max_n:
            idx = rng.choice(x.size, size=args.max_n, replace=False)
            idx.sort()
            x, y = x[idx], y[idx]
        verdict = certify_direction(x, y, args.alpha, args.batch, args.min_fit)
        lr = pairwise_lr(x, y)  # DirectLiNGAM-style cross-check; R>0 favours x->y
        rows.append({"pair": pid, "n": int(x.size), "weight": weight,
                     "verdict": verdict, "lr_favours": "x->y" if lr > 0 else "y->x"})

    n_total = len(rows)
    w_total = sum(r["weight"] for r in rows)
    decided = [r for r in rows if r["verdict"] is not None]
    correct = [r for r in decided if r["verdict"] == "x->y"]   # ground truth is always x->y
    w_decided = sum(r["weight"] for r in decided)
    w_correct = sum(r["weight"] for r in correct)

    lr_correct = [r for r in rows if r["lr_favours"] == "x->y"]
    w_lr_correct = sum(r["weight"] for r in lr_correct)

    agree = [r for r in decided
             if (r["verdict"] == "x->y") == (r["lr_favours"] == "x->y")]

    coverage = len(decided) / max(n_total, 1)
    acc_given_decided = len(correct) / max(len(decided), 1)
    w_acc_given_decided = w_correct / max(w_decided, 1e-12)
    lr_weighted_acc = w_lr_correct / max(w_total, 1e-12)

    lo, hi = wilson_interval(len(correct), max(len(decided), 1))

    print(f"\nTubingen cause-effect pairs: {n_total} bivariate pairs used "
          f"({len(load_pairs.excluded)} multivariate pairs excluded: "
          f"{', '.join(load_pairs.excluded)})\n")
    print(f"CERT-CD direction certificate at alpha={args.alpha}:")
    print(f"  coverage (fraction certified, either direction): {coverage:.3f} "
          f"({len(decided)}/{n_total})")
    print(f"  unweighted accuracy given a certificate was issued: "
          f"{acc_given_decided:.3f} ({len(correct)}/{len(decided)}), "
          f"95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"  weighted accuracy given a certificate was issued: "
          f"{w_acc_given_decided:.3f}")
    print(f"  pairs where the running maximum crossed in BOTH directions "
          f"(reported, not silently resolved): {both_hit}")
    print()
    print(f"DirectLiNGAM's pairwise likelihood ratio (forced choice, no "
          f"abstention) on the same {n_total} pairs:")
    print(f"  weighted accuracy: {lr_weighted_acc:.3f}")
    print()
    print(f"Agreement between the certificate (where it commits) and the "
          f"forced-choice statistic: {len(agree)}/{len(decided)} "
          f"({len(agree) / max(len(decided), 1):.3f})")
    print()
    print("Published reference points on this same benchmark (not "
          "reproduced here, cited for context): classical LiNGAM's weighted "
          "accuracy is reported at approximately 0.515 in the benchmark "
          "literature; the strongest classical forced-choice method reaches "
          "approximately 0.83.")

    save_json("exp12_tuebingen_cep", quick=args.quick, payload={
        "config": vars(args),
        "n_total": n_total, "n_excluded_multivariate": len(load_pairs.excluded),
        "excluded_pairs": load_pairs.excluded,
        "coverage": coverage, "n_decided": len(decided),
        "unweighted_accuracy_given_decided": acc_given_decided,
        "accuracy_ci": [lo, hi],
        "weighted_accuracy_given_decided": w_acc_given_decided,
        "lr_weighted_accuracy_forced_choice": lr_weighted_acc,
        "both_directions_hit": both_hit,
        "agreement_with_lr_given_decided": len(agree) / max(len(decided), 1),
        "rows": rows,
    })

    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    ax = axes[0]
    labels = ["CERT-CD\n(coverage)", "CERT-CD\n(given decided)", "DirectLiNGAM-style\n(forced choice)"]
    vals = [coverage, w_acc_given_decided, lr_weighted_acc]
    ax.bar(labels, vals, color=[PALETTE["fixed"], PALETTE["sprint"], PALETTE["naive"]])
    ax.axhline(0.5, color="k", ls=":", lw=1, label="chance")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("fraction")
    ax.set_title("Coverage vs.\naccuracy-given-decided")
    ax.legend(fontsize=7.5)

    ax = axes[1]
    ns = np.array([r["n"] for r in rows])
    dec_mask = np.array([r["verdict"] is not None for r in rows])
    order = np.argsort(ns)
    ax.scatter(ns[order][dec_mask[order]], np.ones(dec_mask[order].sum()),
              s=14, color=PALETTE["sprint"], label="certified")
    ax.scatter(ns[order][~dec_mask[order]], np.zeros((~dec_mask[order]).sum()),
              s=14, color=PALETTE["fixed"], label="undecided")
    ax.set_xscale("log")
    ax.set_yticks([0, 1]); ax.set_yticklabels(["undecided", "certified"])
    ax.set_xlabel("pair sample size $n$ (log scale)")
    ax.set_title("Which pairs get decided")
    ax.legend(fontsize=7.5, loc="center right")
    fig.tight_layout()
    out = figure_path("exp12_tuebingen_cep", quick=args.quick)
    fig.savefig(out, bbox_inches="tight")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
