"""Two Tuebingen pairs under the frozen exp14 protocol.

Pair 0005 is the first pair on which the agreement certificate commits and is
right; pair 0047 is the only pair on which it commits and is wrong.  For each
pair the figure shows the scatter plot and the confidence sequence for the
running mean of the log-score difference under the constant-scale class.
Data are read from data/tuebingen_cep (fetched by experiments/exp12).
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from common import PALETTE, setup_matplotlib  # noqa: E402
from exp14_tuebingen_relfit import dequantise, MAX_N, SEED  # noqa: E402
from sprint_cd.relative_fit import _GG, cs_radius  # noqa: E402

PAIRS = {"0005": "correct", "0047": "wrong"}


def trace(x, y, m0, seed, batch=50, alpha=0.05, min_warm=100):
    p = np.random.default_rng(seed).permutation(len(x))
    x, y = x[p], y[p]
    x = (x - x[:min_warm].mean()) / (x[:min_warm].std() + 1e-12)
    y = (y - y[:min_warm].mean()) / (y[:min_warm].std() + 1e-12)
    kx = np.quantile(x[:m0], (.2, .4, .6, .8))
    ky = np.quantile(y[:m0], (.2, .4, .6, .8))
    n, s, ds, rows = len(x), m0, [], []
    while s < n:
        e = min(s + batch, n)
        fa = _GG().fit(x[:s], y[:s], kx)
        fb = _GG().fit(y[:s], x[:s], ky)
        ds.extend(fa.logp(x[s:e], y[s:e]) - fb.logp(y[s:e], x[s:e]))
        s = e
        D = np.asarray(ds)
        if len(D) >= 60:
            mu, sd = D.mean(), D.std(ddof=1) + 1e-12
            r = cs_radius(len(D), sd, alpha)
            rows.append((len(D), mu, mu - r, mu + r))
    return x, y, np.array(rows)


def main():
    import matplotlib.pyplot as plt
    from exp12_tuebingen_cep import load_pairs
    setup_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.2))
    for row, (pid, x, y, w) in enumerate(p for p in load_pairs() if p[0] in PAIRS):
        rng = np.random.default_rng(SEED + int(pid))
        if len(x) > MAX_N:
            idx = np.sort(rng.choice(len(x), MAX_N, replace=False))
            x, y = x[idx], y[idx]
        x, y = dequantise(x, rng), dequantise(y, rng)
        m0 = max(100, int(np.ceil(0.2 * len(x))))
        xs, ys, tr = trace(x, y, m0, SEED + int(pid))
        ax = axes[row, 0]
        ax.scatter(xs, ys, s=3, alpha=0.4, color=PALETTE.get("blue", "C0"))
        ax.set_xlabel("cause (standardised)")
        ax.set_ylabel("effect (standardised)")
        ax.set_title(f"pair {pid} ({PAIRS[pid]})", fontsize=10)
        ax = axes[row, 1]
        ax.fill_between(tr[:, 0], tr[:, 2], tr[:, 3], alpha=0.25, color="C1")
        ax.plot(tr[:, 0], tr[:, 1], color="C1")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel("scored observations")
        ax.set_ylabel("mean log-score difference")
    fig.tight_layout()
    out = ROOT / "results" / "tuebingen_demo_pairs.png"
    fig.savefig(out, dpi=180)
    print("wrote", out)


if __name__ == "__main__":
    main()
