"""Shared helpers for the experiment scripts."""

from __future__ import annotations

import json
import pathlib

import numpy as np

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"
RESULTS.mkdir(exist_ok=True)

PALETTE = {
    "sprint": "#1b6ca8",
    "naive": "#c0392b",
    "fixed": "#4a4a4a",
    "oracle": "#2e8b57",
    "accent": "#d68910",
}


def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 140,
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    })
    return plt


def save_json(name: str, payload: dict) -> pathlib.Path:
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=_default))
    return path


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not serialisable: {type(o)}")


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval -- well behaved for the near-zero rates seen here."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(centre - half, 0.0), min(centre + half, 1.0))


def running_cross_products(D: np.ndarray):
    """Cumulative sums and cross-products of a data stream.

    Returns ``(csum, ccross)`` with ``csum[t]`` the sum of the first ``t+1``
    rows and ``ccross[t]`` the corresponding matrix of raw cross-products, so
    that any sufficient statistic can be read off at every time point without
    re-scanning the stream.
    """
    n, d = D.shape
    csum = np.cumsum(D, axis=0)
    outer = D[:, :, None] * D[:, None, :]
    ccross = np.cumsum(outer, axis=0)
    return csum, ccross


def residual_moments_at(csum, ccross, t_index: int, i: int, j: int, cond: tuple[int, ...]):
    """``(sxx, sxy, syy, dof)`` for ``(i, j | cond)`` at monitoring index ``t_index``."""
    n = t_index + 1
    s = csum[t_index]
    C = ccross[t_index] - np.outer(s, s) / n
    k = len(cond)
    dof = n - k - 1
    if dof <= 0:
        return 0.0, 0.0, 0.0, 0
    if k == 0:
        return float(C[i, i]), float(C[i, j]), float(C[j, j]), dof
    idx = np.array(cond)
    Czz = C[np.ix_(idx, idx)]
    Czx = C[np.ix_(idx, [i, j])]
    try:
        sol = np.linalg.solve(Czz, Czx)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(Czz, Czx, rcond=None)[0]
    S = C[np.ix_([i, j], [i, j])] - Czx.T @ sol
    return float(max(S[0, 0], 0)), float(S[0, 1]), float(max(S[1, 1], 0)), dof
