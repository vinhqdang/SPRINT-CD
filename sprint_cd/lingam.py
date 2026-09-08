r"""DirectLiNGAM baseline, for comparison against certified orientation.

Included because CERT-CD's direction certificate is built on the linear
non-Gaussian acyclic model, so the honest comparator is a LiNGAM method rather
than a CPDAG ceiling.  A CPDAG cannot orient an edge outside a v-structure by
construction; DirectLiNGAM can orient essentially everything, using the same
non-Gaussianity assumption the certificate uses.  The interesting quantity is
therefore not orientation *rate* but the arrowhead error rate at a given rate:
DirectLiNGAM commits a direction for every edge with no error control, while
the certificate commits fewer with a time-uniform bound.

Implements the pairwise likelihood-ratio measure of Hyvarinen and Smith (2013)
used by DirectLiNGAM (Shimizu et al., 2011), with the entropy approximation of
Hyvarinen (1998).
"""

from __future__ import annotations

import numpy as np

__all__ = ["pairwise_lr", "direct_lingam_order", "direct_lingam_orient"]

_K1 = 79.047
_K2 = 7.4129
_GAMMA = 0.37457
_H_GAUSS = (1.0 + np.log(2.0 * np.pi)) / 2.0


def _entropy(u: np.ndarray) -> float:
    """Maximum-entropy approximation to differential entropy (Hyvarinen, 1998)."""
    u = np.asarray(u, dtype=float)
    sd = u.std()
    if sd <= 1e-12:
        return -np.inf
    u = (u - u.mean()) / sd
    t1 = float(np.mean(np.log(np.cosh(u))))
    t2 = float(np.mean(u * np.exp(-0.5 * u * u)))
    return _H_GAUSS - _K1 * (t1 - _GAMMA) ** 2 - _K2 * t2 ** 2


def _std(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    sd = u.std()
    return (u - u.mean()) / (sd if sd > 1e-12 else 1.0)


def pairwise_lr(xi: np.ndarray, xj: np.ndarray) -> float:
    """Likelihood-ratio statistic ``R``; ``R > 0`` favours ``xi -> xj``.

    Compares the two directions by the entropy of the putative exogenous
    variable plus the entropy of the corresponding residual.
    """
    x, y = _std(xi), _std(xj)
    rho = float(np.mean(x * y))
    rho = float(np.clip(rho, -0.999999, 0.999999))
    ri = _std(y - rho * x)          # residual of y on x
    rj = _std(x - rho * y)          # residual of x on y
    return (_entropy(y) + _entropy(rj)) - (_entropy(x) + _entropy(ri))


def direct_lingam_order(X: np.ndarray) -> list[int]:
    """Estimate a causal ordering by iterative selection of the most exogenous variable."""
    X = np.asarray(X, dtype=float).copy()
    d = X.shape[1]
    remaining = list(range(d))
    order: list[int] = []
    work = X.copy()

    while len(remaining) > 1:
        scores = []
        for j in remaining:
            tot = 0.0
            for i in remaining:
                if i == j:
                    continue
                r = pairwise_lr(work[:, j], work[:, i])
                tot += min(0.0, r) ** 2      # penalise evidence against j being exogenous
            scores.append(tot)
        root = remaining[int(np.argmin(scores))]
        order.append(root)
        # Remove the selected variable's linear effect from the rest.
        for i in remaining:
            if i == root:
                continue
            xr = work[:, root]
            denom = float(xr @ xr)
            if denom > 1e-12:
                work[:, i] = work[:, i] - (float(xr @ work[:, i]) / denom) * xr
        remaining.remove(root)
    order.extend(remaining)
    return order


def direct_lingam_orient(X: np.ndarray, skeleton_edges) -> dict:
    """Orient a given skeleton by the DirectLiNGAM ordering.

    Returns ``{(i, j): (cause, effect)}``.  Every supplied edge receives a
    direction: the method never abstains, which is exactly the property under
    comparison.
    """
    order = direct_lingam_order(X)
    pos = {v: p for p, v in enumerate(order)}
    return {(i, j): ((i, j) if pos[i] < pos[j] else (j, i))
            for i, j in skeleton_edges}
