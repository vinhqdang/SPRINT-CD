"""Learned cause-effect direction model: multi-score features for a pair (x, y).

Each direction (cause c, effect e) is described by cheap, complementary scores: the
cross-validated error of additive spline regressions of several flexibility levels, the
independence of the residual from the cause (HSIC and a spline dependence of the squared
residual), the residual shape, IGCI entropies, a Gaussian ANM and a heteroscedastic
(location-scale) log-likelihood fitted on a train half and scored on a test half, the
pairwise LiNGAM statistic, and symmetric nuisance features (sample size, discreteness,
correlation).  The final feature vector is the antisymmetric difference between the two
directions plus the symmetric features, so a classifier trained on randomly flipped pairs
predicts P(x -> y).
"""
from __future__ import annotations

import numpy as np

__all__ = ["pair_features", "N_FEATURES"]


def _std(v):
    return (v - v.mean()) / (v.std() + 1e-12)


def _basis(c, df):
    q = np.quantile(c, np.linspace(0, 1, df + 2)[1:-1])
    return np.column_stack([np.ones_like(c), c, c ** 2, c ** 3] + [np.maximum(c - t, 0) ** 3 for t in q])


def _cv_resid(c, e, df, folds=5, seed=0):
    n = len(c)
    B = _basis(c, df)
    fold = np.random.default_rng(seed).permutation(n) % folds
    r = np.empty(n)
    pen = 1e-3 * np.eye(B.shape[1]); pen[0, 0] = 0
    for f in range(folds):
        tr, te = fold != f, fold == f
        beta = np.linalg.solve(B[tr].T @ B[tr] + pen, B[tr].T @ e[tr])
        r[te] = e[te] - B[te] @ beta
    return r


def _hsic(a, b, m=400):
    if len(a) > m:
        idx = np.random.default_rng(1).choice(len(a), m, replace=False)
        a, b = a[idx], b[idx]
    n = len(a)

    def gram(v):
        d = (v[:, None] - v[None, :]) ** 2
        h = np.median(d[d > 0]) if np.any(d > 0) else 1.0
        return np.exp(-d / h)
    K, L = gram(a), gram(b)
    H = np.eye(n) - 1.0 / n
    return float(np.trace(K @ H @ L @ H) / n ** 2)


def _spacing_entropy(v):
    v = np.sort(v)
    d = np.diff(v)
    d = d[d > 0]
    return float(np.mean(np.log(d))) if len(d) else 0.0


def _kurt(v):
    v = _std(v)
    return float(np.mean(v ** 4) - 3)


def _skew(v):
    return float(np.mean(_std(v) ** 3))


def _direction(c, e, seed):
    """Scores for the hypothesis c -> e (c, e standardised)."""
    out = []
    n = len(c)
    resids = {}
    for df in (2, 4, 8):
        r = _cv_resid(c, e, df, seed=seed)
        resids[df] = r
        out.append(np.log(np.mean(r ** 2) + 1e-9))
    r = resids[4]
    rs = _std(r)
    out.append(np.log(_hsic(rs, c) + 1e-9))
    # heteroscedasticity: how well |r| is explained by c
    lr = np.log(r ** 2 + 1e-6)
    rl = _cv_resid(c, lr, 3, seed=seed)
    out.append(1 - np.var(rl) / (np.var(lr) + 1e-12))
    out += [_skew(r), _kurt(r), abs(np.corrcoef(rs ** 2, _std(c) ** 2)[0, 1])]
    # Gaussian ANM and location-scale log-likelihood, train half / test half
    idx = np.random.default_rng(seed).permutation(n)
    tr, te = idx[: n // 2], idx[n // 2:]
    B = _basis(c, 4)
    pen = 1e-3 * np.eye(B.shape[1]); pen[0, 0] = 0
    beta = np.linalg.solve(B[tr].T @ B[tr] + pen, B[tr].T @ e[tr])
    rt = e[tr] - B[tr] @ beta
    s2 = np.mean(rt ** 2) + 1e-9
    rte = e[te] - B[te] @ beta
    ll_g = -0.5 * np.mean(rte ** 2) / s2 - 0.5 * np.log(s2)
    g = np.linalg.solve(B[tr].T @ B[tr] + pen, B[tr].T @ np.log(rt ** 2 + 1e-6))
    ls = np.clip(B[te] @ g, -8, 8)
    ll_h = -0.5 * np.mean(rte ** 2 * np.exp(-ls)) - 0.5 * np.mean(ls)
    # cause marginal log density proxy: spacing entropy
    out += [ll_g, ll_h, _spacing_entropy(c), _skew(c), _kurt(c)]
    return np.array(out)


def pair_features(x, y, seed=0, max_n=1500):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) > max_n:
        idx = np.random.default_rng(seed).choice(len(x), max_n, replace=False)
        x, y = x[idx], y[idx]
    nu_x, nu_y = len(np.unique(x)) / len(x), len(np.unique(y)) / len(y)
    xs, ys = _std(x), _std(y)
    fxy = _direction(xs, ys, seed)
    fyx = _direction(ys, xs, seed)
    mm = lambda v: (v - v.min()) / (v.max() - v.min() + 1e-12)
    h = _spacing_entropy(mm(y)) - _spacing_entropy(mm(x))
    R = float(np.mean(xs * ys)) * float(np.mean(xs * np.tanh(ys) - np.tanh(xs) * ys))
    kurt = 0.5 * (np.mean(xs ** 4) + np.mean(ys ** 4)) - 3
    if kurt < 0:
        R = -R
    sym = [np.log(len(x)), abs(np.corrcoef(xs, ys)[0, 1]), min(nu_x, nu_y), max(nu_x, nu_y)]
    anti = list(fxy - fyx) + [h, R]
    return np.array(anti + list(fxy + fyx) + sym)


N_FEATURES = 2 * 13 + 2 + 4
