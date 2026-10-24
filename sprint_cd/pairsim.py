"""Random cause-effect pair simulator for training the learned direction model."""
from __future__ import annotations

import numpy as np

CAUSE = ["normal", "uniform", "laplace", "exp", "lognormal", "mixture", "beta", "t3"]
NOISE = ["normal", "laplace", "uniform", "t5", "skew", "bimodal"]
MECH = ["linear", "poly2", "poly3", "sin", "tanh", "exp", "abs", "sigmoid", "step", "power", "saturate", "smooth"]


def _draw(rng, kind, n):
    if kind == "normal": return rng.normal(size=n)
    if kind == "uniform": return rng.uniform(-1.7, 1.7, n)
    if kind == "laplace": return rng.laplace(size=n)
    if kind == "exp": return rng.exponential(size=n) - 1
    if kind == "lognormal": return rng.lognormal(0, 0.6, n)
    if kind == "mixture":
        z = rng.random(n) < rng.uniform(0.3, 0.7)
        return np.where(z, rng.normal(-1.5, 0.6, n), rng.normal(1.5, 0.6, n))
    if kind == "beta": return rng.beta(2, 5, n) * 4 - 1
    if kind == "t3": return rng.standard_t(3, n)
    raise ValueError(kind)


def _noise(rng, kind, n):
    if kind == "normal": return rng.normal(size=n)
    if kind == "laplace": return rng.laplace(size=n)
    if kind == "uniform": return rng.uniform(-1.7, 1.7, n)
    if kind == "t5": return rng.standard_t(5, n)
    if kind == "skew": return rng.gamma(2, 1, n) - 2
    if kind == "bimodal":
        z = rng.random(n) < 0.5
        return np.where(z, rng.normal(-1.2, 0.5, n), rng.normal(1.2, 0.5, n))
    raise ValueError(kind)


def _mech(rng, kind, c):
    a = rng.uniform(0.5, 2.5)
    s = (c - c.mean()) / (c.std() + 1e-12)
    if kind == "linear": return a * s
    if kind == "poly2": return a * (s ** 2 - 1) + 0.3 * s
    if kind == "poly3": return a * 0.5 * s ** 3
    if kind == "sin": return np.sin(a * 1.5 * s)
    if kind == "tanh": return np.tanh(a * s)
    if kind == "exp": return np.exp(0.6 * a * s / 1.5)
    if kind == "abs": return np.abs(s) * a
    if kind == "sigmoid": return 1 / (1 + np.exp(-3 * a * s))
    if kind == "step": return np.where(s > rng.uniform(-0.5, 0.5), a, -a) + 0.3 * s
    if kind == "power": return np.sign(s) * np.abs(s) ** rng.uniform(0.4, 1.8)
    if kind == "saturate": return 1 - np.exp(-a * (s - s.min()) / 2)
    if kind == "smooth":
        k = rng.uniform(0.5, 2, 3); p = rng.uniform(0, 6, 3)
        return sum(np.sin(k[i] * s + p[i]) for i in range(3)) / 2
    raise ValueError(kind)


def gen_pair(seed):
    """Returns (x, y) with x the cause (the caller flips at random)."""
    rng = np.random.default_rng([20261400, seed])
    n = int(rng.choice([100, 200, 400, 800, 1500]))
    c = _draw(rng, str(rng.choice(CAUSE)), n)
    if rng.random() < 0.15:  # slow random-walk cause (serial dependence), then shuffled
        c = np.cumsum(rng.normal(size=n)); c = c[rng.permutation(n)]
    f = _mech(rng, str(rng.choice(MECH)), c)
    f = (f - f.mean()) / (f.std() + 1e-12)
    noise = _noise(rng, str(rng.choice(NOISE)), n)
    noise = noise / (noise.std() + 1e-12)
    snr = float(np.exp(rng.uniform(np.log(0.5), np.log(20))))
    structure = rng.choice(["additive", "multiplicative", "locscale", "pnl", "additive", "additive"])
    s = (c - c.mean()) / (c.std() + 1e-12)
    if structure == "additive":
        e = snr * f + noise
    elif structure == "multiplicative":
        e = snr * f * np.exp(0.3 * noise)
    elif structure == "locscale":
        e = snr * f + (0.3 + np.abs(s)) * noise
    else:
        g = [np.tanh, lambda u: u ** 3 / 3, np.exp, lambda u: np.sinh(u / 2)][int(rng.integers(4))]
        e = g((snr * f + noise) / (1 + snr))
    if rng.random() < 0.3:  # measurement noise on both
        c = c + 0.05 * c.std() * rng.normal(size=n)
        e = e + 0.05 * e.std() * rng.normal(size=n)
    if rng.random() < 0.3:  # quantisation, then the usual tie dequantisation
        for v in (c, e):
            pass
        q = float(np.exp(rng.uniform(np.log(0.01), np.log(0.3))))
        c = np.round(c / (q * c.std() + 1e-12)) * q * c.std()
        e = np.round(e / (q * e.std() + 1e-12)) * q * e.std()
        for arr in (c, e):
            u = np.unique(arr)
            if len(u) > 1:
                gap = np.diff(u).min()
                arr += rng.uniform(-0.5, 0.5, n) * gap
    return c, e
