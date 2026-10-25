"""Experiment 56: a factorial simulation of the regimes found in exp49-52, with label-free diagnostics.

Written and committed before running.  Pair: cause c, effect e = f(c) + sigma * eps, then optional measurement noise.
Factors (all crossed, 144 cells, 25 pairs per cell, a random half swapped; seeds 20263200+): mechanism f in {tanh(2c),
c^3/3, sin(1.5c)}; cause marginal in {uniform(-1.7,1.7), standard normal, bimodal 0.5N(-1.5,0.5^2)+0.5N(1.5,0.5^2),
five-level discrete (-2..2, equal probability)}; noise scale in {homoscedastic, sigma = s0 exp(c)/sqrt(mean exp(2c))
(cause-dependent), sigma_i = s0 exp(u_i)/sqrt(mean exp(2u)), u ~ N(0,1) independent of c}, s0 = 0.3 sd(f(c)); measurement
noise in {none, N(0, (0.3 sd)^2) added independently to both variables}; n in {200, 1000}.  Rules: IGCI, RECI, ANM-HSIC,
pairwise LiNGAM (scored), sorted-conditional compression (faithful variant), RESIT, AVICI.  Stored per pair: the
decisions and symmetric (label-free) diagnostics:
  hmin   min of the exp49 heteroscedasticity h over the two directions;
  lu     log fraction of unique values, minimum over the two variables;
  kmin, kmax  min and max of the excess kurtosis of the two (scaled) variables;
  smax   max of the absolute skewness;
  lrho   log |Spearman correlation| ; ln = log n.
These are the inputs of the predictive validation of exp57.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import itertools, json, pathlib, sys, warnings
from multiprocessing import Pool
import numpy as np
from scipy import stats

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
R = HERE.parent / "results"
from sprint_cd.pair_baselines import scored
from sprint_cd.pair_2026 import sorted_compression_faithful
from sprint_cd.sota_wrappers import resit_pair, avici_pairs
import exp49_explain as e49

FS = [lambda c: np.tanh(2 * c), lambda c: c ** 3 / 3, lambda c: np.sin(1.5 * c)]
CAUSES = ["uniform", "normal", "bimodal", "discrete5"]
SCALES = ["homo", "cause", "indep"]


def make(cell, k):
    f_i, cm, sm, meas, n = cell
    rng = np.random.default_rng([20263200, f_i, CAUSES.index(cm), SCALES.index(sm), meas, n, k])
    if cm == "uniform": c = rng.uniform(-1.7, 1.7, n)
    elif cm == "normal": c = rng.normal(size=n)
    elif cm == "bimodal": c = np.where(rng.random(n) < .5, -1.5, 1.5) + 0.5 * rng.normal(size=n)
    else: c = rng.integers(-2, 3, n).astype(float)
    f = FS[f_i](c); s0 = 0.3 * f.std(); eps = rng.normal(size=n); u = rng.normal(size=n)
    z = {"homo": np.zeros(n), "cause": c, "indep": u}[sm]
    sig = s0 * np.exp(z) / np.sqrt(np.mean(np.exp(2 * z)))
    e = f + sig * eps
    x, y = c.copy(), e.copy()
    if meas:
        x = x + 0.3 * x.std() * rng.normal(size=n); y = y + 0.3 * y.std() * rng.normal(size=n)
    flip = rng.random() < .5
    return (y, x) if flip else (x, y), bool(flip)


def feats(x, y):
    zx = (x - x.mean()) / (x.std() + 1e-12); zy = (y - y.mean()) / (y.std() + 1e-12)
    kx, ky = stats.kurtosis(zx), stats.kurtosis(zy)
    ux = len(np.unique(x)) / len(x); uy = len(np.unique(y)) / len(y)
    try: h = min(e49.het(x, y), e49.het(y, x))
    except Exception: h = np.nan
    return {"hmin": float(h), "lu": float(np.log(min(ux, uy))), "kmin": float(min(kx, ky)), "kmax": float(max(kx, ky)),
            "smax": float(max(abs(stats.skew(zx)), abs(stats.skew(zy)))), "lrho": float(np.log(abs(stats.spearmanr(x, y)[0]) + 1e-6)), "ln": float(np.log(len(x)))}


def one(job):
    warnings.filterwarnings("ignore")
    cell, k = job
    (x, y), flip = make(cell, k)
    rng = np.random.default_rng(1)
    xd = x + 1e-6 * x.std() * rng.standard_normal(len(x)); yd = y + 1e-6 * y.std() * rng.standard_normal(len(y))
    sc = scored(xd, yd, 1); m = {a: (float(v[1]) if v[0] == "x->y" else -float(v[1])) for a, v in sc.items()}
    m["Compression"] = sorted_compression_faithful(x, y)
    try: m["RESIT"] = resit_pair(xd, yd)
    except Exception: m["RESIT"] = None
    return {"cell": list(cell), "k": k, "flip": flip, "m": m, "f": feats(xd, yd)}


def main():
    cells = list(itertools.product(range(3), CAUSES, SCALES, (0, 1), (200, 1000)))
    jobs = [(c, k) for c in cells for k in range(25)]
    part = R / "exp56_factorial.partial.json"
    recs = json.loads(part.read_text())["records"] if part.exists() else []
    todo = jobs[len(recs):]
    with Pool(4) as p:
        for s in range(0, len(todo), 400):
            recs.extend(p.map(one, todo[s:s + 400], chunksize=10)); part.write_text(json.dumps({"records": recs})); print(len(recs), "/", len(jobs), flush=True)
    av = avici_pairs([make(j[0], j[1])[0] for j in jobs])
    for r, v in zip(recs, av): r["m"]["AVICI"] = v
    (R / "exp56_factorial.json").write_text(json.dumps({"records": recs})); part.unlink()


if __name__ == "__main__":
    main()
