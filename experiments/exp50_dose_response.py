"""Experiment 50: does heteroscedastic noise cause the loss of the noise-model rules? (simulation, dose-response)

Written and committed before running.  Exp49 found, on Tuebingen, that the pairs the additive-noise rules get wrong have
more heteroscedastic residuals; that is an association.  Here the factor is set by hand.  Pair: z ~ Uniform(-1.7, 1.7)
or Normal(0,1) (half each), cause c = z; effect e = f(c) + sigma(c) * eps with f drawn from {tanh(2c), c^3/3,
sin(1.5c)}, eps ~ N(0,1), sigma(c) = s0 * exp(a c) / sqrt(mean(exp(2 a c))) and s0 = 0.3 sd(f(c)); a in
{0, 0.5, 1, 1.5}; n = 500; 100 pairs per level, a random half swapped (the truth is stored).  Rules: IGCI, RECI,
ANM-HSIC, pairwise LiNGAM (sprint_cd.pair_baselines.scored), RESIT (random forest), AVICI (pretrained).
Prediction: the accuracy of ANM-HSIC and RESIT falls monotonically with a (at least 15 points between a = 0 and
a = 1.5), IGCI, RECI and LiNGAM fall by less than 10 points.  Reported: accuracy per level with Clopper-Pearson
intervals, the exact McNemar test of the change from a = 0 to a = 1.5 is not available (different pairs), so the
two-sample comparison of proportions (Fisher) is used.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json, pathlib, sys, warnings
from multiprocessing import Pool
import numpy as np
from scipy.stats import fisher_exact

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from sprint_cd.pair_baselines import scored
from sprint_cd.sota_wrappers import resit_pair, avici_pairs

LEVELS = [0.0, 0.5, 1.0, 1.5]
N, PER = 500, 100
FS = [lambda c: np.tanh(2 * c), lambda c: c ** 3 / 3, lambda c: np.sin(1.5 * c)]


def make(args):
    lvl, k = args
    rng = np.random.default_rng([20262800, int(LEVELS[lvl] * 10), k])
    c = rng.uniform(-1.7, 1.7, N) if rng.random() < .5 else rng.normal(size=N)
    f = FS[rng.integers(3)](c); s0 = 0.3 * f.std(); a = LEVELS[lvl]
    sig = s0 * np.exp(a * c) / np.sqrt(np.mean(np.exp(2 * a * c)))
    e = f + sig * rng.normal(size=N)
    flip = rng.random() < .5
    return (e, c) if flip else (c, e), flip


def one(args):
    warnings.filterwarnings("ignore")
    (x, y), flip = make(args)
    sc = scored(x, y, 1)
    out = {k: (float(v[1]) if v[0] == "x->y" else -float(v[1])) for k, v in sc.items()}
    try: out["RESIT"] = resit_pair(x, y)
    except Exception: out["RESIT"] = None
    return {"level": args[0], "flip": bool(flip), "m": out}


def main():
    jobs = [(l, k) for l in range(len(LEVELS)) for k in range(PER)]
    with Pool(4) as p: recs = p.map(one, jobs, chunksize=5)
    av = avici_pairs([make(j)[0] for j in jobs])
    for r, v in zip(recs, av): r["m"]["AVICI"] = v
    res = {"levels": LEVELS, "acc": {}}
    for m in recs[0]["m"]:
        res["acc"][m] = []
        for l in range(len(LEVELS)):
            ok = []
            for r in recs:
                if r["level"] != l or r["m"][m] is None: continue
                s = r["m"][m]; truth_xy = not r["flip"]
                ok.append(0.5 if s == 0 else float((s > 0) == truth_xy))
            res["acc"][m].append({"acc": float(np.mean(ok)), "n": len(ok)})
    for m, v in res["acc"].items():
        k0, n0 = round(v[0]["acc"] * v[0]["n"]), v[0]["n"]; k1, n1 = round(v[-1]["acc"] * v[-1]["n"]), v[-1]["n"]
        res["acc"][m].append({"fisher_p_a0_vs_a1.5": float(fisher_exact([[k0, n0 - k0], [k1, n1 - k1]])[1])})
        print(m, [round(x["acc"], 2) for x in v[:-1]], "p", round(res["acc"][m][-1]["fisher_p_a0_vs_a1.5"], 4))
    (HERE.parent / "results" / "exp50_dose_response.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
