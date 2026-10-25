"""Experiment 53: three further real pair collections from the Causal Chambers, and the diagnostics of exp49.

Written and committed before running.  New collections (ground-truth graph of the chamber configuration, directed
edges between variables that are not duplicates, cause = first variable of the edge):
  bernoulli  wt_bernoulli_v1/random_loads_both: wind tunnel, actuators set by independent Bernoulli draws (binary
             causes); i.i.d. rows (200000; 2000 random rows per pair, two seeds).
  changepoints  wt_changepoints_v1, all 10 experiments: actuator changes at random change points (piecewise-constant
             causes); first 2000 rows in time order, one pair set per experiment.
  malus      lt_malus_v1, all 12 experiments (1000 rows each): light tunnel with polarisers (Malus's law, a strongly
             non-additive mechanism); all rows.
Variables: columns of the configuration's graph with at least 2 distinct values, one of each pair of columns with
|corr| > 0.9999.  For every pair: the stored-protocol rules of exp38/42/46 (IGCI, RECI, ANM-HSIC, pairwise LiNGAM via
sprint_cd.pair_baselines.scored; compression (sorted-conditional, dequantised); TRA; RESIT; AVICI), the heteroscedasticity
h of exp49 in the true direction, the marginal features of exp49 and the marginal-only classifier trained on the
Tuebingen pairs (exp49 features) and applied unchanged (no refitting).  Nothing is tuned on these sets.
Prediction (from exp49-52): on pair sets where the cause is an experimenter-set actuator the marginal-only Tuebingen model
is above 0.75 and IGCI/RECI are above the noise-model rules (ANM-HSIC, RESIT).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json, pathlib, sys, warnings
from multiprocessing import Pool
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
R = HERE.parent / "results"
ROOT = str(HERE.parent / "data" / "causalchamber")
from sprint_cd.pair_baselines import scored
from sprint_cd.pair_2026 import sorted_compression, tra_delta
from sprint_cd.sota_wrappers import resit_pair, avici_pairs
import exp49_explain as e49

SETS = [("bernoulli", "wt_bernoulli_v1", ["random_loads_both"], "wt", "standard", "iid"),
        ("changepoints", "wt_changepoints_v1", None, "wt", "standard", "time"),
        ("malus", "lt_malus_v1", None, "lt", "standard", "all")]
NROW = 2000


def extract():
    import causalchamber.ground_truth as gt
    from causalchamber.datasets import Dataset
    pairs = []
    for name, ds, exps, ch, cfg, mode in SETS:
        d = Dataset(ds, root=ROOT, download=True); G = gt.graph(ch, cfg)
        for e in (exps or sorted(d.available_experiments())):
            df = d.get_experiment(e).as_pandas_dataframe()
            var = [c for c in df.columns if c in G.columns and df[c].nunique() >= 2]
            X = df[var].to_numpy(float); keep = []
            C = np.corrcoef(X.T)
            for i in range(len(var)):
                if all(not (abs(C[i, j]) > 0.9999) for j in keep): keep.append(i)
            var = [var[i] for i in keep]; X = X[:, keep]
            sub = G.loc[var, var]
            for a in var:
                for b in var:
                    if sub.loc[a, b] != 1: continue
                    i, j = var.index(a), var.index(b)
                    for seed in ([0, 1] if mode == "iid" else [0]):
                        if mode == "iid":
                            idx = np.random.default_rng([20263100, seed, i, j]).permutation(len(X))[:NROW]
                        elif mode == "time":
                            idx = np.arange(min(NROW, len(X)))
                        else:
                            idx = np.arange(len(X))
                        x, y = X[idx, i], X[idx, j]
                        if np.ptp(x) == 0 or np.ptp(y) == 0: continue
                        pairs.append({"set": name, "exp": e, "id": f"{e}/{a}-{b}/{seed}", "x": x, "y": y})
    return pairs


def job(p):
    warnings.filterwarnings("ignore")
    x, y = p["x"], p["y"]
    rng = np.random.default_rng(1)
    xd = x + 1e-6 * np.ptp(x) * rng.standard_normal(len(x)) if len(np.unique(x)) < len(x) else x
    yd = y + 1e-6 * np.ptp(y) * rng.standard_normal(len(y)) if len(np.unique(y)) < len(y) else y
    sc = scored(xd, yd, 1)
    m = {k: (float(v[1]) if v[0] == "x->y" else -float(v[1])) for k, v in sc.items()}
    for name, fn in (("Compression", lambda: sorted_compression(xd, yd)), ("TRA", lambda: tra_delta(xd, yd, seed=0)), ("RESIT", lambda: resit_pair(xd, yd))):
        try: m[name] = fn()
        except Exception: m[name] = None
    try: h = e49.het(xd, yd)
    except Exception: h = None
    return {"set": p["set"], "exp": p["exp"], "id": p["id"], "n": int(len(x)), "m": m, "h": h, "marg": (e49.marg(xd) - e49.marg(yd)).tolist()}


def main():
    pairs = extract(); print(len(pairs), "pairs", {s[0]: sum(1 for p in pairs if p["set"] == s[0]) for s in SETS}, flush=True)
    with Pool(4) as pool: recs = pool.map(job, pairs, chunksize=4)
    av = avici_pairs([(p["x"], p["y"]) for p in pairs])
    for r, v in zip(recs, av): r["m"]["AVICI"] = v
    (R / "exp53_more_chamber_pairs.json").write_text(json.dumps({"records": recs}))


if __name__ == "__main__":
    main()
