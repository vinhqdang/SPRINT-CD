"""Experiment 48: resolution of the Sachs graph benchmark.

Written and committed before running.  All methods with Sachs skeletons (exp36, 39, 43, 45; 29 methods
and variants).  For each, F1 against the 17-arc consensus skeleton, and a 95 percent bootstrap interval from
10000 resamples of the 55 variable pairs (with replacement).  We report the number of methods whose interval
contains the best method's point estimate, the number of methods whose point estimate is within 0.10 of the best,
and the smallest F1 difference between two methods that exceeds the bootstrap standard error of the difference
(paired resampling) in at least 95 percent of resamples.
"""
import itertools, json, pathlib
import numpy as np
R = pathlib.Path(__file__).resolve().parent.parent / "results"
rng = np.random.default_rng(20262600)
sets = {}
s36 = json.load(open(R / "exp36_graph_benchmark_extra.json"))["sachs"]; truth = {tuple(sorted(e)) for e in s36["truth"]}
for src in (s36["edges"], json.load(open(R / "exp39_graphs_modern.json"))["sachs"]["edges"],
            json.load(open(R / "exp43_bfbic_boss.json"))["sachs"]["edges"]):
    for k, v in src.items():
        if v is not None: sets[k] = v
for r in json.load(open(R / "exp45_sota_graphs.json"))["records"]:
    if r["kind"] == "sachs":
        for k, v in r["edges"].items():
            if v is not None and not k.endswith(" err"): sets[k] = v
pairs = list(itertools.combinations(range(11), 2)); y = np.array([p in truth for p in pairs])
E = {k: np.array([tuple(sorted(p)) in {tuple(sorted(q)) for q in v} for p in pairs]) for k, v in sets.items()}


def f1(pred, idx):
    yy = y[idx]; pp = pred[idx]; tp = (pp & yy).sum(); fp = (pp & ~yy).sum(); fn = (~pp & yy).sum()
    return 2 * tp / max(2 * tp + fp + fn, 1)


full = np.arange(len(pairs)); pt = {k: f1(E[k], full) for k in E}
B = rng.integers(0, len(pairs), (10000, len(pairs)))
boot = {k: np.array([f1(E[k], b) for b in B]) for k in E}
best = max(pt, key=pt.get)
ci = {k: (float(np.quantile(boot[k], .025)), float(np.quantile(boot[k], .975))) for k in E}
out = {"n_methods": len(E), "best": best, "best_f1": pt[best],
       "contain_best": [k for k in E if ci[k][0] <= pt[best] <= ci[k][1]],
       "within_0.10": [k for k in E if pt[best] - pt[k] <= 0.10],
       "f1": pt, "ci": ci}
out["n_contain_best"] = len(out["contain_best"]); out["n_within_0.10"] = len(out["within_0.10"])
# detectable paired difference
ks = list(E); sig = []
for a, b in itertools.combinations(ks, 2):
    d = boot[a] - boot[b]
    if (d > 0).mean() >= .975 or (d < 0).mean() >= .975: sig.append((a, b, abs(pt[a] - pt[b])))
out["n_pairs_compared"] = len(ks) * (len(ks) - 1) // 2; out["n_significant_pairs"] = len(sig)
out["smallest_significant_difference"] = min((s[2] for s in sig), default=None)
(R / "exp48_sachs_resolution.json").write_text(json.dumps(out, indent=1))
print({k: out[k] for k in ("n_methods", "best", "best_f1", "n_contain_best", "n_within_0.10", "n_pairs_compared", "n_significant_pairs", "smallest_significant_difference")})
