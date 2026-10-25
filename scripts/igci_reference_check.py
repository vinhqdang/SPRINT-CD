"""IGCI: our implementation against cdt with a Gaussian and a uniform reference measure."""
import sys, json, warnings
import numpy as np
warnings.filterwarnings("ignore"); sys.path.insert(0, "."); sys.path.insert(0, "experiments")
import exp49_explain as e49, exp53_more_chamber_pairs as e53
from cdt.causality.pairwise import IGCI
from sprint_cd.pair_baselines import igci
pairs = [(s, pid, x, y) for s, pid, x, y in e49.gather()] + [(p["set"], p["id"], p["x"], p["y"]) for p in e53.extract()]
def jit(v, seed):
    r = np.random.default_rng(seed); return v + 1e-6 * np.ptp(v) * r.standard_normal(len(v)) if len(np.unique(v)) < len(v) else v
res = {}
for s, pid, x, y in pairs:
    x, y = jit(x, 1), jit(y, 2); d = res.setdefault(s, {"cdt_gauss": [], "cdt_unif": [], "ours": []})
    d["cdt_gauss"].append(float(IGCI().predict_proba((x, y), ref_measure="gaussian"))); d["cdt_unif"].append(float(IGCI().predict_proba((x, y), ref_measure="uniform")))
    d["ours"].append(1.0 if igci(x, y) == "x->y" else -1.0)
out = {}
for s, d in res.items():
    acc = {k: float(np.mean([0.5 if a == 0 else float(a > 0) for a in v])) for k, v in d.items()}
    ag = {"unif_vs_ours": float(np.mean([(a > 0) == (b > 0) for a, b in zip(d["cdt_unif"], d["ours"])])), "gauss_vs_ours": float(np.mean([(a > 0) == (b > 0) for a, b in zip(d["cdt_gauss"], d["ours"])]))}
    out[s] = {"acc": acc, "agree": ag}; print(s, {k: round(v, 2) for k, v in acc.items()}, {k: round(v, 2) for k, v in ag.items()})
json.dump(out, open("results/igci_reference_check.json", "w"), indent=1)
