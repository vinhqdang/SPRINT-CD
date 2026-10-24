"""Serial dependence of the Causal Chamber recordings: lag-1 autocorrelation of the exogenous
variables (nodes without parents in the chamber graph), implied effective sample size
n (1-r)/(1+r) for the first 4000 rows, and the size of the reference set under an i.i.d. and
a block test (exp30)."""
import json, pathlib, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
import exp27_chamber_fresh as e27
import exp20_causal_chamber as e20
e30 = json.load(open(ROOT / "results" / "exp30_btad_chambers.json"))["records"]
rows = []
for spec in e27.specs():
    X, var, edges = e27.load(spec)
    X = X[:4000]
    has_parent = {b for a, b in edges}
    roots = [i for i in range(len(var)) if i not in has_parent]
    ac = [np.corrcoef(X[:-1, i], X[1:, i])[0, 1] for i in roots if X[:, i].std() > 0]
    r = float(np.nanmean(ac)) if ac else float("nan")
    rec = [x for x in e30 if tuple(x["spec"][:2]) == tuple(spec[:2])][0]
    rows.append({"spec": list(spec[:2]), "roots": len(roots), "ac1": r, "ess": float(4000 * (1 - r) / (1 + r)) if r < 1 else 0.0,
                 "V_iid": len(rec["V_iid"]), "V_block": len(rec["V_block"])})
for di in range(len(e20.DATASETS)):
    X, var, edges = e20.load(*e20.DATASETS[di])
    rng = np.random.default_rng(1); Xs = X[rng.permutation(len(X))[:4000]]
    has_parent = {b for a, b in edges}; roots = [i for i in range(len(var)) if i not in has_parent]
    rows.append({"spec": [e20.DATASETS[di][0], e20.DATASETS[di][1] + " (shuffled)"], "roots": len(roots),
                 "ac1": float(np.mean([np.corrcoef(Xs[:-1, i], Xs[1:, i])[0, 1] for i in roots])), "ess": None})
(ROOT / "results" / "serial_diagnostic.json").write_text(json.dumps(rows, indent=1))
w = [r for r in rows if r["spec"][0] == "wt_walks_v1"]
print("wind walks: mean ac1 %.4f, mean ESS %.0f of 4000, V_iid %.1f vs V_block %.1f" % (
    np.mean([r["ac1"] for r in w]), np.mean([r["ess"] for r in w]), np.mean([r["V_iid"] for r in w]), np.mean([r["V_block"] for r in w])))
for r in rows: print(r)
