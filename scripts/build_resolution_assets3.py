import json, pathlib, itertools
import numpy as np
from scipy.stats import beta
ROOT = pathlib.Path(__file__).resolve().parent.parent; R = ROOT / "results"; OUT = ROOT / "paper" / "resolution"
L = lambda f: json.load(open(R / f))
# 1. universes
d = L("exp47_multiverse.json"); M = d["methods"]
nm = {"LiNGAM-pair": "LiNGAM", "ANM-GP": "ANM-GP", "Compression": "Cmp1", "Compression raw": "Cmp1r", "Compression faithful": "CmpF", "ANM-HSIC": "ANM-H"}
head = " & ".join(nm.get(m, m) for m in M)
rows = []
for u in d["universes"]:
    rows.append(f"{u['weights']} & {u['min_rows']} & {int(u['coverage']*100)} & {u['n_selected']} & {u['min_evaluated']} & " + " & ".join(f"{a:.2f}" for a in u["acc"]) + "\\\\")
open(OUT / "tab_universes.tex", "w").write("\n".join(rows) + "\n"); open(OUT / "tab_universes_head.tex", "w").write(head)
# 2. collections with intervals
A = L("resolution_assets.json"); acc, n = A["acc"], A["n"]
COLL = ["tuebingen", "chamber_iid", "chamber_walk", "sachs", "bernoulli", "changepoints", "malus"]
def cp(k, nn, a=.05):
    return (0 if k == 0 else beta.ppf(a / 2, k, nn - k + 1), 1 if k == nn else beta.ppf(1 - a / 2, k + 1, nn - k))
names = {"LiNGAM-pair": "Pairwise LiNGAM", "ANM-GP": "ANM, Gaussian process", "Compression first": "Compression, first impl.", "Compression faithful": "Compression, faithful"}
rows = []
for m in acc:
    cells = []
    for s in COLL:
        if s in acc[m]:
            lo, hi = cp(round(acc[m][s] * n[m][s]), n[m][s]); cells.append(f"{acc[m][s]:.2f} [{lo:.2f},{hi:.2f}]")
        else: cells.append("--")
    rows.append(names.get(m, m) + " & " + " & ".join(cells) + "\\\\")
open(OUT / "tab_collections_ci.tex", "w").write("\n".join(rows) + "\n")
# 3. factorial per-cell
sim = L("exp56_factorial.json")["records"]; RULES = ["IGCI", "RECI", "ANM-HSIC", "LiNGAM-pair", "Compression", "RESIT", "AVICI"]
def corr(s, flip):
    if s is None or s != s: return np.nan
    return 0.5 if s == 0 else float((s > 0) == (not flip))
cells = {}
for r in sim:
    cells.setdefault(tuple(r["cell"]), []).append([corr(r["m"].get(m), r["flip"]) for m in RULES])
rows = []
mech = ["tanh", "cubic", "sin"]
for c in sorted(cells, key=lambda c: (c[0], ["uniform", "normal", "bimodal", "discrete5"].index(c[1]), ["homo", "cause", "indep"].index(c[2]), c[3], c[4])):
    a = np.nanmean(np.array(cells[c]), 0)
    rows.append(f"{mech[c[0]]} & {c[1]} & {c[2]} & {'y' if c[3] else 'n'} & {c[4]} & " + " & ".join(f"{x:.2f}" for x in a) + "\\\\")
open(OUT / "tab_factorial_cells.tex", "w").write("\n".join(rows) + "\n")
# 4. Sachs resolution
b = L("exp48_sachs_resolution.json")
rows = []
for m, v in sorted(b["f1"].items(), key=lambda kv: -kv[1]):
    lo, hi = b["ci"][m]; label = m.replace('_', '\\_').replace('a=', '$\\alpha=$'); rows.append(label + f" & {v:.2f} & [{lo:.2f}, {hi:.2f}]\\\\")
open(OUT / "tab_sachs_ci.tex", "w").write("\n".join(rows) + "\n")
print(len(cells), len(rows))
