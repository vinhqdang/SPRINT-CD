import json, pathlib
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parent.parent; R = ROOT / "results"; OUT = ROOT / "paper" / "resolution"
A = json.load(open(R / "resolution_assets.json")); corr, ids = A["corr"], A["ids"]
COLL = ["tuebingen", "chamber_iid", "chamber_walk", "sachs", "bernoulli", "changepoints", "malus"]
def cluster(s, i):
    if s in ("tuebingen", "sachs"): return i
    if s == "chamber_iid": return i.rsplit("/", 1)[0]
    if s == "chamber_walk": return i.split("/")[0]
    return i.split("/")[1]
rng = np.random.default_rng(11); out = {}; rows = []
names = {"LiNGAM-pair": "Pairwise LiNGAM", "ANM-GP": "ANM, Gaussian process", "Compression first": "Compression, first impl.", "Compression faithful": "Compression, faithful"}
ncl = {}
for s in COLL:
    cl = np.array([cluster(s, i) for i in ids[s]]); u = np.unique(cl); ncl[s] = len(u)
for m, d in corr.items():
    cells = []
    for s in COLL:
        v = np.array([np.nan if x is None else x for x in d.get(s, [])], float)
        if not len(v) or np.all(np.isnan(v)): cells.append("--"); continue
        cl = np.array([cluster(s, i) for i in ids[s]]); u = np.unique(cl)
        sums = {c: (np.nansum(v[cl == c]), np.sum(~np.isnan(v[cl == c]))) for c in u}
        boots = []
        for _ in range(3000):
            pick = rng.choice(len(u), len(u)); num = sum(sums[u[k]][0] for k in pick); den = sum(sums[u[k]][1] for k in pick)
            boots.append(num / den if den else np.nan)
        lo, hi = np.nanquantile(boots, [.025, .975]); mean = np.nanmean(v)
        out.setdefault(m, {})[s] = [float(mean), float(lo), float(hi)]
        cells.append(f"{mean:.2f} [{lo:.2f},{hi:.2f}]")
    rows.append(names.get(m, m) + " & " + " & ".join(cells) + "\\\\")
open(OUT / "tab_collections_ci.tex", "w").write("\n".join(rows))
json.dump({"ci": out, "n_clusters": ncl}, open(R / "cluster_bootstrap.json", "w"), indent=1)
print(ncl)
walk_names = {i.split("/")[1] for i in ids["chamber_walk"]}; print("distinct walk names", len(walk_names))
import numpy as np
K = np.array(A["kendall"], float); P = np.array(A["kendall_perm_p"], float)
lab = ["Tuebingen", "iid", "walks", "Sachs", "Bernoulli", "Changepoints", "Malus"]
tex = []
for i in range(7):
    tex.append(lab[i] + " & " + " & ".join("--" if i == j or np.isnan(K[i, j]) else f"{K[i, j]:+.2f} ({P[i, j]:.2f})" for j in range(7)) + "\\\\")
open(OUT / "tab_kendall.tex", "w").write("\n".join(tex).replace("Tuebingen", "T\\\"ubingen"))
print("\n".join(tex))
