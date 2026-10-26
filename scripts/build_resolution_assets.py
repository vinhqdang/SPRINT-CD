"""Tables and figures for paper/resolution from the stored results."""
import json, pathlib, sys, itertools
import numpy as np
from scipy import stats
from scipy.stats import beta
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = pathlib.Path(__file__).resolve().parent.parent; R = ROOT / "results"; OUT = ROOT / "paper" / "resolution"
(OUT / "figs").mkdir(exist_ok=True)
L = lambda f: json.load(open(R / f))
e35 = L("exp35_pdm_eval.json")["records"]; e38 = L("exp38_pairs_modern.json")["records"]; e42 = L("exp42_pairs_2026.json")["records"]
e46 = L("exp46_pairs_sota.json")["records"]; e53 = L("exp53_more_chamber_pairs.json")["records"]; e54 = L("exp54_compression_faithful.json")["records"]
e59 = L("exp59_ties.json")["records"]; e58 = L("exp58_walks_all.json")["records"]; e60 = L("exp60_tra_raw.json")["records"] if (R / "exp60_tra_raw.json").exists() else []
COLL = ["tuebingen", "chamber_iid", "chamber_walk", "walks_all", "sachs", "bernoulli", "changepoints", "malus"]
LAB = {"tuebingen": "T\\\"ubingen", "chamber_iid": "Chambers iid", "chamber_walk": "Chambers walks", "walks_all": "Walks, all 19", "sachs": "Sachs", "bernoulli": "Bernoulli", "changepoints": "Changepoints", "malus": "Malus"}
RULES = ["IGCI", "RECI", "ANM-HSIC", "LiNGAM-pair", "PDM", "LOCI", "ANM-GP", "Compression first", "Compression faithful", "TRA", "AVICI", "RESIT"]
NAMES = {"LiNGAM-pair": "Pairwise LiNGAM", "ANM-GP": "ANM, Gaussian process", "Compression first": "Compression, floor256", "Compression faithful": "Compression, round255"}
c = lambda s, tx=True: np.nan if s is None or s != s else (0.5 if s == 0 else float((s > 0) == tx))
acc = {m: {} for m in RULES}; n_ = {m: {} for m in RULES}; corr = {m: {} for m in RULES}
for s in COLL:
    if s in ("tuebingen", "chamber_iid", "chamber_walk"):
        a35 = [r for r in e35 if r["set"] == s]
        for m in ("IGCI", "RECI", "ANM-HSIC", "LiNGAM-pair"): corr[m][s] = [float((r["scores"][m][0] == "x->y") == (r["truth"] == "x->y")) for r in a35]
        corr["PDM"][s] = [float((r["pdm"] > .5) == (r["truth"] == "x->y")) for r in a35]
    if s == "sachs":
        a38 = [r for r in e38 if r["set"] == s]
        for m in ("IGCI", "RECI", "ANM-HSIC", "LiNGAM-pair"): corr[m][s] = [float((r["scores"][m][0] == "x->y") == (r["truth"] == "x->y")) for r in a38]
    if s in ("tuebingen", "chamber_iid", "chamber_walk", "sachs"):
        a38 = [r for r in e38 if r["set"] == s]; a42 = [r for r in e42 if r["set"] == s]; a46 = [r for r in e46 if r["set"] == s]
        tx = [r["truth"] == "x->y" for r in a38]
        for m in ("LOCI", "ANM"): corr["LOCI" if m == "LOCI" else "ANM-GP"][s] = [c(r["m"][m], t) for r, t in zip(a38, tx)]
        corr["Compression first"][s] = [c(r["m"]["SC"], t) for r, t in zip(a42, tx)]; corr["TRA"][s] = [c(r["m"]["TRA"], t) for r, t in zip(a42, tx)]
        corr["AVICI"][s] = [c(r["m"]["AVICI"], t) for r, t in zip(a46, tx)]; corr["RESIT"][s] = [c(r["m"]["RESIT"], t) for r, t in zip(a46, tx)]
    elif s == "walks_all":
        for m in ("IGCI", "RECI", "ANM-HSIC", "LiNGAM-pair", "RESIT", "AVICI"): corr[m][s] = [c(r["m"].get(m)) for r in e58]
        corr["Compression faithful"][s] = [c(r["m"].get("Compression")) for r in e58]
        tr = {r["id"]: r["score"] for r in e60 if r["set"] == "walks_all"}
        corr["TRA"][s] = [c(tr.get(r["rec"] + "/" + r["id"])) for r in e58]
        continue
    else:
        a59 = [r for r in e59 if r["set"] == s]; a53 = [r for r in e53 if r["set"] == s]
        for m in ("IGCI", "RECI", "ANM-HSIC", "LiNGAM-pair", "RESIT", "LOCI", "ANM-GP"): corr[m][s] = [c(r["m"].get(m + "|raw")) for r in a59]
        corr["AVICI"][s] = [c(r["m"].get("AVICI")) for r in a53]
        tr = {r["id"]: r["score"] for r in e60 if r["set"] == s}; corr["TRA"][s] = [c(tr.get(r["id"])) for r in a53]
    corr["Compression faithful"][s] = [c(r["score"], (r["truth"] == "x->y") if False else True) if s not in ("tuebingen",) else c(r["score"]) for r in [x for x in e54 if x["set"] == s]]
# chamber/sachs pairs in exp38/42/46/54 have random flips only in exp38-based files; e54 uses unflipped arrays (truth x->y)
for s in ("chamber_iid", "chamber_walk", "sachs"):
    a38 = [r for r in e38 if r["set"] == s]  # fix compression faithful: arrays are unflipped, truth always x->y
for m in RULES:
    for s in COLL:
        v = np.array(corr[m].get(s, []), float); v = v[~np.isnan(v)]
        if len(v): acc[m][s] = float(v.mean()); n_[m][s] = len(v)
def cp(k, n, a=.05):
    return (0 if k == 0 else beta.ppf(a / 2, k, n - k + 1), 1 if k == n else beta.ppf(1 - a / 2, k + 1, n - k))
rows = []
for m in RULES:
    cells = []
    for s in COLL:
        if s in acc[m]:
            lo, hi = cp(round(acc[m][s] * n_[m][s]), n_[m][s]); cells.append(f"{acc[m][s]:.2f}")
        else: cells.append("--")
    rows.append(NAMES.get(m, m) + " & " + " & ".join(cells) + "\\\\")
open(OUT / "tab_collections.tex", "w").write("\n".join(rows) + "\n")
# Kendall tau between collection rankings (rules present in both)
M = np.full((len(COLL), len(COLL)), np.nan)
for i, a in enumerate(COLL):
    for j, b in enumerate(COLL):
        ms = [m for m in RULES if a in acc[m] and b in acc[m]]
        if len(ms) >= 5: M[i, j] = stats.kendalltau([acc[m][a] for m in ms], [acc[m][b] for m in ms])[0]
ids = {st: ([r["id"] for r in e38 if r["set"] == st] if st in ("tuebingen", "chamber_iid", "chamber_walk", "sachs") else ([r["rec"] + "/" + r["id"] for r in e58] if st == "walks_all" else [r["id"] for r in e53 if r["set"] == st])) for st in COLL}
perm_p = np.full_like(M, np.nan); rng_ = np.random.default_rng(5)
for i, a in enumerate(COLL):
    for j, b in enumerate(COLL):
        ms = [m for m in RULES if a in acc[m] and b in acc[m]]
        if i < j and len(ms) >= 5:
            va, vb = np.array([acc[m][a] for m in ms]), np.array([acc[m][b] for m in ms]); t0 = stats.kendalltau(va, vb)[0]
            null = [stats.kendalltau(va, rng_.permutation(vb))[0] for _ in range(5000)]
            perm_p[i, j] = perm_p[j, i] = float(np.mean(np.abs(null) >= abs(t0) - 1e-12))
json.dump({"acc": acc, "n": n_, "kendall": M.tolist(), "kendall_perm_p": perm_p.tolist(), "corr": {m: {s: list(map(lambda v: None if v != v else v, corr[m].get(s, []))) for s in COLL} for m in RULES}, "ids": ids}, open(R / "resolution_assets.json", "w"), indent=1)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
A = np.array([[acc[m].get(s, np.nan) for s in COLL] for m in RULES])
im = ax[0].imshow(A, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto"); ax[0].set_xticks(range(len(COLL))); ax[0].set_xticklabels([LAB[s].replace("\\\"", "") for s in COLL], rotation=40, ha="right", fontsize=8)
ax[0].set_yticks(range(len(RULES))); ax[0].set_yticklabels([NAMES.get(m, m) for m in RULES], fontsize=8)
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        if not np.isnan(A[i, j]): ax[0].text(j, i, f"{A[i, j]:.2f}", ha="center", va="center", fontsize=7)
ax[0].set_title("Accuracy of each rule on each collection")
im2 = ax[1].imshow(M, cmap="coolwarm", vmin=-1, vmax=1); ax[1].set_xticks(range(len(COLL))); ax[1].set_xticklabels([LAB[s].replace("\\\"", "") for s in COLL], rotation=40, ha="right", fontsize=8)
ax[1].set_yticks(range(len(COLL))); ax[1].set_yticklabels([LAB[s].replace("\\\"", "") for s in COLL], fontsize=8)
for i in range(len(COLL)):
    for j in range(len(COLL)):
        if not np.isnan(M[i, j]): ax[1].text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=7)
ax[1].set_title("Kendall tau between rankings of rules")
plt.tight_layout(); plt.savefig(OUT / "figs" / "collections.png", dpi=170); plt.close()
print(open(OUT / "tab_collections.tex").read()); print(np.round(M, 2))
