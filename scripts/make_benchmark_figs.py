"""Figures for the benchmark paper."""
import json, pathlib, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
from common import setup_matplotlib
import matplotlib.pyplot as plt
setup_matplotlib()
OUT = ROOT / "paper" / "bench" / "figs"
G = json.load(open(ROOT / "results" / "benchmark_graph_summary.json"))
P = json.load(open(ROOT / "results" / "exp35_summary.json"))
D = json.load(open(ROOT / "results" / "serial_diagnostic.json"))

# Fig 1: pair accuracy
fig, axes = plt.subplots(1, 4, figsize=(11, 3.2), sharey=True)
names = {"tuebingen": "Tübingen (102)", "chamber_iid": "Chambers, i.i.d. (94)", "chamber_walk": "Chambers, walks (122)", "heldout": "held-out simulated (270)"}
rules = ["PDM", "IGCI", "RECI", "ANM-HSIC", "LiNGAM-pair"]
for ax, (k, t) in zip(axes, names.items()):
    v = [P[k][r]["acc"] for r in rules]
    ci = np.array([P[k][r]["ci"] for r in rules])
    ax.bar(range(len(rules)), v, color=["C3", "0.6", "C0", "0.6", "0.6"], yerr=[np.array(v) - ci[:, 0], ci[:, 1] - np.array(v)], capsize=2)
    ax.axhline(0.5, color="k", lw=0.6, ls=":")
    ax.set_xticks(range(len(rules))); ax.set_xticklabels(rules, rotation=40, ha="right", fontsize=7)
    ax.set_title(t, fontsize=8.5)
axes[0].set_ylabel("forced-choice accuracy")
fig.tight_layout(); fig.savefig(OUT / "pairs_accuracy.png", dpi=180)

# Fig 2: graph F1
groups = list(G.keys())
meths = ["PC a=0.05", "GES-BIC", "DirectLiNGAM", "NOTEARS", "PC-GCM a=0.05", "BTAD", "LRG", "CERT-CD nonparametric"]
lab = ["PC", "GES", "LiNGAM", "NOTEARS", "PC-GCM", "BTAD", "LRG", "CERT-CD"]
fig, ax = plt.subplots(figsize=(8.5, 3.4))
w = 0.1
for i, (m, l) in enumerate(zip(meths, lab)):
    v = [G[g].get(m, {"F1": np.nan})["F1"] for g in groups]
    ax.bar(np.arange(len(groups)) + (i - 3.5) * w, v, w, label=l)
ax.set_xticks(range(len(groups)))
ax.set_xticklabels(["Sachs\n(1)", "Chambers i.i.d.\n(2 rec. x 10)", "Chambers\nESS > 1000 (2)", "Chambers\nESS < 20 (17)"], fontsize=8)
ax.set_ylabel("skeleton F1 (strict)"); ax.legend(fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.18))
fig.tight_layout(); fig.savefig(OUT / "graph_f1.png", dpi=180)

# Fig 3: effective sample size and reference set
rows = [r for r in D if r["ess"] is not None and "V_iid" in r]
fig, axes = plt.subplots(1, 2, figsize=(8, 3.1))
ess = np.array([max(r["ess"], 1) for r in rows])
axes[0].scatter(range(len(rows)), ess, s=14); axes[0].set_yscale("log"); axes[0].axhline(4000, color="k", lw=0.6, ls=":")
axes[0].set_xlabel("recording"); axes[0].set_ylabel("effective sample size of the exogenous variables\n(out of 4000 rows)", fontsize=7.5)
vi = np.array([r["V_iid"] for r in rows]); vb = np.array([r["V_block"] for r in rows])
axes[1].scatter(vi, vb, s=14); lim = max(vi.max(), vb.max()) + 2; axes[1].plot([0, lim], [0, lim], "k:", lw=0.6)
axes[1].set_xlabel("pairs rejected by an i.i.d. test"); axes[1].set_ylabel("pairs rejected by a block test")
fig.tight_layout(); fig.savefig(OUT / "serial_dependence.png", dpi=180)
print("ok")
