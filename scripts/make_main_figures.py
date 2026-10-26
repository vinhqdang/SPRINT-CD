import json, pathlib
import numpy as np
from scipy.stats import beta, norm
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
ROOT = pathlib.Path(__file__).resolve().parent.parent; R = ROOT / "results"; F = ROOT / "paper" / "resolution_jci" / "figs"
L = lambda f: json.load(open(R / f))
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
C = {"blue": "#2c7fb8", "red": "#d95f02", "green": "#1b9e77", "grey": "#888888", "purple": "#7570b3"}

# ---------- Fig 1: overview
fig, ax = plt.subplots(figsize=(10, 3.4)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 34)
def box(x, y, w, h, txt, col):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=col, ec="#333", lw=0.8)); ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=8.2)
def arrow(x1, y1, x2, y2): ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10, color="#333", lw=1))
box(1, 13, 17, 12, "8 real pair\ncollections\n(Tübingen, Sachs,\n6 chamber sets)", "#dbe9f6")
box(23, 13, 17, 12, "12 direction rules\n(IGCI, RECI, ANM,\nLiNGAM, LOCI, AVICI,\ncompression, TRA, ...)", "#fde5cf")
box(45, 13, 17, 12, "stored decision\nof every rule on\nevery pair\n(raw arrays)", "#e5f2dc")
arrow(18.5, 19, 22.5, 19); arrow(40.5, 19, 44.5, 19)
box(68, 25, 30, 7, "§3 resolution: noise, clusters, multiverse", "#f3f3f3")
box(68, 16, 30, 7, "§4 conventions: rounding, ties, reference", "#f3f3f3")
box(68, 7, 30, 7, "§5 mechanisms: marginals, noise regimes", "#f3f3f3")
box(68, -2, 30, 7, "§6 recommendations (Algorithm 1)", "#f9e0e0")
for y in (28.5, 19.5, 10.5): arrow(62.8, 19, 67.5, y)
arrow(62.8, 19, 67.5, 1.5)
ax.set_ylim(-4, 34); plt.savefig(F / "fig1_overview.png", bbox_inches="tight", dpi=170); plt.close()

# ---------- Fig 3: resolution (forest + detectable difference)
d = L("exp47_multiverse.json")
names = {"LiNGAM-pair": "Pairwise LiNGAM", "ANM-GP": "ANM (GP)", "Compression": "Compr. floor256", "Compression raw": "Compr. floor256 raw", "Compression faithful": "Compr. round255"}
def cp(k, n, a=.05):
    return (0 if k == 0 else beta.ppf(a / 2, k, n - k + 1), 1 if k == n else beta.ppf(1 - a / 2, k + 1, n - k))
ms = [m for m in d["methods"] if m not in ("Compression raw",)]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.9), gridspec_kw={"width_ratios": [1.25, 1]})
ys = np.arange(len(ms))[::-1]
for y, m in zip(ys, ms):
    nn = 82 if m == "TRA" else 102; a = d["accuracy"][m]; lo, hi = cp(round(a * nn), nn)
    ax[0].plot([lo, hi], [y, y], color=C["blue"], lw=2); ax[0].plot(a, y, "o", color=C["red"], ms=5)
ax[0].axvline(.5, ls=":", c="k", lw=.8); ax[0].set_yticks(ys); ax[0].set_yticklabels([names.get(m, m) for m in ms]); ax[0].set_xlabel("accuracy on 102 Tübingen pairs (95% interval)"); ax[0].set_xlim(.25, .85)
ax[0].set_title("(a) the intervals overlap", fontsize=9)
nn = np.arange(20, 3000, 10); dd = 0.42
for lab, al, col in (("α=0.05", .05, C["blue"]), ("Bonferroni, 78 comparisons", .05 / 78, C["red"])):
    ax[1].plot(nn, (norm.ppf(1 - al / 2) + norm.ppf(.8)) * np.sqrt(dd / nn), color=col, label=lab)
ax[1].axvline(102, ls="--", c=C["grey"]); ax[1].text(110, .62, "Tübingen\n(n = 102)", fontsize=8, color=C["grey"])
for dl in (.10, .05): ax[1].plot([7.85 * dd / dl ** 2], [dl], "ks", ms=4); ax[1].text(7.85 * dd / dl ** 2 * 1.25, dl + (.07 if dl == .05 else .02), f"{int(round(7.85 * dd / dl ** 2, -1))} pairs\nfor Δ = {dl:.2f}", fontsize=8)
ax[1].set_xscale("log"); ax[1].set_xlabel("number of pairs n"); ax[1].set_ylabel("smallest difference detected\nwith 80% power"); ax[1].set_ylim(0, .7); ax[1].legend(frameon=False, fontsize=8)
ax[1].set_title("(b) resolution of a paired test", fontsize=9)
plt.tight_layout(); plt.savefig(F / "fig3_resolution.png", dpi=170); plt.close()

# ---------- Fig 4: reporting freedom
e61 = L("exp61_protocol_multiverse.json")["records"]
rules61 = [("IGCI", "IGCI (uniform ref.)"), ("cdt-IGCI", "IGCI (cdt, Gaussian ref.)"), ("RECI", "RECI"), ("cdt-RECI", "RECI (cdt)"), ("ANM-HSIC", "ANM-HSIC"), ("LiNGAM-pair", "Pairwise LiNGAM"), ("Compression", "Compression (round255)"), ("cdt-CDS", "CDS (cdt)")]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))
ms2 = [m for m in d["methods"] if m not in ("Compression raw", "Compression")]
ys = np.arange(len(ms2))[::-1]
for y, m in zip(ys, ms2):
    mv = d["multiverse"][m]; ax[0].plot([mv["acc_min"], mv["acc_max"]], [y, y], color=C["blue"], lw=5, alpha=.6, solid_capstyle="butt"); ax[0].plot(d["accuracy"][m], y, "o", color=C["red"], ms=5)
ax[0].set_yticks(ys); ax[0].set_yticklabels([names.get(m, m) for m in ms2]); ax[0].axvline(.5, ls=":", c="k", lw=.8); ax[0].set_xlabel("accuracy range over the ten reporting universes"); ax[0].set_title("(a) ten reporting universes\n(weights, row threshold, coverage)", fontsize=9)
ys = np.arange(len(rules61))[::-1]
for y, (m, lab) in zip(ys, rules61):
    u = [r["res"][m]["unweighted"] for r in e61]; w = [r["res"][m]["weighted"] for r in e61]
    ax[1].plot([min(u), max(u)], [y + .15, y + .15], color=C["blue"], lw=4, alpha=.6, solid_capstyle="butt"); ax[1].plot([min(w), max(w)], [y - .15, y - .15], color=C["green"], lw=4, alpha=.6, solid_capstyle="butt")
ax[1].set_yticks(ys); ax[1].set_yticklabels([l for _, l in rules61]); ax[1].axvline(.5, ls=":", c="k", lw=.8); ax[1].set_xlabel("accuracy range over 50 preprocessing configurations")
ax[1].set_title("(b) 50 preprocessing configurations\n(blue: unweighted, green: weighted)", fontsize=9)
ax[0].set_xlim(.35, .95); ax[1].set_xlim(.35, .85)
plt.tight_layout(); plt.savefig(F / "fig4_freedom.png", dpi=170); plt.close()

# ---------- Fig 5: conventions
t = L("exp59_ties.json")["records"]; cols = ["tuebingen", "chamber_iid", "chamber_walk", "sachs", "bernoulli", "changepoints", "malus"]; lab = ["Tübingen", "i.i.d.", "walks", "Sachs", "Bernoulli", "Chgpts", "Malus"]
def acc(rule, cond, s):
    v = [x["m"].get(f"{rule}|{cond}") for x in t if x["set"] == s]; v = [a for a in v if a is not None and a == a]
    return np.mean([0.5 if a == 0 else float(a > 0) for a in v])
fig, ax = plt.subplots(1, 2, figsize=(10, 3.7), gridspec_kw={"width_ratios": [1.2, 1]})
x = np.arange(len(cols)); w = .38
ax[0].bar(x - w / 2, [acc("IGCI", "raw", s) for s in cols], w, color=C["blue"], label="ties kept (raw)"); ax[0].bar(x + w / 2, [acc("IGCI", "jitter", s) for s in cols], w, color=C["red"], label="ties broken (jitter 1e-6)")
ax[0].axhline(.5, ls=":", c="k", lw=.8); ax[0].set_xticks(x); ax[0].set_xticklabels(lab, rotation=30, ha="right"); ax[0].set_ylabel("IGCI accuracy"); ax[0].legend(frameon=False, fontsize=8, loc="upper center", ncol=2, bbox_to_anchor=(.5, 1.0)); ax[0].set_ylim(0, 1.18); ax[0].set_title("(a) one line of code reverses IGCI", fontsize=9)
g = L("exp54_audit_conventions.json"); order = [(m, c) for m in ("floor256", "floor255", "round255") for c in (500, 1000, 2000, 4000, 8000, 1000000)]
cols3 = {"floor256": C["red"], "floor255": C["purple"], "round255": C["green"]}
for i, (m, c) in enumerate(order): ax[1].bar(i, g[f"{m}_cap{c}"]["weighted"], color=cols3[m])
ax[1].axhline(.747, c="k", ls="--", lw=1); ax[1].text(0, .752, "published 74.7%", fontsize=8)
ax[1].set_ylim(.55, .8); ax[1].set_xticks([2.5, 8.5, 14.5]); ax[1].set_xticklabels(["floor(256q)", "floor(255q)", "round(255q)"]); ax[1].set_ylabel("weighted accuracy (Tübingen)"); ax[1].set_title("(b) quantiser × row cap: 18 conventions", fontsize=9)
plt.tight_layout(); plt.savefig(F / "fig5_conventions.png", dpi=170); plt.close()

# ---------- Fig 6: mechanisms (3 panels)
st = L("exp49_strata.json")["acc"]; e52 = L("exp52_controls.json"); a = e52["A"]["all5"]
fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.6), gridspec_kw={"width_ratios": [1, 1.25, 1]})
sets = ["tuebingen", "chamber_iid", "chamber_walk", "sachs"]; T = np.full((4, 4), np.nan)
for i, s in enumerate(sets):
    for j, tt in enumerate(sets):
        if i != j: T[i, j] = a[f"transfer_{s}_to_{tt}"]
ax[0].imshow(T, cmap="RdYlGn", vmin=0, vmax=1)
for i in range(4):
    for j in range(4):
        if not np.isnan(T[i, j]): ax[0].text(j, i, f"{T[i, j]:.2f}", ha="center", va="center", fontsize=8)
ax[0].set_xticks(range(4)); ax[0].set_xticklabels(["Tüb", "iid", "walk", "Sachs"]); ax[0].set_yticks(range(4)); ax[0].set_yticklabels(["Tüb", "iid", "walk", "Sachs"]); ax[0].set_xlabel("test"); ax[0].set_ylabel("train"); ax[0].set_title("(a) marginal-only classifier", fontsize=9)
ms3 = ["ANM-HSIC", "RESIT", "ANM-GP", "TRA", "LOCI", "RECI", "IGCI"]; w = .27
for i, (tt, col) in enumerate(zip(["low", "mid", "high"], ["#7fbf7b", "#e0e0e0", "#af8dc3"])):
    ax[1].bar(np.arange(len(ms3)) + (i - 1) * w, [st[m][i] for m in ms3], w, label=f"{tt} h", color=col, edgecolor="k", lw=.4)
ax[1].axhline(.5, ls=":", c="k", lw=.8); ax[1].set_xticks(range(len(ms3))); ax[1].set_xticklabels(["ANM-\nHSIC", "RESIT", "ANM\n(GP)", "TRA", "LOCI", "RECI", "IGCI"], fontsize=8); ax[1].set_ylabel("Tübingen accuracy"); ax[1].legend(frameon=False, fontsize=8, ncol=3, loc="upper right"); ax[1].set_title("(b) by residual heteroscedasticity", fontsize=9); ax[1].set_ylim(0, 1.15)
lv = [0, .5, 1, 1.5]
for m, col in (("ANM-HSIC", C["red"]), ("IGCI", C["blue"]), ("RECI", C["green"])):
    ax[2].plot(lv, e52["B"]["hetero"][m], "-o", color=col, label=m + " (scale depends on cause)", ms=4); ax[2].plot(lv, e52["B"]["independent"][m], "--s", color=col, alpha=.6, ms=3)
ax[2].axhline(.5, ls=":", c="k", lw=.8); ax[2].set_xlabel("strength a of scale variation"); ax[2].set_ylabel("accuracy"); ax[2].set_title("(c) matched manipulation\n(solid: depends on cause; dashed: control)", fontsize=9); from matplotlib.lines import Line2D
ax[2].legend(handles=[Line2D([0],[0],color=C["red"],marker="o",label="ANM-HSIC"),Line2D([0],[0],color=C["blue"],marker="o",label="IGCI"),Line2D([0],[0],color=C["green"],marker="o",label="RECI")],frameon=False,fontsize=8,loc="center right")
plt.tight_layout(); plt.savefig(F / "fig6_mechanisms.png", dpi=170); plt.close()
print("ok")
