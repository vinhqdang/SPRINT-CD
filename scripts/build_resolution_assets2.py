import json, pathlib, itertools
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = pathlib.Path(__file__).resolve().parent.parent; R = ROOT / "results"; OUT = ROOT / "paper" / "resolution"
L = lambda f: json.load(open(R / f))
sim = L("exp56_factorial.json")["records"]
RULES = ["IGCI", "RECI", "ANM-HSIC", "LiNGAM-pair", "Compression", "RESIT", "AVICI"]
def corr(s, flip):
    if s is None or s != s: return np.nan
    return 0.5 if s == 0 else float((s > 0) == (not flip))
C = np.array([[corr(r["m"].get(m), r["flip"]) for m in RULES] for r in sim], float)
cell = np.array([r["cell"] for r in sim], dtype=object)
fac = {"mechanism": (0, ["tanh", "cubic", "sin"], lambda v: v), "cause marginal": (1, ["uniform", "normal", "bimodal", "discrete5"], str), "noise scale": (2, ["homo", "cause", "indep"], str),
       "measurement noise": (3, [0, 1], int), "n": (4, [200, 1000], int)}
lines = []
for name, (k, levels, cast) in fac.items():
    for lv_i, lv in enumerate(levels):
        sel = np.array([ (r["cell"][k] == lv_i) if name == "mechanism" else (r["cell"][k] == lv) for r in sim])
        lines.append((name, str(lv), np.nanmean(C[sel], 0)))
rows = []
for name, lv, a in lines:
    rows.append(f"{name} & {lv} & " + " & ".join(f"{x:.2f}" for x in a) + "\\\\")
open(OUT / "tab_factorial.tex", "w").write("\n".join(rows).replace("homo", "homoscedastic").replace("cause\\", "cause-dependent\\").replace(" indep ", " independent scale ") + "\n")
# interaction: scale x measurement
inter = []
for sc_i, sc_n in enumerate(["homoscedastic", "cause-dependent", "independent scale"]):
    for me in (0, 1):
        sel = np.array([(r["cell"][2] == ["homo", "cause", "indep"][sc_i]) and r["cell"][3] == me for r in sim])
        inter.append(f"{sc_n} & {'yes' if me else 'no'} & " + " & ".join(f"{x:.2f}" for x in np.nanmean(C[sel], 0)) + "\\\\")
open(OUT / "tab_factorial_inter.tex", "w").write("\n".join(inter) + "\n")
print(open(OUT / "tab_factorial.tex").read()); print(open(OUT / "tab_factorial_inter.tex").read())
# figures
e50 = L("exp50_dose_response.json"); e52 = L("exp52_controls.json")
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6)); lv = [0, .5, 1, 1.5]
for m, col in (("ANM-HSIC", "C3"), ("IGCI", "C0"), ("RECI", "C2")):
    ax[0].plot(lv, e52["B"]["hetero"][m], "-o", color=col, label=m); ax[1].plot(lv, e52["B"]["independent"][m], "-o", color=col, label=m)
for a_, t in zip(ax, ["Noise scale depends on the cause", "Independent random scale (control)"]):
    a_.set_xlabel("strength a"); a_.set_ylabel("accuracy"); a_.axhline(.5, ls=":", c="grey"); a_.set_title(t, fontsize=9); a_.set_ylim(0, 1.02)
ax[0].legend(fontsize=8); plt.tight_layout(); plt.savefig(OUT / "figs" / "dose.png", dpi=170); plt.close()
st = L("exp49_strata.json")["acc"]; ms = ["ANM-HSIC", "RESIT", "ANM-GP", "TRA", "LOCI", "RECI", "IGCI", "Compression raw", "AVICI"]
fig, ax = plt.subplots(figsize=(8, 3.6)); w = .27
for i, (t, col) in enumerate(zip(["low h", "middle", "high h"], ["#7fbf7b", "#f7f7f7", "#af8dc3"])):
    ax.bar(np.arange(len(ms)) + (i - 1) * w, [st[m][i] for m in ms], w, label=t, color=col, edgecolor="k", lw=.4)
ax.axhline(.5, ls=":", c="grey"); ax.set_xticks(range(len(ms))); ax.set_xticklabels(ms, rotation=30, ha="right", fontsize=8); ax.set_ylabel("accuracy"); ax.legend(fontsize=8); ax.set_title("Tübingen by tercile of residual heteroscedasticity", fontsize=9)
plt.tight_layout(); plt.savefig(OUT / "figs" / "strata.png", dpi=170); plt.close()
a = L("exp52_controls.json")["A"]["all5"]; sets = ["tuebingen", "chamber_iid", "chamber_walk", "sachs"]
T = np.full((4, 4), np.nan)
for i, s in enumerate(sets):
    for j, t in enumerate(sets):
        if i != j: T[i, j] = a[f"transfer_{s}_to_{t}"]
fig, ax = plt.subplots(figsize=(4.8, 4)); im = ax.imshow(T, cmap="RdYlGn", vmin=0, vmax=1)
for i in range(4):
    for j in range(4):
        if not np.isnan(T[i, j]): ax.text(j, i, f"{T[i, j]:.2f}", ha="center", va="center", fontsize=9)
ax.set_xticks(range(4)); ax.set_xticklabels(["Tüb", "iid", "walk", "Sachs"]); ax.set_yticks(range(4)); ax.set_yticklabels(["Tüb", "iid", "walk", "Sachs"]); ax.set_xlabel("test collection"); ax.set_ylabel("training collection"); ax.set_title("Marginal-only classifier: transfer", fontsize=9)
plt.tight_layout(); plt.savefig(OUT / "figs" / "transfer.png", dpi=170); plt.close()
