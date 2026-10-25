import sys, json, pathlib, warnings
import numpy as np
from scipy import stats
warnings.filterwarnings("ignore")
sys.path.insert(0, ".."); sys.path.insert(0, "."); sys.path.insert(0, "experiments")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import exp49_explain as e49, exp53_more_chamber_pairs as e53
ROOT = pathlib.Path(".").resolve(); OUT = ROOT / "paper" / "resolution"
real = {}
for s, pid, x, y in e49.gather(): real.setdefault(s, []).append((pid, x, y))
for p in e53.extract(): real.setdefault(p["set"], []).append((p["id"], p["x"], p["y"]))
COLL = ["tuebingen", "chamber_iid", "chamber_walk", "sachs", "bernoulli", "changepoints", "malus"]
LAB = {"tuebingen": "T\\\"ubingen", "chamber_iid": "Chambers iid", "chamber_walk": "Chambers walks", "sachs": "Sachs", "bernoulli": "Bernoulli", "changepoints": "Changepoints", "malus": "Malus"}
rows = []; diag = {}
for s in COLL:
    L = real[s]; n = [len(x) for _, x, y in L]
    rho = [abs(stats.spearmanr(x, y)[0]) for _, x, y in L]
    h = []; kc = []; ke = []; fewc = []; ties = []
    for _, x, y in L:
        try: h.append(e49.het(x, y))
        except Exception: pass
        kc.append(stats.kurtosis((x - x.mean()) / (x.std() + 1e-12))); ke.append(stats.kurtosis((y - y.mean()) / (y.std() + 1e-12)))
        fewc.append(len(np.unique(x)) <= 10); ties.append(len(np.unique(x)) < 0.9 * len(x) or len(np.unique(y)) < 0.9 * len(y))
    diag[s] = dict(n=int(np.median(n)), rho=float(np.nanmedian(rho)), h=float(np.nanmedian(h)), kc=float(np.median(kc)), ke=float(np.median(ke)), few=float(np.mean(fewc)), ties=float(np.mean(ties)))
    d = diag[s]
    rows.append(f"{LAB[s]} & {len(L)} & {d['n']} & {d['rho']:.2f} & {d['h']:.2f} & {d['kc']:.1f} & {d['ke']:.1f} & {d['few']:.2f} & {d['ties']:.2f}\\\\")
(OUT / "tab_diag.tex").write_text("\n".join(rows)); json.dump(diag, open("results/collection_diagnostics.json", "w"), indent=1)
rng = np.random.default_rng(7)
fig, axs = plt.subplots(2, 4, figsize=(12, 5.4)); axs = axs.ravel()
for ax, s in zip(axs, COLL):
    pid, x, y = real[s][rng.integers(len(real[s]))]
    idx = rng.choice(len(x), min(len(x), 800), replace=False)
    ax.scatter(x[idx], y[idx], s=4, alpha=.5); ax.set_title(f"{LAB[s].replace(chr(92)+chr(34), 'u')}: {str(pid)[:28]}", fontsize=7); ax.tick_params(labelsize=6); ax.set_xlabel("cause", fontsize=7); ax.set_ylabel("effect", fontsize=7)
axs[-1].axis("off")
plt.tight_layout(); plt.savefig(OUT / "figs" / "examples.png", dpi=160)
print("\n".join(rows))
