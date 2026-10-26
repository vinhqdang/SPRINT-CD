"""Additional checks requested on review: common-pair McNemar family, tie-break seed variance of the multiverse,
Holm correction of the Kendall and Mann-Whitney families, cluster-aware (sign-flip) tests, analytic power."""
import json, sys, itertools, warnings
import numpy as np
from scipy import stats
from scipy.stats import binomtest, norm
warnings.filterwarnings("ignore"); sys.path.insert(0, "."); sys.path.insert(0, "experiments")
R = "results/"
out = {}
import exp47_multiverse as e47
ids, w, n, S = e47.load(); M = list(S[ids[0]])
sc = np.array([[S[i][m] if S[i][m] is not None and S[i][m] == S[i][m] else np.nan for m in M] for i in ids])
ok = np.where(np.isnan(sc), np.nan, np.where(sc > 0, 1.0, np.where(sc < 0, 0.0, 0.5)))
FAM = ["IGCI", "RECI", "ANM-HSIC", "LiNGAM-pair", "LOCI", "ANM-GP", "Compression faithful", "TRA", "AVICI", "RESIT"]
fi = [M.index(m) for m in FAM]
# 1. common-pair set, ties counted as half in both cells of the discordance table
common = ~np.isnan(ok[:, fi]).any(1); out["common_pairs"] = int(common.sum())
pv = {}
for a, b in itertools.combinations(fi, 2):
    x, y = ok[common, a], ok[common, b]
    b01 = float(((x == 1) & (y == 0)).sum() + 0.5 * (((x == 1) & (y == .5)) | ((x == .5) & (y == 0))).sum()); b10 = float(((x == 0) & (y == 1)).sum() + 0.5 * (((x == 0) & (y == .5)) | ((x == .5) & (y == 1))).sum())
    k = int(round(b01)); N = int(round(b01 + b10)); pv[(M[a], M[b])] = binomtest(k, N, .5).pvalue if N else 1.0
order = sorted(pv, key=pv.get); run = 0; holm = {}
for k, key in enumerate(order):
    run = max(run, min(1, pv[key] * (len(order) - k))); holm[key] = run
out["common_family"] = {"n_comparisons": len(pv), "raw_below_0.05": int(sum(v < .05 for v in pv.values())), "holm_below_0.05": int(sum(v < .05 for v in holm.values())), "smallest_raw": min(pv.values()), "smallest_pair": list(min(pv, key=pv.get))}
# 2. tie-break seed variance of the multiverse
CAT = {}
def universes(seed):
    rng = np.random.default_rng(seed); tie = rng.random(sc.shape); res = []
    for wt, minrows, cov in itertools.product(("official", "none"), (0, 250), (1.0, .75, .5)):
        if minrows == 250 and cov == 0.5: continue
        sel = n >= minrows; wv = w if wt == "official" else np.ones_like(w); accs = []
        for j in range(len(M)):
            idx = np.where(sel & ~np.isnan(ok[:, j]))[0]
            if cov < 1.0:
                keep = max(1, int(round(cov * len(idx)))); o = np.lexsort((tie[idx, j], -np.abs(sc[idx, j]))); idx = idx[o[:keep]]
            accs.append((ok[idx, j] * wv[idx]).sum() / wv[idx].sum())
        res.append(accs)
    return np.array(res)
reps = [universes(s) for s in range(100, 150)]
A = np.array(reps)  # seeds x universes x rules
rng_ = A.max(1) - A.min(1)
first = np.array([[(a[u] >= a[u].max() - 1e-12).astype(int) for u in range(a.shape[0])] for a in A]).sum(1)
cf = M.index("Compression faithful"); jr = M.index("RECI")
out["tiebreak"] = {"n_universes": int(A.shape[1]), "range_mean": dict(zip(M, rng_.mean(0).round(3).tolist())), "range_sd": dict(zip(M, rng_.std(0).round(3).tolist())), "first_count_mean": dict(zip(M, first.mean(0).round(2).tolist())), "first_count_sd": dict(zip(M, first.std(0).round(2).tolist())),
                    "faithful_above_RECI_mean": float((A[:, :, cf] > A[:, :, jr]).sum(1).mean())}
# first counts without the faithful variant (other variants of compression remain)
fi2 = [j for j in range(len(M)) if j != cf]
first2 = np.array([[(a[u][fi2] >= a[u][fi2].max() - 1e-12).astype(int) for u in range(a.shape[0])] for a in A]).sum(1)
out["tiebreak"]["first_count_without_faithful_mean"] = {M[j]: float(first2.mean(0)[k]) for k, j in enumerate(fi2)}
# 3. Holm for Kendall and Mann-Whitney
asset = json.load(open(R + "resolution_assets.json")); P = np.array(asset["kendall_perm_p"], float); K = np.array(asset["kendall"], float)
iu = [(i, j) for i in range(8) for j in range(i + 1, 8) if not np.isnan(P[i, j])]
ps = np.array([P[i, j] for i, j in iu]); o = np.argsort(ps); run = 0; adj = np.empty(len(ps))
for k, idx in enumerate(o): run = max(run, min(1, ps[idx] * (len(ps) - k))); adj[idx] = run
out["kendall_holm"] = {"n": len(ps), "raw_below_0.05": int((ps < .05).sum()), "holm_below_0.05": int((adj < .05).sum()), "smallest_adjusted": float(adj.min())}
mw = json.load(open(R + "exp49_explain.json"))["h2"]["methods"]; names = list(mw); ps = np.array([mw[m]["p_h"] for m in names]); o = np.argsort(ps); run = 0; adj = np.empty(len(ps))
for k, idx in enumerate(o): run = max(run, min(1, ps[idx] * (len(ps) - k))); adj[idx] = run
out["mannwhitney_holm"] = {m: [float(p), float(a)] for m, p, a in zip(names, ps, adj)}
# 4. cluster-aware sign-flip tests
ci = json.load(open(R + "cluster_bootstrap.json")); corr = asset["corr"]; ids_ = asset["ids"]
def cluster(s, i):
    if s in ("tuebingen", "sachs"): return i
    if s == "chamber_iid": return i.rsplit("/", 1)[0]
    if s == "chamber_walk": return i.split("/")[0]
    if s == "walks_all": return "/".join(i.split("/")[:2])
    return i.split("/")[1]
def flip_test(s, r1, r2, reps=20000, seed=0):
    a = np.array([np.nan if v is None else v for v in corr[r1][s]], float); b = np.array([np.nan if v is None else v for v in corr[r2][s]], float)
    cl = np.array([cluster(s, i) for i in ids_[s]]); u = np.unique(cl)
    d = np.array([np.nanmean(a[cl == c] - b[cl == c]) for c in u]); d = d[~np.isnan(d)]
    obs = d.mean(); rng = np.random.default_rng(seed); null = np.array([(d * rng.choice([-1, 1], len(d))).mean() for _ in range(reps)])
    return {"clusters": int(len(d)), "mean_diff": float(obs), "p": float((np.abs(null) >= abs(obs) - 1e-12).mean())}
out["cluster_flip"] = {}
for s in ("chamber_iid", "chamber_walk", "walks_all"):
    for r1, r2 in (("IGCI", "RECI"), ("RECI", "ANM-HSIC"), ("IGCI", "ANM-HSIC"), ("RECI", "AVICI"), ("RECI", "LiNGAM-pair")):
        out["cluster_flip"][f"{s}:{r1}-{r2}"] = flip_test(s, r1, r2)
# 5. analytic detectable difference
d = 0.42
for nm, alpha in (("unadjusted", .05), ("bonferroni78", .05 / 78)):
    out[f"analytic_detectable_{nm}"] = float((norm.ppf(1 - alpha / 2) + norm.ppf(.8)) * np.sqrt(d / 102))
out["required_n"] = {f"delta{dl}": float(((norm.ppf(.975) + norm.ppf(.8)) ** 2) * d / dl ** 2) for dl in (0.02, 0.05, 0.10)}
json.dump(out, open(R + "final_checks.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k not in ("tiebreak",)}, indent=1)[:3500])
print({k: v for k, v in out["tiebreak"].items() if k in ("n_universes", "faithful_above_RECI_mean")}); print("range sd", out["tiebreak"]["range_sd"]); print("first mean", out["tiebreak"]["first_count_mean"])
