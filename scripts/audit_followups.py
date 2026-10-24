"""Numbers cited in the benchmark paper that need their own script (audit follow-ups)."""
import json, pathlib, sys, warnings
import numpy as np
from scipy import stats
warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
L = lambda f: json.load(open(ROOT / "results" / f))
out = {}

# 1. LRG within-recording AUC (strict labels), wind walks
from sklearn.metrics import roc_auc_score
base = L("exp30_btad_chambers.json")["records"]; lrg = {tuple(r["spec"][:2]): r for r in L("exp32_lrg_chambers.json")["records"]}
aucs = []
for r in base:
    if r["spec"][0] != "wt_walks_v1": continue
    T = {tuple(sorted(e)) for e in r["truth"]}; l = lrg[tuple(r["spec"][:2])]
    y = np.array([1 if tuple(p) in T else 0 for p in l["pairs"]]); aucs.append(roc_auc_score(y, np.array(l["prob"])))
out["lrg_auc_wind_walks"] = {"mean": float(np.mean(aucs)), "n": len(aucs), "min": float(np.min(aucs)), "max": float(np.max(aucs))}

# 2. Cross-correlation of load_in -> rpm_in at lags -6..6, wind walks (first 4000 rows)
import exp27_chamber_fresh as e27
cc = []
for spec in e27.specs():
    if spec[0] != "wt_walks_v1": continue
    X, var, _ = e27.load(spec); X = X[:4000]
    a, b = X[:, var.index("load_in")], X[:, var.index("rpm_in")]; a = (a - a.mean()) / a.std(); b = (b - b.mean()) / b.std()
    cc.append([np.mean(a[max(0, -l):len(a) - max(0, l)] * b[max(0, l):len(b) - max(0, -l)]) for l in range(-6, 7)])
cc = np.array(cc)
out["ccf_load_in_rpm_in_wind_walks"] = {"lags": list(range(-6, 7)), "mean": cc.mean(0).tolist(), "range_over_lags_mean": float(cc.mean(0).max() - cc.mean(0).min())}

# 3. interventional shift test (KS) on lt_interventions_standard_v1
import re
import causalchamber.ground_truth as gt
from causalchamber.datasets import Dataset
d = Dataset("lt_interventions_standard_v1", root=str(ROOT / "data" / "causalchamber"), download=False)
G = gt.graph("lt", "standard"); nodes = list(G.columns); A = G.values.astype(int); R = A.copy()
for _ in range(len(nodes)): R = ((R + R @ A) > 0).astype(int)
ref = d.get_experiment("uniform_reference").as_pandas_dataframe(); tp = fp = fn = used = 0
for e in sorted(d.available_experiments()):
    m = re.match(r"uniform_(.+?)_(weak|mid|strong)$", e)
    if not m or m.group(1) not in nodes: continue
    tgt = m.group(1); df = d.get_experiment(e).as_pandas_dataframe()
    var = [c for c in nodes if c in df.columns and c in ref.columns and ref[c].nunique() >= 20 and c != tgt]
    shifted = {c for c in var if stats.ks_2samp(df[c], ref[c]).pvalue < 1e-6}
    desc = {c for c in var if R[nodes.index(tgt), nodes.index(c)] > 0}
    tp += len(shifted & desc); fp += len(shifted - desc); fn += len(desc - shifted); used += 1
out["interventional_shift_test"] = {"experiments": used, "tp": tp, "fp": fp, "fn": fn}

# 4. Paired tests on the 15 wind walks, BTAD vs every method, Holm-corrected (strict F1)
e36 = L("exp36_graph_benchmark_extra.json"); e39 = L("exp39_graphs_modern.json"); e32 = L("exp32_lrg_chambers.json")["records"]


def f1(E, T):
    E = {tuple(sorted(p)) for p in E}; tp = len(E & T); fp = len(E - T); P = tp / max(tp + fp, 1); Rr = tp / len(T)
    return 2 * P * Rr / (P + Rr) if P + Rr else 0.0


per = {}
for r in base:
    if r["spec"][0] != "wt_walks_v1": continue
    key = tuple(r["spec"][:2]); T = {tuple(sorted(e)) for e in r["truth"]}
    for m, v in r["m"].items(): per.setdefault(m, []).append(f1(v["edges"], T))
    per.setdefault("LRG", []).append(f1(lrg[key]["edges"], T))
    for x in e36["notears_chambers"]:
        if x["kind"] == "walk" and tuple(x["spec"][:2]) == key: per.setdefault("NOTEARS", []).append(f1(x["edges"], T))
    for x in e39["chambers"]:
        if x["kind"] == "walk" and tuple(x["spec"][:2]) == key:
            for m in ("DAGMA", "PCMCI+"): per.setdefault(m, []).append(f1(x["edges"][m], T))
ps = {}
for m, v in per.items():
    if m == "BTAD": continue
    a, b = np.array(per["BTAD"]), np.array(v); ps[m] = {"p": float(stats.wilcoxon(a, b).pvalue), "BTAD_minus_other": float((a - b).mean())}
order = sorted(ps, key=lambda k: ps[k]["p"]); k = len(order)
for i, m in enumerate(order): ps[m]["holm_p"] = min(1.0, max(ps[o]["p"] * (k - j) for j, o in enumerate(order[:i + 1])))
out["wilcoxon_BTAD_vs_others_wind_walks"] = ps

# 5. McNemar (exact) RECI vs others on the pair sets, and ANM ties
e35 = L("exp35_pdm_eval.json")["records"]; e38 = L("exp38_pairs_modern.json")["records"]
m38 = {(r["set"], r["id"]): r for r in e38}


def correct(r, m):
    if m == "PDM": return (r["pdm"] > 0.5) == (r["truth"] == "x->y")
    if m in ("LOCI", "ANM"):
        s = m38[(r["set"], r["id"])]["m"][m]
        if s is None: return False
        return (s > 0) == (r["truth"] == "x->y")
    return r["scores"][m][0] == r["truth"]


mc = {}
for s in ("tuebingen", "chamber_iid", "chamber_walk"):
    rs = [r for r in e35 if r["set"] == s]; mc[s] = {}
    for m in ("PDM", "IGCI", "ANM-HSIC", "LiNGAM-pair", "LOCI", "ANM"):
        a = np.array([correct(r, "RECI") for r in rs]); b = np.array([correct(r, m) for r in rs])
        n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
        mc[s][m] = {"RECI_only": n01, "other_only": n10, "p": float(stats.binomtest(n01, n01 + n10, 0.5).pvalue) if n01 + n10 else 1.0}
out["mcnemar_RECI_vs"] = mc
ties = {}
for s in ("tuebingen", "chamber_iid", "chamber_walk", "sachs"):
    rs = [r for r in e38 if r["set"] == s]
    t = [r for r in rs if r["m"]["ANM"] == 0.0]
    ok = [((r["m"]["ANM"] > 0) == (r["truth"] == "x->y")) if r["m"]["ANM"] != 0.0 else 0.5 for r in rs]
    ties[s] = {"ties": len(t), "acc_ties_half": float(np.mean(ok)), "n": len(rs)}
out["anm_gp_ties"] = ties

# 6. V ratios per group and composition of the chamber walk pair set
div = L("serial_diagnostic.json"); ess = {tuple(r["spec"][:2]): r for r in div if r["ess"] is not None}
def grp(f):
    rows = [r for r in div if r["ess"] is not None and f(r)]
    return {"n": len(rows), "V_iid": float(np.mean([r["V_iid"] for r in rows])), "V_block": float(np.mean([r["V_block"] for r in rows]))}
out["V_ratio"] = {"wind_walks_15": grp(lambda r: r["spec"][0] == "wt_walks_v1"), "ess_lt20_17": grp(lambda r: r["ess"] < 20), "all_19": grp(lambda r: True),
                  "hatch_0": grp(lambda r: r["spec"][1] == "hatch_0"), "color_mix": grp(lambda r: r["spec"][1] == "color_mix")}
comp = {}
for r in e35:
    if r["set"] == "chamber_walk": comp[r["id"].split("/")[0]] = comp.get(r["id"].split("/")[0], 0) + 1
out["chamber_walk_pair_composition"] = comp

# 7. bootstrap over recordings (17 low-ESS) for strict F1 of the best methods
rng = np.random.default_rng(0); bs = {}
G = L("benchmark_graph_summary.json")
(ROOT / "results" / "audit_followups.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k not in ("wilcoxon_BTAD_vs_others_wind_walks", "mcnemar_RECI_vs")}, indent=1)[:2500])
print({m: (round(v["p"], 4), round(v["holm_p"], 4), round(v["BTAD_minus_other"], 3)) for m, v in out["wilcoxon_BTAD_vs_others_wind_walks"].items()})
print(json.dumps(out["mcnemar_RECI_vs"]))
