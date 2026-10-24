"""Aggregate all graph-benchmark results into one summary (skeleton precision / recall / F1, strict)."""
import json, pathlib
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parent.parent
L = lambda f: json.load(open(ROOT / "results" / f))
e20, e23, e33, e37, e30, e32, e36, diag = (L("exp20_causal_chamber.json"), L("exp23_chamber_nonparam.json"),
    L("exp33_lrg_iid_chambers.json"), L("exp37_pcgcm_iid_chambers.json"), L("exp30_btad_chambers.json")["records"],
    L("exp32_lrg_chambers.json")["records"], L("exp36_graph_benchmark_extra.json"), L("serial_diagnostic.json"))


def sc(E, T):
    E = {tuple(sorted(p)) for p in E}; tp = len(E & T); fp = len(E - T)
    P = tp / max(tp + fp, 1); R = tp / len(T); return [P, R, 2 * P * R / (P + R) if P + R else 0.0, fp, len(E)]


def agg(v): 
    v = np.array(v); return {"P": float(v[:, 0].mean()), "R": float(v[:, 1].mean()), "F1": float(v[:, 2].mean()), "fp": float(v[:, 3].mean()), "n": len(v)}


M = ["PC a=0.05", "PC a=0.01", "GES-BIC", "DirectLiNGAM", "NOTEARS", "PC-GCM a=0.05", "PC-GCM a=0.01", "BTAD", "LRG", "CERT-CD nonparametric"]
out = {}
# Sachs
s = e36["sachs"]; T = {tuple(sorted(e)) for e in s["truth"]}
out["Sachs"] = {m: agg([sc(s["edges"][k], T)]) for m in M for k in [m if m != "CERT-CD nonparametric" else "CERT-CD nonparametric k=3"] if s["edges"].get(k) is not None}
# iid chambers (2 recordings, 10 subsamples each; pooled over both)
iid = {m: [] for m in M}
for di in (0, 1):
    recs = [r for r in e20["records"] if r["ds"] == di]; T = {tuple(sorted(e)) for e in recs[0]["truth"]}
    for r in recs:
        for m, key in (("PC a=0.05", "PC a=0.05"), ("PC a=0.01", "PC a=0.01"), ("GES-BIC", "GES-BIC"), ("DirectLiNGAM", "DirectLiNGAM")):
            est = r["methods"][key]["edges"]
            if est and est[-1] is not None: iid[m].append(sc(est[-1], T))
    for r in e23["records"]:
        if r["ds"] == di: iid["CERT-CD nonparametric"].append(sc(r["nonparam"][-1], T))
    for r in e33["records"]:
        if r["ds"] == di: iid["LRG"].append(sc(r["LRG"], T)); iid["BTAD"].append(sc(r["BTAD"], T))
    for r in e37["records"]:
        if r["ds"] == di: iid["PC-GCM a=0.05"].append(sc(r["a05"], T)); iid["PC-GCM a=0.01"].append(sc(r["a01"], T))
    for r in e36["notears_chambers"]:
        if r["kind"] == "iid" and r["ds"] == di and r["edges"] is not None: iid["NOTEARS"].append(sc(r["edges"], T))
out["Chambers iid (2 recordings x 10 subsamples)"] = {m: agg(v) for m, v in iid.items() if v}
# fresh recordings, stratified by effective sample size (ESS from serial diagnostic)
ess = {tuple(r["spec"][:2]): r["ess"] for r in diag if r["ess"] is not None}
groups = {"Chambers, 2 recordings with ESS > 1000": lambda s: ess[tuple(s[:2])] > 1000,
          "Chambers, 17 recordings with ESS < 20": lambda s: ess[tuple(s[:2])] < 20}
for g, f in groups.items():
    acc = {m: [] for m in M}
    for r in e30:
        if not f(r["spec"]): continue
        T = {tuple(sorted(e)) for e in r["truth"]}
        for m, key in (("PC a=0.05", "PC a=0.05"), ("PC a=0.01", "PC a=0.01"), ("GES-BIC", "GES-BIC"), ("DirectLiNGAM", "DirectLiNGAM"),
                       ("PC-GCM a=0.05", "PC-GCM a=0.05"), ("PC-GCM a=0.01", "PC-GCM a=0.01"), ("BTAD", "BTAD"), ("CERT-CD nonparametric", "CERT-CD nonparametric k=3")):
            est = r["m"][key]["edges"]
            if est is not None: acc[m].append(sc(est, T))
        l = [x for x in e32 if tuple(x["spec"][:2]) == tuple(r["spec"][:2])][0]; acc["LRG"].append(sc(l["edges"], T))
        n_ = [x for x in e36["notears_chambers"] if x["kind"] == "walk" and tuple(x["spec"][:2]) == tuple(r["spec"][:2])][0]
        if n_["edges"] is not None: acc["NOTEARS"].append(sc(n_["edges"], T))
    out[g] = {m: agg(v) for m, v in acc.items() if v}
(ROOT / "results" / "benchmark_graph_summary.json").write_text(json.dumps(out, indent=1))
for g, row in out.items():
    print(g)
    for m, v in row.items(): print(f"   {m:24s} P {v['P']:.2f} R {v['R']:.2f} F1 {v['F1']:.2f} fp {v['fp']:.1f} (n={v['n']})")
