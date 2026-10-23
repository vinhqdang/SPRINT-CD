"""Summaries for exp22 (nonlinear simulation) and exp23 (chamber, nonparametric)."""
import json, pathlib, itertools
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parent.parent


def wilson(k, n, z=1.96):
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(max(0, c - h), 3), round(min(1, c + h), 3)]


out = {}
d = json.load(open(ROOT / "results" / "exp22_nonparametric_adjacency.json"))
s22 = {}
for mech in ("linear", "nonlinear"):
    for noise in ("laplace", "gaussian"):
        rs = [r for r in d["records"] if r["mech"] == mech and r["noise"] == noise]
        row = {"dags": len(rs)}
        for m in ("linear", "nonparam"):
            ef = tp = fp = fn = 0
            for r in rs:
                T = {tuple(e) for e in r["truth"]}
                ef += any({tuple(p) for p in g} - T for g in r[m])
                E = {tuple(p) for p in r[m][-1]}
                tp += len(E & T); fp += len(E - T); fn += len(T - E)
            row[m] = {"ever_false": ef, "ci": wilson(ef, len(rs)),
                      "precision": tp / (tp + fp) if tp + fp else None, "recall": tp / (tp + fn)}
        s22[f"{mech}/{noise}"] = row
out["exp22"] = s22

d23 = json.load(open(ROOT / "results" / "exp23_chamber_nonparam.json"))
e20 = json.load(open(ROOT / "results" / "exp20_causal_chamber.json"))
s23 = {}
for di in range(len(e20["config"]["datasets"])):
    V = {(a, b) for a, b, _ in d23["reference_set"][di]}
    recs20 = [r for r in e20["records"] if r["ds"] == di]
    T = {tuple(sorted(e)) for e in recs20[0]["truth"]}
    row = {"graph_edges": len(T), "reference_pairs": len(V)}
    methods = {m: [r["methods"][m]["edges"] for r in recs20] for m in recs20[0]["methods"]}
    methods["CERT-CD nonparametric"] = [r["nonparam"] for r in d23["records"] if r["ds"] == di]
    for m, runs in methods.items():
        res = {}
        for name, allowed in (("strict", T), ("adjusted", T | V)):
            ef = ok = 0; tp = fp = 0
            for est in runs:
                if est is None or any(e is None for e in est): continue
                ok += 1
                ef += any({tuple(sorted(p)) for p in g} - allowed for g in est)
                E = {tuple(sorted(p)) for p in est[-1]}
                tp += len(E & allowed); fp += len(E - allowed)
            res[name] = {"ever_false": ef, "runs": ok, "ci": wilson(ef, ok),
                         "false_per_run": fp / ok, "certified_per_run": (tp + fp) / ok,
                         "precision": tp / (tp + fp) if tp + fp else None}
        rec = np.mean([len({tuple(sorted(p)) for p in est[-1]} & T) / len(T) for est in runs if est and all(e is not None for e in est)])
        res["recall_of_graph_edges"] = float(rec)
        row[m] = res
    s23[f"dataset{di}"] = row
out["exp23"] = s23
(ROOT / "results" / "exp22_23_summary.json").write_text(json.dumps(out, indent=1))
for k, v in s22.items():
    print(k, v["dags"], {m: (v[m]["ever_false"], round(v[m]["precision"] or 0, 3), round(v[m]["recall"], 3)) for m in ("linear", "nonparam")})
for k, v in s23.items():
    print(k, "graph edges", v["graph_edges"], "reference pairs", v["reference_pairs"])
    for m, r in v.items():
        if isinstance(r, dict):
            print(f"  {m:22s} strict ever-false {r['strict']['ever_false']}/{r['strict']['runs']} fp/run {r['strict']['false_per_run']:.1f} | adjusted ever-false {r['adjusted']['ever_false']} fp/run {r['adjusted']['false_per_run']:.1f} certified/run {r['adjusted']['certified_per_run']:.1f} | recall {r['recall_of_graph_edges']:.2f}")
