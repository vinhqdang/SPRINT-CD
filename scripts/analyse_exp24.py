"""Score exp24 (k = 3) against the chamber graph and the exp23 reference set."""
import json, pathlib
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parent.parent
d24 = json.load(open(ROOT / "results" / "exp24_chamber_k3.json"))
d23 = json.load(open(ROOT / "results" / "exp23_chamber_nonparam.json"))
e20 = json.load(open(ROOT / "results" / "exp20_causal_chamber.json"))


def wilson(k, n, z=1.96):
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(max(0, c - h), 3), round(min(1, c + h), 3)]


out = {}
for di in range(len(e20["config"]["datasets"])):
    V = {(a, b) for a, b, _ in d23["reference_set"][di]}
    T = {tuple(sorted(e)) for e in [r for r in e20["records"] if r["ds"] == di][0]["truth"]}
    row = {"graph_edges": len(T), "reference_pairs": len(V)}
    for arm in ("nonparam_k3", "linear_k3"):
        runs = [r[arm] for r in d24["records"] if r["ds"] == di]
        res = {}
        for name, allowed in (("strict", T), ("adjusted", T | V)):
            ef = fp = tp = 0
            for est in runs:
                ef += any({tuple(sorted(p)) for p in g} - allowed for g in est)
                E = {tuple(sorted(p)) for p in est[-1]}
                tp += len(E & allowed); fp += len(E - allowed)
            res[name] = {"ever_false": ef, "runs": len(runs), "ci": wilson(ef, len(runs)),
                         "false_per_run": fp / len(runs), "certified_per_run": (tp + fp) / len(runs)}
        res["recall_of_graph_edges"] = float(np.mean([len({tuple(sorted(p)) for p in est[-1]} & T) / len(T) for est in runs]))
        res["reference_pairs_certified_per_run"] = float(np.mean([len({tuple(sorted(p)) for p in est[-1]} & V) for est in runs]))
        row[arm] = res
    out[f"dataset{di}"] = row
(ROOT / "results" / "exp24_summary.json").write_text(json.dumps(out, indent=1))
for k, v in out.items():
    print(k, v["graph_edges"], "graph edges,", v["reference_pairs"], "reference pairs")
    for a in ("nonparam_k3", "linear_k3"):
        r = v[a]
        print(f"  {a:12s} strict {r['strict']['ever_false']}/{r['strict']['runs']} fp/run {r['strict']['false_per_run']:.1f} | adjusted {r['adjusted']['ever_false']}/{r['adjusted']['runs']} {r['adjusted']['ci']} fp/run {r['adjusted']['false_per_run']:.1f} cert/run {r['adjusted']['certified_per_run']:.1f} | recall {r['recall_of_graph_edges']:.2f} | V certified/run {r['reference_pairs_certified_per_run']:.1f}")
