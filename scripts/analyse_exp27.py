"""Summaries for exp27 (untouched Causal Chamber recordings)."""
import json, pathlib
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parent.parent
d = json.load(open(ROOT / "results" / "exp27_chamber_fresh.json"))


def wilson(k, n, z=1.96):
    if n == 0: return [0, 1]
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(max(0, c - h), 3), round(min(1, c + h), 3)]


groups = {"wind tunnel, random walks": lambda s: s[0] == "wt_walks_v1",
          "wind tunnel, pressure control": lambda s: s[0] == "wt_pressure_control_v1",
          "light tunnel, walks": lambda s: s[0] == "lt_walks_v1"}
out = {}
for gname, f in groups.items():
    recs = [r for r in d["records"] if f(r["spec"])]
    row = {"recordings": len(recs), "mean_vars": float(np.mean([len(r["vars"]) for r in recs])),
           "mean_graph_edges": float(np.mean([len(r["truth"]) for r in recs])),
           "mean_reference_pairs": float(np.mean([len(r["reference"]) for r in recs]))}
    for m in recs[0]["methods"]:
        res = {}
        for name in ("strict", "adjusted"):
            ever = fp = cert = ok = 0; rec = []
            for r in recs:
                est = r["methods"][m]["edges"]
                if est is None or any(e is None for e in est):
                    continue
                T = {tuple(sorted(e)) for e in r["truth"]}
                V = {(a, b) for a, b, _ in r["reference"]}
                allowed = T if name == "strict" else T | V
                ok += 1
                ever += any({tuple(sorted(p)) for p in g} - allowed for g in est)
                E = {tuple(sorted(p)) for p in est[-1]}
                fp += len(E - allowed); cert += len(E)
                rec.append(len(E & T) / max(len(T), 1))
            res[name] = {"ever_false": ever, "runs": ok, "ci": wilson(ever, ok),
                         "false_per_run": fp / max(ok, 1), "certified_per_run": cert / max(ok, 1)}
            res["recall"] = float(np.mean(rec)) if rec else None
        row[m] = res
    out[gname] = row
(ROOT / "results" / "exp27_summary.json").write_text(json.dumps(out, indent=1))
for g, row in out.items():
    print(g, {k: round(v, 1) for k, v in row.items() if not isinstance(v, dict)})
    for m, r in row.items():
        if isinstance(r, dict):
            print(f"  {m:27s} strict ever {r['strict']['ever_false']}/{r['strict']['runs']} fp/run {r['strict']['false_per_run']:.1f} | adjusted ever {r['adjusted']['ever_false']}/{r['adjusted']['runs']} fp/run {r['adjusted']['false_per_run']:.1f} cert/run {r['adjusted']['certified_per_run']:.1f} | recall {r['recall']:.2f}")
