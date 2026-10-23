"""Summaries for exp20 (Causal Chambers graphs) and exp21 (chamber pair directions)."""
import json, pathlib
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parent.parent


def wilson(k, n, z=1.96):
    if n == 0: return [0.0, 1.0]
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(max(0, c - h), 3), round(min(1, c + h), 3)]


d = json.load(open(ROOT / "results" / "exp20_causal_chamber.json"))
grid = d["config"]["grid"]
out = {}
for di, (ds, ex, ch) in enumerate(d["config"]["datasets"]):
    recs = [r for r in d["records"] if r["ds"] == di]
    row = {"vars": len(recs[0]["vars"]), "true_edges": len(recs[0]["truth"]), "reps": len(recs)}
    T = {tuple(sorted(e)) for e in recs[0]["truth"]}
    for m in recs[0]["methods"]:
        ever = 0; ok = 0; tp = fp = fn = 0; byn = {n: [0, 0, 0] for n in grid}
        for r in recs:
            est = r["methods"][m]["edges"]
            if est is None or any(e is None for e in est): continue
            ok += 1
            ever += any({tuple(sorted(p)) for p in e} - T for e in est)
            for gi, n in enumerate(grid):
                E = {tuple(sorted(p)) for p in est[gi]}
                byn[n][0] += len(T & E); byn[n][1] += len(E - T); byn[n][2] += len(T - E)
        ps = {n: (v[0] / (v[0] + v[1]) if v[0] + v[1] else None, v[0] / (v[0] + v[2])) for n, v in byn.items()}
        row[m] = {"runs": ok, "ever_false": ever, "ever_false_ci": wilson(ever, ok), "pr_by_n": ps,
                  "false_final": byn[grid[-1]][1] / max(ok, 1), "tp_final": byn[grid[-1]][0] / max(ok, 1)}
    # orientation of certified edges
    ori = [o for r in recs for o in r["orientation"]]
    row["orientation"] = {k: [sum(o["res"] is not None and o["res"][k] is not None for o in ori),
                              sum(o["res"] is not None and o["res"][k] == o["truth"] for o in ori)]
                          for k in ("agree", "gg", "ggh")}
    row["orientation"]["pairs"] = len(ori)
    out[f"{ds}/{ex}"] = row
p = ROOT / "results" / "exp21_chamber_pairs.json"
if p.exists():
    rs = json.load(open(p))["records"]; o21 = {}
    for di in sorted({r["ds"] for r in rs}):
        sub = [r for r in rs if r["ds"] == di]; row = {"pairs": len(sub)}
        for k in ("agree", "gg", "ggh"):
            dec = [r for r in sub if r["relfit"] and r["relfit"][k] is not None]
            row[k] = [len(dec), sum(r["relfit"][k] == r["truth"] for r in dec)]
        for k in ("forced_gg", "forced_ggh"):
            sc = [r for r in sub if r["relfit"]]
            row[k] = (float(np.mean([r["relfit"][k] == r["truth"] for r in sc])) if sc else None, len(sc))
        for b in sub[0]["base"]:
            row[b] = float(np.mean([r["base"][b] == r["truth"] for r in sub]))
        o21[di] = row
    out["exp21"] = o21
(ROOT / "results" / "exp20_summary.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1)[:6000])
