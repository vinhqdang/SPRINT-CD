import json, pathlib
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parent.parent
d = json.load(open(ROOT / "results" / "exp29_btad_sim.json"))["records"]
out = {}
for ex in (0, 1):
    for phi in (0.0, 0.9, 0.99):
        rs = [r for r in d if r["extras"] == ex and r["phi"] == phi]
        row = {"graphs": len(rs)}
        for m in rs[0]["m"]:
            v = np.array([r["m"][m] for r in rs if r["m"][m] is not None])
            row[m] = {"P": float(v[:, 0].mean()), "R": float(v[:, 1].mean()), "F1": float(v[:, 2].mean()),
                      "F1_se": float(v[:, 2].std(ddof=1) / np.sqrt(len(v))), "n": len(v)}
        out[f"extras={ex},phi={phi}"] = row
(ROOT / "results" / "exp29_summary.json").write_text(json.dumps(out, indent=1))
for k, row in out.items():
    print(k, row["graphs"])
    for m, v in row.items():
        if isinstance(v, dict):
            print(f"   {m:15s} P {v['P']:.2f} R {v['R']:.2f} F1 {v['F1']:.2f} (se {v['F1_se']:.2f}, n {v['n']})")
