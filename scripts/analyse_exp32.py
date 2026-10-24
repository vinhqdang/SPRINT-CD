import json, pathlib
import numpy as np
from scipy import stats
ROOT = pathlib.Path(__file__).resolve().parent.parent
base = json.load(open(ROOT / "results" / "exp30_btad_chambers.json"))["records"]
lrg = {tuple(r["spec"][:2]): r for r in json.load(open(ROOT / "results" / "exp32_lrg_chambers.json"))["records"]}
groups = {"wind walks": lambda s: s[0] == "wt_walks_v1", "pressure control": lambda s: s[0] == "wt_pressure_control_v1",
          "light walks": lambda s: s[0] == "lt_walks_v1", "all": lambda s: True}


def score(r, edges, kind):
    T = {tuple(sorted(e)) for e in r["truth"]}
    V = {(a, b) for a, b, _ in r["V_block"]} if kind == "adjusted" else set()
    E = {tuple(sorted(p)) for p in edges}
    tp = len(E & (T | V)); fp = len(E - (T | V)); tpT = len(E & T)
    P = tp / max(tp + fp, 1); R = tpT / max(len(T), 1)
    return P, R, (2 * P * R / (P + R) if P + R else 0.0), fp


out = {}; per = {}
for g, f in groups.items():
    rs = [r for r in base if f(r["spec"])]
    row = {"recordings": len(rs)}
    methods = list(rs[0]["m"]) + ["LRG"]
    for m in methods:
        row[m] = {}
        for kind in ("strict", "adjusted"):
            v = []
            for r in rs:
                est = lrg[tuple(r["spec"][:2])]["edges"] if m == "LRG" else r["m"][m]["edges"]
                v.append(score(r, est, kind))
            v = np.array(v)
            row[m][kind] = {"P": float(v[:, 0].mean()), "R": float(v[:, 1].mean()), "F1": float(v[:, 2].mean()), "fp": float(v[:, 3].mean())}
            per[(g, m, kind)] = v[:, 2]
    out[g] = row
tests = {}
for m in base[0]["m"]:
    a, b = per[("wind walks", "LRG", "strict")], per[("wind walks", m, "strict")]
    tests[m] = float(stats.wilcoxon(a, b).pvalue) if np.any(a != b) else 1.0
out["wilcoxon_strictF1_windwalks_LRG_vs"] = tests
(ROOT / "results" / "exp32_summary.json").write_text(json.dumps(out, indent=1))
for g, row in out.items():
    if g.startswith("wilcoxon"): print(g, {k: round(v, 4) for k, v in row.items()}); continue
    print(g, row["recordings"])
    for m, v in row.items():
        if isinstance(v, dict):
            s, a = v["strict"], v["adjusted"]
            print(f"   {m:27s} strict P {s['P']:.2f} R {s['R']:.2f} F1 {s['F1']:.2f} fp {s['fp']:.1f} | adj P {a['P']:.2f} F1 {a['F1']:.2f} fp {a['fp']:.1f}")
