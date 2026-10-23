import json, pathlib
import numpy as np
from scipy import stats
ROOT = pathlib.Path(__file__).resolve().parent.parent
d = json.load(open(ROOT / "results" / "exp30_btad_chambers.json"))["records"]
groups = {"wind walks": lambda s: s[0] == "wt_walks_v1", "pressure control": lambda s: s[0] == "wt_pressure_control_v1",
          "light walks": lambda s: s[0] == "lt_walks_v1", "all": lambda s: True}


def prf(E, allowed):
    tp = len(E & allowed); fp = len(E - allowed)
    return tp, fp


out = {}
per = {}
for g, f in groups.items():
    rs = [r for r in d if f(r["spec"])]
    row = {"recordings": len(rs)}
    for m in rs[0]["m"]:
        agg = {}
        for name in ("strict", "adjusted_block", "adjusted_iid"):
            P = []; R = []; F = []; fps = []
            for r in rs:
                est = r["m"][m]["edges"]
                if est is None: continue
                T = {tuple(sorted(e)) for e in r["truth"]}
                V = {(a, b) for a, b, _ in (r["V_block"] if name == "adjusted_block" else r["V_iid"])} if name != "strict" else set()
                E = {tuple(sorted(p)) for p in est}
                tp, fp = prf(E, T | V); tpT = len(E & T)
                p = tp / max(tp + fp, 1); rc = tpT / max(len(T), 1)
                P.append(p); R.append(rc); F.append(2 * p * rc / (p + rc) if p + rc else 0); fps.append(fp)
            agg[name] = {"P": float(np.mean(P)), "R": float(np.mean(R)), "F1": float(np.mean(F)), "fp_per_rec": float(np.mean(fps)), "n": len(P)}
            per[(g, m, name)] = F
        row[m] = agg
    out[g] = row
# paired comparison on wind walks: BTAD vs each, strict F1
tests = {}
for m in d[0]["m"]:
    if m == "BTAD": continue
    a = np.array(per[("wind walks", "BTAD", "strict")]); b = np.array(per[("wind walks", m, "strict")])
    tests[m] = float(stats.wilcoxon(a, b).pvalue) if np.any(a != b) else 1.0
out["wilcoxon_p_strictF1_windwalks_BTAD_vs"] = tests
(ROOT / "results" / "exp30_summary.json").write_text(json.dumps(out, indent=1))
for g, row in out.items():
    if g.startswith("wilcoxon"): print(g, {k: round(v, 4) for k, v in row.items()}); continue
    print(g, row["recordings"])
    for m, v in row.items():
        if isinstance(v, dict):
            s, ab = v["strict"], v["adjusted_block"]
            print(f"   {m:27s} strict P {s['P']:.2f} R {s['R']:.2f} F1 {s['F1']:.2f} fp {s['fp_per_rec']:.1f} | adj(block) P {ab['P']:.2f} F1 {ab['F1']:.2f} fp {ab['fp_per_rec']:.1f}")
