import json, pathlib
import numpy as np
from scipy import stats
ROOT = pathlib.Path(__file__).resolve().parent.parent
rec = json.load(open(ROOT / "results" / "exp35_pdm_eval.json"))["records"]
rules = ["IGCI", "RECI", "ANM-HSIC", "LiNGAM-pair"]


def cp(k, n, a=0.05):
    if n == 0: return [0, 1]
    lo = stats.beta.ppf(a / 2, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(1 - a / 2, k + 1, n - k) if k < n else 1.0
    return [round(float(lo), 3), round(float(hi), 3)]


out = {}
for s in ("tuebingen", "chamber_iid", "chamber_walk", "heldout"):
    rs = [r for r in rec if r["set"] == s]
    w = np.array([r["weight"] for r in rs]); n = len(rs)
    row = {"pairs": n}
    ok = np.array([(r["pdm"] > 0.5) == (r["truth"] == "x->y") for r in rs])
    row["PDM"] = {"acc": float(ok.mean()), "ci": cp(int(ok.sum()), n), "w_acc": float((w * ok).sum() / w.sum())}
    for rule in rules:
        okr = np.array([r["scores"][rule][0] == r["truth"] for r in rs])
        row[rule] = {"acc": float(okr.mean()), "ci": cp(int(okr.sum()), n), "w_acc": float((w * okr).sum() / w.sum())}
    # selective: top fraction by confidence
    conf = np.array([abs(r["pdm"] - 0.5) for r in rs])
    sel = {}
    for cov in (0.25, 0.5, 0.75):
        k = max(1, int(cov * n)); idx = np.argsort(-conf)[:k]
        d = {"PDM": f"{int(ok[idx].sum())}/{k}"}
        for rule in rules:
            c = np.array([r["scores"][rule][1] for r in rs]); okr = np.array([r["scores"][rule][0] == r["truth"] for r in rs])
            j = np.argsort(-c)[:k]; d[rule] = f"{int(okr[j].sum())}/{k}"
        sel[f"top{int(cov*100)}"] = d
    row["selective"] = sel
    out[s] = row
(ROOT / "results" / "exp35_summary.json").write_text(json.dumps(out, indent=1))
for s, row in out.items():
    print(s, row["pairs"])
    for m in ["PDM"] + rules:
        v = row[m]; print(f"   {m:12s} acc {v['acc']:.3f} {v['ci']} weighted {v['w_acc']:.3f}")
    for k, d in row["selective"].items(): print("   ", k, d)
