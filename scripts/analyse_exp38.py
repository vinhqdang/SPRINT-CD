import json, pathlib
import numpy as np
from scipy import stats
ROOT = pathlib.Path(__file__).resolve().parent.parent
rec = json.load(open(ROOT / "results" / "exp38_pairs_modern.json"))["records"]
old = {(r["set"], r["id"]): r for r in json.load(open(ROOT / "results" / "exp35_pdm_eval.json"))["records"]}


def cp(k, n, a=0.05):
    lo = stats.beta.ppf(a / 2, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(1 - a / 2, k + 1, n - k) if k < n else 1.0
    return [round(float(lo), 3), round(float(hi), 3)]


out = {}
for s in ("tuebingen", "chamber_iid", "chamber_walk", "sachs"):
    rs = [r for r in rec if r["set"] == s]
    row = {"pairs": len(rs)}
    w = np.array([r.get("weight", 1.0) for r in rs])
    for m in ("LOCI", "ANM"):
        ok = [(None if r["m"][m] is None else (0.5 if r["m"][m] == 0.0 else float((r["m"][m] > 0) == (r["truth"] == "x->y")))) for r in rs]  # exact ties count half
        valid = [o for o in ok if o is not None]
        v = np.array([float(o) if o is not None else 0.0 for o in ok])
        row[m] = {"acc": float(np.mean(valid)), "n_valid": len(valid), "ci": cp(int(round(sum(valid))), len(valid)),
                  "w_acc": float((w * v).sum() / w[[o is not None for o in ok]].sum())}
    if s == "sachs":
        for rule in ("IGCI", "RECI", "ANM-HSIC", "LiNGAM-pair"):
            okr = np.array([r["scores"][rule][0] == r["truth"] for r in rs]); row[rule] = {"acc": float(okr.mean()), "ci": cp(int(okr.sum()), len(rs))}
        okp = np.array([(r["pdm"] > 0.5) == (r["truth"] == "x->y") for r in rs]); row["PDM"] = {"acc": float(okp.mean()), "ci": cp(int(okp.sum()), len(rs))}
    out[s] = row
(ROOT / "results" / "exp38_summary.json").write_text(json.dumps(out, indent=1))
for s, row in out.items():
    print(s, row["pairs"])
    for m, v in row.items():
        if isinstance(v, dict): print(f"   {m:12s} acc {v['acc']:.3f} {v['ci']}" + (f" valid {v['n_valid']} weighted {v['w_acc']:.3f}" if 'n_valid' in v else ""))
