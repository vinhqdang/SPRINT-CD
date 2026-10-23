"""Summaries of exp17 (scaling, ablation): ever-false rate, final precision/recall, runtime."""
import json, pathlib, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parent.parent
from statistics import median


def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"),) * 2
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return round(c - h, 3), round(c + h, 3)


def summarise(path, key):
    d = json.load(open(path)); out = {}
    groups = {}
    for r in d["records"]:
        groups.setdefault(r[key] if key == "d" else 8, []).append(r)
    for g, recs in sorted(groups.items()):
        out[g] = {}
        for m in recs[0]["methods"]:
            ef = tp = fp = fn = und = ok = 0; secs = []
            for r in recs:
                mm = r["methods"][m]; T = {tuple(p) for p in r["truth"]}
                est = mm["edges"]
                if est is None or any(e is None for e in est): continue
                ok += 1; secs.append(mm["sec"])
                ever = any({tuple(p) for p in e} - T for e in est)
                ef += ever
                E = {tuple(p) for p in est[-1]}
                tp += len(T & E); fp += len(E - T); fn += len(T - E)
                if mm.get("undecided"): und += mm["undecided"][-1] / (r["d"] * (r["d"] - 1) / 2)
            out[g][m] = {"dags": ok, "ever_false": ef / ok if ok else None,
                         "ever_false_ci": wilson(ef, ok), "precision": tp / (tp + fp) if tp + fp else None,
                         "recall": tp / (tp + fn) if tp + fn else None,
                         "undecided": und / ok if ok else None, "median_sec": median(secs) if secs else None}
    return out


if __name__ == "__main__":
    res = {}
    for name, key in (("scaling", "d"), ("ablation", "r")):
        p = ROOT / "results" / f"exp17_{name}.json"
        if p.exists():
            res[name] = summarise(p, key)
    (ROOT / "results" / "exp17_summary.json").write_text(json.dumps(res, indent=1))
    for name, s in res.items():
        print(name)
        for g, ms in s.items():
            for m, v in ms.items():
                print(f"  d={g} {m:15s} dags={v['dags']} everfalse={v['ever_false']:.3f} {v['ever_false_ci']} P={v['precision']:.3f} R={v['recall']:.3f} und={v['undecided']} sec={v['median_sec']}" if v['dags'] else f"  d={g} {m} none")
