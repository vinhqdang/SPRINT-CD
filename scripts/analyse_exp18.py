"""Summaries of exp18 (held-out families, n sweep, Tuebingen baselines)."""
import json, pathlib
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parent.parent


def wilson(k, n, z=1.96):
    if n == 0: return [0.0, 1.0]
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(max(0, c - h), 3), round(min(1, c + h), 3)]


def summarise(recs):
    groups = {}
    for r in recs:
        groups.setdefault((r["fam"], r["n"]), []).append(r)
    out = {}
    for (fam, n), rs in groups.items():
        row = {"reps": len(rs)}
        for kind in ("agree", "gg", "ggh"):
            dec = [r for r in rs if r["relfit"] and r["relfit"][kind] is not None]
            wrong = sum(r["relfit"][kind] != r["truth"] for r in dec)
            row[kind] = {"decided": len(dec), "wrong": wrong, "wrong_ci": wilson(wrong, len(rs))}
        for kind in ("forced_gg", "forced_ggh"):
            sc = [r for r in rs if r["relfit"]]
            row[kind] = float(np.mean([r["relfit"][kind] == r["truth"] for r in sc])) if sc else None
        for b in rs[0]["base"]:
            row[b] = float(np.mean([r["base"][b] == r["truth"] for r in rs]))
        out[f"{fam}@{n}"] = row
    return out


res = {}
for name in ("sim", "sweep"):
    p = ROOT / "results" / f"exp18_{name}.json"
    if p.exists():
        res[name] = summarise(json.load(open(p))["records"])
(ROOT / "results" / "exp18_summary.json").write_text(json.dumps(res, indent=1))
for name, s in res.items():
    print(name)
    for k, v in s.items():
        print(f"  {k:18s} agree {v['agree']['decided']}/{v['reps']} wrong {v['agree']['wrong']} | gg {v['gg']['decided']} w{v['gg']['wrong']} | ggh {v['ggh']['decided']} w{v['ggh']['wrong']} | forced gg {v['forced_gg']:.2f} ggh {v['forced_ggh']:.2f} | IGCI {v['IGCI']:.2f} RECI {v['RECI']:.2f} ANM {v['ANM-HSIC']:.2f} LiNG {v['LiNGAM-pair']:.2f}")
