"""Spread of the Tuebingen counts across random seeds (exp19) plus the exp14 seed."""
import json, pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
d = json.load(open(ROOT / "results" / "exp19_tuebingen_seeds.json"))
out = {}
for seed in d["seeds"]:
    rs = [r for r in d["records"] if r["seed"] == seed and r["r"] is not None]
    row = {"scored": len(rs)}
    for kind in ("agree", "gg", "ggh"):
        dec = [r for r in rs if r["r"][kind] is not None]
        row[kind] = {"decided": len(dec), "correct": sum(r["r"][kind] == "x->y" for r in dec)}
    for kind in ("forced_gg", "forced_ggh"):
        row[kind] = sum(r["r"][kind] == "x->y" for r in rs) / len(rs)
    out[seed] = row
a = json.load(open(ROOT / "results" / "exp14_tuebingen_relfit.json"))["kinds"]
out["exp14_seed"] = {k: {"decided": a[k]["decided"], "correct": a[k]["correct"]} for k in ("agree", "gg", "ggh")}
(ROOT / "results" / "exp19_summary.json").write_text(json.dumps(out, indent=1))
for s, r in out.items():
    print(s, r)
