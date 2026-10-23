"""Accuracy of the pair rules at the coverage of the certificate (matched-coverage comparison)."""
import json, pathlib
import numpy as np
from scipy import stats
ROOT = pathlib.Path(__file__).resolve().parent.parent
rec = json.load(open(ROOT / "results" / "exp28_selective_orientation.json"))["records"]
t14 = json.load(open(ROOT / "results" / "exp14_tuebingen_relfit.json"))["rows"]
scored_t = {r["pair"] for r in t14}
rules = ["IGCI", "RECI", "ANM-HSIC", "LiNGAM-pair"]


def cp(k, n, a=0.05):
    if n == 0: return [0, 1]
    lo = stats.beta.ppf(a / 2, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(1 - a / 2, k + 1, n - k) if k < n else 1.0
    return [round(float(lo), 3), round(float(hi), 3)]


def topk(rs, rule, k):
    order = sorted(rs, key=lambda r: -r["scores"][rule][1])[:k]
    ok = sum(r["scores"][rule][0] == r["truth"] for r in order)
    return ok, len(order)


out = {}
# Tuebingen
rs = [r for r in rec if r["set"] == "tuebingen" and r["id"] in scored_t]
ours = {"agree": (12, 11), "gg": (23, 19), "ggh": (18, 14)}
out["tuebingen"] = {"pairs": len(rs), "certificate": ours,
                    "rules": {rule: {c: topk(rs, rule, k) for c, (k, _) in ours.items()} for rule in rules}}
# held-out simulated
rs = [r for r in rec if r["set"] == "heldout"]
e18 = json.load(open(ROOT / "results" / "exp18_sim.json"))["records"]
ours = {}
for kind in ("agree", "gg", "ggh"):
    dec = [x for x in e18 if x["relfit"] and x["relfit"][kind] is not None]
    ours[kind] = (len(dec), sum(x["relfit"][kind] == x["truth"] for x in dec))
out["heldout"] = {"pairs": len(rs), "certificate": ours,
                  "rules": {rule: {c: topk(rs, rule, k) for c, (k, _) in ours.items()} for rule in rules}}
# chamber
rs = [r for r in rec if r["set"] == "chamber"]
e21 = json.load(open(ROOT / "results" / "exp21_chamber_pairs.json"))["records"]
ours = {}
for kind in ("agree", "gg", "ggh"):
    dec = [x for x in e21 if x["relfit"] and x["relfit"][kind] is not None]
    ours[kind] = (len(dec), sum(x["relfit"][kind] == x["truth"] for x in dec))
out["chamber"] = {"pairs": len(rs), "certificate": ours,
                  "rules": {rule: {c: topk(rs, rule, k) for c, (k, _) in ours.items()} for rule in rules}}
(ROOT / "results" / "exp28_summary.json").write_text(json.dumps(out, indent=1))
for s, v in out.items():
    print(s, v["pairs"], "pairs; certificate (decided, correct):", v["certificate"])
    for rule, d in v["rules"].items():
        print("  ", rule, {c: f"{a}/{b}" for c, (a, b) in d.items()})
