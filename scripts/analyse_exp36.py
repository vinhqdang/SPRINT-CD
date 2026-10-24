import json, pathlib
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parent.parent
d = json.load(open(ROOT / "results" / "exp36_graph_benchmark_extra.json"))
s = d["sachs"]; T = {tuple(sorted(e)) for e in s["truth"]}
out = {"sachs": {}}
for m, edges in s["edges"].items():
    if edges is None: out["sachs"][m] = None; continue
    E = {tuple(sorted(p)) for p in edges}; tp = len(E & T); fp = len(E - T)
    P = tp / max(tp + fp, 1); R = tp / len(T); out["sachs"][m] = {"edges": len(E), "P": P, "R": R, "F1": 2 * P * R / (P + R) if P + R else 0, "fp": fp}
# NOTEARS on chambers
e20 = json.load(open(ROOT / "results" / "exp20_causal_chamber.json"))
e30 = json.load(open(ROOT / "results" / "exp30_btad_chambers.json"))["records"]


def sc(E, T):
    E = {tuple(sorted(p)) for p in E}; tp = len(E & T); fp = len(E - T)
    P = tp / max(tp + fp, 1); R = tp / len(T); return [P, R, 2 * P * R / (P + R) if P + R else 0, fp]


iid = {0: [], 1: []}
for r in d["notears_chambers"]:
    if r["kind"] != "iid" or r["edges"] is None: continue
    rec = [x for x in e20["records"] if x["ds"] == r["ds"]][0]
    iid[r["ds"]].append(sc(r["edges"], {tuple(sorted(e)) for e in rec["truth"]}))
walk = []
for r in d["notears_chambers"]:
    if r["kind"] != "walk" or r["edges"] is None: continue
    rec = [x for x in e30 if tuple(x["spec"][:2]) == tuple(r["spec"][:2])][0]
    walk.append((r["spec"][0], sc(r["edges"], {tuple(sorted(e)) for e in rec["truth"]})))
out["notears_iid"] = {k: np.mean(v, axis=0).tolist() for k, v in iid.items()}
out["notears_walk_wind"] = np.mean([x for s_, x in walk if s_ == "wt_walks_v1"], axis=0).tolist()
out["notears_walk_all"] = np.mean([x for s_, x in walk], axis=0).tolist()
(ROOT / "results" / "exp36_summary.json").write_text(json.dumps(out, indent=1))
print("SACHS (17 consensus edges)")
for m, v in out["sachs"].items():
    print(f"  {m:27s}", v if v is None else f"edges {v['edges']} P {v['P']:.2f} R {v['R']:.2f} F1 {v['F1']:.2f} fp {v['fp']}")
print("NOTEARS iid chambers [P,R,F1,fp]", {k: np.round(v, 2).tolist() for k, v in out["notears_iid"].items()})
print("NOTEARS walks wind", np.round(out["notears_walk_wind"], 2).tolist(), "all", np.round(out["notears_walk_all"], 2).tolist())
