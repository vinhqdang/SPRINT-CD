"""Compression baseline on the 102 Tuebingen pairs under different quantisation conventions and row caps (audit table)."""
import json, sys, bz2, warnings
import numpy as np
warnings.filterwarnings("ignore"); sys.path.insert(0, "."); sys.path.insert(0, "experiments")
from exp12_tuebingen_cep import load_pairs
P = [(np.asarray(x, float), np.asarray(y, float), w) for pid, x, y, w in load_pairs()]
def q(v, mode):
    lo, hi = v.min(), v.max(); u = (v - lo) / (hi - lo) if hi > lo else np.zeros(len(v))
    return {"floor256": np.minimum((u * 256).astype(int), 255), "floor255": np.floor(u * 255).astype(int), "round255": np.round(u * 255).astype(int)}[mode].astype(np.uint8)
dc = lambda a: len(bz2.compress(np.diff(a.astype(np.int16)).astype(np.int8).tobytes(), 9))
def sc(k, v):
    o = np.argsort(k, kind="stable"); return len(bz2.compress(np.diff(v[o].astype(np.int16)).astype(np.int8).tobytes(), 9))
out = {}
for mode in ("floor256", "floor255", "round255"):
    for cap in (500, 1000, 2000, 4000, 8000, 10 ** 6):
        ok = []; w = []
        for x, y, wt in P:
            if len(x) > cap: idx = np.linspace(0, len(x) - 1, cap).astype(int); x, y = x[idx], y[idx]
            qx, qy = q(x, mode), q(y, mode); s = sc(qy, qx) / max(dc(qx), 1) - sc(qx, qy) / max(dc(qy), 1)
            ok.append(1.0 if s > 0 else 0.0 if s < 0 else .5); w.append(wt)
        ok = np.array(ok); w = np.array(w); out[f"{mode}_cap{cap}"] = {"pairs_correct": float(ok.sum()), "unweighted": float(ok.mean()), "weighted": float((ok * w).sum() / w.sum())}
json.dump(out, open("results/exp54_audit_conventions.json", "w"), indent=1)
for k, v in out.items(): print(k, v)
