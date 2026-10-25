"""Experiment 54: sorted-conditional compression with the quantisation convention that reproduces its published figure.

Written and committed before running.  Exp42 used floor(256 q) quantisation, rows capped at 2000 and (for the 'raw' variant)
the capped rows; the published description does not fix the rounding.  A check on the Tuebingen pairs (all rows, linear
thinning to 8000, official weights) shows that round(255 q) gives exactly the published 68 of 102 pairs and 74.7% weighted
accuracy, while floor(256 q) gives 67 and 68.8% (and floor(255 q) 71 and 70.5%).  This experiment re-scores the faithful
variant (sprint_cd.pair_2026.sorted_compression_faithful) on every pair set used in the study: Tuebingen on its raw rows,
the chamber sets of exp38, the Sachs pairs and the three further collections of exp53 (raw rows, as stored arrays).  No
other setting is varied.  Stored: the per-pair score (positive = x -> y).
"""
import json, pathlib, sys, warnings
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from sprint_cd.pair_2026 import sorted_compression_faithful as scf
import exp49_explain as e49
import exp53_more_chamber_pairs as e53


def main():
    warnings.filterwarnings("ignore")
    from exp12_tuebingen_cep import load_pairs
    recs = []
    for pid, x, y, w in load_pairs():
        recs.append({"set": "tuebingen", "id": pid, "weight": w, "score": scf(x, y)})
    for s, pid, x, y in e49.gather():
        if s != "tuebingen": recs.append({"set": s, "id": pid, "score": scf(x, y)})
    for p in e53.extract():
        recs.append({"set": p["set"], "id": p["id"], "score": scf(p["x"], p["y"])})
    (HERE.parent / "results" / "exp54_compression_faithful.json").write_text(json.dumps({"records": recs}))
    for st in sorted({r["set"] for r in recs}):
        R = [r for r in recs if r["set"] == st]; ok = np.array([0.5 if r["score"] == 0 else float(r["score"] > 0) for r in R])
        line = f"{st:13s} n={len(R)} acc={ok.mean():.3f}"
        if st == "tuebingen": w = np.array([r["weight"] for r in R]); line += f" weighted={(ok * w).sum() / w.sum():.3f}"
        print(line)


if __name__ == "__main__":
    main()
