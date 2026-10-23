"""Per-sample-size precision, recall and undecided fraction from exp13's raw records."""
import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
d = json.load(open(ROOT / "results" / "exp13_monitoring_fwer.json"))
grid = d["config"]["grid"]
out = {}
for cond in d["config"]["conditions"]:
    out[cond] = {}
    for m in d["config"]["methods"]:
        rows = []
        for gi, n in enumerate(grid):
            tp = fp = fn = 0
            for rec in d["raw"]:
                est = rec["conds"][cond]["edges"][m][gi]
                if est is None:
                    continue
                T = {tuple(p) for p in rec["truth"]}
                E = {tuple(p) for p in est}
                tp += len(T & E); fp += len(E - T); fn += len(T - E)
            prec = tp / (tp + fp) if tp + fp else float("nan")
            rec_ = tp / (tp + fn) if tp + fn else float("nan")
            rows.append({"n": n, "precision": prec, "recall": rec_, "tp": tp, "fp": fp, "fn": fn})
        out[cond][m] = rows
(ROOT / "results" / "exp13_by_n.json").write_text(json.dumps(out, indent=1))
for cond in out:
    print(cond)
    for m, rows in out[cond].items():
        print(f"  {m:14s}", " ".join(f"{r['n']}:{r['precision']:.2f}/{r['recall']:.2f}" for r in rows))
