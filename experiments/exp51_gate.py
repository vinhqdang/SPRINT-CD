"""Experiment 51: an assumption-gated direction rule (GATE), frozen on simulation, tested once on real pairs.

Written and committed before running.  Exp49-50 showed that the additive-noise rules are near-perfect where the
residuals are homoscedastic and reversed where they are strongly heteroscedastic, while RECI/IGCI are helped by
heteroscedasticity.  GATE reads this off the data:
  h_min(x, y) = min(h(x->y), h(y->x)), with h(c->e) = |Spearman(|r|, c)| for the cross-fitted additive-spline
  residual r of e on c (exp49 definition; 6 knots, ridge 1, two folds).  h_min is symmetric in x and y, so it does
  not use the label.
  GATE(x, y) = ANM-HSIC direction if h_min <= t, else the RECI direction.
t is the 95th percentile of h_min on 200 homoscedastic simulated pairs (the a = 0 generator of exp50, seeds 20262900+k,
n = 500); it is frozen before any real pair is touched.  The ANM-HSIC and RECI directions of the real pairs are the
stored ones of exp35 (same preprocessing).  Evaluated on Tuebingen (102; unweighted and official weights), chamber
i.i.d. (94), chamber walks (122), Sachs (17); exact McNemar test of GATE against RECI and against ANM-HSIC.
Prediction: Tuebingen unweighted accuracy >= 0.70; chambers within 0.05 of RECI.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json, pathlib, sys, warnings
import numpy as np
from scipy.stats import binomtest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
R = HERE.parent / "results"
import exp49_explain as e49
import exp50_dose_response as e50


def hmin(x, y):
    return min(e49.het(x, y), e49.het(y, x))


def main():
    warnings.filterwarnings("ignore")
    cal = []
    for k in range(200):
        rng = np.random.default_rng([20262900, k])
        c = rng.uniform(-1.7, 1.7, 500) if rng.random() < .5 else rng.normal(size=500)
        f = e50.FS[rng.integers(3)](c); e = f + 0.3 * f.std() * rng.normal(size=500)
        cal.append(hmin(c, e))
    t = float(np.quantile(cal, .95)); print("threshold", t, flush=True)
    data = e49.gather()
    key = {}
    for s, pid, x, y in data:
        key.setdefault(s, []).append((pid, hmin(x, y)))
    e35 = json.load(open(R / "exp35_pdm_eval.json"))["records"]
    out = {"t": t, "sets": {}}
    for s in ("tuebingen", "chamber_iid", "chamber_walk", "sachs"):
        recs = [r for r in e35 if r["set"] == s]
        hs = [h for _, h in key[s]]
        # order of data == order of records inside each set (same loaders); check ids for tuebingen
        if s == "tuebingen": assert [r["id"] for r in recs] == [p for p, _ in key[s]]
        n = min(len(recs), len(hs))
        ok = {"GATE": [], "RECI": [], "ANM-HSIC": []}; wt = []
        for r, h in zip(recs[:n], hs[:n]):
            truth_xy = r["truth"] == "x->y"
            def corr(m):
                lab = r["scores"][m][0]; return float((lab == "x->y") == truth_xy)
            ok["RECI"].append(corr("RECI")); ok["ANM-HSIC"].append(corr("ANM-HSIC"))
            ok["GATE"].append(corr("ANM-HSIC") if h <= t else corr("RECI")); wt.append(r.get("weight", 1) or 1)
        a = {m: float(np.mean(v)) for m, v in ok.items()}
        res = {"n": n, "acc": a, "share_gated_to_anm": float(np.mean([h <= t for h in hs[:n]]))}
        if s == "tuebingen":
            w = np.array(wt); res["weighted"] = {m: float((np.array(v) * w).sum() / w.sum()) for m, v in ok.items()}
        for m in ("RECI", "ANM-HSIC"):
            g, o = np.array(ok["GATE"]), np.array(ok[m]); b01 = int(((g == 1) & (o == 0)).sum()); b10 = int(((g == 0) & (o == 1)).sum())
            res[f"mcnemar_vs_{m}"] = {"gate_only": b01, "other_only": b10, "p": binomtest(b01, b01 + b10, .5).pvalue if b01 + b10 else 1.0}
        out["sets"][s] = res; print(s, json.dumps(res), flush=True)
    (R / "exp51_gate.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
