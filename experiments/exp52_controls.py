"""Experiment 52: controls for experiments 49-50 (leakage-free marginal classifier, matched dose-response with marginal control).

Written and committed before running.
(A) Marginal-only classifier without leakage through shared variables.  Same features as exp49 (skewness, kurtosis,
    spacing entropy, log fraction of unique values, 1-99% range), same logistic regression.  Protocols:
    (A1) chamber i.i.d.: train on one recording's pairs, test on the other's (and conversely); chamber walks: leave one
    recording out (five recordings in the data; an earlier wording said 19); Sachs: leave one protein out (train on pairs involving neither protein of the test pair);
    Tuebingen: leave one pair out.  (A2) cross-set transfer: train on all pairs of one set, test on each other set.
    (A3) the same with the log-unique feature removed (it separates discrete actuators from continuous sensors and
    is a measurement artefact when sets are preprocessed differently).  Prediction: accuracy on the chambers stays
    above 0.80 under A1/A3, otherwise the 'solved from marginals' claim is withdrawn.
(B) Dose-response with matched pairs and a marginal control.  Same generator as exp50 but for each (a, k) the same
    c, f and standard normal eps are used at every level, and a second arm replaces the heteroscedastic scale
    sigma(c) = s0 exp(a c)/sqrt(mean exp(2ac)) by an independent random scale sigma_i = s0 exp(a u_i)/sqrt(mean exp(2 a u))
    with u_i ~ N(0,1) independent of c: the effect then has the same heavy-tailed marginal mixture but no
    dependence of the noise scale on the cause.  Rules: IGCI, RECI, ANM-HSIC (RESIT and AVICI omitted).  100 pairs per
    level and arm, a random half swapped.  Prediction: the rise of IGCI and RECI with a is present in the heteroscedastic
    arm and absent (< 5 points) in the independent-scale arm; ANM-HSIC falls in the heteroscedastic arm only.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json, pathlib, sys, warnings
from multiprocessing import Pool
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
R = HERE.parent / "results"
import exp49_explain as e49
import exp50_dose_response as e50
from sprint_cd.pair_baselines import scored


def fit_pred(Ftr, Fte):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    X = np.vstack([Ftr, -Ftr]); y = np.r_[np.ones(len(Ftr)), np.zeros(len(Ftr))]
    sc = StandardScaler().fit(X); clf = LogisticRegression(C=1.0, max_iter=1000).fit(sc.transform(X), y)
    return clf.decision_function(sc.transform(Fte)) > 0


def partA():
    warnings.filterwarnings("ignore")
    data = e49.gather(); sets = {}
    for s, pid, x, y in data:
        sets.setdefault(s, []).append((pid, e49.marg(x) - e49.marg(y)))
    out = {}
    for drop in (None, 3):
        keep = [i for i in range(5) if i != drop]; tag = "all5" if drop is None else "no_unique"
        F = {s: np.array([f for _, f in l])[:, keep] for s, l in sets.items()}
        ids = {s: [p for p, _ in l] for s, l in sets.items()}
        r = {}
        # chamber iid: wind (di 0) vs light (di 1), id prefix "di/"
        d = np.array([int(p.split("/")[0]) for p in ids["chamber_iid"]]); Fi = F["chamber_iid"]
        r["chamber_iid_cross_recording"] = float(np.mean(np.r_[fit_pred(Fi[d == 0], Fi[d == 1]), fit_pred(Fi[d == 1], Fi[d == 0])]))
        k = np.array([int(p.split("/")[0]) for p in ids["chamber_walk"]]); Fw = F["chamber_walk"]; c = []
        for g in np.unique(k): c += list(fit_pred(Fw[k != g], Fw[k == g]))
        r["chamber_walk_leave_recording_out"] = float(np.mean(c))
        names = [p.split("->") for p in ids["sachs"]]; Fs = F["sachs"]; c = []
        for i, (a, b) in enumerate(names):
            tr = [j for j, (u, v) in enumerate(names) if u not in (a, b) and v not in (a, b)]
            c.append(float(fit_pred(Fs[tr], Fs[i:i + 1])[0]) if len(tr) >= 3 else np.nan)
        r["sachs_leave_proteins_out"] = float(np.nanmean(c)); r["sachs_n_evaluated"] = int(np.sum(~np.isnan(c)))
        Ft = F["tuebingen"]; r["tuebingen_leave_one_out"] = float(np.mean([fit_pred(np.delete(Ft, i, 0), Ft[i:i + 1])[0] for i in range(len(Ft))]))
        for a_ in F:
            for b_ in F:
                if a_ != b_: r[f"transfer_{a_}_to_{b_}"] = float(np.mean(fit_pred(F[a_], F[b_])))
        out[tag] = r; print("A", tag, json.dumps({k_: round(v, 3) for k_, v in r.items()}), flush=True)
    return out


def make_b(args):
    arm, lvl, k = args
    a = e50.LEVELS[lvl]
    rng = np.random.default_rng([20263000, k])
    c = rng.uniform(-1.7, 1.7, 500) if rng.random() < .5 else rng.normal(size=500)
    f = e50.FS[rng.integers(3)](c); eps = rng.normal(size=500); u = rng.normal(size=500); flip = rng.random() < .5
    s0 = 0.3 * f.std()
    z = c if arm == "hetero" else u
    sig = s0 * np.exp(a * z) / np.sqrt(np.mean(np.exp(2 * a * z)))
    e = f + sig * eps
    return ((e, c) if flip else (c, e)), flip


def one_b(args):
    warnings.filterwarnings("ignore")
    (x, y), flip = make_b(args)
    sc = scored(x, y, 1)
    return {"arm": args[0], "level": args[1], "flip": bool(flip), "m": {k: (float(v[1]) if v[0] == "x->y" else -float(v[1])) for k, v in sc.items() if k in ("IGCI", "RECI", "ANM-HSIC")}}


def partB():
    jobs = [(arm, l, k) for arm in ("hetero", "independent") for l in range(4) for k in range(100)]
    with Pool(4) as p: recs = p.map(one_b, jobs, chunksize=5)
    out = {}
    for arm in ("hetero", "independent"):
        out[arm] = {}
        for m in ("IGCI", "RECI", "ANM-HSIC"):
            out[arm][m] = [float(np.mean([0.5 if r["m"][m] == 0 else float((r["m"][m] > 0) == (not r["flip"])) for r in recs if r["arm"] == arm and r["level"] == l])) for l in range(4)]
            print("B", arm, m, [round(x, 2) for x in out[arm][m]], flush=True)
    return out


if __name__ == "__main__":
    res = {"A": partA(), "B": partB()}
    (R / "exp52_controls.json").write_text(json.dumps(res, indent=1))
