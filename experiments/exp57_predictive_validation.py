"""Experiment 57: does the factorial map of exp56 predict rule accuracy on the real collections?

Written and committed before running.  Seven real pair collections (Tuebingen 102, chamber i.i.d. 94, chamber walks 122,
Sachs 17, bernoulli 16, changepoints 70, malus 48 pair-runs).  Rules: IGCI, RECI, ANM-HSIC, pairwise LiNGAM, compression
(faithful), RESIT, AVICI; their stored decisions (exp35, 38, 42, 46, 53, 54).  For every real pair the label-free
diagnostics of exp56 (hmin, lu, kmin, kmax, smax, lrho, ln) are computed from the arrays.  Prediction for a real pair:
the mean correctness of each rule among its k = 100 nearest simulated pairs (Euclidean distance on diagnostics
standardised with the simulation's mean and sd).  Outcomes, pre-registered:
 (1) over the 7 x 7 = 49 (collection, rule) cells, the Spearman correlation between predicted and observed accuracy
     (success: >= 0.5) and the RMSE against the RMSE of the constant prediction (each rule's overall simulation accuracy)
     (success: smaller);
 (2) for each collection, whether the rule predicted to be best is among the two best observed (success: >= 4 of 7
     collections);
 (3) pair-level AUC of the predicted correctness for the observed correctness, pooled per rule, against 0.5.
Nothing is fitted to real data.
"""
import json, pathlib, sys, warnings
import numpy as np
from scipy import stats

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
R = HERE.parent / "results"
import exp49_explain as e49
import exp53_more_chamber_pairs as e53
import exp56_factorial as e56

RULES = ["IGCI", "RECI", "ANM-HSIC", "LiNGAM-pair", "Compression", "RESIT", "AVICI"]
FEATS = ["hmin", "lu", "kmin", "kmax", "smax", "lrho", "ln"]


def corr_of(score, truth_xy=True):
    if score is None or score != score: return np.nan
    return 0.5 if score == 0 else float((score > 0) == truth_xy)


def main():
    warnings.filterwarnings("ignore")
    sim = json.load(open(R / "exp56_factorial.json"))["records"]
    Fs = np.array([[r["f"][k] for k in FEATS] for r in sim], float)
    Cs = np.array([[corr_of(r["m"].get(m), not r["flip"]) for m in RULES] for r in sim], float)
    ok = ~np.isnan(Fs).any(1); Fs, Cs = Fs[ok], Cs[ok]
    mu, sd = Fs.mean(0), Fs.std(0) + 1e-9; Z = (Fs - mu) / sd
    real = {}
    data = e49.gather()
    for s, pid, x, y in data:
        real.setdefault(s, []).append({"id": pid, "x": x, "y": y})
    for p in e53.extract(): real.setdefault(p["set"], []).append({"id": p["id"], "x": p["x"], "y": p["y"]})
    # observed decisions
    e35 = {s: [r for r in json.load(open(R / "exp35_pdm_eval.json"))["records"] if r["set"] == s] for s in ("tuebingen", "chamber_iid", "chamber_walk")}
    e38 = json.load(open(R / "exp38_pairs_modern.json"))["records"]
    e42 = json.load(open(R / "exp42_pairs_2026.json"))["records"]; e46 = json.load(open(R / "exp46_pairs_sota.json"))["records"]
    e53r = json.load(open(R / "exp53_more_chamber_pairs.json"))["records"]; e54 = json.load(open(R / "exp54_compression_faithful.json"))["records"]
    obs = {}
    for s in ("tuebingen", "chamber_iid", "chamber_walk", "sachs"):
        a38 = [r for r in e38 if r["set"] == s]; a46 = [r for r in e46 if r["set"] == s]; a54 = [r for r in e54 if r["set"] == s]
        assert len(a38) == len(real[s]) == len(a46) == len(a54), (s, len(a38), len(real[s]))
        rows = []
        for k in range(len(a38)):
            truth_xy = a38[k]["truth"] == "x->y"
            if s == "sachs": sc = a38[k]["scores"]
            else:
                sc = e35[s][k]["scores"]; assert e35[s][k]["truth"] == a38[k]["truth"]
            d = {m: float((sc[m][0] == "x->y") == truth_xy) for m in ("IGCI", "RECI", "ANM-HSIC", "LiNGAM-pair")}
            d["Compression"] = corr_of(a54[k]["score"], truth_xy); d["RESIT"] = corr_of(a46[k]["m"]["RESIT"], truth_xy); d["AVICI"] = corr_of(a46[k]["m"]["AVICI"], truth_xy)
            rows.append(d)
        obs[s] = rows
    for s in ("bernoulli", "changepoints", "malus"):
        a53 = [r for r in e53r if r["set"] == s]; a54 = [r for r in e54 if r["set"] == s]
        assert len(a53) == len(real[s]) == len(a54)
        obs[s] = [dict({m: corr_of(r["m"].get(m)) for m in ("IGCI", "RECI", "ANM-HSIC", "LiNGAM-pair", "RESIT", "AVICI")}, Compression=corr_of(b["score"])) for r, b in zip(a53, a54)]
    out = {"cells": {}, "rules": RULES}
    pred_cells, obs_cells, const = [], [], np.nanmean(Cs, 0)
    pair_scores = {m: ([], []) for m in RULES}
    for s, lst in real.items():
        P = []
        for p in lst:
            f = e56.feats(p["x"], p["y"]); v = np.array([f[k] for k in FEATS], float)
            if np.isnan(v).any(): v = np.where(np.isnan(v), mu, v)
            z = (v - mu) / sd; idx = np.argsort(((Z - z) ** 2).sum(1))[:100]
            P.append(np.nanmean(Cs[idx], 0))
        P = np.array(P); O = np.array([[r[m] for m in RULES] for r in obs[s]], float)
        out["cells"][s] = {"n": len(lst), "pred": np.nanmean(P, 0).tolist(), "obs": np.nanmean(O, 0).tolist()}
        for j, m in enumerate(RULES):
            v = ~np.isnan(O[:, j]); pair_scores[m][0].extend(P[v, j]); pair_scores[m][1].extend(O[v, j])
        pred_cells += list(np.nanmean(P, 0)); obs_cells += list(np.nanmean(O, 0))
        print(s, len(lst), "pred", np.round(np.nanmean(P, 0), 2), "obs", np.round(np.nanmean(O, 0), 2), flush=True)
    pc, oc = np.array(pred_cells), np.array(obs_cells); v = ~np.isnan(oc)
    cc = np.tile(const, len(real))
    out["spearman"] = float(stats.spearmanr(pc[v], oc[v])[0]); out["pearson"] = float(stats.pearsonr(pc[v], oc[v])[0])
    out["rmse_model"] = float(np.sqrt(np.mean((pc[v] - oc[v]) ** 2))); out["rmse_constant"] = float(np.sqrt(np.mean((cc[v] - oc[v]) ** 2)))
    best = 0
    for s, c in out["cells"].items():
        pr, ob = np.array(c["pred"]), np.array(c["obs"]); top2 = set(np.argsort(-np.nan_to_num(ob, nan=-1))[:2]); hit = int(np.argmax(pr) in top2); best += hit; c["pred_best"] = RULES[int(np.argmax(pr))]; c["obs_top2"] = [RULES[i] for i in top2]
    out["top1_in_top2"] = best
    from sklearn.metrics import roc_auc_score
    out["pair_auc"] = {}
    for m in RULES:
        p_, o_ = np.array(pair_scores[m][0]), np.array(pair_scores[m][1]); keep = (o_ == 0) | (o_ == 1)
        out["pair_auc"][m] = float(roc_auc_score(o_[keep], p_[keep])) if len(set(o_[keep])) > 1 else None
    print({k: out[k] for k in ("spearman", "pearson", "rmse_model", "rmse_constant", "top1_in_top2", "pair_auc")})
    (R / "exp57_predictive_validation.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
