"""Experiment 47: how much of the spread between pair methods is sampling noise and reporting choices?

Written and committed before running.  Inputs are the per-pair decisions already produced for the 102
univariate Tuebingen pairs (exp35, 38, 42, 46; scores: positive = x -> y, ties half correct; PDM score
0.5 is neutral, its probability of x -> y minus 0.5 is used).  Twelve methods: IGCI, RECI, ANM-HSIC, pairwise
LiNGAM, PDM, LOCI, ANM (GP), compression (dequantised and raw rows), TRA, AVICI, RESIT.
 (A) Sampling noise: all 66 paired comparisons with the exact McNemar test, Holm over the family; the
     smallest accuracy difference detectable with 80% power at n = 102 (simulated, discordance rate from the data).
 (B) Multiverse of reporting choices: weighting {official weights, none} x subset {all 102, >= 250 rows,
     >= 500, >= 1000} x coverage {100, 75, 50, 25 percent most confident by |score|} = 32 universes.  For each
     method: accuracy range, best and worst rank (ties share the best rank) and the number of universes in which it
     ranks first, and in which it ranks above RECI.
 (C) Selective subsets: 10000 random subsets of 50 pairs; for each method the mean subset accuracy, and the expected
     maximum over m = 1, 3, 10 independently drawn subsets (the accuracy a researcher would report after
     trying m subsets); the probability that a method beats RECI on a random subset.
"""
import itertools, json, pathlib, sys
import numpy as np
from scipy.stats import binomtest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
R = HERE.parent / "results"
rng = np.random.default_rng(20262500)


def load():
    from exp12_tuebingen_cep import load_pairs
    nrows = {pid: len(x) for pid, x, y, w in load_pairs()}
    e35 = [r for r in json.load(open(R / "exp35_pdm_eval.json"))["records"] if r["set"] == "tuebingen"]
    e38 = {r["id"]: r for r in json.load(open(R / "exp38_pairs_modern.json"))["records"] if r["set"] == "tuebingen"}
    e42 = {r["id"]: r for r in json.load(open(R / "exp42_pairs_2026.json"))["records"] if r["set"] == "tuebingen"}
    e46 = {r["id"]: r for r in json.load(open(R / "exp46_pairs_sota.json"))["records"] if r["set"] == "tuebingen"}
    ids, w, n = [], [], []
    S = {}
    for r in e35:
        i = r["id"]; ids.append(i); w.append(r["weight"]); n.append(nrows[i])
        row = {k: (v[1] if v[0] == "x->y" else -v[1]) for k, v in r["scores"].items()}
        row["PDM"] = r["pdm"] - 0.5
        row["LOCI"] = e38[i]["m"]["LOCI"]; row["ANM-GP"] = e38[i]["m"]["ANM"]
        row["Compression"] = e42[i]["m"]["SC"]; row["Compression raw"] = e42[i]["m"]["SCraw"]; row["TRA"] = e42[i]["m"]["TRA"]
        row["AVICI"] = e46[i]["m"]["AVICI"]; row["RESIT"] = e46[i]["m"]["RESIT"]
        S[i] = row
    return ids, np.array(w), np.array(n), S


def main():
    ids, w, n, S = load()
    M = list(S[ids[0]].keys())
    sc = np.array([[S[i][m] if S[i][m] is not None and S[i][m] == S[i][m] else np.nan for m in M] for i in ids])  # (102, M)
    # IGCI/RECI score sign convention: scored() returns a magnitude with the direction label; handled in load()
    ok = np.where(np.isnan(sc), 0.5, np.where(sc > 0, 1.0, np.where(sc < 0, 0.0, 0.5)))   # truth is always x -> y
    out = {"methods": M, "n_pairs": len(ids)}
    # (A)
    acc = ok.mean(0); out["accuracy"] = dict(zip(M, acc.tolist()))
    pv = {}
    for a, b in itertools.combinations(range(len(M)), 2):
        b01 = int(((ok[:, a] == 1) & (ok[:, b] == 0)).sum()); b10 = int(((ok[:, a] == 0) & (ok[:, b] == 1)).sum())
        pv[(a, b)] = binomtest(b01, b01 + b10, 0.5).pvalue if b01 + b10 else 1.0
    order = sorted(pv, key=pv.get); m_ = len(order); holm = {}; run = 0.0
    for k, key in enumerate(order):
        run = max(run, min(1.0, pv[key] * (m_ - k))); holm[key] = run
    out["pairs_tested"] = m_; out["raw_p_below_0.05"] = int(sum(v < .05 for v in pv.values())); out["holm_p_below_0.05"] = int(sum(v < .05 for v in holm.values()))
    out["smallest_p_holm"] = min(holm.values())
    # detectable difference at 80% power (paired, discordance rate d from data)
    disc = np.mean([((ok[:, a] != ok[:, b]) & (ok[:, a] != .5) & (ok[:, b] != .5)).mean() for a, b in itertools.combinations(range(len(M)), 2)])
    det = None
    for delta in np.arange(0.02, 0.4, 0.01):
        pa = (disc + delta) / 2; hits = 0
        for _ in range(400):
            k = rng.binomial(len(ids), disc); x = rng.binomial(k, pa / disc) if disc else 0
            hits += binomtest(int(x), int(k), 0.5).pvalue < 0.05 if k else 0
        if hits / 400 >= 0.8: det = float(delta); break
    out["discordance_rate"] = float(disc); out["detectable_difference_80pct_power"] = det
    # (B)
    uni = []
    for wt, minrows, cov in itertools.product(("official", "none"), (0, 250, 500, 1000), (1.0, .75, .5, .25)):
        sel = n >= minrows; wv = w if wt == "official" else np.ones_like(w)
        accs = []
        for j in range(len(M)):
            s = np.abs(np.nan_to_num(sc[:, j], nan=0.0)); idx = np.where(sel)[0]
            if cov < 1.0:
                thr = np.quantile(s[idx], 1 - cov); idx = idx[s[idx] >= thr]
            accs.append(float((ok[idx, j] * wv[idx]).sum() / wv[idx].sum()))
        uni.append({"weights": wt, "min_rows": minrows, "coverage": cov, "n": int(sel.sum()), "acc": accs})
    A = np.array([u["acc"] for u in uni]); rk = (-A).argsort(1).argsort(1) + 1
    best_rank = np.array([[1 + (A[u] > A[u, j] + 1e-12).sum() for j in range(len(M))] for u in range(len(uni))])
    jr = M.index("RECI")
    out["multiverse"] = {m: {"acc_min": float(A[:, j].min()), "acc_max": float(A[:, j].max()), "best_rank": int(best_rank[:, j].min()),
                              "worst_rank": int(best_rank[:, j].max()), "universes_first": int((best_rank[:, j] == 1).sum()),
                              "universes_above_RECI": int((A[:, j] > A[:, jr] + 1e-12).sum())} for j, m in enumerate(M)}
    out["n_universes"] = len(uni); out["universes"] = uni
    # (C)
    idx_all = np.arange(len(ids)); subs = [rng.choice(idx_all, 50, replace=False) for _ in range(10000)]
    SA = np.array([ok[s].mean(0) for s in subs])
    out["subsets"] = {}
    for j, m in enumerate(M):
        v = SA[:, j]; e = {}
        for mm in (1, 3, 10):
            k = len(v) // mm * mm; e[str(mm)] = float(v[:k].reshape(-1, mm).max(1).mean())
        out["subsets"][m] = {"mean": float(v.mean()), "sd": float(v.std()), "expected_max": e, "p_beats_RECI": float((v > SA[:, jr]).mean())}
    (R / "exp47_multiverse.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("accuracy", "pairs_tested", "raw_p_below_0.05", "holm_p_below_0.05", "smallest_p_holm", "discordance_rate", "detectable_difference_80pct_power")}, indent=1))
    for m in M: print(f"{m:16s}", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out["multiverse"][m].items()}, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out["subsets"][m].items() if k != "expected_max"}, {k: round(v, 3) for k, v in out["subsets"][m]["expected_max"].items()})


if __name__ == "__main__":
    main()
