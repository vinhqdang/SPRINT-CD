"""Experiment 49: why do direction rules that exploit dependence lose to RECI/IGCI on real pairs?

Written and committed before running.  Two hypotheses, each with a prediction that can fail.
H1 (marginal shortcut): on the real pair sets a classifier that sees only the two marginal distributions (never the
   joint) predicts the direction about as well as the best direction rule, so a benchmark can be solved without any
   dependence structure.  Features per variable after min-max scaling to [0,1]: skewness, excess kurtosis, Vasicek
   spacing entropy (m = sqrt(n)), log fraction of unique values, (q99 - q01) range ratio; the pair feature is
   f(x) - f(y), the label is 1 for the true orientation and 0 for the swapped copy (both copies enter).  Logistic
   regression on standardised features, 5-fold cross-validation with the folds assigned to pairs (both copies of a pair
   in the same fold), 50 repetitions.  Sets: Tuebingen (102), chamber i.i.d. (94 pair-runs, folds over the 47 edges),
   chamber walks (122), Sachs (17).  Prediction: marginal-only accuracy is within 0.05 of RECI on the chamber sets and
   not above chance on Tuebingen (the Tuebingen cause is not marginally distinguishable).
H2 (assumption violation): the additive-noise family loses accuracy where the model's assumption fails.  On Tuebingen,
   for each pair the residual heteroscedasticity h = |Spearman(|r|, x)| of a cross-fitted additive-spline (6 knots, ridge) regression
   of y on x (true direction) and the fraction of unique values u.  For each of the twelve rules of exp47, the
   Mann-Whitney test of h (and of u) between correctly and wrongly oriented pairs, one-sided in the direction 'more
   violation, more errors'; and, pooled over the four noise-model rules (ANM-HSIC, ANM-GP, RESIT, LOCI), a logistic
   regression of correctness on standardised h and log n (u is constant after dequantisation and is dropped from the regression; its Mann-Whitney columns are uninformative).  Prediction: h is higher on the pairs the noise-model rules get
   wrong (p < 0.05 for at least two of the four), and not for RECI.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json, pathlib, sys, warnings
import numpy as np
from scipy import stats

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
R = HERE.parent / "results"
NMAX = 2000


def gather():
    warnings.filterwarnings("ignore")
    import exp20_causal_chamber as e20, exp21_chamber_pairs as e21, exp27_chamber_fresh as e27, exp10_sachs as e10
    from exp12_tuebingen_cep import load_pairs
    from exp14_tuebingen_relfit import SEED, dequantise, MAX_N
    out = []
    for pid, x, y, w in load_pairs():
        rng = np.random.default_rng(SEED + int(pid))
        if len(x) > MAX_N:
            idx = np.sort(rng.choice(len(x), MAX_N, replace=False)); x, y = x[idx], y[idx]
        out.append(("tuebingen", pid, dequantise(x, rng), dequantise(y, rng)))
    for di in range(len(e20.DATASETS)):
        X, var, edges = e20.load(*e20.DATASETS[di])
        for (i, j) in edges:
            for seed in e21.SEEDS:
                rng = np.random.default_rng([seed, di, i, j]); idx = rng.permutation(len(X))[:NMAX]
                out.append(("chamber_iid", f"{di}/{i}-{j}", X[idx, i], X[idx, j]))
    sp = e27.specs(); seen = set()
    for k, s in enumerate(sp):
        key = (s[0], s[3]) if s[0] != "lt_walks_v1" else (s[0], s[1])
        if key in seen: continue
        seen.add(key)
        X, var, edges = e27.load(s)
        for (i, j) in edges:
            rng = np.random.default_rng([20261500, k, i, j]); idx = rng.permutation(len(X))[:NMAX]
            out.append(("chamber_walk", f"{k}/{i}-{j}", X[idx, i], X[idx, j]))
    X, names = e10.load_sachs(); X = np.log(np.maximum(X, 1e-3))
    for a, b in e10.CONSENSUS:
        out.append(("sachs", f"{a}->{b}", X[:, names.index(a)], X[:, names.index(b)]))
    return out


def marg(v):
    v = np.asarray(v, float); n = len(v)
    z = (v - v.min()) / max(v.max() - v.min(), 1e-12)
    s = np.sort(z); m = max(int(np.sqrt(n)), 1)
    sp = s[np.minimum(np.arange(n) + m, n - 1)] - s[np.maximum(np.arange(n) - m, 0)]
    ent = np.mean(np.log(np.maximum(sp * n / (2 * m), 1e-12)))
    q = np.quantile(z, [.01, .99])
    return np.array([stats.skew(z), stats.kurtosis(z), ent, np.log(len(np.unique(v)) / n), q[1] - q[0]])


def cv_acc(F, group, rng, reps=50):
    """F: (n, 5) antisymmetric features for the true orientation; both copies are used with +-F."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    ug = np.unique(group); accs = []
    for _ in range(reps):
        perm = rng.permutation(len(ug)); fold = {g: k % 5 for k, g in zip(range(len(ug)), ug[perm])}
        fi = np.array([fold[g] for g in group]); correct = np.zeros(len(F))
        for k in range(5):
            tr, te = fi != k, fi == k
            X = np.vstack([F[tr], -F[tr]]); y = np.r_[np.ones(tr.sum()), np.zeros(tr.sum())]
            sc = StandardScaler().fit(X); clf = LogisticRegression(C=1.0, max_iter=1000).fit(sc.transform(X), y)
            correct[te] = clf.decision_function(sc.transform(F[te])) > 0
        accs.append(correct.mean())
    return float(np.mean(accs)), float(np.std(accs))


def het(x, y, seed=0):
    """Cross-fitted additive-spline regression of y on x (6 knots, cubic, ridge 1), then |Spearman(|r|, x)|."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import SplineTransformer
    z = ((x - x.mean()) / x.std()).reshape(-1, 1); B = SplineTransformer(n_knots=6, degree=3).fit_transform(z)
    res = np.empty(len(x)); idx = np.random.default_rng(seed).permutation(len(x)); h = len(x) // 2
    for tr, te in ((idx[:h], idx[h:]), (idx[h:], idx[:h])):
        m = Ridge(alpha=1.0).fit(B[tr], y[tr]); res[te] = y[te] - m.predict(B[te])
    return float(abs(stats.spearmanr(np.abs(res), x)[0]))


def main():
    rng = np.random.default_rng(20262700)
    data = gather(); out = {"h1": {}, "h2": {}}
    sets = {}
    for s, pid, x, y in data:
        sets.setdefault(s, []).append((pid, marg(x) - marg(y)))
    rec = {k: v for k, v in json.load(open(R / "exp38_summary.json")).items()} if False else None
    ref = {"tuebingen": 0.608, "chamber_iid": 0.979, "chamber_walk": 0.943, "sachs": 0.529}   # RECI, from exp35
    for s, lst in sets.items():
        F = np.array([f for _, f in lst]); grp = np.array([p for p, _ in lst])
        a, sd = cv_acc(F, grp, rng)
        out["h1"][s] = {"n": len(lst), "marginal_only_cv_accuracy": a, "sd_over_splits": sd, "RECI": ref[s]}
        print("H1", s, len(lst), round(a, 3), "RECI", ref[s], flush=True)
    # H2
    from exp47_multiverse import load
    ids, w, n, S = load()
    M = list(S[ids[0]].keys())
    tue = {pid: (x, y) for s, pid, x, y in data if s == "tuebingen"}
    H = np.array([het(*tue[i]) for i in ids]); U = np.array([len(np.unique(np.r_[tue[i][0]])) / len(tue[i][0]) for i in ids])
    out["h2"]["methods"] = {}
    for m in M:
        sc = np.array([S[i][m] if S[i][m] is not None and S[i][m] == S[i][m] else np.nan for i in ids])
        ok = np.where(np.isnan(sc), np.nan, np.where(sc > 0, 1.0, np.where(sc < 0, 0.0, 0.5)))
        v = ~np.isnan(ok); good, bad = H[v & (ok == 1)], H[v & (ok == 0)]
        p = stats.mannwhitneyu(bad, good, alternative="greater").pvalue if len(good) and len(bad) else None
        pu = stats.mannwhitneyu(U[v & (ok == 0)], U[v & (ok == 1)], alternative="less").pvalue
        out["h2"]["methods"][m] = {"mean_h_wrong": float(bad.mean()), "mean_h_right": float(good.mean()), "p_h": float(p), "p_unique": float(pu), "n_wrong": int(len(bad)), "n_right": int(len(good))}
        print("H2", m, {k: round(x, 3) for k, x in out["h2"]["methods"][m].items()}, flush=True)
    import statsmodels.api as sm
    rows = []
    for m in ("ANM-HSIC", "ANM-GP", "RESIT", "LOCI"):
        sc = np.array([S[i][m] if S[i][m] is not None and S[i][m] == S[i][m] else np.nan for i in ids])
        ok = np.where(np.isnan(sc), np.nan, np.where(sc > 0, 1.0, np.where(sc < 0, 0.0, 0.5)))
        for k in range(len(ids)):
            if ok[k] in (0.0, 1.0): rows.append((ok[k], H[k], np.log(n[k])))
    A = np.array(rows); Xz = (A[:, 1:] - A[:, 1:].mean(0)) / A[:, 1:].std(0)
    fit = sm.Logit(A[:, 0], sm.add_constant(Xz)).fit(disp=0)
    out["h2"]["pooled_logit"] = {"coef_h_logn": fit.params[1:].tolist(), "p": fit.pvalues[1:].tolist(), "n": len(A)}
    print("pooled", out["h2"]["pooled_logit"])
    (R / "exp49_explain.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
