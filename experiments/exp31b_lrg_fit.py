"""Fit the LRG classifier on exp31 features; choose the threshold on validation; report held-out simulation metrics."""
import json, pathlib, sys
import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = pathlib.Path(__file__).resolve().parent.parent
z = np.load(ROOT / "results" / "exp31_features.npz")
seed, F, y = z["seed"], z["F"], z["y"]
ntr, nva, nte = int(z["n_train"]), int(z["n_val"]), int(z["n_test"])
grp = lambda lo, hi: (seed >= lo) & (seed < hi)
tr, va, te = grp(0, ntr), grp(ntr, ntr + nva), grp(ntr + nva, ntr + nva + nte)
clf = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=300, l2_regularization=1.0, random_state=0)
clf.fit(F[tr], y[tr])


def prf_by_dataset(mask, thr, p):
    out = []
    for s in np.unique(seed[mask]):
        m = mask & (seed == s)
        pred = p[m] >= thr; t = y[m] == 1
        tp = (pred & t).sum(); fp = (pred & ~t).sum(); fn = (~pred & t).sum()
        P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
        out.append([P, R, 2 * P * R / (P + R) if P + R else 0.0])
    return np.mean(out, axis=0)


pv = clf.predict_proba(F[va])[:, 1]
p_all = np.zeros(len(y)); p_all[va] = pv
thr_grid = np.linspace(0.1, 0.9, 33)
f1s = [prf_by_dataset(va, t, p_all)[2] for t in thr_grid]
thr = float(thr_grid[int(np.argmax(f1s))])
p_all[te] = clf.predict_proba(F[te])[:, 1]
res = {"threshold": thr, "val_F1": float(max(f1s)), "test_P_R_F1": prf_by_dataset(te, thr, p_all).tolist(),
       "test_P_R_F1_at_0.5": prf_by_dataset(te, 0.5, p_all).tolist(), "positives_train": float(y[tr].mean())}
joblib.dump({"model": clf, "threshold": thr}, ROOT / "results" / "lrg_model.joblib")
(ROOT / "results" / "exp31_summary.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
