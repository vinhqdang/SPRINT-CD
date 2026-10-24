"""Fit the pair-direction classifier on exp34 features; report held-out simulated accuracy."""
import json, pathlib
import numpy as np, joblib
from sklearn.ensemble import HistGradientBoostingClassifier
ROOT = pathlib.Path(__file__).resolve().parent.parent
z = np.load(ROOT / "results" / "exp34_pair_features.npz")
F, y = z["F"], z["y"]; ntr, nva, nte = int(z["n_train"]), int(z["n_val"]), int(z["n_test"])
clf = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.05, max_iter=500, l2_regularization=1.0, random_state=0)
clf.fit(F[:ntr], y[:ntr])
acc = lambda s: float(np.mean((clf.predict_proba(F[s])[:, 1] > 0.5) == (y[s] == 1)))
va, te = slice(ntr, ntr + nva), slice(ntr + nva, ntr + nva + nte)
res = {"val_acc": acc(va), "test_acc": acc(te)}
p = clf.predict_proba(F[te])[:, 1]; conf = np.abs(p - 0.5); order = np.argsort(-conf)
for cov in (0.25, 0.5, 0.75):
    k = int(cov * len(p)); sel = order[:k]
    res[f"test_acc_top{int(cov*100)}"] = float(np.mean((p[sel] > 0.5) == (y[te][sel] == 1)))
joblib.dump(clf, ROOT / "results" / "pdm_model.joblib")
(ROOT / "results" / "exp34_summary.json").write_text(json.dumps(res, indent=1)); print(res)
