"""Experiment 30: BTAD on the untouched Causal Chamber recordings, time-ordered.

Written and committed before running.  Same 19 recordings and variable rule as exp27
(none used to design BTAD; BTAD's settings were fixed on simulated data: K = 60,
alpha = .2, max blanket 8, gap .05).  Unlike exp27 the rows are NOT shuffled: every
method sees the first 4000 rows of the recording in time order, so that serial
dependence is present for all of them.  Methods: BTAD, PC (Fisher-z .05, .01), GES-BIC,
DirectLiNGAM, PC with the iid nonparametric test (.05, .01), and CERT-CD with the
nonparametric primitive (k = 3, streamed over the same rows).
Reference set V_block: non-adjacent pairs whose graph-implied conditional independence
is rejected by a block test (40 contiguous blocks of the LAST 4000 rows of the recording,
t statistic of the block means of the residual products, |t| > 4); an iid version
(|z| > 5 on the same rows) is also stored.  Scores: strict (graph edges) and adjusted
(graph edges plus V_block), precision, recall, F1.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import itertools, json, pathlib, sys, warnings
from multiprocessing import Pool
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from sprint_cd.btad import btad_skeleton
from sprint_cd.npadj import NPAdjacency
from sprint_cd.pc_gcm import pc_gcm_skeleton
import exp27_chamber_fresh as e27
from exp20_causal_chamber import pc, ges, lingam
from exp23_chamber_nonparam import _resid

NROWS, NB = 4000, 40


def reference(X, edges, d):
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    A = np.zeros((d, d), bool)
    for a, b in edges:
        A[a, b] = True
    R = A.copy()
    for _ in range(d):
        R = R | (R.astype(int) @ A.astype(int) > 0)
    Vb, Vi = [], []
    for i, j in itertools.combinations(range(d), 2):
        if A[i, j] or A[j, i]:
            continue
        if R[i, j]:
            i, j = j, i
        if R[i, j]:
            continue
        pa = list(np.flatnonzero(A[:, i]))
        z = _resid(X[:, i], X[:, pa]) * _resid(X[:, j], X[:, pa])
        iid = z.mean() / z.std() * np.sqrt(len(z))
        bm = np.array([b.mean() for b in np.array_split(z, NB)])
        tb = bm.mean() / (bm.std(ddof=1) / np.sqrt(NB) + 1e-12)
        if abs(iid) > 5:
            Vi.append([min(i, j), max(i, j), float(iid)])
        if abs(tb) > 4:
            Vb.append([min(i, j), max(i, j), float(tb)])
    return Vb, Vi


def one(idx):
    warnings.filterwarnings("ignore")
    spec = e27.specs()[idx]
    X, var, edges = e27.load(spec)
    XA, XB = X[:NROWS], X[-NROWS:]
    Z = (XA - XA[:100].mean(0)) / (XA[:100].std(0, ddof=1) + 1e-12)
    d = Z.shape[1]
    Vb, Vi = reference(XB, edges, d)
    res = {"spec": list(spec), "vars": var, "truth": [list(e) for e in edges], "V_block": Vb, "V_iid": Vi, "m": {}}

    def np_cert():
        a = NPAdjacency(d, 0.05 / (d * (d - 1) / 2), k=3, warm=100)
        for s in range(0, NROWS, 100):
            a.update(Z[s:s + 100])
        return [list(p) for p in a.certified_edges()]

    for name, fn in (("BTAD", lambda: btad_skeleton(Z, K=60, alpha=0.2, max_blanket=8)),
                     ("PC a=0.05", lambda: pc(Z, .05)), ("PC a=0.01", lambda: pc(Z, .01)),
                     ("GES-BIC", lambda: ges(Z)), ("DirectLiNGAM", lambda: lingam(Z)),
                     ("PC-GCM a=0.05", lambda: pc_gcm_skeleton(Z, .05, 3)),
                     ("PC-GCM a=0.01", lambda: pc_gcm_skeleton(Z, .01, 3)),
                     ("CERT-CD nonparametric k=3", np_cert)):
        v, e = e27.guarded(fn, 900)
        res["m"][name] = {"edges": v, "err": e}
    return res


def main():
    n = len(e27.specs())
    part = HERE.parent / "results" / "exp30_btad_chambers.partial.json"
    recs = json.loads(part.read_text())["records"] if part.exists() else []
    done = {tuple(r["spec"][:2]) for r in recs}
    sp = e27.specs()
    jobs = [i for i in range(n) if tuple(sp[i][:2]) not in done]
    with Pool(4) as p:
        for rec in p.imap_unordered(one, jobs, chunksize=1):
            recs.append(rec)
            part.write_text(json.dumps({"records": recs}))
            print(len(recs), "/", n, flush=True)
    (HERE.parent / "results" / "exp30_btad_chambers.json").write_text(json.dumps({"records": recs}))
    part.unlink()


if __name__ == "__main__":
    main()
