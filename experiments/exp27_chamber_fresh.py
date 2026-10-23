"""Experiment 27: untouched Causal Chamber recordings, final algorithm frozen.

Written and committed before running.  None of these recordings was used in
exp20-exp25.  The algorithm settings are the ones in the paper: CERT-CD with the
nonparametric primitive (k = 3, alpha_A = .05, orientation off), CERT-CD with the
linear primitive (k = 3), PC (Fisher-z, alpha .05 and .01), GES-BIC, DirectLiNGAM,
and PC with the nonparametric test of sprint_cd.pc_gcm (depth <= 3, alpha .05, .01).

Recordings (rule fixed in advance): all experiments of wt_walks_v1 named
actuators_random_walk_* with >= 8000 rows (15 independent runs of the wind tunnel,
all actuators random-walking), wt_pressure_control_v1/hatch_0 (pressure-control
configuration), and every experiment of lt_walks_v1 (light tunnel); all with >= 8000 rows so that each half has >= 4000.
Variables: columns of the chamber graph with >= 20 distinct values; of two
columns with |corr| >= 0.9999 the second is dropped (duplicate measurements).
Rows are shuffled once (seed fixed) and split in halves: the algorithms see
4000 rows drawn from half A (nested prefixes 250 ... 4000); half B (up to 20000
rows) is used only to build the reference set V of non-adjacent pairs whose
graph-implied conditional independence (parents of the non-descendant, additive
spline regression) is rejected at |z| > 5.  V is therefore independent of the
rows the algorithms monitor.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import itertools, json, pathlib, signal, sys, time, warnings
from multiprocessing import Pool
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from sprint_cd.certcd import CertCD, CertCDConfig
from sprint_cd.npadj import NPAdjacency
from sprint_cd.pc_gcm import pc_gcm_skeleton
import exp20_causal_chamber as e20
from exp23_chamber_nonparam import _resid

ROOT = e20.ROOT
GRID = [250, 500, 1000, 2000, 4000]
SEED = 20261111
K = 3


_SPECS = []


def specs():
    if _SPECS:
        return _SPECS
    from causalchamber.datasets import Dataset
    out = []
    d = Dataset("wt_walks_v1", root=str(ROOT), download=True)
    for e in sorted(d.available_experiments()):
        if e.startswith("actuators_random_walk_"):
            out.append(("wt_walks_v1", e, "wt", "standard"))
    out.append(("wt_pressure_control_v1", "hatch_0", "wt", "pressure-control"))
    d = Dataset("lt_walks_v1", root=str(ROOT), download=True)
    for e in sorted(d.available_experiments()):
        out.append(("lt_walks_v1", e, "lt", "standard"))
    out = [sp for sp in out if _rows(sp) >= 8000]
    _SPECS.extend(out)
    return _SPECS


def _rows(sp):
    from causalchamber.datasets import Dataset
    return len(Dataset(sp[0], root=str(ROOT), download=True).get_experiment(sp[1]).as_pandas_dataframe())


def load(spec):
    import causalchamber.ground_truth as gt
    from causalchamber.datasets import Dataset
    ds, ex, ch, cfg = spec
    df = Dataset(ds, root=str(ROOT), download=True).get_experiment(ex).as_pandas_dataframe()
    G = gt.graph(ch, cfg)
    var = [c for c in df.columns if c in G.columns and df[c].nunique() >= 20]
    X = df[var].to_numpy(float)
    keep = []
    C = np.corrcoef(X.T)
    for i in range(len(var)):
        if all(abs(C[i, j]) < 0.9999 for j in keep):
            keep.append(i)
    var = [var[i] for i in keep]
    X = X[:, keep]
    sub = G.loc[var, var]
    edges = [(var.index(a), var.index(b)) for a in var for b in var if sub.loc[a, b] == 1]
    return X, var, edges


class _TO(Exception):
    pass


def guarded(fn, sec):
    def h(*_):
        raise _TO()
    signal.signal(signal.SIGALRM, h)
    signal.setitimer(signal.ITIMER_REAL, sec)
    try:
        return fn(), None
    except _TO:
        return None, "timeout"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"[:150]
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def reference_set(X, edges, d):
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    A = np.zeros((d, d), bool)
    for a, b in edges:
        A[a, b] = True
    R = A.copy()
    for _ in range(d):
        R = R | (R.astype(int) @ A.astype(int) > 0)
    V = []
    for i, j in itertools.combinations(range(d), 2):
        if A[i, j] or A[j, i]:
            continue
        if R[i, j]:
            i, j = j, i
        if R[i, j]:
            continue
        pa = list(np.flatnonzero(A[:, i]))
        z = _resid(X[:, i], X[:, pa]) * _resid(X[:, j], X[:, pa])
        t = z.mean() / z.std() * np.sqrt(len(z))
        if abs(t) > 5:
            V.append([min(i, j), max(i, j), float(t)])
    return V


def one(idx):
    warnings.filterwarnings("ignore")
    spec = specs()[idx]
    X, var, edges = load(spec)
    n = len(X)
    perm = np.random.default_rng([SEED, idx]).permutation(n)
    A_rows, B_rows = perm[: n // 2], perm[n // 2:][:20000]
    XA = X[A_rows][:4000]
    Xs = (XA - XA[:100].mean(0)) / (XA[:100].std(0, ddof=1) + 1e-12)
    d = Xs.shape[1]
    res = {"spec": list(spec), "vars": var, "truth": [list(e) for e in edges], "n_total": n,
           "reference": reference_set(X[B_rows], edges, d), "methods": {}}
    from exp20_causal_chamber import pc, ges, lingam

    def stream_np():
        a = NPAdjacency(d, 0.05 / (d * (d - 1) / 2), k=K, warm=100)
        cur, out = 0, []
        for g in GRID:
            for s in range(cur, g, 100):
                a.update(Xs[s:min(s + 100, g)])
            cur = g
            out.append([list(p) for p in a.certified_edges()])
        return out

    def stream_lin():
        a = CertCD(d, CertCDConfig(alpha=0.1, max_order=K, warmup=50, orient=False))
        a.warm_up(Xs[:50]); cur, out = 50, []
        for g in GRID:
            for s in range(cur, g, 100):
                a.update(Xs[s:min(s + 100, g)])
            cur = g
            out.append([list(p) for p in a.certified_edges()])
        return out

    for name, fn in (("CERT-CD nonparametric k=3", stream_np), ("CERT-CD linear k=3", stream_lin)):
        v, e = guarded(fn, 3000)
        res["methods"][name] = {"edges": v, "err": e}
    per_n = {"PC a=0.05": lambda X: pc(X, .05), "PC a=0.01": lambda X: pc(X, .01),
             "GES-BIC": ges, "DirectLiNGAM": lingam,
             "PC-GCM a=0.05": lambda X: pc_gcm_skeleton(X, .05, 3),
             "PC-GCM a=0.01": lambda X: pc_gcm_skeleton(X, .01, 3)}
    for name, fn in per_n.items():
        o, er = [], []
        for g in GRID:
            v, e = guarded(lambda: fn(Xs[:g]), 600)
            o.append(v); er.append(e)
        res["methods"][name] = {"edges": o, "err": er}
    return res


def main():
    n = len(specs())
    part = HERE.parent / "results" / "exp27_chamber_fresh.partial.json"
    recs = json.loads(part.read_text())["records"] if part.exists() else []
    done = {tuple(r["spec"][:2]) for r in recs}
    allspecs = specs()
    jobs = [i for i in range(n) if tuple(allspecs[i][:2]) not in done]
    with Pool(4) as p:
        for rec in p.imap_unordered(one, jobs, chunksize=1):
            recs.append(rec)
            part.write_text(json.dumps({"records": recs}))
            print(len(recs), "/", n, flush=True)
    (HERE.parent / "results" / "exp27_chamber_fresh.json").write_text(
        json.dumps({"grid": GRID, "k": K, "records": recs}))
    part.unlink()


if __name__ == "__main__":
    main()
