"""Experiment 20: real physical data with a known causal graph (Causal Chambers).

Protocol, fixed before running.

Data (Gamella, Peters, Buhlmann, Nature Mach. Intell. 2025), downloaded on first
use: (a) wind tunnel, dataset wt_bernoulli_v1, experiment random_loads_intake;
(b) light tunnel, dataset lt_interventions_standard_v1, experiment
uniform_reference.  (The wind-tunnel experiment random_loads_both was dropped
before any result was seen, because load_in and load_out are identical.)  Variables
are the columns that appear in the chamber's ground-truth graph and take at
least 20 distinct values in the experiment (constant settings are dropped,
which creates no latent confounder because every chamber variable is a node).
The truth is the chamber graph restricted to these variables.

Replications: R random subsamples of NMAX rows (rows are randomly permuted
first, so serial dependence of the recording is broken), nested prefixes
n = 250, 500, 1000, 2000, 4000.  CERT-CD (alpha = .1, k = 2, orientation off
inside the stream), PC (.05, .01), GES-BIC, DirectLiNGAM, as in exp13.
Orientation of each certified edge is assessed separately by the bivariate
relative-fit certificate on the pair (marginal, at most 2000 rows).
Reported: P(ever a false edge) over the grid, precision/recall of the skeleton
at the final n, and orientation counts.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse, itertools, json, pathlib, signal, sys, time, warnings
from multiprocessing import Pool
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from sprint_cd.certcd import CertCD, CertCDConfig
from sprint_cd.relative_fit import certify_direction

GRID = [250, 500, 1000, 2000, 4000]
NMAX = 4000
WARM = 50
BATCH = 100
SEED = 20261030
# random_loads_both is excluded before any result was seen: load_in and load_out
# are identical in that experiment (correlation 1.0), so the two nodes cannot be
# distinguished from data and the correlation matrix is singular.
DATASETS = [("wt_bernoulli_v1", "random_loads_intake", "wt"),
            ("lt_interventions_standard_v1", "uniform_reference", "lt")]
ROOT = HERE.parent / "data" / "causalchamber"


def load(ds, ex, ch):
    import causalchamber.ground_truth as gt
    from causalchamber.datasets import Dataset
    d = Dataset(ds, root=str(ROOT), download=True)
    df = d.get_experiment(ex).as_pandas_dataframe()
    G = gt.graph(ch, "standard")
    var = [c for c in df.columns if c in G.columns and df[c].nunique() >= 20]
    sub = G.loc[var, var]
    edges = [(var.index(a), var.index(b)) for a in var for b in var if sub.loc[a, b] == 1]
    return df[var].to_numpy(float), var, edges


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
        return None, f"{type(e).__name__}: {e}"[:160]
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def adj(M):
    M = np.asarray(M)
    return [[i, j] for i, j in itertools.combinations(range(M.shape[0]), 2) if M[i, j] != 0 or M[j, i] != 0]


def pc(X, a):
    from causallearn.search.ConstraintBased.PC import pc as _pc
    from causallearn.utils.cit import fisherz
    return adj(_pc(X, a, indep_test=fisherz, show_progress=False).G.graph)


def ges(X):
    from causallearn.search.ScoreBased.GES import ges as _g
    return adj(_g(X, score_func="local_score_BIC")["G"].graph)


def lingam(X):
    from causallearn.search.FCMBased.lingam import DirectLiNGAM
    m = DirectLiNGAM(); m.fit(X)
    B = np.asarray(m.adjacency_matrix_); d = X.shape[1]
    return [[i, j] for i, j in itertools.combinations(range(d), 2) if abs(B[i, j]) > .05 or abs(B[j, i]) > .05]


def stream(Xs):
    d = Xs.shape[1]
    a = CertCD(d, CertCDConfig(alpha=0.1, max_order=2, warmup=WARM, orient=False))
    a.warm_up(Xs[:WARM]); cur, out = WARM, []
    for n in GRID:
        for s in range(cur, n, BATCH):
            a.update(Xs[s:min(s + BATCH, n)])
        cur = n
        out.append([list(p) for p in a.certified_edges()])
    return out


def one(job):
    di, r = job
    warnings.filterwarnings("ignore")
    X, var, edges = load(*DATASETS[di])
    rng = np.random.default_rng([SEED, di, r])
    idx = rng.permutation(len(X))[:NMAX]
    X = X[idx]
    Xs = (X - X[:100].mean(0)) / (X[:100].std(0, ddof=1) + 1e-12)
    d = Xs.shape[1]
    res = {"ds": di, "r": r, "vars": var, "truth": [list(e) for e in edges], "methods": {}}
    t = time.time()
    v, e = guarded(lambda: stream(Xs), 3000)
    res["methods"]["CERT-CD"] = {"edges": v, "err": e, "sec": time.time() - t}
    for name, fn in (("PC a=0.05", lambda X: pc(X, .05)), ("PC a=0.01", lambda X: pc(X, .01)),
                     ("GES-BIC", ges), ("DirectLiNGAM", lingam)):
        o, er = [], []
        for n in GRID:
            v, e = guarded(lambda: fn(Xs[:n]), 300)
            o.append(v); er.append(e)
        res["methods"][name] = {"edges": o, "err": er}
    # orientation of certified edges at the final n
    ori = []
    if res["methods"]["CERT-CD"]["edges"]:
        T = {e for e in edges}
        for i, j in res["methods"]["CERT-CD"]["edges"][-1]:
            truth = "x->y" if (i, j) in T else ("y->x" if (j, i) in T else None)
            if truth is None:
                continue
            c = certify_direction(Xs[:2000, i], Xs[:2000, j], seed=SEED + 17 * i + j)
            ori.append({"pair": [i, j], "truth": truth, "res": c})
    res["orientation"] = ori
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    part = HERE.parent / "results" / "exp20_causal_chamber.partial.json"
    recs = json.loads(part.read_text())["records"] if part.exists() else []
    done = {(r["ds"], r["r"]) for r in recs}
    jobs = [(di, r) for di in range(len(DATASETS)) for r in range(a.reps) if (di, r) not in done]
    for di in range(len(DATASETS)):
        load(*DATASETS[di])  # download once before the pool starts
    with Pool(a.workers) as p:
        for rec in p.imap_unordered(one, jobs, chunksize=1):
            recs.append(rec)
            part.write_text(json.dumps({"records": recs}))
            print(len(recs), flush=True)
    out = {"config": {"reps": a.reps, "grid": GRID, "nmax": NMAX, "seed": SEED, "datasets": DATASETS},
           "records": sorted(recs, key=lambda r: (r["ds"], r["r"]))}
    (HERE.parent / "results" / "exp20_causal_chamber.json").write_text(json.dumps(out))
    part.unlink()


if __name__ == "__main__":
    main()
