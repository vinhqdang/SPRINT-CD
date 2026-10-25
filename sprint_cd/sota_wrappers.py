"""Skeleton wrappers around published causal-discovery packages, library defaults unless stated.

Every function takes a standardised data matrix and returns a list of [i, j] (i < j) skeleton edges.
AVICI runs in a separate virtual environment (pinned old JAX) at /tmp/venv_avici.
"""
import itertools, os, subprocess, tempfile
import numpy as np

AVICI_PY = "/tmp/venv_avici/bin/python"
_AV = r'''
import sys, numpy as np, avici
m = avici.load_pretrained(download="scm-v0")
x = np.load(sys.argv[1]); np.save(sys.argv[2], np.asarray(m(x=x)))
'''


def _skel(W, thr=0.0):
    W = np.nan_to_num(np.asarray(W, float))
    d = W.shape[0]
    return [[i, j] for i, j in itertools.combinations(range(d), 2) if abs(W[i, j]) > thr or abs(W[j, i]) > thr]


def avici_skeleton(X, thr=0.5, max_n=2000):
    X = np.asarray(X, float)
    if len(X) > max_n:
        X = X[np.random.default_rng(0).choice(len(X), max_n, replace=False)]
    with tempfile.TemporaryDirectory() as t:
        np.save(f"{t}/x.npy", X)
        subprocess.run([AVICI_PY, "-c", _AV, f"{t}/x.npy", f"{t}/g.npy"], check=True, capture_output=True, timeout=1500)
        P = np.load(f"{t}/g.npy")
    return _skel(P, thr)


def golem_skeleton(X):
    from castle.algorithms import GOLEM
    m = GOLEM(num_iter=10000); m.learn(X)
    return _skel(m.causal_matrix)


def notears_mlp_skeleton(X):
    from castle.algorithms import NotearsNonlinear
    m = NotearsNonlinear(); m.learn(X)
    return _skel(m.causal_matrix)


def grandag_skeleton(X):
    from castle.algorithms import GraNDAG
    m = GraNDAG(input_dim=X.shape[1]); m.learn(X)
    return _skel(m.causal_matrix)


def daggnn_skeleton(X):
    from castle.algorithms import DAG_GNN
    m = DAG_GNN(); m.learn(X)
    return _skel(m.causal_matrix)


def _cl(X, fn):
    G = fn(X, score_func="local_score_BIC")
    return _skel(G.graph if hasattr(G, "graph") else G["G"].graph)


def grasp_skeleton(X):
    from causallearn.search.PermutationBased.GRaSP import grasp
    return _cl(X, grasp)


def boss_skeleton(X):
    from causallearn.search.PermutationBased.BOSS import boss
    return _cl(X, boss)


def resit_skeleton(X):
    import lingam
    from sklearn.ensemble import RandomForestRegressor
    m = lingam.RESIT(regressor=RandomForestRegressor(max_depth=4, random_state=0), alpha=0.01); m.fit(X)
    return _skel(m.adjacency_matrix_)


def camuv_skeleton(X):
    import lingam
    m = lingam.CAMUV(alpha=0.01, num_explanatory_vals=3); m.fit(X)
    return _skel(m.adjacency_matrix_)


METHODS = {"AVICI": avici_skeleton, "GOLEM": golem_skeleton, "NOTEARS-MLP": notears_mlp_skeleton,
           "GraN-DAG": grandag_skeleton, "DAG-GNN": daggnn_skeleton, "GRaSP": grasp_skeleton, "BOSS": boss_skeleton,
           "RESIT": resit_skeleton, "CAM-UV": camuv_skeleton}
