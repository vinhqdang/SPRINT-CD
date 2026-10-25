import sys, json, numpy as np, pandas as pd, itertools, networkx as nx
from scipy.stats import ks_2samp
import causalchamber.ground_truth as gt
from causalchamber.datasets import Dataset
from sklearn.metrics import roc_auc_score
d=Dataset('lt_interventions_standard_v1',root='data/causalchamber',download=True)
G=gt.graph('lt','standard')
ref=d.get_experiment('uniform_reference').as_pandas_dataframe()
var=[c for c in G.columns if c in ref.columns and ref[c].nunique()>=20]
D=nx.DiGraph(); D.add_nodes_from(G.columns)
for a in G.columns:
    for b in G.columns:
        if G.loc[a,b]==1: D.add_edge(a,b)
anc={(a,b) for a in var for b in var if a!=b and nx.has_path(D,a,b)}
direct={(a,b) for a in var for b in var if D.has_edge(a,b)}
print(len(var),'ancestral pairs',len(anc),'direct in G among obs',len(direct))
tg={}
for e in d.available_experiments():
    if e=='uniform_reference': continue
    t=e.replace('uniform_','').rsplit('_',1)[0]
    if t in var: tg.setdefault(t,[]).append(e)
print({t:len(v) for t,v in tg.items()})
R=ref[var].to_numpy(float)
y=[];s=[]
for t,es in tg.items():
    Xi=np.vstack([d.get_experiment(e).as_pandas_dataframe()[var].to_numpy(float) for e in es])
    for v in var:
        if v==t: continue
        y.append((t,v) in anc); s.append(ks_2samp(R[:,var.index(v)],Xi[:,var.index(v)]).statistic)
print('KS-shift AUC for ancestors:',roc_auc_score(y,s),'pos',sum(y),'of',len(y))
P=[];Y=[]
for t,es in tg.items():
    Xi=np.vstack([d.get_experiment(e).as_pandas_dataframe()[var].to_numpy(float) for e in es])
    for v in var:
        if v==t: continue
        P.append(ks_2samp(R[:,var.index(v)],Xi[:,var.index(v)]).pvalue); Y.append((t,v) in anc)
P=np.array(P);Y=np.array(Y)
for a in (0.05,0.01,1e-4,1e-6):
    pr=P<a/len(P)*1 if a<0.05 else P<a/len(P)
    tp=(pr&Y).sum(); print('bonf',a,'pred',pr.sum(),'tp',tp,'prec',tp/max(pr.sum(),1),'rec',tp/Y.sum())
o=np.argsort(P); print('top39 hits',Y[o[:39]].sum())
print('--- missed by marginal KS:')
pairs=[(t,v) for t,es in tg.items() for v in var if v!=t]
for (t,v),p,yy in zip(pairs,P,Y):
    if yy and p>=0.05/len(P): print(t,v,'p=%.3g'%p)
from sklearn.ensemble import HistGradientBoostingRegressor as H
import warnings; warnings.filterwarnings('ignore')
res=[]
shift={}
for t,es in tg.items():
    Xi=np.vstack([d.get_experiment(e).as_pandas_dataframe()[var].to_numpy(float) for e in es]); 
    ps={v:ks_2samp(R[:,var.index(v)],Xi[:,var.index(v)]).pvalue for v in var if v!=t}
    nd=[v for v in var if v!=t and ps[v]>1e-3]  # estimated non-descendants
    Pc=[]
    for v in var:
        if v==t: continue
        Z=[var.index(u) for u in nd if u!=v]
        if not Z: Pc.append((t,v,ps[v])); continue
        m=H(max_iter=100,max_depth=3).fit(R[:5000][:,Z],R[:5000,var.index(v)])
        r0=R[5000:,var.index(v)]-m.predict(R[5000:][:,Z]); r1=Xi[:,var.index(v)]-m.predict(Xi[:,Z])
        Pc.append((t,v,ks_2samp(r0,r1).pvalue))
    res+=Pc
Pc=np.array([x[2] for x in res]); Yc=np.array([(x[0],x[1]) in anc for x in res])
for a in (0.05,):
    pr=Pc<a/len(Pc); tp=(pr&Yc).sum(); print('cond bonf pred',pr.sum(),'tp',tp,'prec',tp/max(pr.sum(),1),'rec',tp/Yc.sum())
print('AUC cond',roc_auc_score(Yc,-np.log(Pc+1e-300)))

json.dump({"n_vars": len(var), "ancestral_pairs": len(anc), "direct_pairs": len(direct), "tested_pairs": len(Y), "ks_auc": float(roc_auc_score(Y, -np.log(P + 1e-300))),
           "bonferroni_0.05": {"predicted": int((P < 0.05 / len(P)).sum()), "tp": int(((P < 0.05 / len(P)) & Y).sum()), "recall": float(((P < 0.05 / len(P)) & Y).sum() / Y.sum()), "precision": float(((P < 0.05 / len(P)) & Y).sum() / max((P < 0.05 / len(P)).sum(), 1))},
           "cond_bonferroni": {"predicted": int((Pc < 0.05 / len(Pc)).sum()), "tp": int(((Pc < 0.05 / len(Pc)) & Yc).sum()), "recall": float(((Pc < 0.05 / len(Pc)) & Yc).sum() / Yc.sum()), "precision": float(((Pc < 0.05 / len(Pc)) & Yc).sum() / max((Pc < 0.05 / len(Pc)).sum(), 1)), "auc": float(roc_auc_score(Yc, -np.log(Pc + 1e-300)))},
           "missed_pairs": [[a, b, float(p)] for (a, b), p, yy in zip(pairs, P, Y) if yy and p >= 0.05 / len(P)]}, open("results/lt_interventions_check.json", "w"), indent=1)
