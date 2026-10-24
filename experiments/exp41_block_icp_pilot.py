"""Experiment 41 (negative result): block-ICP pilot on five wind-tunnel walk recordings (development only).
Result: strict F1 about 0.00-0.07 (recall <= 0.04) at K = 20/40 blocks and alpha = .05/.2, far below the
pre-set criterion (F1 > 0.42); the method was abandoned and not evaluated on the other recordings."""
import sys, numpy as np, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0,'/home/user/SPRINT-CD'); sys.path.insert(0,'/home/user/SPRINT-CD/experiments')
import exp27_chamber_fresh as e27
from sprint_cd.blockicp import block_icp_skeleton
sp=e27.specs(); dev=[s for s in sp if s[1] in ('actuators_random_walk_10','actuators_random_walk_11','actuators_random_walk_12','actuators_random_walk_13','actuators_random_walk_14')]
res=[]
for spec in dev:
    X,var,edges=e27.load(spec); X=X[:4000]; Z=(X-X[:100].mean(0))/(X[:100].std(0,ddof=1)+1e-12)
    T={tuple(sorted(e)) for e in edges}
    for K,al in ((20,.05),(40,.05),(20,.2)):
        t=time.time(); E={tuple(sorted(p)) for p in block_icp_skeleton(Z,K=K,alpha=al)}
        tp=len(E&T); fp=len(E-T); P=tp/max(tp+fp,1); R=tp/len(T); F=2*P*R/(P+R) if P+R else 0
        res.append((spec[1],K,al,round(P,2),round(R,2),round(F,2),fp,round(time.time()-t,1)))
        print(res[-1],flush=True)
