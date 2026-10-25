import sys, json, numpy as np
sys.path.insert(0,'.'); sys.path.insert(0,'experiments')
from exp47_multiverse import load
import exp49_explain as e49
from exp12_tuebingen_cep import load_pairs
from exp14_tuebingen_relfit import SEED, dequantise, MAX_N
ids,w,n,S=load(); M=list(S[ids[0]].keys())
tue={}
for pid,x,y,wt in load_pairs():
    rng=np.random.default_rng(SEED+int(pid))
    if len(x)>MAX_N:
        idx=np.sort(rng.choice(len(x),MAX_N,replace=False)); x,y=x[idx],y[idx]
    tue[pid]=(dequantise(x,rng),dequantise(y,rng))
H=np.array([e49.het(*tue[i]) for i in ids]); q=np.quantile(H,[1/3,2/3]); st=np.digitize(H,q)
out={"h_tercile_bounds":q.tolist(),"acc":{}}
print("h tercile bounds",np.round(q,3))
for m in M:
    sc=np.array([S[i][m] if S[i][m] is not None and S[i][m]==S[i][m] else np.nan for i in ids])
    ok=np.where(np.isnan(sc),np.nan,np.where(sc>0,1.0,np.where(sc<0,0.0,0.5)))
    r=[float(np.nanmean(ok[st==k])) for k in range(3)]; out["acc"][m]=r
    print(f"{m:16s}",[round(x,2) for x in r])
json.dump(out,open('results/exp49_strata.json','w'),indent=1)
