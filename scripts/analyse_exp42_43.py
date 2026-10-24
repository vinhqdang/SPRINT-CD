import json, itertools, numpy as np, sys
sys.path.insert(0,'.'); sys.path.insert(0,'experiments')
r=json.load(open('results/exp42_pairs_2026.json'))['records']
print("== pairs: accuracy (None -> excluded, ties half) ==")
for st in ['tuebingen','chamber_iid','chamber_walk','sachs']:
    R=[x for x in r if x['set']==st]
    for m in ['SC','SCraw','TRA']:
        c=w=n=0.0
        for x in R:
            s=x['m'].get(m)
            if s is None or s!=s: continue
            pred='x->y' if s>0 else 'y->x'
            ok=pred==x['truth'] if s!=0 else 0.5
            wt=x.get('weight',1) or 1
            c+=ok; w+=ok*wt; n+=1
        W=sum((x.get('weight',1) or 1) for x in R if x['m'].get(m) is not None)
        print(f"{st:13s}{m:6s} n={int(n):4d}/{len(R)} acc={c/max(n,1):.3f}"+(f" weighted={w/W:.3f}" if st=='tuebingen' else ""))
g=json.load(open('results/exp43_bfbic_boss.json'))
import exp10_sachs as e10, exp20_causal_chamber as e20, exp27_chamber_fresh as e27
def f1(E,T):
    E={tuple(sorted(q)) for q in E}; T={tuple(sorted(q)) for q in T}
    tp=len(E&T); p=tp/max(len(E),1); rc=tp/max(len(T),1)
    return 2*p*rc/(p+rc) if p+rc else 0, len(E), tp
s=g['sachs']; print("Sachs BF-BIC+BOSS F1,|E|,TP:",f1(s['edges']['BF-BIC+BOSS'] or [],s['truth']),"truth",len(s['truth']))
iid=[];walk=[]
for x in g['chambers']:
    if x['kind']=='iid':
        _,var,edges=e20.load(*e20.DATASETS[x['ds']]); iid.append(f1(x['edges'] or [],edges)[0])
    else:
        sp=tuple(x['spec']); _,var,edges=e27.load(sp); walk.append(f1(x['edges'] or [],edges)[0])
print("Chambers iid mean F1",np.mean(iid),len(iid),"walks mean strict F1",np.mean(walk),len(walk))
