import json
r=json.load(open('results/exp46_pairs_sota.json'))['records']
for st in ['tuebingen','chamber_iid','chamber_walk','sachs']:
    R=[x for x in r if x['set']==st]
    for m in ['AVICI','RESIT']:
        c=w=n=W=0.0
        for x in R:
            s=x['m'].get(m)
            if s is None or s!=s: continue
            ok=(0.5 if s==0 else float((s>0)==(x['truth']=='x->y')))
            wt=x.get('weight',1) or 1
            c+=ok; w+=ok*wt; n+=1; W+=wt
        print(f"{st:13s}{m:6s} n={int(n)}/{len(R)} acc={c/max(n,1):.3f}"+(f" weighted={w/W:.3f}" if st=='tuebingen' else ""))
