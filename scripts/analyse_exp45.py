import json, collections, numpy as np
e20=json.load(open('results/exp20_causal_chamber.json')); e30=json.load(open('results/exp30_btad_chambers.json'))['records']
diag=json.load(open('results/serial_diagnostic.json')) if __import__('os').path.exists('results/serial_diagnostic.json') else []
ess={tuple(r['spec'][:2]):r['ess'] for r in diag if r.get('ess') is not None}
def sc(E,T):
    E={tuple(sorted(p)) for p in E}; tp=len(E&T); fp=len(E-T)
    P=tp/max(tp+fp,1); R=tp/len(T); return [P,R,2*P*R/(P+R) if P+R else 0,fp]
d=json.load(open('results/exp45_sota_graphs.json'))['records'] if __import__('os').path.exists('results/exp45_sota_graphs.json') else json.load(open('results/exp45_sota_graphs.partial.json'))['records']
cells=collections.defaultdict(lambda: collections.defaultdict(list)); to=collections.Counter()
for r in d:
    if r['kind']=='sachs': T={tuple(sorted(e)) for e in r['truth']}; cell='Sachs'
    elif r['kind']=='iid':
        T={tuple(sorted(e)) for e in [x for x in e20['records'] if x['ds']==r['ds']][0]['truth']}; cell='Chambers iid'
    else:
        rec=[x for x in e30 if tuple(x['spec'][:2])==tuple(r['spec'][:2])][0]; T={tuple(sorted(e)) for e in rec['truth']}
        e=ess.get(tuple(r['spec'][:2])); cell='ESS>1000' if e and e>1000 else 'ESS<20' if e and e<20 else 'other walk'
    for m,v in r['edges'].items():
        if m.endswith(' err'): continue
        if v is None: to[(cell,m)]+=1; continue
        cells[cell][m].append(sc(v,T))
out={}
for cell,ms in cells.items():
    print(cell)
    for m,v in ms.items():
        a=np.mean(v,0); out.setdefault(cell,{})[m]=a.tolist()+[len(v),to[(cell,m)]]
        print(f"   {m:12s} P {a[0]:.2f} R {a[1]:.2f} F1 {a[2]:.2f} fp {a[3]:.1f} (n={len(v)}, timeouts {to[(cell,m)]})")
json.dump(out,open('results/exp45_summary.json','w'),indent=1)
