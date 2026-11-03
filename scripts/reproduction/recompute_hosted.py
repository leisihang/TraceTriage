#!/usr/bin/env python3
"""Offline hosted scoring, validation-fitted calibration, and frozen mu sweep."""
import argparse,csv,json
from pathlib import Path
import numpy as np

CONFIGS={'claude_opus47_weak':('claude-opus-4.7','weak'),'claude_opus47_strong':('claude-opus-4.7','strong'),'gpt55_high_weak':('gpt-5.5-high','weak'),'gpt55_high_strong':('gpt-5.5-high','strong'),'deepseek_v4_flash_weak':('deepseek-v4-flash','weak'),'deepseek_v4_flash_strong':('deepseek-v4-flash','strong')}
def read(p):
    with p.open(newline='',encoding='utf8') as f:return list(csv.DictReader(f))
def write(p,rows):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def joined(root,cfg,ds,split):
    truth=read(root/'reproduction_inputs/truth'/ds/(split+'.csv'))
    pred=read(root/'reproduction_inputs/hosted'/cfg/ds/(split+'.csv'))
    byid={r['prefix_id']:r for r in pred}
    assert len(byid)==len(pred)==len(truth)
    assert set(byid)=={r['prefix_id'] for r in truth}
    truth.sort(key=lambda r:r['prefix_id'])
    y=np.array([int(r['strict_success']) for r in truth]);rem=np.array([float(r['remaining_seconds']) for r in truth])
    pp=[byid[r['prefix_id']] for r in truth]
    assert np.array_equal(y,[int(r['strict_success']) for r in pp])
    return y,rem,np.array([float(r['success_prob']) for r in pp]),np.array([int(r['pred_stop']) for r in pp],dtype=bool)
def metric(y,rem,p,stop,cost=0.,mu=0.):
    return dict(accuracy=float(np.mean((p>=.5)==y)),frs=float(stop[y==0].mean()) if (y==0).any() else 0.,fsr=float(stop[y==1].mean()) if (y==1).any() else 0.,net_noapi=float(np.mean(stop*(1-2*y)*rem-mu*(~stop)*(1-y)*rem)),net=float(np.mean(stop*(1-2*y)*rem-mu*(~stop)*(1-y)*rem)-cost))
def choose(y,rem,p,cap):
    best=None
    for i in range(96):
        t=i/100;m=metric(y,rem,p,p<t)
        if m['fsr']>cap:continue
        if best is None or m['net']>best[1]['net'] or (abs(m['net']-best[1]['net'])<1e-12 and m['fsr']<best[1]['fsr']):best=(t,m)
    assert best is not None
    return best[0]
def fit(mode,x,y):
    if mode=='raw':return lambda z:z
    if len(np.unique(y))==1:return lambda z:np.full(len(z),float(y[0]))
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    if mode=='platt':
        m=LogisticRegression(C=1e6,solver='lbfgs',max_iter=5000,tol=1e-10).fit(x.reshape(-1,1),y)
        return lambda z:m.predict_proba(z.reshape(-1,1))[:,1]
    m=IsotonicRegression(out_of_bounds='clip').fit(x,y)
    return m.predict

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--cap',type=float,default=.10);ap.add_argument('--raw-only',action='store_true');a=ap.parse_args()
    reference=read(a.root/'reference_results/hosted_canonical_per_dataset.csv')
    lookup={(r['model'],r['prompt'],r['dataset']):r for r in reference}
    detailed=[];sweep=[]
    for cfg,(model,prompt) in CONFIGS.items():
        for ds in sorted(p.name for p in (a.root/'reproduction_inputs/truth').iterdir() if p.is_dir()):
            yv,rv,pv,_=joined(a.root,cfg,ds,'val');yt,rt,pt,direct=joined(a.root,cfg,ds,'test')
            ref=lookup[(model,prompt,ds)];cost=float(ref['test_latency_s'])
            raw_tau=choose(yv,rv,pv,a.cap);raw_stop=pt<raw_tau
            raw=metric(yt,rt,pt,raw_stop,cost)
            if a.cap==.10:
                assert abs(raw_tau-float(ref['tau']))<1e-12,(cfg,ds,raw_tau,ref['tau'])
                assert abs(raw['net']-float(ref['net_charged']))<1e-7
            for mode in (('raw',) if a.raw_only else ('raw','platt','isotonic')):
                fun=fit(mode,pv,yv);cv,ct=fun(pv),fun(pt);tau=choose(yv,rv,cv,a.cap)
                m=metric(yt,rt,ct,ct<tau,cost)
                detailed.append(dict(config_id=cfg,calibration=mode,dataset=ds,threshold=tau,changed_decisions=int(np.count_nonzero((ct<tau)!=raw_stop)),**m))
            m=metric(yt,rt,pt,direct,cost)
            detailed.append(dict(config_id=cfg,calibration='direct',dataset=ds,threshold='',changed_decisions=int(np.count_nonzero(direct!=raw_stop)),**m))
            for mu in (0.,.25,.5,1.):sweep.append(dict(config_id=cfg,dataset=ds,mu=mu,**metric(yt,rt,pt,raw_stop,cost,mu)))
    macro=[]
    for cfg in CONFIGS:
        for mode in (('raw','direct') if a.raw_only else ('raw','platt','isotonic','direct')):
            rr=[r for r in detailed if r['config_id']==cfg and r['calibration']==mode]
            assert len(rr)==10
            macro.append(dict(config_id=cfg,calibration=mode,**{k:float(np.mean([r[k] for r in rr])) for k in ['accuracy','frs','fsr','net_noapi','net']},changed_decisions=sum(r['changed_decisions'] for r in rr)))
    if not a.raw_only and a.cap==.10:
        gold={(r['config_id'],r['calibration']):r for r in read(a.root/'reference_results/hosted_calibration.csv')}
        for row in macro:
            if row['calibration']=='direct':continue
            g=gold[row['config_id'],row['calibration']]
            for key,refkey in [('accuracy','macro_accuracy'),('frs','macro_frs'),('fsr','macro_fsr'),('net_noapi','macro_net_no_latency_s'),('net','macro_net_charged_s')]:
                assert abs(row[key]-float(g[refkey]))<1e-6,(row['config_id'],row['calibration'],key,row[key],g[refkey])
            assert row['changed_decisions']==int(g['test_rows_changed_vs_raw'])
    write(a.out/'hosted_per_dataset.csv',detailed);write(a.out/'hosted_macro.csv',macro);write(a.out/'hosted_mu_per_dataset.csv',sweep)
    for row in macro:print(row['config_id'],row['calibration'],round(row['net'],8),row['changed_decisions'])
if __name__=='__main__':main()
