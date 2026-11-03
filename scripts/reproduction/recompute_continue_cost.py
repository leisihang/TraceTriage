#!/usr/bin/env python3
"""Score frozen default Qwen/hosted policies under two-sided continue cost."""
import argparse,csv
from pathlib import Path
import numpy as np
from recompute_bootstrap import load
from recompute_hosted import CONFIGS,read,write,joined,choose,metric

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    data,_,_,_=load(a.root);rr=[]
    hosted_lookup={(r['model'],r['prompt'],r['dataset']):r for r in read(a.root/'reference_results/hosted_canonical_per_dataset.csv')}
    for mu in (0.,.25,.5,1.):
        seed_values=[]
        for j in range(3):
            cells=[]
            for ds,d in data.items():
                yy=np.array([int(x['strict_success']) for x in d['truth']]);rem=np.array([float(x['remaining_seconds']) for x in d['truth']])
                # Unstopped failures have zero default utility and negative utility when mu>0.
                base=a.root/'reproduction_inputs/qwen'/f'seed{(42,123,3407)[j]}'/ds
                import json
                setting=next(x for x in json.loads((base/'metrics_summary.json').read_text())['lambda_sweep'] if x['lambda_false_stop']==1)
                pred=sorted(read(base/'predictions.csv'),key=lambda x:x['prefix_id'])
                stop=np.array([float(x['success_prob'])<setting['stop_threshold'] for x in pred])
                cells.append(float(np.mean(d['values'][:,j]-mu*(~stop)*(1-yy)*rem)-d['costs'][j]))
            seed_values.append(float(np.mean(cells)))
        qwen=float(np.mean(seed_values));hosted=[]
        for cfg,(model,prompt) in CONFIGS.items():
            cells=[]
            for ds in data:
                yv,rv,pv,_=joined(a.root,cfg,ds,'val');yt,rt,pt,_=joined(a.root,cfg,ds,'test')
                tau=choose(yv,rv,pv,.10)
                cost=float(hosted_lookup[(model,prompt,ds)]['test_latency_s'])
                cells.append(metric(yt,rt,pt,pt<tau,cost,mu)['net'])
            hosted.append((float(np.mean(cells)),cfg))
        best,cfg=max(hosted)
        rr.append(dict(mu=mu,qwen_net=qwen,qwen_seed_sd=float(np.std(seed_values,ddof=1)),best_hosted_net=best,best_hosted_config=cfg,gap=qwen-best))
    expected=[67.11585142748324,35.2448620182859,3.373872609088551,-60.36810620930613]
    assert np.allclose([r['qwen_net'] for r in rr],expected,atol=1e-7,rtol=0)
    write(a.out/'continue_cost.csv',rr)
    for r in rr: print(r)
if __name__=='__main__':main()
