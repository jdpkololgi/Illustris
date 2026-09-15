"""Read-only post-fit clean-input and antithetic-noise error localization."""
import argparse
import json
import math
import os
from pathlib import Path
import time

import h5py
import numpy as np
import torch
from workflows.sbi import e2e_wide_pipeline as p
from workflows.sbi.e2e_wide_continue import checked_binding
from workflows.sbi.e2e_wide_denoising_audit import Bands, TRAIN384, clean_estimate
from workflows.sbi.e2e_wide_research_canary import preflight, NORMALIZATION
from workflows.sbi.e2e_fixed_noise_test import build
from workflows.sbi.e2e_fixed_noise_report import verify


def decomposition(bands,plus_error,minus_error,clean_error):
    fp,fm,f0=map(bands.fft,(plus_error,minus_error,clean_error))
    even=(fp+fm)/2;odd=(fp-fm)/2
    power=lambda a:bands.cross(a,a)
    pe,po,p0=power(even),power(odd),power(f0)
    paired=(power(fp)+power(fm))/2
    np.testing.assert_allclose(paired,pe+po,rtol=1e-10,atol=1e-18)
    return dict(paired_error_power=paired.tolist(),even_error_power=pe.tolist(),odd_error_power=po.tolist(),
                clean_input_error_power=p0.tolist(),even_minus_clean_error_power=power(even-f0).tolist(),
                even_fraction=(pe/np.maximum(paired,1e-30)).tolist(),
                clean_power_over_paired=(p0/np.maximum(paired,1e-30)).tolist(),
                even_clean_correlation=(bands.cross(even,f0)/np.sqrt(np.maximum(pe*p0,1e-60))).tolist())


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('input',type=Path);ap.add_argument('output',type=Path)
    args=ap.parse_args();device=p.runtime();data=json.loads(args.input.read_text());root=args.input.parent
    if not data['complete'] or len(data['results'])!=8:
        raise ValueError('completed registered experiment required')
    checked=verify(data,root);c,_,_,_=preflight();ds=p.dataset_for(c,NORMALIZATION)
    binding=checked_binding(TRAIN384,p.provenance(c,ds))
    parent_path=TRAIN384/'diffusion_fine/step_000384.pt'
    if p.sha256(parent_path)!=data['registration']['parent_sha256']:
        raise ValueError('parent hash mismatch')
    parent=p.load_checkpoint(parent_path,binding,'fine','diffusion')
    cfg=data['registration']['config'];panel=data['registration']['panel'];items={};started=time.monotonic()
    for group in ('fit','transfer'):
        for anchor in panel[group+'_anchors']:
            item=ds[next(i for i,r in enumerate(ds.rows) if r['anchor_id']==anchor)]
            items[anchor]=dict(target=p.tensor(item['fine_target'],device),condition=p.tensor(item['fine_condition'],device),
                               wide=p.tensor(item['coarse_condition'],device),group=group)
    bands=Bands(96,3.383,[0,.08,.16,.32,np.inf]);rows=[];scale=ds.normalization['targets']['fine']['std']
    for branch in data['results']:
        arm=branch['arm'];ratio=branch['ratio'];label=f'{arm}_{ratio}'
        model=build(arm,c,parent,cfg,device)
        expected={**binding,'fixed_noise':dict(registration=data['registration'],ratio=ratio,arm=arm)}
        state=p.load_checkpoint(root/label/'update_000512.pt',expected,'fine','diffusion')
        model.load_state_dict(state['model']);model.eval()
        a=1/math.sqrt(1+ratio**2);b=ratio*a;t=2*math.atan(ratio)/math.pi
        for anchor,item in items.items():
            if time.monotonic()-started>600:
                raise TimeoutError('response check limited to ten minutes')
            y=item['target'];seed=p.seed_for(panel['seed'],anchor,'evaluation-0','fine')
            noise=torch.randn(y.shape,device=device,generator=torch.Generator(device=device).manual_seed(seed))
            reconstructed=[]
            for x in (a*y,a*y-b*noise):
                pred=model(x,y.new_tensor([t]),item['condition'],wide_condition=item['wide'])
                reconstructed.append(clean_estimate('diffusion',x,pred,t)[0,0].cpu().numpy())
            with h5py.File(root/f'{arm}_{ratio}_{anchor}.h5','r') as f:
                if int(f.attrs['noise_seed'])!=seed:
                    raise ValueError('saved positive-noise reconstruction seed mismatch')
                plus=f['normalized_clean_fine'][:]
            truth=y[0,0].cpu().numpy();zero,minus=reconstructed
            metrics=decomposition(bands,(plus-truth)*scale,(minus-truth)*scale,(zero-truth)*scale)
            metrics['clean_input_gain']=bands.compare(zero,truth)['gain']
            rows.append(dict(arm=arm,ratio=ratio,anchor_id=anchor,group=item['group'],seed=seed,metrics=metrics))
        del model,state
    preflight();verify(data,root)
    summary={}
    for branch in data['results']:
        for group in ('fit','transfer'):
            selected=[r for r in rows if r['arm']==branch['arm'] and r['ratio']==branch['ratio'] and r['group']==group]
            summary[f'{branch["arm"]}/{branch["ratio"]}/{group}']={
                k:np.median([r['metrics'][k] for r in selected],axis=0).tolist() for k in selected[0]['metrics']}
    p.write_json(args.output,dict(complete=True,claim='post-fit descriptive localization, not a changed gate',
                 main_sha256=p.sha256(args.input),source_sha256=p.sha256(__file__),
                 report_source_sha256=p.sha256(p.REPO/'workflows/sbi/e2e_fixed_noise_report.py'),job_id=os.environ['SLURM_JOB_ID'],
                 verification=checked,rows=rows,summary=summary,forward_calls=96,
                 elapsed_seconds=time.monotonic()-started,training_updates=0,heldout_payloads_read=False))
    print('FIXED RESPONSE COMPLETE',flush=True)


if __name__=='__main__':
    main()
