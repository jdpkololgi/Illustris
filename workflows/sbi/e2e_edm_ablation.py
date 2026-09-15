"""Matched diffusion-only denoiser ablations; frozen E2E parents stay untouched."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import time

import numpy as np
import torch

from workflows.sbi import e2e_wide_pipeline as p
from workflows.sbi.e2e_wide_continue import checked_binding, equal_state
from workflows.sbi.e2e_wide_denoising_audit import Bands, TRAIN384
from workflows.sbi.e2e_wide_research_canary import preflight, NORMALIZATION
from workflows.sbi.e2e_fine_learning_test import evaluate, aggregate, gates, draws, prediction_loss
from workflows.sbi.e2e_edm_ablation_models import NoiseAdaptedNet, edm_loss

CONFIG=p.REPO/'configs/e2e_edm_ablation_20260915.json'
PANEL=p.REPO/'configs/e2e_fine_learning_20260915.json'


def sample_time(arm,seed,cfg):
    rng=torch.Generator().manual_seed(seed)
    if arm['noise']=='uniform_time':
        t=float(torch.rand((),generator=rng))
        return t,math.tan(t*math.pi/2)
    sigma=math.exp(cfg['P_mean']+cfg['P_std']*float(torch.randn((),generator=rng)))
    return 2*math.atan(sigma)/math.pi,sigma


def grad_norm(parameters):
    values=[torch.sum(x.grad.detach()**2) for x in parameters if x.grad is not None]
    return float(torch.sqrt(torch.stack(values).sum())) if values else 0.


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    device=p.runtime();c,_,_,_=preflight()
    cfg=json.loads(CONFIG.read_text());panel=json.loads(PANEL.read_text())
    if cfg['updates']!=512 or cfg['sigma_data']!=1 or cfg['heldout_access'] or cfg['automatic_extension']:
        raise ValueError('unsupported ablation contract')
    ds=p.dataset_for(c,NORMALIZATION);base=p.provenance(c,ds)
    parent_binding=checked_binding(TRAIN384,base)
    path=TRAIN384/'diffusion_fine/step_000384.pt'
    state=p.load_checkpoint(path,parent_binding,'fine','diffusion')
    out=p.output_path(c,args.output);out.mkdir(parents=True,exist_ok=False)
    sources=[Path(__file__),CONFIG,PANEL,p.REPO/'workflows/sbi/e2e_edm_ablation_models.py',
             p.REPO/'workflows/sbi/e2e_fine_learning_test.py',p.REPO/'workflows/sbi/e2e_wide_denoising_audit.py']
    source_hashes={str(x.relative_to(p.REPO)):p.sha256(x) for x in sources}
    registration=dict(config=cfg,panel=panel,source_sha256=source_hashes,
                      parent_sha256=p.sha256(path),parent_binding_sha256=p.digest(parent_binding),
                      git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=p.REPO,text=True).strip(),
                      job_id=os.environ['SLURM_JOB_ID'],node=socket.gethostname(),
                      claim='warm-started training-only diffusion ablations, no E2E promotion',
                      heldout_payloads_read=False,training_ready=False)
    p.write_json(out/'ABLATION_STARTED.json',registration)
    started=time.monotonic();items={}
    for group in ('fit','transfer'):
        for anchor in panel[group+'_anchors']:
            index=next(i for i,r in enumerate(ds.rows) if r['anchor_id']==anchor)
            item=ds[index];fine=ds.inverse_target(item['fine_target'][0],'fine')
            up=p.coarse_to_fine(ds.inverse_target(item['coarse_target'][0],'coarse'))
            items[anchor]=dict(target=p.tensor(item['fine_target'],device),
                               condition=p.tensor(item['fine_condition'],device),wide=p.tensor(item['coarse_condition'],device),
                               target_np=item['fine_target'][0],scale=ds.normalization['targets']['fine']['std'],
                               up_coarse=up,truth=up+fine,group=group)
    bands=Bands(96,3.383,[0,.08,.16,.32,np.inf]);baseline=None
    results=[];fields=[];checkpoint_hashes={}
    for arm in cfg['arms']:
        torch.manual_seed(cfg['adapter_seed'])
        original=p.build_model(c,'fine',device);original.load_state_dict(state['model'])
        model=NoiseAdaptedNet(original,film=arm['film'],receptive=arm['receptive']).to(device)
        if not equal_state(model.base.state_dict(),state['model']):
            raise ValueError('parent weights changed at initialization')
        optimizer=torch.optim.AdamW(model.parameters(),lr=c['training']['learning_rate'],weight_decay=c['training']['weight_decay'])
        if arm['optimizer']=='inherit':
            optimizer.load_state_dict(copy.deepcopy(state['optimizer']))
            if not equal_state(optimizer.state_dict(),state['optimizer']):
                raise ValueError('parent optimizer mismatch')
        # Full-size identity-init check at all evaluation noise levels, before updates.
        item=items[panel['fit_anchors'][0]];initial_error=0.
        with torch.no_grad():
            for ratio in panel['ratios']:
                t=item['target'].new_tensor([2*math.atan(ratio)/math.pi])
                a=original(item['target'],t,item['condition'],wide_condition=item['wide'])
                b=model(item['target'],t,item['condition'],wide_condition=item['wide'])
                initial_error=max(initial_error,float(torch.max(torch.abs(a-b))))
                torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-6)
        branch=out/arm['name'];branch.mkdir()
        binding={**parent_binding,'edm_ablation':dict(registration=registration,arm=arm)}
        history=copy.deepcopy(state['history']);training=[];curve=[]
        for update in range(cfg['updates']+1):
            if time.monotonic()-started>cfg['maximum_work_seconds']:
                raise TimeoutError('bounded work budget reached; no automatic retry')
            if update in cfg['evaluate_at']:
                probes=evaluate(model,items,'diffusion',panel,device,bands)
                if update==0:
                    if baseline is None:
                        baseline=probes
                        fields.extend(draws(model,items,ds,'diffusion','parent',c,panel,out,device,bands))
                    else:
                        if probes!=baseline:
                            raise ValueError('nonidentical paired initial probes')
                curve.append(dict(update=update,summary=aggregate(probes),probes=probes))
                p.write_json(branch/f'probe_{update:03d}.json',curve[-1])
                print(f'ABLATION {arm["name"]} update={update} '
                      f'noise05={curve[-1]["summary"]["fit/0.05"]["noise_amplitude"][-1]:.6f} '
                      f'noise02={curve[-1]["summary"]["fit/0.2"]["noise_amplitude"][-1]:.6f}',flush=True)
            if update==cfg['updates']:
                fields.extend(draws(model,items,ds,'diffusion',arm['name'],c,panel,out,device,bands));break
            anchor=panel['fit_anchors'][update%3];item=items[anchor]
            seed=p.seed_for(cfg['train_seed'],anchor,update,'noise')
            generator=torch.Generator(device=device).manual_seed(seed)
            noise=torch.randn(item['target'].shape,device=device,generator=generator)
            t,sigma=sample_time(arm,p.seed_for(cfg['train_seed'],anchor,update,'time'),cfg)
            model.train();optimizer.zero_grad(set_to_none=True)
            if arm['noise']=='uniform_time':
                loss=prediction_loss(model,item['target'],noise,t,item['condition'],item['wide'],'diffusion')
            else:
                loss=edm_loss(model,item['target'],noise,sigma,item['condition'],item['wide'])
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite loss')
            loss.backward()
            norms=dict(base=grad_norm(model.base.parameters()),
                       film=grad_norm(model.film.parameters()) if model.film is not None else 0.,
                       receptive=grad_norm(model.receptive.parameters()) if model.receptive is not None else 0.)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),arm['clip'],error_if_nonfinite=True)
            optimizer.step()
            record=dict(step=385+update,diagnostic_update=update+1,anchor_id=anchor,noise_seed=seed,
                        time=t,sigma=sigma,loss=float(loss.detach()),gradient_norm_before_clip=float(norm),gradient_groups=norms)
            history.append(record);training.append(record)
            if update+1 in cfg['evaluate_at']:
                path=branch/f'step_{385+update:06d}.pt'
                p.save_checkpoint(path,model=model,optimizer=optimizer,generator=generator,binding=binding,
                                  stage='fine',method='diffusion',step=385+update,history=history)
                checkpoint_hashes[f'{arm["name"]}/{update+1}']=p.sha256(path)
        result=dict(arm=arm,curve=curve,training=training,gate=gates(baseline,curve[-1]['probes'],panel),
                    parameter_count=sum(x.numel() for x in model.parameters()),initial_max_abs=initial_error)
        p.write_json(branch/'BRANCH_COMPLETE.json',result);results.append(result)
        del model,optimizer,original
    preflight()
    if checked_binding(TRAIN384,p.provenance(c,ds))!=parent_binding or any(p.sha256(p.REPO/k)!=v for k,v in source_hashes.items()):
        raise ValueError('provenance drift')
    if len(results)!=6 or len(fields)!=42 or len(checkpoint_hashes)!=18:
        raise ValueError('incomplete registered ablation')
    p.write_json(out/'ABLATION_COMPLETE.json',dict(registration=registration,results=results,fields=fields,
                 checkpoints=checkpoint_hashes,elapsed_seconds=time.monotonic()-started,complete=True,
                 training_ready=False,calibration_pass=None))
    print('EDM ABLATION COMPLETE',flush=True)


if __name__=='__main__':
    main()
