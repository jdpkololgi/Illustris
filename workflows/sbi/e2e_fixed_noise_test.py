"""Bounded fixed-sigma denoiser capability experiment, training phases only."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import time

import h5py
import numpy as np
import torch

from workflows.sbi import e2e_wide_pipeline as p
from workflows.sbi.e2e_wide_continue import checked_binding
from workflows.sbi.e2e_wide_denoising_audit import Bands, TRAIN384, clean_estimate
from workflows.sbi.e2e_wide_research_canary import preflight, NORMALIZATION
from workflows.sbi.e2e_fine_learning_test import evaluate, aggregate, gates
from workflows.sbi.e2e_fixed_noise_models import ResidualAdapter, HighResolutionNet, LinearReference, fixed_loss

CONFIG=p.REPO/'configs/e2e_fixed_noise_20260915.json'
PANEL=p.REPO/'configs/e2e_fine_learning_20260915.json'


def build(arm,c,parent,cfg,device):
    torch.manual_seed(cfg['model_seed'])
    base=p.build_model(c,'fine',device)
    if arm=='vp_parent':
        base.load_state_dict(parent['model']); return base
    if arm=='residual_highres':
        return ResidualAdapter(HighResolutionNet(base,cfg['highres_width'],cfg['highres_blocks'])).to(device)
    torch.nn.init.zeros_(base.output.weight);torch.nn.init.zeros_(base.output.bias)
    if arm=='vp_fresh':
        return base
    if arm=='residual_unet':
        return ResidualAdapter(base)
    raise ValueError('unknown fixed-noise arm')


@torch.no_grad()
def save_cleaned(model,items,ratio,label,panel,out):
    records=[];a=1/math.sqrt(1+ratio**2);b=ratio*a;t=2*math.atan(ratio)/math.pi
    model.eval()
    for anchor,item in items.items():
        target=item['target'];seed=p.seed_for(panel['seed'],anchor,'evaluation-0','fine')
        noise=torch.randn(target.shape,device=target.device,generator=torch.Generator(device=target.device).manual_seed(seed))
        x=a*target+b*noise;pred=model(x,target.new_tensor([t]),item['condition'],wide_condition=item['wide'])
        clean=clean_estimate('diffusion',x,pred,t)[0,0].cpu().numpy()
        if not np.isfinite(clean).all():
            raise ValueError('nonfinite controlled reconstruction')
        path=out/f'{label}_{ratio}_{anchor}.h5'
        with h5py.File(path,'x') as f:
            f.attrs['controlled_denoising_only']=True;f.attrs['ratio']=ratio;f.attrs['noise_seed']=seed
            f.attrs['physical_scale']=item['scale'];f.create_dataset('normalized_clean_fine',data=clean)
        records.append(dict(path=path.name,sha256=p.sha256(path),anchor_id=anchor,ratio=ratio,label=label))
    return records


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    device=p.runtime();c,_,_,_=preflight();cfg=json.loads(CONFIG.read_text());panel=json.loads(PANEL.read_text())
    if cfg['updates']!=512 or cfg['ratios']!=[.05,.2] or cfg['heldout_access'] or cfg['full_diffusion_sampling']:
        raise ValueError('unsupported fixed-noise contract')
    ds=p.dataset_for(c,NORMALIZATION);binding=checked_binding(TRAIN384,p.provenance(c,ds))
    parent_path=TRAIN384/'diffusion_fine/step_000384.pt';parent=p.load_checkpoint(parent_path,binding,'fine','diffusion')
    out=p.output_path(c,args.output);out.mkdir(parents=True,exist_ok=False)
    sources=[Path(__file__),CONFIG,PANEL,p.REPO/'workflows/sbi/e2e_fixed_noise_models.py',
             p.REPO/'workflows/sbi/e2e_fine_learning_test.py',p.REPO/'workflows/sbi/e2e_wide_denoising_audit.py']
    registration=dict(config=cfg,panel=panel,source_sha256={str(x.relative_to(p.REPO)):p.sha256(x) for x in sources},
                      parent_sha256=p.sha256(parent_path),parent_binding_sha256=p.digest(binding),
                      git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=p.REPO,text=True).strip(),
                      job_id=os.environ['SLURM_JOB_ID'],node=socket.gethostname(),heldout_payloads_read=False,
                      training_ready=False,full_diffusion_sampling=False)
    p.write_json(out/'FIXED_STARTED.json',registration);started=time.monotonic();items={}
    for group in ('fit','transfer'):
        for anchor in panel[group+'_anchors']:
            item=ds[next(i for i,r in enumerate(ds.rows) if r['anchor_id']==anchor)]
            items[anchor]=dict(target=p.tensor(item['fine_target'],device),condition=p.tensor(item['fine_condition'],device),
                               wide=p.tensor(item['coarse_condition'],device),target_np=item['fine_target'][0],
                               scale=ds.normalization['targets']['fine']['std'],group=group)
    bands=Bands(96,3.383,[0,.08,.16,.32,np.inf]);model=build('vp_parent',c,parent,cfg,device)
    baseline=evaluate(model,items,'diffusion',panel,device,bands);del model
    references=[];fields=[];results=[];checkpoints={}
    for label in ('identity','lowpass'):
        model=LinearReference(low=cfg['reference_low_h_mpc'],high=cfg['reference_high_h_mpc'],identity=label=='identity')
        probes=evaluate(model,items,'diffusion',panel,device,bands)
        noiseless={anchor:bands.compare(model.clean(item['target'])[0,0].cpu().numpy(),item['target_np'])
                   for anchor,item in items.items()}
        references.append(dict(label=label,probes=probes,summary=aggregate(probes),gate=gates(baseline,probes,panel),noiseless=noiseless))
        for ratio in cfg['ratios']:
            fields.extend(save_cleaned(model,items,ratio,label,panel,out))
        p.write_json(out/f'REFERENCE_{label}.json',references[-1])
        print('REFERENCE',label,json.dumps(references[-1]['gate']),flush=True)
    for ratio in cfg['ratios']:
        for arm in cfg['arms']:
            branch_start=time.monotonic();label=f'{arm}_{ratio}';branch=out/label;branch.mkdir()
            model=build(arm,c,parent,cfg,device);residual=arm.startswith('residual')
            optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
            experiment_binding={**binding,'fixed_noise':dict(registration=registration,ratio=ratio,arm=arm)}
            curve=[];training=[]
            for update in range(cfg['updates']+1):
                if time.monotonic()-started>cfg['maximum_work_seconds']:
                    raise TimeoutError('fixed-noise work cap reached; no automatic extension')
                if update in cfg['evaluate_at']:
                    probes=evaluate(model,items,'diffusion',panel,device,bands)
                    curve.append(dict(update=update,probes=probes,summary=aggregate(probes)))
                    p.write_json(branch/f'probe_{update:03d}.json',curve[-1])
                    print('FIXED',label,'update',update,'noise',curve[-1]['summary'][f'fit/{ratio}']['noise_amplitude'][-1],flush=True)
                if update==cfg['updates']:
                    fields.extend(save_cleaned(model,items,ratio,arm,panel,out));break
                anchor=panel['fit_anchors'][update%3];item=items[anchor]
                seed=p.seed_for(cfg['train_seed'],anchor,update,'noise')
                generator=torch.Generator(device=device).manual_seed(seed)
                noise=torch.randn(item['target'].shape,device=device,generator=generator)
                model.train();optimizer.zero_grad(set_to_none=True)
                loss=fixed_loss(model,item['target'],noise,ratio,item['condition'],item['wide'],residual)
                if not torch.isfinite(loss):
                    raise FloatingPointError('nonfinite loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['clip'],error_if_nonfinite=True)
                optimizer.step();training.append(dict(update=update+1,anchor_id=anchor,noise_seed=seed,
                                                      loss=float(loss.detach()),gradient_norm_before_clip=float(norm)))
                if update+1 in cfg['evaluate_at']:
                    path=branch/f'update_{update+1:06d}.pt'
                    p.save_checkpoint(path,model=model,optimizer=optimizer,generator=generator,binding=experiment_binding,
                                      stage='fine',method='diffusion',step=update+1,history=training)
                    checkpoints[str(path.relative_to(out))]=p.sha256(path)
            gate_panel=copy.deepcopy(panel);gate_panel['gate']['near_clean_ratios']=[ratio]
            result=dict(arm=arm,ratio=ratio,curve=curve,training=training,
                        gate=gates(baseline,curve[-1]['probes'],gate_panel),
                        parameters=sum(x.numel() for x in model.parameters()),elapsed_seconds=time.monotonic()-branch_start)
            results.append(result);p.write_json(branch/'BRANCH_COMPLETE.json',result);del model,optimizer
    preflight()
    if checked_binding(TRAIN384,p.provenance(c,ds))!=binding or p.sha256(parent_path)!=registration['parent_sha256']:
        raise ValueError('parent provenance drift')
    if any(p.sha256(p.REPO/k)!=v for k,v in registration['source_sha256'].items()):
        raise ValueError('experiment source drift')
    if len(results)!=8 or len(fields)!=72 or len(checkpoints)!=24:
        raise ValueError('incomplete registered test')
    p.write_json(out/'FIXED_COMPLETE.json',dict(registration=registration,baseline=baseline,references=references,
                 results=results,fields=fields,checkpoints=checkpoints,elapsed_seconds=time.monotonic()-started,
                 complete=True,training_ready=False,calibration_pass=None))
    print('FIXED NOISE COMPLETE',flush=True)


if __name__=='__main__':
    main()
