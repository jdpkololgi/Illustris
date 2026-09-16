"""Paired one-step Adam interventions, not a training extension or repaired model."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import torch
from workflows.sbi import e2e_wide_pipeline as p
from workflows.sbi import e2e_durable as durable
from workflows.sbi import e2e_clean_limit as base
from workflows.sbi import e2e_diversity_norm as data
from workflows.sbi.e2e_oracle_conflict import verify
from workflows.sbi.e2e_loss_conflict import components,gradient_vectors,remove_conflicting_component
from workflows.sbi.e2e_wide_continue import equal_state


def assign_gradient(parameters,vector):
    start=0
    for parameter in parameters:
        end=start+parameter.numel()
        parameter.grad=vector[start:end].reshape(parameter.shape).to(parameter).clone()
        start=end
    if start!=vector.numel():raise ValueError('gradient shape mismatch')


@torch.no_grad()
def evaluate(model,item,seed):
    y=item['target'];gen=torch.Generator(device=y.device).manual_seed(seed)
    noises=[torch.randn(y.shape,device=y.device,generator=gen) for _ in range(2)]
    rows=[];model.eval()
    for q in (.005,.01,.05):
        a=1/math.sqrt(1+q*q);b=q*a;t=y.new_tensor([2*math.atan(q)/math.pi])
        denoise=lambda x:a*x-b*model(x,t,item['condition'],wide_condition=item['wide'])
        rows.append(dict(sigma=q,clean_mse=float((denoise(a*y)-y).double().square().mean()),
            noisy_mse=sum(float((denoise(a*y+b*e)-y).double().square().mean()) for e in noises)/len(noises)))
    return rows


def run(root):
    device=p.runtime();m=verify(root);s=m['spec'];old=Path(s['parent_root'])
    previous=json.loads((old/'MANIFEST.json').read_text())['spec']
    clean_root=Path(previous['parent_root'])
    if p.sha256(clean_root/'MANIFEST.json')!=previous['parent_manifest_sha256']:raise ValueError('parent data contract drift')
    oldspec=json.loads((clean_root/'MANIFEST.json').read_text())['spec'];raw=Path(oldspec['parent_root'])
    if p.sha256(raw/'PREPARED.json')!=oldspec['prepared_sha256']:raise ValueError('prepared drift')
    prepared,items=data.load_items(raw,device);records=[]
    for seed in s['replicas']:
        parent=m['parents'][f'{seed}/control'];branch=old/f'replica_{seed}/control'
        if p.sha256(branch/'COMPLETE.json')!=parent['complete_sha256']:raise ValueError('parent receipt drift')
        state,pointer=durable.load(branch,parent['binding'])
        if pointer!=parent['checkpoint']:raise ValueError('checkpoint drift')
        cfg,model,opt,gen=base.make(prepared,seed,device);base.restore(state,model,opt,gen)
        parameters=list(model.parameters())
        for row in prepared['selection']['train'][:3]:
            anchor=row['anchor_id'];item=items[anchor]
            eval_seed=p.seed_for(916225,anchor,0,'step-evaluation')
            before=evaluate(model,item,eval_seed)
            for q in (.005,.01,.05):
                base.restore(state,model,opt,gen);model.eval()
                rng=torch.Generator(device=device).manual_seed(p.seed_for(916224,anchor,q,'step-gradient'))
                eps=torch.randn(item['target'].shape,device=device,generator=rng)
                aux=torch.randn(item['target'].shape,device=device,generator=rng)
                gradients=gradient_vectors(components(model,item,eps,aux,q),parameters)
                primary=gradients['denoising'];identity=gradients['identity']
                origin=torch.cat([x.detach().flatten() for x in parameters]).double()
                choices={'denoising_only':primary,'identity_strong':primary+identity,
                         'projected_identity':primary+remove_conflicting_component(primary,identity)}
                control_delta=None
                for name,gradient in choices.items():
                    base.restore(state,model,opt,gen);opt.zero_grad(set_to_none=True)
                    assign_gradient(parameters,gradient)
                    norm=torch.nn.utils.clip_grad_norm_(parameters,cfg['clip'],error_if_nonfinite=True)
                    opt.step()
                    delta=torch.cat([x.detach().flatten() for x in parameters]).double()-origin
                    if name=='denoising_only':control_delta=delta.clone()
                    after=evaluate(model,item,eval_seed)
                    records.append(dict(replica=seed,anchor_id=anchor,phase=row['phase'],gradient_sigma=q,variant=name,
                        raw_primary_dot_aux=float((primary*(gradient-primary)).sum()),
                        actual_primary_dot_displacement=float((primary*delta).sum()),
                        incremental_primary_dot_vs_control=float((primary*(delta-control_delta)).sum()),
                        gradient_norm_before_clipping=float(norm),before=before,after=after))
                base.restore(state,model,opt,gen)
                del gradients,primary,identity,choices,origin,control_delta
        base.restore(state,model,opt,gen)
        if not equal_state(model.state_dict(),state['model']) or not equal_state(opt.state_dict(),state['optimizer']):
            raise ValueError('failed to restore original checkpoint')
        print('ADAM STEP PROBES',seed,flush=True)
    verify(root)
    durable.publish_json(root/'ADAM_STEPS.json',dict(complete=True,manifest_sha256=p.sha256(root/'MANIFEST.json'),
        records=records,job_id=os.environ.get('SLURM_JOB_ID'),saved_weights=False,
        scope='54 independent one-step copies of two control checkpoints; three fitting fields, fresh evaluation noise, no transfer or convergence claim.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    run(parser.parse_args().root)
