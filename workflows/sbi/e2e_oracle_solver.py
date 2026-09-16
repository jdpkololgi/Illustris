"""Matched-NFE VP probability-flow Heun candidate; no production replacement."""
import argparse
import json
import math
import os
from pathlib import Path
import socket
import torch
from workflows.sbi import e2e_wide_pipeline as p
from workflows.sbi import e2e_durable as durable
from workflows.sbi.e2e_oracle_conflict import verify
from workflows.sbi.e2e_analytic_fields import spectrum,filt,GaussianVelocity
from workflows.sbi.e2e_wide_models import sample_field


@torch.no_grad()
def sample_vp_heun(model,condition,steps,generator,wide_condition=None):
    """dx/dt=(pi/2)*v, integrate t=1->0; two evaluations per interval.

    This is a VP probability-flow derivative, not the straight CFM velocity.
    No clipping/churn/extra denoise. Model must be finite at both endpoints.
    """
    if not isinstance(steps,int) or isinstance(steps,bool) or steps<1:
        raise ValueError('positive integer steps required')
    if condition.ndim!=5 or condition.shape[1]<1 or not isinstance(generator,torch.Generator):
        raise ValueError('5D condition and addressed generator required')
    x=torch.randn(condition[:,:1].shape,device=condition.device,dtype=condition.dtype,generator=generator)
    before=model.training;model.eval()
    def velocity(x,t):
        time=x.new_full((x.shape[0],),t)
        if wide_condition is None:return model(x,time,condition)
        return model(x,time,condition,wide_condition=wide_condition)
    try:
        h=-1./steps
        for i in range(steps):
            t=1-i/steps;tn=1-(i+1)/steps
            v=math.pi/2*velocity(x,t)
            end=math.pi/2*velocity(x+h*v,tn)
            x=x+.5*h*(v+end)
            if not torch.isfinite(x).all():raise FloatingPointError('nonfinite VP Heun path')
    finally:model.train(before)
    return x


def run(root):
    device=p.runtime();m=verify(root);n=m['spec']['oracle_grid'];count=m['spec']['oracle_draws']
    power=spectrum(n,device=device);condition=torch.zeros(count,1,n,n,n,device=device,dtype=torch.float64)
    condition+=.15*torch.sin(2*math.pi*torch.arange(n,device=device)/n)[None,None,:,None,None]
    gen=lambda:torch.Generator(device=device).manual_seed(916220)
    white=torch.randn(condition.shape,device=device,dtype=condition.dtype,generator=gen())
    truth=condition+filt(white,power.sqrt());oracle=GaussianVelocity(power,'diffusion');records=[]
    for nfe in (32,128,512):
        old=sample_field(oracle,condition,'diffusion',nfe,gen())
        new=sample_vp_heun(oracle,condition,nfe//2,gen())
        relative=lambda x,y:float((x-y).square().mean().sqrt()/y.square().mean().sqrt())
        records.append(dict(nfe=nfe,ddim_relative_rms=relative(old,truth),heun_relative_rms=relative(new,truth),
            ddim_log_density_relative_rms=relative(torch.expm1(old-.5*power.mean()),torch.expm1(truth-.5*power.mean())),
            heun_log_density_relative_rms=relative(torch.expm1(new-.5*power.mean()),torch.expm1(truth-.5*power.mean()))))
    passed=all(r['heun_relative_rms']<r['ddim_relative_rms'] and
               r['heun_log_density_relative_rms']<r['ddim_log_density_relative_rms'] for r in records)
    verify(root)
    durable.publish_json(root/'SOLVER.json',dict(passed=passed,records=records,
        manifest_sha256=p.sha256(root/'MANIFEST.json'),job_id=os.environ.get('SLURM_JOB_ID'),node=socket.gethostname(),
        scope='Matched-NFE analytic Gaussian/log-Gaussian oracle only; no neural checkpoint or E2E production change.'))
    print('SOLVER',passed,json.dumps(records),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    run(parser.parse_args().root)
