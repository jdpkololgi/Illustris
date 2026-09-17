"""Full-size GPU replay, observation-only generation, and bounded cost forecast."""
import argparse
import gc
import os
from pathlib import Path
import socket
import time

import numpy as np
import torch

from workflows.sbi import e2e_durable as durable,e2e_wide_pipeline as existing
from workflows.sbi.e2e_wide_continue import equal_state
from workflows.sbi.e2e_vdm_assessment import atomic_npz
from workflows.sbi.e2e_vdm_context_data import spec,output_root,read_json
from workflows.sbi.e2e_vdm_context_dataset import Products
from workflows.sbi.e2e_vdm_context_models import coupled_sample,decode_density,block_mean
from workflows.sbi.e2e_vdm_context_sample import expand_condition,with_coarse
from workflows.sbi.e2e_vdm_context_tasks import coarse_cache_key
from workflows.sbi.e2e_vdm_context_train import TrainingCache,new_model,update_model,checkpoint,restore,verify_manifest


def forecast(timings,ledger,c,overhead=1.15):
    training=sum(timings[f'{arm}_fine']['update_seconds']*c['updates']*2 for arm in c['arms'])
    training+=timings['D_coarse']['update_seconds']*c['updates']*2
    inference=0.
    for task in ledger['tasks']:
        inference+=task['count']*task['steps']/250*timings[task['arm']+'_fine']['draw_seconds']
    parents={}
    for task in ledger['tasks']:
        if task['arm']=='D' and task['coarse_mode']=='sampled':
            for draw in range(task['count']):
                key=coarse_cache_key('D',task['replica'],task['checkpoint'],task['domain'],draw,task['steps'],task['purpose'])
                parents[key]=task['steps']/250
    inference+=sum(parents.values())*timings['D_coarse']['draw_seconds']
    # Include the registered smoke allowance plus15% scheduling/I/O/recovery margin.
    total=(training+inference)*overhead/3600+4
    return dict(training_gpu_hours=training/3600,inference_gpu_hours=inference/3600,
                overhead_factor=overhead,smoke_allowance_gpu_hours=4,forecast_gpu_hours=total,
                within_ceiling=total<=c['budget']['gpu_hours'])


def smoke(root):
    root=output_root(root)
    device=existing.runtime()
    verify_manifest(root)
    if (root/'SMOKE.json').exists():
        raise FileExistsError('immutable full-size smoke already recorded')
    c=spec()
    folder=root/'smoke'
    folder.mkdir(exist_ok=False)
    observations=Products(root,['ph000'],targets=False)
    small=TrainingCache(root,'A')
    large=TrainingCache(root,'B')
    anchor=small.ids[0]
    base=observations.condition(anchor,device=device)
    # The public inference reader must reject targets even on training phases.
    try:
        observations.raw_targets(anchor)
    except PermissionError:
        pass
    else:
        raise ValueError('observation-only target firewall failed')
    timings={}
    coarse_draws=None
    for arm,factor in [('D','coarse'),('A','fine'),('B','fine'),('C','fine'),('D','fine')]:
        data=small if arm=='A' else large
        data.arm=arm
        model,opt,gen=new_model(c,0,arm,factor,device)
        branch=folder/f'{arm}_{factor}'
        branch.mkdir()
        history=[]
        update_times=[]
        torch.cuda.reset_peak_memory_stats()
        for step in range(8):
            x,condition=data.batch(step,0,factor,device)
            torch.cuda.synchronize()
            started=time.monotonic()
            history.append(update_model(model,opt,gen,x,condition,c,step))
            torch.cuda.synchronize()
            update_times.append(time.monotonic()-started)
        binding=dict(manifest_sha256=existing.sha256(root/'MANIFEST.json'),arm=arm,factor=factor,smoke=True)
        checkpoint(branch,model,opt,gen,binding,factor,8,history)
        x,condition=data.batch(8,0,factor,device)
        expected=update_model(model,opt,gen,x,condition,c,8)
        replica,optimizer,generator=new_model(c,0,arm,factor,device)
        restore(branch,binding,replica,optimizer,generator,factor)
        actual=update_model(replica,optimizer,generator,x,condition,c,8)
        if (expected!=actual or not equal_state(model.state_dict(),replica.state_dict())
                or not equal_state(opt.state_dict(),optimizer.state_dict())):
            raise ValueError('full-size GPU optimizer/RNG replay mismatch')
        del replica,optimizer,generator,opt,gen,x,condition
        gc.collect()
        model.eval()
        if factor=='coarse':
            condition=base.wide.expand(8,-1,-1,-1,-1)
        elif arm=='D':
            condition=with_coarse(base,coarse_draws,observations.chart,[0,0,0],'sampled')
        else:
            condition=expand_condition(base,8)
        seeds=list(range(301,309))
        torch.cuda.synchronize()
        started=time.monotonic()
        z=coupled_sample(model,condition,250,seeds)
        torch.cuda.synchronize()
        sampling_seconds=time.monotonic()-started
        if factor=='coarse':
            scalar_condition=base.wide
        elif arm=='D':
            scalar_condition=with_coarse(base,coarse_draws[:1],observations.chart,[0,0,0],'sampled')
        else:
            scalar_condition=base
        scalar=coupled_sample(model,scalar_condition,250,seeds[:1])
        relative=float((z[:1]-scalar).square().mean().sqrt()/scalar.square().mean().sqrt().clamp_min(1e-10))
        if relative>2e-4:
            raise ValueError('actual-network scalar/batch trajectories differ beyond tolerance')
        chart=observations.chart
        if factor=='coarse':
            coarse_draws=torch.exp(z.double()*chart['coarse']['std']+chart['coarse']['mean'])
            rho=coarse_draws
        elif arm=='D':
            coarse_local=coarse_draws[:,:,18:30,18:30,18:30]
            rho=decode_density(coarse_local,z.double()*chart['residual']['std'])
            if float((block_mean(rho)-coarse_local).abs().max())>2e-6:
                raise ValueError('generated coarse/fine mass mismatch')
        else:
            rho=torch.exp(z.double()*chart['fine']['std']+chart['fine']['mean'])
        if not torch.isfinite(rho).all() or (rho<=0).any():
            raise FloatingPointError('nonfinite/nonpositive full-size smoke draw')
        arrays=rho[:,0].cpu().numpy()
        path=atomic_npz(branch,'smoke-draws',dict(rho=arrays))
        with np.load(path,allow_pickle=False) as replay:
            if not np.array_equal(replay['rho'],arrays):
                raise ValueError('actual GPU draw serialization replay failed')
        timings[f'{arm}_{factor}']=dict(update_seconds=float(np.median(update_times[2:])),
            draw_seconds=sampling_seconds/8,batch_sampling_seconds=sampling_seconds,
            scalar_batch_relative_rms=relative,exact_checkpoint_replay=True,
            peak_gpu_bytes=torch.cuda.max_memory_allocated(),parameters=sum(p.numel() for p in model.parameters()),
            draw_file=str(path.relative_to(root)),draw_sha256=existing.sha256(path))
        print('GPU_SMOKE',arm,factor,timings[f'{arm}_{factor}'],flush=True)
        del model,z,scalar,condition,scalar_condition,rho,arrays
        gc.collect()
    estimate=forecast(timings,read_json(root/'DRAW_LEDGER.json'),c)
    result=dict(manifest_sha256=existing.sha256(root/'MANIFEST.json'),timings=timings,**estimate,
        passed=estimate['within_ceiling'],observation_only_inference=True,full_size=True,
        canonical_batch=8,steps=250,job=os.environ['SLURM_JOB_ID'],node=socket.gethostname(),
        scientific_fit=False,scientific_draws=False)
    durable.publish_json(root/'SMOKE.json',result)
    if not result['passed']:
        raise RuntimeError('measured forecast exceeds approved112GPUh; no matrix launch')
    print('SMOKE_PASSED',estimate,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True,type=Path)
    smoke(p.parse_args().root)


if __name__=='__main__':
    main()
