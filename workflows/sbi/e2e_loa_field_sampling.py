"""Frozen CFM weights with explicitly checked numerical sampling precision."""
from dataclasses import replace
import argparse
import json
import os
from pathlib import Path
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_views as views
from workflows.sbi import e2e_coupled_operators as op
from workflows.sbi import e2e_cfm_pilot_evaluate as evaluate
from workflows.sbi.e2e_coupled_benchmark_models import FieldCondition,sample
from workflows.sbi.e2e_vdm_context_models import replicate
from workflows.sbi.e2e_coupled_physical_gate import eigenvalues


def setup(tf32=False):
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=tf32;torch.backends.cudnn.allow_tf32=tf32
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)


@torch.no_grad()
def generate(models,arrays,chart,address,first=0,count=2,nfe=128,legacy=False):
    if nfe%2:raise ValueError('even NFE required for Heun')
    t=lambda x:torch.as_tensor(x,device='cuda',dtype=torch.float32)[None].expand(count,*x.shape)
    base=FieldCondition(t(views.scale_channels(arrays['joint'],chart['joint'])),
        t(views.scale_channels(arrays['wide'],chart['wide'])),t(np.zeros(3,dtype='f4')),'wide')
    def seeds(stage):
        return [int(c.digest(['loa-gallery-v1',17,i,stage] if legacy else
                            ['loa-atlas-v1',17,address,i,stage])[:15],16) for i in range(first,first+count)]
    z=sample(models['coarse'],base,'cfm',nfe//2,seeds('coarse'))
    crop=tuple(slice(*v) for v in op.layout()['joint_coarse_crop_in_wide'])
    fine=replace(base,region='joint',coarse_joint=replicate(z[(slice(None),slice(None),*crop)]),
                 coarse_wide=z,coarse_source='sampled')
    u=sample(models['fine'],fine,'cfm',nfe//2,seeds('fine'))
    rho=[];wide=[];errors=[]
    for a,b in zip(z.cpu().numpy(),u.cpu().numpy()):
        w=np.exp(a[0].astype(float)*chart['coarse_logrho']['std'][0]+chart['coarse_logrho']['mean'][0])
        r=op.decode(w[crop],b[0].astype(float)*chart['fine_residual']['std'][0])
        error=float(np.max(abs(op.mean_pool(r)-w[crop])/w[crop]))
        if not np.isfinite(r).all() or not np.isfinite(w).all() or r.min()<=0 or w.min()<=0 or error>2e-6:
            raise ValueError('positive mass-conserving decode failed')
        rho.append(r);wide.append(w);errors.append(error)
    return dict(rho=np.asarray(rho),wide=np.asarray(wide),mass_error=np.asarray(errors))


def benchmark(output):
    c.require_compute();output.mkdir(parents=True,exist_ok=False)
    source=Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/cfm_gallery_20260930_v2')
    old=json.loads((source/'COMPLETE.json').read_text())
    if c.sha256(source/'loa_draw000.npz')!=old['loa']['chunks']['loa_draw000.npz']:
        raise ValueError('original draw changed')
    with np.load(source/'loa_conditions.npz') as f:arrays={k:f[k] for k in f.files}
    setup(False);models,hashes,normalizer=evaluate.load_models(17,13312,26624);chart=views.load_chart(normalizer)
    timing={};results={}
    for name,tf32,batch in [('fp32',False,2),('tf32',True,2),('tf32_batch4',True,4)]:
        setup(tf32);torch.cuda.reset_peak_memory_stats();start=time.monotonic()
        result=generate(models,arrays,chart,'legacy',count=batch,legacy=True)
        torch.cuda.synchronize();elapsed=time.monotonic()-start
        timing[name]=dict(seconds=elapsed,draws=batch,seconds_per_draw=elapsed/batch,
                          peak_gpu_bytes=torch.cuda.max_memory_allocated())
        results[name]=result;np.savez_compressed(output/(name+'.npz'),**result)
        print(json.dumps(dict(benchmark=name,**timing[name])),flush=True)
    with np.load(source/'loa_draw000.npz') as f:replay=bool(np.array_equal(results['fp32']['rho'],f['rho']))
    checks={}
    base=results['fp32'];crop=op.layout()['joint_coarse_crop_in_wide']
    base_eigen=[eigenvalues(op.consistent_tensor(r-1,w-1,crop,workers=4)[16:48,16:32,16:32])
                for r,w in zip(base['rho'],base['wide'])]
    for name in ('tf32','tf32_batch4'):
        value=results[name];check={}
        for key in ('rho','wide'):
            scale=float(np.sqrt(np.mean((base[key][0]-base[key][1])**2)/2))
            err=float(np.sqrt(np.mean((value[key][:2]-base[key])**2)))
            check[key+'_rmse_over_spread']=err/max(scale,1e-12)
        eigen=[eigenvalues(op.consistent_tensor(r-1,w-1,crop,workers=4)[16:48,16:32,16:32])
               for r,w in zip(value['rho'][:2],value['wide'][:2])]
        check['class_disagreement']=float(np.mean((np.asarray(eigen)>.2).sum(-1)!=(np.asarray(base_eigen)>.2).sum(-1)))
        check['passed']=max(check.values())<.01
        checks[name]=check
    record=dict(timing=timing,checks=checks,original_density_exact_replay=replay,
        passed=replay and all(v['passed'] for v in checks.values()),checkpoints=hashes,normalizer=normalizer,
        source=c.file_record(__file__,content_hash=True),
        limits='Numerical checks on two matched draws, not calibration or all-tile precision validation.')
    c.atomic_json(output/'BENCHMARK.json',record);print(json.dumps(record),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();benchmark(a.output)
