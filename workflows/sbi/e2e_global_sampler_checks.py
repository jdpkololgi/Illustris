"""Real-network sampler parity, precision and NFE controls (no truth access)."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import torch
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_views as views
from workflows.sbi import e2e_coupled_operators as op
from workflows.sbi import e2e_cfm_pilot_evaluate as old
from workflows.sbi import e2e_loa_field_sampling as sampling
from workflows.sbi.e2e_global_sampler import generate, integrate
from workflows.sbi.e2e_global_mock_evaluate import load_arrays, density_statistics
from workflows.sbi.e2e_coupled_benchmark_models import FieldCondition, sample
from workflows.sbi.e2e_vdm_context_models import project, replicate


def original_parity(models,arrays,chart):
    t=lambda x:torch.as_tensor(x,device='cuda',dtype=torch.float32)[None].expand(2,*x.shape)
    base=FieldCondition(t(views.scale_channels(arrays['joint'],chart['joint'])),
                        t(views.scale_channels(arrays['wide'],chart['wide'])),
                        t(np.zeros(3,'f4')),'wide')
    crop=tuple(slice(*s) for s in op.layout()['joint_coarse_crop_in_wide'])
    record={};seeds=[239,841]
    for stage,shape in [('coarse',(48,48,48)),('fine',(64,48,48))]:
        condition=base if stage=='coarse' else replace(base,region='joint',coarse_wide=z,
                    coarse_joint=replicate(z[(slice(None),slice(None),*crop)]),coarse_source='sampled')
        noise=torch.cat([torch.randn((1,1,*shape),device='cuda',generator=
                    torch.Generator(device='cuda').manual_seed(seed)) for seed in seeds])
        if stage=='fine':noise=project(noise)
        reference=sample(models[stage],condition,'cfm',64,seeds)
        actual=integrate(models[stage],stage,[condition,condition],[noise,noise],128,False)[0]
        error=float((reference-actual).abs().max())
        record[stage]=dict(maxabs=error,exact=bool(torch.equal(reference,actual)))
        if stage=='coarse':z=reference
    if not all(v['exact'] for v in record.values()):raise ValueError('Original-sampler parity failed')
    return record


def check(inputs,samples,out):
    c.require_compute();out.mkdir(parents=True,exist_ok=False);sampling.setup(True)
    models,hashes,normalizer=old.load_models(17,13312,26624);chart=views.load_chart(normalizer)
    panel=json.loads((inputs/'PREPARED.json').read_text())
    selected=['ph016_NGC_s0_interior_00','ph016_SGC_s0_boundary_00',
              'ph017_NGC_s3_boundary_00','ph017_SGC_s3_interior_00']
    checks=[];parity=None
    for identity in selected:
        case=next(v for v in panel['cases'] if v['pair_id']==identity)
        arrays=[load_arrays(v) for v in case['conditions']]
        if parity is None:parity=original_parity(models,arrays[0],chart)
        marker=json.loads((samples/identity/'COMPLETE.json').read_text())
        item=next(v for v in marker['outputs'] if Path(v['path']).stem=='shared_state')
        saved=load_arrays(item)
        sampling.setup(True);reference=generate(models,arrays,chart,identity,count=2)
        for variant,nfe,tf32 in [('nfe256',256,True),('strict_fp32',128,False)]:
            sampling.setup(tf32)
            value=generate(models,arrays,chart,identity,count=2,nfe=nfe)
            sampling.setup(True)
            metrics={}
            for key in ('density','regional','seam_density','seam_mass'):
                a=np.stack([density_statistics(r)[key] for r in reference['rho'][0]])
                b=np.stack([density_statistics(r)[key] for r in value['rho'][0]])
                spread=float(np.sqrt(np.mean(saved[key].var(0,ddof=1))))
                metrics[key]=float(np.sqrt(np.mean((a-b)**2))/max(spread,1e-12))
            checks.append(dict(case=identity,variant=variant,rmse_over_32draw_sd=metrics,
                               below_one_percent=max(metrics.values())<.01))
            print(identity,variant,metrics,flush=True)
    record=dict(original_sampler_exact_parity=parity,checks=checks,checkpoints=hashes,
                normalizer=normalizer,passed=all(v['below_one_percent'] for v in checks),
                scope='Two matched draws on four fixed cases; numerical controls, not calibration.')
    c.atomic_json(out/'CHECKS.json',record)
    if not record['passed']:raise ValueError('Numerical refinement/precision concern requires review')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True,type=Path)
    p.add_argument('--samples',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    a=p.parse_args();check(a.inputs,a.samples,a.out)
