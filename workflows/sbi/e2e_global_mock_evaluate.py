"""Matched two-window posterior calibration; ph016/ph017 development only."""
import argparse
import json
import os
from pathlib import Path
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_operators as op
from workflows.sbi import e2e_coupled_target_products as targets
from workflows.sbi import e2e_cfm_pilot_evaluate as old
from workflows.sbi.e2e_global_mock_prepare import PHASES, guard
from workflows.sbi.e2e_vdm_context_metrics import calibration, fair_energy, variogram_score


def density_statistics(rho):
    delta = np.asarray(rho, dtype=float)-1
    seam = delta[40:56,16:32,16:32]
    sides = np.array([seam[:8].mean(), seam[8:].mean()])
    # Four block-zero orthogonal x modes on each side of the seam.
    basis = op.project(np.broadcast_to(np.cos(np.pi*(np.arange(8)+.5)/8)[:,None,None],(8,8,8)))
    basis /= np.linalg.norm(basis)
    probes = np.array([np.sum(seam[x:x+8,y:y+8,z:z+8]*basis)
                       for x in (0,8) for y in (0,8) for z in (0,8)])
    return dict(density=targets.owned_core_arrays(delta),
                regional=old.regional(delta), original_probes=old.features(delta),
                seam_density=seam, seam_mass=sides,
                seam_sum_difference=np.array([sides.mean(), sides[1]-sides[0]]),
                seam_probes=probes)


def tidal_statistics(rho, wide):
    from workflows.sbi.e2e_loa_field_tensor import consistent
    from workflows.sbi.e2e_coupled_physical_gate import eigenvalues
    field = consistent(rho, wide)
    return eigenvalues(targets.owned_core_arrays(field)).reshape(2,4096,3)[:,::16]


def metric_summary(draws, truth):
    value = calibration(draws, truth)
    return dict(crps=float(value['crps'].mean()), bias=float(value['bias'].mean()),
                mse=float(np.mean(value['bias']**2)), variance=float(np.mean(value['std']**2)),
                attainable=value['attainable'],
                coverage={str(p):float(value[f'covered_{p}'].mean()) for p in (.5,.68,.9,.95)},
                width90=float(value['width_0.9'].mean()),
                rank_histogram=np.histogram(value['rank'],np.linspace(0,1,9))[0].tolist(),
                targets=int(np.asarray(truth).size))


def controls(draws=32):
    rng=np.random.default_rng(8841)
    truth=rng.normal(size=20000)
    x=rng.normal(size=(draws,20000))
    rows={label:metric_summary(value,truth) for label,value in
          [('oracle',x),('biased',x+1.5),('underdispersed',x*.3)]}
    reference=rows['oracle']['attainable']['0.9']
    if abs(rows['oracle']['coverage']['0.9']-reference)>.02:
        raise ValueError('Finite-ensemble oracle control failed')
    if any(rows[k]['coverage']['0.9']>reference-.15 for k in ('biased','underdispersed')):
        raise ValueError('Calibration controls not sensitive')
    # Correlated four-dimensional oracle; shuffle second half independently.
    good,bad=[],[]
    for _ in range(200):
        z=rng.normal(size=(draws+1,4)); z[:,2:]=.95*z[:,:2]+np.sqrt(1-.95**2)*z[:,2:]
        y,x=z[0],z[1:]; shuffled=x.copy();shuffled[:,2:]=x[rng.permutation(draws),2:]
        good.append(variogram_score(x,y));bad.append(variogram_score(shuffled,y))
    if np.mean(bad)<=np.mean(good):
        raise ValueError('Dependence control not sensitive')
    return dict(marginal=rows,dependence=dict(oracle=float(np.mean(good)),
                shuffled=float(np.mean(bad))),passed=True,synthetic=True)


def load_arrays(item):
    if c.sha256(item['path'])!=item['sha256']:
        raise ValueError('Prepared input changed')
    with np.load(item['path']) as f:
        return {k:f[k] for k in f.files}


def probe_parity(models,chart,out):
    """Replay the existing Loa feasibility output exactly with factored engine."""
    import torch
    from workflows.sbi.e2e_global_sampler import generate
    from workflows.sbi.e2e_vdm_context_models import project
    root=Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/loa_global_feasibility_20261001_v1/overlap')
    receipt=json.loads((root/'CONDITIONS.json').read_text())
    case=next(v for v in receipt['cases'] if v['cap']=='NGC')
    arrays=[load_arrays(dict(path=str(root/f'NGC_{i}.npz'),sha256=h)) for i,h in enumerate(case['hashes'])]
    noise={}
    for stage,shape in [('coarse',(56,48,48)),('fine',(96,48,48))]:
        values=[]
        for d in range(4):
            seed=int(c.digest(['global-probe-v1','NGC',stage,d])[:15],16)
            values.append(torch.randn((1,1,*shape),device='cuda',generator=torch.Generator(device='cuda').manual_seed(seed)))
        value=torch.cat(values);noise[stage]=project(value) if stage=='fine' else value
    result=generate(models,arrays,chart,'parity',count=4,base_noise=noise)
    with np.load(root/'NGC_shared_state_every_step.npz') as f:
        errors=[float(np.max(abs(result['rho'][i].astype('f4')-f[f'rho{i}'][:,0]))) for i in range(2)]
    record=dict(exact_replay=errors==[0.,0.],maxabs=errors,
                original_probe_sha256=c.sha256(root/'PROBE.json'))
    c.atomic_json(out/'PARITY.json',record)
    if not record['exact_replay']:
        raise ValueError('Factored sampler differs from original feasibility probe')


def run(root,out,draws=32,limit=None,parity=False):
    import torch
    from workflows.sbi import e2e_loa_field_sampling as sampling
    from workflows.sbi import e2e_coupled_views as views
    from workflows.sbi import e2e_global_sampler as sampler
    c.require_compute()
    if draws<4 or draws%4:
        raise ValueError('Draw count must be a multiple of four')
    prepared=json.loads((root/'PREPARED.json').read_text())
    if prepared['phases']!=list(PHASES) or len(prepared['cases'])!=32:
        raise ValueError('Wrong mock panel')
    out.mkdir(parents=True,exist_ok=True)
    sampling.setup(True)
    models,hashes,normalizer=old.load_models(17,13312,26624)
    chart=views.load_chart(normalizer)
    binding=dict(draws=draws,nfe=128,batch=4,checkpoints=hashes,normalizer=normalizer,
                 inputs=c.sha256(root/'PREPARED.json'),sampler=c.sha256(sampler.__file__),
                 runner=c.sha256(__file__),precision='TF32 network / float64 decode and FFT')
    bp=out/'BINDING.json'
    if bp.exists():
        if json.loads(bp.read_text())!=binding:raise ValueError('Resume binding changed')
    else:c.atomic_json(bp,binding)
    if parity:probe_parity(models,chart,out)
    completed=[]
    for case in prepared['cases'][:limit]:
        guard(case['phase']); directory=out/case['pair_id'];directory.mkdir(exist_ok=True)
        marker=directory/'COMPLETE.json'
        if marker.exists():
            record=json.loads(marker.read_text())
            if record['binding']!=c.digest(binding):raise ValueError('Case resume binding changed')
            for item in record['outputs']:load_arrays(item)
            completed.append(record);continue
        arrays=[load_arrays(v) for v in case['conditions']]
        statistics={k:[] for k in ('original','independent_tiled','shared_state')}
        numerical=[];start=time.monotonic()
        for first in range(0,draws,4):
            for variant in ('independent','shared_state'):
                value=sampler.generate(models,arrays,chart,case['pair_id'],first,4,128,variant)
                numerical.append(dict(first=first,variant=variant,
                                      **{k:value[k] for k in ('max_mass_error','overlap_maxabs','overlap_rmse')}))
                for i in range(4):
                    left,right=value['rho'][0][i],value['rho'][1][i]
                    name='original' if variant=='independent' else 'shared_state'
                    features=density_statistics(left)
                    features['tidal']=tidal_statistics(left,value['wide'][0][i])
                    statistics[name].append(features)
                    if variant=='independent':
                        stitched=np.concatenate([left[:48],right[16:32]],axis=0)
                        statistics['independent_tiled'].append(density_statistics(stitched))
        outputs=[]
        for name,rows in statistics.items():
            path=directory/f'{name}.npz'
            np.savez_compressed(path,**{key:np.stack([v[key] for v in rows]) for key in rows[0]})
            outputs.append(c.file_record(path,content_hash=True))
        record=dict(pair_id=case['pair_id'],phase=case['phase'],binding=c.digest(binding),
                    outputs=outputs,numerical=numerical,seconds=time.monotonic()-start,
                    job=os.environ['SLURM_JOB_ID'])
        c.atomic_json(marker,record);completed.append(record)
        print(json.dumps(dict(case=case['pair_id'],seconds=record['seconds'],completed=len(completed))),flush=True)
    c.atomic_json(out/'RUN_COMPLETE.json',dict(cases=len(completed),panel_complete=len(completed)==32,
                  draws=draws,binding=c.digest(binding),job=os.environ['SLURM_JOB_ID']))


def score(root,samples,out):
    c.require_compute();out.mkdir(parents=True,exist_ok=False)
    prepared=json.loads((root/'PREPARED.json').read_text())
    binding=json.loads((samples/'BINDING.json').read_text())
    if c.sha256(root/'PREPARED.json')!=binding['inputs']:raise ValueError('Scoring panel changed')
    control=controls(binding['draws']);c.atomic_json(out/'CONTROLS.json',control)
    results=[]
    for case in prepared['cases']:
        guard(case['phase']);marker=samples/case['pair_id']/'COMPLETE.json'
        if not marker.exists():continue
        record=json.loads(marker.read_text())
        if record['binding']!=c.digest(binding):raise ValueError('Scoring binding mismatch')
        target=load_arrays(case['truth']);truth=density_statistics(target['rho_joint'])
        from workflows.sbi.e2e_coupled_physical_gate import eigenvalues
        truth['tidal']=eigenvalues(target['fullbox_tensor_cores']).reshape(2,4096,3)[:,::16]
        row={k:case[k] for k in ('pair_id','phase','cap','shell','support_stratum','seam_support')}
        row['methods']={}
        for item in record['outputs']:
            name=Path(item['path']).stem;data=load_arrays(item)
            metrics={k:metric_summary(v,truth[k]) for k,v in data.items()}
            metrics['core_mass']=metric_summary(data['regional'][:,:2],truth['regional'][:2])
            metrics['block_mass']=metric_summary(data['regional'][:,2:10],truth['regional'][2:10])
            metrics['fine_mass']=metric_summary(data['regional'][:,10:],truth['regional'][10:])
            metrics['seam_energy']=float(fair_energy(data['seam_probes'],truth['seam_probes']))
            metrics['seam_variogram']=variogram_score(data['seam_probes'],truth['seam_probes'])
            shuffled=data['seam_probes'].copy()
            shuffled[:,4:]=shuffled[np.random.default_rng(914).permutation(len(shuffled)),4:]
            metrics['seam_variogram_shuffled']=variogram_score(shuffled,truth['seam_probes'])
            metrics['seam_covariance']=np.cov(data['seam_mass'].T).tolist()
            metrics['seam_truth']=truth['seam_mass'].tolist()
            metrics['seam_mean']=data['seam_mass'].mean(0).tolist()
            if 'tidal' in data:
                metrics['eigengap']=metric_summary(np.diff(data['tidal'],axis=-1),np.diff(truth['tidal'],axis=-1))
                for threshold in (0.,.2):
                    labels=(truth['tidal']>threshold).sum(-1)
                    classes=(data['tidal']>threshold).sum(-1)
                    prob=np.stack([(classes==k).mean(0) for k in range(4)],axis=-1)
                    metrics[f'class_brier_{threshold}']=float(np.mean(np.sum((prob-np.eye(4)[labels])**2,axis=-1)))
                    metrics[f'class_offset_{threshold}']=(prob.mean((0,1))-np.eye(4)[labels].mean((0,1))).tolist()
            row['methods'][name]=metrics
        results.append(row)
    c.atomic_json(out/'SCORES.json',dict(cases=results,panel_complete=len(results)==32,
                  independent_phases=len(set(v['phase'] for v in results)),draws=binding['draws'],
                  binding=binding,job=os.environ['SLURM_JOB_ID'],posterior_qualified=False))
    print('scored',len(results),'cases',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run','score'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--samples',type=Path);p.add_argument('--draws',type=int,default=32)
    p.add_argument('--limit',type=int);p.add_argument('--parity',action='store_true');a=p.parse_args()
    run(a.root,a.out,a.draws,a.limit,a.parity) if a.action=='run' else score(a.root,a.samples,a.out)
