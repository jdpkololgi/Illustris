"""Development-only paired CFM posterior assessment with durable draw chunks."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from scipy import fft
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_coordinates as coord
from workflows.sbi import e2e_coupled_views as views
from workflows.sbi import e2e_coupled_operators as op
from workflows.sbi import e2e_coupled_target_products as targets
from workflows.sbi.e2e_coupled_product_audit import read_target
from workflows.sbi.e2e_coupled_physical_gate import eigenvalues
from workflows.sbi.e2e_coupled_benchmark_models import CoupledBackbone,FieldCondition,sample
from workflows.sbi.e2e_vdm_context_models import replicate
from workflows.sbi.e2e_vdm_context_metrics import calibration,fair_energy
from workflows.sbi.e2e_coupled_cfm_pilot import training_ids,pair

TRAIN_ROOT=Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/cfm_pilot_20260924_v1')
EVAL_PHASES=c.DEVELOPMENT


def development(phase):
    if phase not in EVAL_PHASES:raise PermissionError('phase outside explicitly selected evaluation panel')


def draw_seed(seed,phase,pair_id,index,stage):
    # Deliberately excludes checkpoint and NFE; includes coarse/fine stage.
    return int(c.digest([seed,phase,pair_id,index,stage,'pilot-eval-v1'])[:15],16)


def probes():
    x=(np.arange(16)+.5)/16;out=[]
    for mode in ((1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1),(1,1,1)):
        axes=[np.cos(np.pi*k*x) for k in mode]
        value=op.project(axes[0][:,None,None]*axes[1][None,:,None]*axes[2][None,None,:])
        out.append(value/np.linalg.norm(value))
    return np.asarray(out)


def features(delta):
    owned=targets.owned_core_arrays(delta)
    return np.einsum('cxyz,kxyz->ck',owned,probes()).ravel()


def regional(delta):
    block=[delta[x:x+24,y:y+24,z:z+24].mean() for x in (8,32) for y in (0,24) for z in (0,24)]
    fine=[delta[x:x+12,y:y+12,z:z+12].mean() for x in (17,33) for y in (17,19) for z in (17,19)]
    core=targets.owned_core_arrays(delta).mean((1,2,3))
    return np.r_[core,block,fine]


def matched_variogram(draws,truth):
    m=len(draws);moment=abs(draws[:,:7]-draws[:,7:])**.5
    target=abs(truth[:7]-truth[7:])**.5
    return float(np.mean((target-moment.mean(0))**2-moment.var(0,ddof=1)/m))


def summary(draws,truth):
    value=calibration(draws,truth)
    return dict(crps=float(value['crps'].mean()),bias=float(value['bias'].mean()),
        mean_squared_error=float(np.mean(value['bias']**2)),posterior_variance=float(np.mean(value['std']**2)),
        attainable=value['attainable'],coverage={str(p):float(value[f'covered_{p}'].mean()) for p in (.5,.68,.9,.95)},
        width90=float(value['width_0.9'].mean()),rank_histogram=np.histogram(value['rank'],np.linspace(0,1,9))[0].tolist())


def load_models(seed,step):
    models={};hashes={};normalizer=None
    for stage in ('coarse','fine'):
        path=TRAIN_ROOT/f'{stage}_{seed}'/f'CHECKPOINT_{step:06d}.pt'
        hashes[stage]=c.sha256(path);state=torch.load(path,map_location='cpu',weights_only=False)
        b=state['binding']
        if state['step']!=step or b['stage']!=stage or b['seed']!=seed or b['confirmation_access']:
            raise ValueError('invalid checkpoint identity')
        if normalizer is not None and normalizer!=b['normalizer']:raise ValueError('factor chart mismatch')
        normalizer=b['normalizer']
        import inspect
        if b['model']!=c.sha256(inspect.getfile(CoupledBackbone)):raise ValueError('checkpoint architecture drift')
        model=CoupledBackbone(stage,'wide' if stage=='coarse' else 'joint',24,3).cuda().eval()
        model.load_state_dict(state['ema']);models[stage]=model
    return models,hashes,normalizer


def reference_scales(normalizer):
    rows=[];ids=training_ids();selected=[];chart=views.load_chart(normalizer)
    for phase in c.TRAIN:
        choices=[i for i in ids if i[0]==phase]
        for index in (0,64):
            item=choices[index];_,target=pair(item,normalizer)
            rho=views.decode_view(target['coarse_logrho'],target['fine_residual'],phase,(0,0,0),chart)
            rows.append(features(rho-1));selected.append(item)
    scale=np.std(rows,axis=0,ddof=1)
    if np.min(scale)<=1e-10:raise ValueError('degenerate training probe scale')
    return scale,selected


def physical(z,u,phase,chart):
    rho=views.decode_view(z,u,phase,(0,0,0),chart)
    wide=np.exp(z[0].astype(float)*chart['coarse_logrho']['std'][0]+chart['coarse_logrho']['mean'][0])
    _,crop=views.crop_slices(phase,(0,0,0));expected=wide[crop]
    error=float(np.max(abs(op.mean_pool(rho)-expected)/expected))
    if not np.isfinite(rho).all() or not np.isfinite(wide).all() or min(rho.min(),wide.min())<=0 or error>2e-6:
        raise FloatingPointError('positive mass-conserving decode failed')
    return rho,wide,error


@torch.no_grad()
def generate(models,observation,phase,pair_id,seed,first,nfe,chart):
    development(phase)
    device='cuda';t=lambda x:torch.as_tensor(x,device=device,dtype=torch.float32)[None].expand(2,*x.shape)
    base=FieldCondition(t(observation['joint']),t(observation['wide']),t(observation['joint_center_from_wide_mpc_h']),'wide')
    seeds=lambda stage:[draw_seed(seed,phase,pair_id,i,stage) for i in (first,first+1)]
    z=sample(models['coarse'],base,'cfm',nfe//2,seeds('coarse'))
    _,crop=views.crop_slices(phase,(0,0,0))
    fine=replace(base,region='joint',coarse_joint=replicate(z[(slice(None),slice(None),*crop)]),
        coarse_wide=z,coarse_source='sampled')
    u=sample(models['fine'],fine,'cfm',nfe//2,seeds('fine'))
    z=z.cpu().numpy();u=u.cpu().numpy()
    values=[physical(a,b,phase,chart) for a,b in zip(z,u)]
    return dict(rho=np.stack([v[0] for v in values]),wide=np.stack([v[1] for v in values]),
        mass_error=np.array([v[2] for v in values]))


def truth(phase,pair_id):
    development(phase) # guard BEFORE any target IO
    directory=coord.ROOT/'targets'/phase
    receipt=coord.verify_receipt(directory/f'{pair_id}.json',payload=False)
    if receipt['phase']!=phase or receipt['pair_id']!=pair_id or len(receipt['outputs'])!=1:
        raise ValueError('target identity mismatch')
    item=receipt['outputs'][0];path=c.guarded(item['path'],phase)
    if path.parent!=directory.resolve() or c.sha256(path)!=item['sha256']:raise ValueError('target changed')
    return read_target(path,phase,pair_id),item['sha256']


def spectra(draws,target):
    # Common unwindowed rectangular pseudo-spectrum; no spherical-grid substitution.
    shape=target.shape
    axes=np.meshgrid(*[2*np.pi*fft.fftfreq(n,d=6.766) for n in shape],indexing='ij',sparse=True)
    k=np.sqrt(sum(v*v for v in axes));edges=(0,.04,.08,.16,.32,1.)
    ft=fft.fftn(target,norm='ortho');fm=fft.fftn(draws.mean(0),norm='ortho')
    power=np.mean([abs(fft.fftn(v,norm='ortho'))**2 for v in draws],axis=0)
    mean=abs(fm)**2;residual=power-mean;m=len(draws);result=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        mask=(k>lo)&(k<=hi);tp=float(np.mean(abs(ft[mask])**2));mp=float(mean[mask].mean())
        result.append(dict(band=[lo,hi],modes=int(mask.sum()),truth_power=tp,
            sample_power=float(power[mask].mean()),mean_power=mp,
            residual_power_unbiased=float(residual[mask].mean()*m/(m-1)),
            mean_error_power=float(np.mean(abs(fm[mask]-ft[mask])**2)),
            cross_correlation=float(np.real(fm[mask]*ft[mask].conj()).mean()/np.sqrt(max(mp*tp,1e-60)))))
    return result


def score(values,target,scale):
    delta=values['rho']-1;wide=values['wide']-1;truth_delta=target['rho_joint']-1
    density=np.stack([targets.owned_core_arrays(v) for v in delta])
    target_density=targets.owned_core_arrays(truth_delta)
    region=np.stack([regional(v) for v in delta]);target_region=regional(truth_delta)
    # Fixed 256 probes per core, not data-selected; full physical operator used.
    index=np.arange(0,4096,16);eigen=[]
    crop=op.layout()['joint_coarse_crop_in_wide']
    for d,w in zip(delta,wide):
        tensor=op.consistent_tensor(d,w,crop,workers=2)
        owned=targets.owned_core_arrays(tensor)
        relative=np.max(abs(owned[...,[0,3,5]].sum(-1)-targets.owned_core_arrays(d)))/max(1.,np.max(abs(d)))
        if relative>2e-6:raise ValueError('physical trace identity failed')
        eigen.append(eigenvalues(owned).reshape(2,4096,3)[:,index])
    eigen=np.asarray(eigen);true_eigen=eigenvalues(target['fullbox_tensor_cores']).reshape(2,4096,3)[:,index]
    vector=np.stack([features(v) for v in delta])/scale;true_vector=features(truth_delta)/scale
    classes=(eigen>0).sum(-1);labels=(true_eigen>0).sum(-1)
    probabilities=np.stack([(classes==k).mean(0) for k in range(4)],axis=-1)
    return dict(density=summary(density,target_density),core_mass=summary(region[:,:2],target_region[:2]),
        block_mass=summary(region[:,2:10],target_region[2:10]),fine_mass=summary(region[:,10:],target_region[10:]),
        tidal=summary(eigen,true_eigen),eigengap=summary(np.diff(eigen,axis=-1),np.diff(true_eigen,axis=-1)),
        class_brier=float(np.mean(np.sum((probabilities-np.eye(4)[labels])**2,axis=-1))),
        class_calibration_offset=(probabilities.mean((0,1))-np.eye(4)[labels].mean((0,1))).tolist(),
        joint_energy=float(fair_energy(vector,true_vector)),matched_variogram=matched_variogram(vector,true_vector),
        region_truth=target_region.tolist(),region_mean=region.mean(0).tolist(),region_std=region.std(0,ddof=1).tolist(),
        spectra=spectra(delta,truth_delta),max_mass_error=float(values['mass_error'].max()))


def run(a):
    global EVAL_PHASES
    EVAL_PHASES=('ph014','ph015') if a.panel=='replication' else c.DEVELOPMENT
    if a.panel=='replication' and a.step not in (13312,26624):raise ValueError('replication checkpoints frozen')
    c.require_compute();torch.set_num_threads(4)
    if torch.cuda.device_count()!=1:raise RuntimeError('one visible GPU per worker')
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    if a.seed not in (17,29) or a.step not in (6656,13312,26624):raise ValueError('unregistered assessment cell')
    root=Path(a.output)/f'seed{a.seed}_step{a.step}';root.mkdir(parents=True,exist_ok=True)
    models,hashes,normalizer=load_models(a.seed,a.step);chart=views.load_chart(normalizer)
    scale,fit_ids=reference_scales(normalizer)
    binding=dict(checkpoints=hashes,normalizer=normalizer,runner=c.sha256(__file__),seed=a.seed,step=a.step,
        weights='ema',draws=32,nfe=128,refinement_draws=8,refinement_nfe=256,probe_scale=scale.tolist(),fit_ids=fit_ids,
        evaluation_phases=list(EVAL_PHASES),panel=a.panel)
    marker=root/'BINDING.json'
    if marker.exists():
        if json.loads(marker.read_text())!=json.loads(json.dumps(binding)):raise ValueError('resume binding mismatch')
    else:c.atomic_json(marker,binding)
    start=time.monotonic();completed=[]
    for phase in EVAL_PHASES:
        development(phase)
        receipt=coord.verify_receipt(coord.ROOT/'conditions'/phase/'CONDITIONS_COMPLETE.json',payload=False)
        ids=sorted(Path(item['path']).stem for item in receipt['pair_receipts'])
        if len(ids)!=16:raise ValueError('development panel incomplete')
        for pair_id in ids:
            observation=views.load_observations(phase,pair_id,normalizer)
            for nfe,count in [(128,32)]+([(256,8)] if pair_id==ids[0] else []):
                folder=root/phase/pair_id/f'nfe{nfe}';folder.mkdir(parents=True,exist_ok=True)
                if (folder/'COMPLETE.json').exists():
                    previous=json.loads((folder/'COMPLETE.json').read_text())
                    if previous['binding']!=c.digest(binding):raise ValueError('completed case binding changed')
                    for name,checksum in previous['chunks'].items():
                        if c.sha256(folder/name)!=checksum:raise ValueError('completed draw chunk changed')
                    completed.append(str(folder));continue
                values=[];chunks={}
                for first in range(0,count,2):
                    path=folder/f'draw{first:03d}.npz';record=path.with_suffix('.json')
                    if record.exists():
                        previous=json.loads(record.read_text())
                        if previous['binding']!=c.digest(binding) or c.sha256(path)!=previous['sha256']:
                            raise ValueError('draw receipt mismatch')
                        with np.load(path) as saved:val={k:saved[k] for k in saved.files}
                    else:
                        if time.monotonic()-start>a.seconds:
                            c.atomic_json(root/'PAUSED.json',dict(phase=phase,pair_id=pair_id,nfe=nfe,first=first),replace=True)
                            return
                        tick=time.monotonic();val=generate(models,observation,phase,pair_id,a.seed,first,nfe,chart)
                        replay=False
                        if phase==EVAL_PHASES[0] and pair_id==ids[0] and first==0 and nfe==128:
                            repeated=generate(models,observation,phase,pair_id,a.seed,first,nfe,chart)
                            if any(not np.array_equal(val[k],repeated[k]) for k in val):
                                raise ValueError('addressed draw replay differs')
                            replay=True
                        tmp=path.with_suffix('.tmp.npz');np.savez(tmp,**val);os.replace(tmp,path)
                        c.atomic_json(record,dict(binding=c.digest(binding),sha256=c.sha256(path),
                            phase=phase,pair_id=pair_id,ids=[first,first+1],nfe=nfe,seconds=time.monotonic()-tick,
                            exact_in_process_replay_passed=replay))
                        if first%8==0:print('DRAW_CHUNK',phase,pair_id,nfe,first,'seconds',time.monotonic()-tick,flush=True)
                    chunks[path.name]=c.sha256(path);values.append(val)
                joined={k:np.concatenate([v[k] for v in values]) for k in values[0]}
                # All draw generation above receives observations only; truth enters here.
                actual,truth_sha=truth(phase,pair_id)
                result=dict(phase=phase,pair_id=pair_id,nfe=nfe,draws=count,truth_sha256=truth_sha,
                    chunks=chunks,binding=c.digest(binding),scores=score(joined,actual,scale))
                if nfe==256:
                    main=folder.parent/'nfe128';base=[]
                    for first in range(0,8,2):
                        with np.load(main/f'draw{first:03d}.npz') as saved:base.append({k:saved[k] for k in saved.files})
                    base={k:np.concatenate([v[k] for v in base]) for k in base[0]}
                    result['paired_base8_scores']=score(base,actual,scale)
                    result['paired_field_rms']=float(np.sqrt(np.mean((joined['rho']-base['rho'])**2)))
                c.atomic_json(folder/'COMPLETE.json',result,replace=True);completed.append(str(folder))
                print('CASE',phase,pair_id,nfe,'elapsed',time.monotonic()-start,flush=True)
    c.atomic_json(root/'COMPLETE.json',dict(cases=completed,binding=c.digest(binding),seconds=time.monotonic()-start),replace=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--step',type=int,required=True)
    p.add_argument('--output',required=True);p.add_argument('--seconds',type=int,default=13800)
    p.add_argument('--panel',choices=('development','replication'),default='development');run(p.parse_args())
