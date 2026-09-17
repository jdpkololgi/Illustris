"""Physical-distribution and sampler diagnostics for the frozen VDM matrix."""
import argparse
from pathlib import Path

import numpy as np

from workflows.sbi import e2e_durable as durable, e2e_wide_pipeline as existing
from workflows.sbi.e2e_field_build_products import require_compute,tensor_from_delta,eigs
from workflows.sbi.e2e_wide_denoising_audit import Bands
from workflows.sbi.e2e_vdm_context_data import read_json,output_root,spec
from workflows.sbi.e2e_vdm_context_dataset import Products,observation_summary
from workflows.sbi.e2e_vdm_context_metrics import calibration,fair_energy,central_order_interval,variogram_score
from workflows.sbi.e2e_vdm_context_physics import composite_tensor
from workflows.sbi.e2e_vdm_context_sample import read_array_receipt,MICROBATCH
from workflows.sbi.e2e_vdm_context_tasks import coarse_cache_key,draw_tasks
from workflows.sbi.e2e_vdm_context_train import verify_launch

CORE=(slice(16,32),)*3
EDGES=[0,.04,.08,.16,.32,np.inf]
WIDE_EDGES=[0,.01,.02,.04,.08,np.inf]


def checkpoint_hash(root,task,factor):
    branch=root/'models'/f"{task['arm']}_{factor}_seed{task['replica']}"
    pointer=read_json(branch/f"CHECKPOINT_{task['checkpoint']:06d}.json")
    path=(branch/pointer['path']).resolve()
    if branch.resolve() not in path.parents or read_json(path.parent/'COMMITTED.json')!=pointer:
        raise ValueError('invalid committed checkpoint pointer')
    return pointer['sha256']


def ensemble(root,task,count=None):
    count=task['count'] if count is None else count
    folder=root/'draws'/task['task_id']
    if count%MICROBATCH or count>task['count']:
        raise ValueError('invalid scientific draw count')
    complete=None
    if count==task['count']:
        complete=read_json(folder/'COMPLETE.json')
        if complete['task']!=task or complete['manifest_sha256']!=existing.sha256(root/'MANIFEST.json'):
            raise ValueError('case completion drift')
    values,receipts=[],{}
    for first in range(task['start'],task['start']+count,MICROBATCH):
        path=folder/f'{first:06d}.json'
        digest=existing.sha256(path)
        if complete is not None and complete['chunks'][str(first)]!=digest:
            raise ValueError('completed chunk changed')
        binding=dict(manifest_sha256=existing.sha256(root/'MANIFEST.json'),task=task,
            ids=list(range(first,first+MICROBATCH)),checkpoint_sha256=checkpoint_hash(root,task,'fine'),
            coarse_checkpoint_sha256=checkpoint_hash(root,task,'coarse') if task['arm']=='D' else None)
        array,_=read_array_receipt(folder,path,binding,'delta')
        if array.shape!=(MICROBATCH,48,48,48):
            raise ValueError('scientific field shape drift')
        values.append(array)
        receipts[str(path.relative_to(root))]=digest
    return np.concatenate(values),receipts


def parent_ensemble(root,task):
    if task['arm']!='D':
        raise ValueError('only D has wide matter draws')
    count=spec()['draws_joint'] if task['coarse_mode']=='fixed_mean' else task['count']
    reference=dict(task,purpose='joint' if task['purpose'].startswith('joint') else task['purpose'])
    values=[]
    for first in range(0,count,MICROBATCH):
        folder=root/'parents'/str(Path(coarse_cache_key('D',task['replica'],task['checkpoint'],task['domain'],first,task['steps'],reference['purpose'])).parent)
        binding=dict(manifest_sha256=existing.sha256(root/'MANIFEST.json'),
            checkpoint_sha256=checkpoint_hash(root,task,'coarse'),domain=task['domain'],
            replica=task['replica'],checkpoint=task['checkpoint'],steps=task['steps'],purpose=reference['purpose'],
            ids=list(range(first,first+MICROBATCH)))
        array,_=read_array_receipt(folder,folder/f'{first:06d}.json',binding,'rho')
        values.append(array)
    values=np.concatenate(values)
    return np.repeat(values.mean(0,keepdims=True),task['count'],axis=0) if task['coarse_mode']=='fixed_mean' else values


def density_spectra(draws,truth,cell=6.766,edges=EDGES):
    b=Bands(len(truth),cell,edges)
    ft=b.fft(truth)
    mean=draws.mean(0)
    fm=b.fft(mean)
    mean_power=b.cross(fm,fm)
    truth_power=b.cross(ft,ft)
    power,cross,residual=[],[],[]
    for value in draws:
        f=b.fft(value)
        power.append(b.cross(f,f))
        cross.append(b.cross(f,ft))
        residual.append(b.cross(f-fm,f-fm))
    power,cross,residual=map(np.asarray,(power,cross,residual))
    closure=power.mean(0)-mean_power-residual.mean(0)
    if np.max(np.abs(closure)/np.maximum(power.mean(0),1e-30))>1e-9:
        raise ValueError('sample/mean/residual power decomposition failed')
    return dict(truth_power=truth_power.tolist(),mean_sample_power=power.mean(0).tolist(),
        posterior_mean_power=mean_power.tolist(),posterior_residual_power_population=residual.mean(0).tolist(),
        posterior_mean_power_mc_corrected=(mean_power-residual.mean(0)/(len(draws)-1)).tolist(),
        posterior_residual_power_unbiased=(residual.mean(0)*len(draws)/(len(draws)-1)).tolist(),
        mean_sample_power_ratio=(power.mean(0)/np.maximum(truth_power,1e-30)).tolist(),
        correlation_posterior_mean=(b.cross(fm,ft)/np.sqrt(np.maximum(mean_power*truth_power,1e-60))).tolist(),
        mean_sample_correlation=(cross/np.sqrt(np.maximum(power*truth_power,1e-60))).mean(0).tolist(),
        power_decomposition_max_relative_error=float(np.max(np.abs(closure)/np.maximum(power.mean(0),1e-30)))),power


def core_features(delta,tensor):
    eigen=eigs(tensor)
    return np.concatenate([delta[...,None],eigen,np.diff(eigen,axis=-1)],axis=-1).reshape(-1,6)


def closure_tensor(delta,closure):
    if closure=='periodic':
        return tensor_from_delta(delta,6.766)
    if closure not in ('zero','reflect'):
        raise ValueError('unknown boundary closure')
    x=np.pad(delta,24,mode='constant' if closure=='zero' else 'reflect')
    return tensor_from_delta(x,6.766)[24:72,24:72,24:72]


def calibration_summary(draws,truth,scales,mask=None):
    c=calibration(draws,truth)
    mask=np.ones(len(truth),dtype=bool) if mask is None else np.asarray(mask,dtype=bool)
    if not mask.any():
        return dict(probes=0)
    take=lambda key:np.asarray(c[key])[mask]
    crps=take('crps').mean(0)
    result=dict(probes=int(mask.sum()),draws=len(draws),bias=take('bias').mean(0).tolist(),
        rmse_mean=np.sqrt(np.mean(take('bias')**2,axis=0)).tolist(),
        rms_spread=np.sqrt(np.mean(take('std')**2,axis=0)).tolist(),
        crps=crps.tolist(),standardized_crps=(crps/scales).tolist(),
        attainable_coverage=c['attainable'],coverage={},width={},rank_histogram=[])
    for level in (.5,.68,.9,.95):
        result['coverage'][str(level)]=take(f'covered_{level}').mean(0).tolist()
        result['width'][str(level)]=take(f'width_{level}').mean(0).tolist()
    for column in range(truth.shape[-1]):
        result['rank_histogram'].append(np.histogram(take('rank')[:,column],bins=np.linspace(0,1,17))[0].tolist())
    result['tidal_joint_energy']=float(fair_energy(draws[:,mask,1:4]/scales[1:4],truth[mask,1:4]/scales[1:4]).mean())
    result['all_six_energy']=float(fair_energy(draws[:,mask]/scales,truth[mask]/scales).mean())
    result['caveat']='correlated grid probes, not independent posterior experiments'
    return result


def paired_refinement(a,b,power_a,power_b,seed=917,bootstrap=512,core=CORE):
    """Common-noise draw-pair bootstrap; no voxel independence assumption."""
    lo,hi,target=central_order_interval(len(a),.9)
    def width(x):
        selected=x if core is None else x[(slice(None),*core)]
        ordered=np.partition(selected.reshape(len(x),-1),(lo,hi),axis=0)
        return float((ordered[hi]-ordered[lo]).mean())
    pa,pb=power_a.mean(0),power_b.mean(0)
    power=np.abs(pb-pa)/np.maximum(np.abs(pb),1e-30)
    width_change=abs(width(b)-width(a))/max(width(b),1e-30)
    rng=np.random.default_rng(seed)
    power_bounds,width_bounds=[],[]
    for _ in range(bootstrap):
        ids=rng.integers(len(a),size=len(a))
        first,second=power_a[ids].mean(0),power_b[ids].mean(0)
        power_bounds.append(np.max(np.abs(second-first)/np.maximum(np.abs(second),1e-30)))
        width_bounds.append(abs(width(b[ids])-width(a[ids]))/max(width(b[ids]),1e-30))
    return dict(max_relative_power_change=float(power.max()),relative_width_change=float(width_change),
        mc95_power_bound=float(np.quantile(power_bounds,.95)),mc95_width_bound=float(np.quantile(width_bounds,.95)),
        attainable_width_coverage=target,
        coupled_field_rms_over_spread=float(np.sqrt(np.mean((a-b)**2))/max(np.sqrt(np.mean(b.var(0,ddof=1))),1e-30)))


def paired_feature_widths(a,b,seed=917,bootstrap=512):
    """Draw x correlated probes x components; bootstrap whole paired draws."""
    a,b=np.asarray(a),np.asarray(b)
    if a.shape!=b.shape or a.ndim!=3:
        raise ValueError('paired feature ensembles required')
    lo,hi,target=central_order_interval(len(a),.9)
    def width(x):
        order=np.partition(x,(lo,hi),axis=0)
        return (order[hi]-order[lo]).mean(0)
    change=lambda x,y:np.abs(width(y)-width(x))/np.maximum(width(y),1e-30)
    rng=np.random.default_rng(seed)
    bounds=[]
    for _ in range(bootstrap):
        ids=rng.integers(len(a),size=len(a))
        bounds.append(change(a[ids],b[ids]))
    return dict(relative_width_change=change(a,b).tolist(),
                mc95_width_bound=np.quantile(bounds,.95,axis=0).tolist(),attainable_width_coverage=target)


def refinement_pass(early,late,early_tidal,late_tidal,wide=None):
    checks=[max(early['max_relative_power_change'],early['relative_width_change'])<.05,
            max(late['max_relative_power_change'],late['relative_width_change'])<.02,
            max(late['mc95_power_bound'],late['mc95_width_bound'])<.05,
            max(early_tidal['relative_width_change'])<.05,
            max(late_tidal['relative_width_change'])<.02,
            max(late_tidal['mc95_width_bound'])<.05]
    if wide is not None:
        e,l=wide['early'],wide['late']
        checks.extend([max(e['max_relative_power_change'],e['relative_width_change'])<.05,
                       max(l['max_relative_power_change'],l['relative_width_change'])<.02,
                       max(l['mc95_power_bound'],l['mc95_width_bound'])<.05])
    return bool(all(checks))


def sampler_gate(root):
    require_compute()
    root=output_root(root)
    verify_launch(root)
    ledger=read_json(root/'DRAW_LEDGER.json')
    tasks=ledger['tasks']
    records,inputs=[],{}
    truth=Products(root,['ph004'],targets=True)
    for arm in spec()['arms']:
        for replica in spec()['replicas']:
            for anchor in ledger['panels']['refinement']:
                values,powers,features,coarse,coarse_power={},{},{},{},{}
                y=truth.raw_targets(anchor)['rho']-1
                for steps in (250,500,1000):
                    task=next(t for t in tasks if t['arm']==arm and t['replica']==replica
                              and t['anchor']==anchor and t['checkpoint']==20480 and t['steps']==steps and t['purpose']=='main')
                    values[steps],receipts=ensemble(root,task,count=8)
                    inputs.update(receipts)
                    _,powers[steps]=density_spectra(values[steps],y)
                    if arm=='D':
                        coarse[steps]=parent_ensemble(root,dict(task,count=8))
                        _,coarse_power[steps]=density_spectra(coarse[steps]-1,truth.raw_targets(anchor)['coarse']-1,
                                                            cell=27.064,edges=WIDE_EDGES)
                    tensors=(composite_tensor(x,coarse[steps][i]-1)[CORE] if arm=='D'
                             else closure_tensor(x,'periodic')[CORE] for i,x in enumerate(values[steps]))
                    features[steps]=np.stack([core_features(x[CORE],tensor)[:,1:]
                                             for x,tensor in zip(values[steps],tensors)])
                early=paired_refinement(values[250],values[500],powers[250],powers[500])
                late=paired_refinement(values[500],values[1000],powers[500],powers[1000])
                et=paired_feature_widths(features[250],features[500])
                lt=paired_feature_widths(features[500],features[1000])
                wide=None if arm!='D' else dict(
                    early=paired_refinement(coarse[250],coarse[500],coarse_power[250],coarse_power[500],core=None),
                    late=paired_refinement(coarse[500],coarse[1000],coarse_power[500],coarse_power[1000],core=None))
                passed=refinement_pass(early,late,et,lt,wide)
                records.append(dict(arm=arm,replica=replica,anchor=anchor,early=early,late=late,
                                    tidal_early=et,tidal_late=lt,wide=wide,passed=passed))
    folder=root/'analysis'
    folder.mkdir(exist_ok=True)
    result=dict(manifest_sha256=existing.sha256(root/'MANIFEST.json'),inputs=inputs,records=records,
                passed=all(r['passed'] for r in records),automatic_sampler_search=False)
    durable.publish_json(folder/'SAMPLER_GATE.json',result)
    print('SAMPLER_GATE',result['passed'],flush=True)
    if not result['passed']:
        raise RuntimeError('new-model sampler gate failed; main sampling stopped')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['sampler-gate'])
    p.add_argument('--root',required=True,type=Path)
    sampler_gate(p.parse_args().root)


if __name__=='__main__':
    main()
