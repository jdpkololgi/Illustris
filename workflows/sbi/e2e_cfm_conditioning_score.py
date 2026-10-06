"""Registered Tier 2 development scores and paired conditional-MC decisions.

No fitting, threshold changes, phase resampling, or posterior qualification.
The three independent fraction coordinates follow Stage C's joint statistic.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

RUN=Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/cfm_conditioning_augmentation_20261006_v3')
BASE=RUN.parent/'coherent_exterior_20261002_v1'
sys.path.insert(0,str(BASE/'source_78156ac3417555ef269f'))
sys.path.insert(1,str(RUN/'source/workflows/sbi'))
import numpy as np
from workflows.sbi import e2e_coupled_contract as c
import e2e_stage_a_calibration as a
import e2e_stage_b_metrics as bm
import e2e_stage_b_score as bs
import e2e_stage_c_mechanism_metrics as cm
import e2e_stage_c_mechanism as cs
import e2e_cfm_conditioning_gates as gates
import e2e_cfm_conditioning_metrics as m

SETTINGS=[('parent',0.),('control',0.)]+[('augmented',s) for s in (0.,.025,.05,.1,.2)]
METHODS=('original','corrected')


def read(path):return json.loads(Path(path).read_text())


def verify(record):
    if c.sha256(record['path'])!=record['sha256']:raise ValueError('Changed source: '+record['path'])


def armkey(arm,strength):return f'{arm}_s{strength:g}'


def load_bank(arm,strength):
    eigen=[];rho=[];truth=[];truthrho=[];weights=[];cases=[];sources=[]
    for phase in a.PHASES:
        for cap in ('NGC','SGC'):
            ee=[];rr=[];meta=None
            for first in range(0,32,4):
                path=RUN/'cores'/phase/cap/arm/f's{strength:g}'/f'd{first:03d}/COMPLETE.json'
                rec=read(path)
                if (not rec['passed'] or rec['smoke'] or rec['phase']!=phase or rec['cap']!=cap
                        or rec['first']!=first or rec['arm']!=arm or rec['strength']!=strength or rec['draws']!=4):
                    raise ValueError('Incomplete or mismatched core panel')
                verify(rec['output']);verify(rec['binding']);sources.append(c.file_record(path,content_hash=True))
                with np.load(rec['output']['path']) as f:
                    if f['eigen'].shape!=(2,8,4,2,4096,3) or f['density'].shape!=(8,4,2,4096):
                        raise ValueError('Core bank shape changed')
                    ee.append(f['eigen']);rr.append(f['density'])
                    names=list(f['pair_id'])
                    current=[f[k] for k in ('truth_eigen','truth_density','galaxy_counts')]
                    if first==0:fixed=current;meta=rec['cases'];expected=names
                    else:
                        if rec['cases']!=meta or names!=expected:raise ValueError('Core order changed')
                        for x,y in zip(fixed,current):np.testing.assert_array_equal(x,y)
            eigen.append(np.concatenate(ee,axis=2));rho.append(np.concatenate(rr,axis=1))
            truth.append(fixed[0]);truthrho.append(fixed[1]);weights.append(fixed[2]);cases.extend(meta)
    if len(cases)!=32 or len({r['pair_id'] for r in cases})!=32:raise ValueError('Registered 32-pair panel required')
    return (np.concatenate(eigen,axis=1),np.concatenate(rho),np.concatenate(truth),
            np.concatenate(truthrho),np.concatenate(weights),cases,sources)


def score_arm(arm,strength):
    if (arm,strength) not in SETTINGS:raise ValueError('Unregistered setting')
    c.require_compute();started=time.monotonic();key=armkey(arm,strength)
    dest=RUN/'scores'/key;dest.mkdir(parents=True,exist_ok=False)
    eigen,rho,truth,trho,weights,cases,sources=load_bank(arm,strength)
    index,counts=m.resamples();records=[];fractions={};actual={};kernels={};boot={}
    means=eigen.mean(-2);tmean=truth.mean(-2)
    for mi,method in enumerate(METHODS):
        labels=(eigen[mi]>.2).sum(-1)
        for weight,w in (('galaxy',weights),('volume',np.ones_like(weights))):
            xx=[];yy=[]
            for j in range(32):
                x,_=bm.region_fractions(labels[j],w[j]);y,_=bm.region_fractions((truth[j]>.2).sum(-1)[None],w[j])
                xx.append(x);yy.append(y[0])
            fractions[method+'/'+weight]=np.asarray(xx);actual[weight]=np.asarray(yy)
    for j,row in enumerate(cases):
        metrics={'cell_density':a.scalar_scores(rho[j],trho[j]),
                 'regional_density':a.scalar_scores(rho[j].mean(-1),trho[j].mean(-1)),
                 'paired_density':a.vector_scores(rho[j].mean(-1),trho[j].mean(-1))}
        regional={};reliability={}
        for mi,method in enumerate(METHODS):
            metrics.update({f'{method}/physical/{name}':v for name,v in a.eig_metrics(means[mi,j],tmean[j]).items()})
            for k in range(3):metrics[f'{method}/physical/cell_lambda{k+1}']=a.scalar_scores(eigen[mi,j,...,k],truth[j,...,k])
            for threshold in (.2,0.):
                labels=(eigen[mi,j]>threshold).sum(-1);target=(truth[j]>threshold).sum(-1)
                probability=np.stack([(labels==k).mean(0) for k in range(4)],-1)
                ids=np.repeat([row['pair_id']+'/core0',row['pair_id']+'/core1'],4096)
                for weight,w in (('volume',np.ones_like(weights[j])),('galaxy',weights[j])):
                    regional[f'{method}/{threshold}/{weight}']=bs.regional(labels,target,w)
                    # Preserve the original Stage B pooled-galaxy reliability.
                    select=np.repeat(np.arange(8192),w.reshape(-1).astype(int))
                    reliability[f'{method}/{threshold}/{weight}']=bm.reliability(
                        probability.reshape(-1,4)[select],target.reshape(-1)[select],ids[select],draws=32)
        records.append(dict(**row,metrics=metrics,regional=regional,reliability=reliability))
    groups=a.aggregate(records);reports=[];moment_reports=[]
    for phase in a.PHASES:
        phaseidx=np.array([j for j,r in enumerate(cases) if r['phase']==phase]);meta=[cases[j] for j in phaseidx]
        for group,localidx in cs.selections(meta):
            ix=phaseidx[localidx];rr=[records[j] for j in ix]
            reports.append(dict(phase=phase,group=group,pairs=len(ix),
                regional={k:bs.summarize_regional([r['regional'][k] for r in rr]) for k in rr[0]['regional']},
                reliability={k:bm.merge_reliability([r['reliability'][k] for r in rr]) for k in rr[0]['reliability']}))
            for method in METHODS:
                for weight in ('galaxy','volume'):
                    x=fractions[method+'/'+weight][ix];y=actual[weight][ix]
                    moment_reports.append(dict(phase=phase,group=group,operator=method,weighting=weight,
                        classwise=cm.core_moments(x,y),joint=cm.moments(x[...,:3].reshape(len(x),32,6),y[...,:3].reshape(len(y),6)),
                        cross_core=cm.cross_core(x,y)))
        interior=np.array([j for j in phaseidx if cases[j]['support_stratum']=='interior'])
        if len(interior)!=8:raise ValueError('Registered Stage C interior selection changed')
        for mi,method in enumerate(METHODS):
            gap=np.diff(means[mi,phaseidx],axis=-1).reshape(len(phaseidx),32,4)
            tg=np.diff(tmean[phaseidx],axis=-1).reshape(len(phaseidx),4)
            frac=fractions[method+'/galaxy'][interior,...,:3].reshape(len(interior),32,6)
            tf=actual['galaxy'][interior,...,:3].reshape(len(interior),6)
            for target,kernel in [('gap_energy',m.energy_kernel(gap,tg)),('interior_fraction_energy',m.energy_kernel(frac,tf))]:
                prefix=phase+'/'+method+'/'+target;kernels[prefix]=kernel
            lab=(eigen[mi,phaseidx]>.2).sum(-1).transpose(1,0,2,3).reshape(32,-1)
            target=(truth[phaseidx]>.2).sum(-1).reshape(-1)
            for name,w in [('cell',np.ones_like(target)),('galaxy',weights[phaseidx].reshape(-1))]:
                kernels[phase+'/'+method+'/'+name+'_brier']=m.brier_kernel(lab,target,w)
            boot[phase+'/'+method+'/interior_ratio']=m.ratio_samples(frac,tf,counts)
        for name,x,y in [('cell_density',rho[phaseidx],trho[phaseidx])]+[
                (f'cell_lambda{k+1}',eigen[1,phaseidx,...,k],truth[phaseidx,...,k]) for k in range(3)]:
            values=x.transpose(1,0,2,3).reshape(32,-1);target=y.reshape(-1)
            boot[phase+'/'+name+'/curve']=m.cell_curve_samples(values,target,counts)
        print(json.dumps(dict(stage='score_arm',arm=key,phase=phase,seconds=time.monotonic()-started)),flush=True)
    # Parent score reconciliation independently checks the weighted aggregation.
    parent_checks=0
    if arm=='parent':
        old_a=read(a.ROOT/'a2/COMPLETE.json');old_b=read(bs.p.ROOT/'score/COMPLETE.json')
        for group in groups:
            prior=next(q for q in old_a['groups'] if q['phase']==group['phase'] and q['group']==group['group'])
            for k,q in group['metrics'].items():
                if k not in prior['metrics']:continue
                old=prior['metrics'][k]
                for fld in ('coverage','bias','crps','energy'):
                    if fld not in q:continue
                    if isinstance(q[fld],dict):np.testing.assert_allclose(list(q[fld].values()),list(old[fld].values()),rtol=0,atol=1e-12)
                    else:np.testing.assert_allclose(q[fld],old[fld],rtol=0,atol=1e-12)
                    parent_checks+=1
        for phase in a.PHASES:
            prior=next(q for q in old_b['groups'] if q['phase']==phase and q['group']=='all')
            for method in METHODS:
                got=m.score_from_kernel(kernels[phase+'/'+method+'/galaxy_brier'])
                np.testing.assert_allclose(got,prior['reliability'][method+'/0.2']['fair_brier'],rtol=0,atol=1e-12);parent_checks+=1
    arrays={}
    for name,(first,pair) in kernels.items():arrays[name+'/first']=first;arrays[name+'/pair']=pair
    arrays.update(boot)
    with (dest/'MC_KERNELS.npz').open('xb') as stream:np.savez_compressed(stream,**arrays)
    result=dict(passed=True,panel_complete=True,arm=arm,strength=strength,draws=32,cases=records,groups=groups,
        class_reports=reports,moments=moment_reports,parent_reconciliation_checks=parent_checks,
        proper_scores={k:m.score_from_kernel(v) for k,v in kernels.items()},
        mc_sensitivity={k:dict(lower=np.quantile(v,.05,axis=0).tolist(),upper=np.quantile(v,.95,axis=0).tolist()) for k,v in boot.items()},
        mc=c.file_record(dest/'MC_KERNELS.npz',content_hash=True),inputs=sources,
        sources=[c.file_record(q,content_hash=True) for q in (__file__,m.__file__,a.__file__,bm.__file__,bs.__file__,cm.__file__,cs.__file__)],
        seconds=time.monotonic()-started,job=os.environ['SLURM_JOB_ID'],posterior_qualified=False)
    c.atomic_json(dest/'COMPLETE.json',result)


def summarize():
    c.require_compute();dest=RUN/'decision';dest.mkdir(exist_ok=False)
    points={};arrays={};sources=[];oracle=[]
    for arm,s in SETTINGS:
        path=RUN/'scores'/armkey(arm,s)/'COMPLETE.json';q=read(path)
        if not q['passed'] or not q['panel_complete'] or q['arm']!=arm or q['strength']!=s:raise ValueError('Score bank mismatch')
        verify(q['mc']);points[armkey(arm,s)]=q;sources.append(c.file_record(path,content_hash=True))
        with np.load(q['mc']['path']) as f:arrays[armkey(arm,s)]={k:f[k] for k in f.files}
    for row in points['parent_s0']['cases']:
        path=RUN/'oracle'/row['pair_id']/'COMPLETE.json';q=read(path)
        if not q['passed'] or q['smoke'] or q['draws']!=32 or not q['exact_parent_replay']:raise ValueError('Oracle panel incomplete')
        oracle.append(q);sources.append(c.file_record(path,content_hash=True))
    def oracle_bank(key,field):
        rows=[]
        for phase in a.PHASES:
            selected=[q for q in oracle if q['phase']==phase]
            if len(selected)!=16:raise ValueError('Oracle phase panel incomplete')
            rows.append(np.mean([q['parent'][field] if key=='parent' else q['arms'][key][field] for q in selected],axis=0))
        return np.moveaxis(np.asarray(rows),0,1)
    truthp=np.asarray([np.mean([q['truth_power'] for q in oracle if q['phase']==phase],axis=0) for phase in a.PHASES])[:,1:3]
    truthw=np.asarray([np.mean([q['truth_width'] for q in oracle if q['phase']==phase]) for phase in a.PHASES])
    refs={name:oracle_bank('parent' if name=='parent' else 'control/s0','width') for name in ('parent','control')}
    oracle_results={}
    for arm,s in SETTINGS:
        name='parent' if arm=='parent' else f'{arm}/s{s:g}'
        oracle_results[armkey(arm,s)]=gates.oracle_screen(oracle_bank(name,'power')[:,:,1:3],oracle_bank(name,'width'),truthp,truthw,refs)
    _,counts=m.resamples();decisions=[]
    for arm,s in SETTINGS:
        key=armkey(arm,s);q=points[key];bank=arrays[key];control=arrays['control_s0'];checks=[];comparisons={}
        for phase in a.PHASES:
            group=next(g for g in q['groups'] if g['phase']==phase and g['group']=='all')
            for feature in ['cell_density']+[f'corrected/physical/cell_lambda{k}' for k in (1,2,3)]:
                stat=group['metrics'][feature]
                checks.append(dict(name=phase+'/'+feature,decision=stat['screen_0p05'],coverage=stat['coverage'],reference=stat['reference']))
            moment=next(v for v in q['moments'] if v['phase']==phase and v['group']=='support_stratum=interior'
                and v['operator']=='corrected' and v['weighting']=='galaxy')['joint']
            ratio=moment['trace_adjusted_mse_over_variance']
            checks.append(dict(name=phase+'/interior_fraction_trace_ratio',point=ratio,limits=[.67,1.5],
                decision='pass' if ratio is not None and .67<=ratio<=1.5 else 'fail'))
            for method in METHODS:
                for score in ('gap_energy','interior_fraction_energy','cell_brier','galaxy_brier'):
                    name=phase+'/'+method+'/'+score
                    kernel=lambda v:(v[name+'/first'],v[name+'/pair'])
                    comparison=m.deterioration(kernel(bank),kernel(control),counts)
                    comparison['versus_parent']=m.deterioration(kernel(bank),kernel(arrays['parent_s0']),counts)
                    comparisons[name]=comparison
                    if method=='corrected':checks.append(dict(name=name,**comparison))
        decision=m.combine([v['decision'] for v in checks]+[oracle_results[key]['decision']]
            +([oracle_results['augmented_s0']['decision']] if arm=='augmented' else []))
        decisions.append(dict(arm=arm,strength=s,decision=decision,checks=checks,proper=comparisons,
            oracle=oracle_results[key],clean_oracle_required=arm=='augmented'))
    selected=next((q['strength'] for q in decisions if q['arm']=='augmented' and q['decision']=='pass'),None)
    result=dict(passed=True,evaluation_complete=True,selected_strength=selected,
        scientific_decision='development_candidate_pass' if selected is not None else 'no_setting_passes_all_registered_gates',
        settings=decisions,inputs=sources,bootstrap_replicates=4096,bootstrap_seed=6100617,
        source=c.file_record(__file__,content_hash=True),metrics=c.file_record(m.__file__,content_hash=True),
        scope='Two reused phases, one trained seed; pointwise conditional MC, not independent/population/familywise confirmation.',
        posterior_qualified=False,confirmation_access=False)
    c.atomic_json(dest/'COMPLETE.json',result)
    print(json.dumps(dict(scientific_decision=result['scientific_decision'],selected_strength=selected)),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['arm','summarize'])
    ap.add_argument('--arm',choices=['parent','control','augmented']);ap.add_argument('--strength',type=float,default=0.)
    args=ap.parse_args()
    if args.action=='arm':score_arm(args.arm,args.strength)
    else:summarize()
