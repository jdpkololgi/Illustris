"""B1 galaxy reliability and B2 tied regional fractions, matched operators."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
import e2e_stage_b_prepare as p
import e2e_stage_b_metrics as m
import numpy as np
c=p.c;a=p.a


def regional(labels,truth,weights):
    draws,denom=m.region_fractions(labels,weights)
    actual,_=m.region_fractions(truth[None],weights)
    valid=denom>0
    if not valid.any():return dict(status='empty_support',denominator=denom.tolist())
    marginal=m.fraction_score(draws[:,valid],actual[0,valid])
    result=dict(denominator=denom.tolist(),valid_cores=valid.tolist(),marginal=marginal)
    if valid.all():
        result['joint_pair']=m.joint_fraction_score(draws[...,:3],actual[0,:,:3])
        result['difference']=m.fraction_score(draws[:,1]-draws[:,0],actual[0,1]-actual[0,0])
        result['joint_difference']=m.joint_fraction_score(draws[:,1,:3]-draws[:,0,:3],actual[0,1,:3]-actual[0,0,:3])
    else:result['joint_status']='unavailable_empty_core'
    return result


def summarize_regional(records):
    usable=[q for q in records if 'marginal' in q]
    if not usable:return dict(status='empty_support',cases=0)
    marginals=[q['marginal'] for q in usable]
    result=dict(cases=len(usable),cores=sum(sum(q['valid_cores']) for q in usable),
        missing_cases=len(records)-len(usable),coverage={},inclusive_coverage={},width={})
    # Each available core is equally weighted; all empty cases stay explicit.
    for key in ('bias','sd','crps','truth','mean'):
        result[key]=np.concatenate([np.asarray(q[key]) for q in marginals],axis=0).mean(0).tolist()
    for level in m.LEVELS:
        for key in ('coverage','inclusive_coverage','width'):
            result[key][str(level)]=np.concatenate([np.asarray(q[key][str(level)]) for q in marginals],axis=0).mean(0).tolist()
    result['reference']={str(level):level for level in m.LEVELS}
    result['screen_0p05']=['pass' if max(abs(result['coverage'][str(level)][k]-level) for level in m.LEVELS)<=.05 else 'fail' for k in range(4)]
    for key in ('joint_pair','joint_difference'):
        qq=[q[key] for q in usable if key in q]
        result[key]=dict(cases=len(qq),energy=float(np.mean([q['energy'] for q in qq])) if qq else None,
            coverage={str(level):float(np.mean([q['coverage'][str(level)] for q in qq])) if qq else None for level in m.LEVELS},
            degenerate_cases=sum(q['degenerate'] for q in qq))
        result[key]['screen_0p05']='pass' if qq and max(abs(result[key]['coverage'][str(level)]-level) for level in m.LEVELS)<=.05 else ('fail' if qq else 'unavailable')
    differences=[q['difference'] for q in usable if 'difference' in q]
    result['difference']=dict(cases=len(differences),
        bias=np.mean([q['bias'] for q in differences],axis=0).tolist() if differences else None,
        crps=np.mean([q['crps'] for q in differences],axis=0).tolist() if differences else None,
        coverage={str(level):np.mean([q['coverage'][str(level)] for q in differences],axis=0).tolist() if differences else None for level in m.LEVELS})
    result['evidence']='insufficient_independent_phases_for_qualification'
    return result


def controls():
    rng=np.random.default_rng(61007)
    ensembles=[rng.multinomial(8,[.15,.45,.3,.1],size=(33,2))/8,
               rng.dirichlet([2,3,4,1],size=(33,2)),np.tile([0.,0.,0.,1.],(33,2,1))]
    checks=[]
    for z in ensembles:
        scalar=[];joint=[]
        for i in range(len(z)):
            x=np.delete(z,i,axis=0)
            scalar.append(m.fraction_score(x,z[i]))
            joint.append(m.joint_fraction_score(x[...,:3],z[i,:,:3]))
        for level in m.LEVELS:
            np.testing.assert_allclose(np.mean([q['coverage'][str(level)] for q in scalar],axis=0),level,atol=1e-13)
            np.testing.assert_allclose(np.mean([q['coverage'][str(level)] for q in joint]),level,atol=1e-13)
        checks.append(dict(exact_member_permutations=33,dimensions=6,discrete=bool(np.unique(z).size<50),passed=True))
    probs=np.array([.1,.2,.5,.2]);draw=rng.choice(4,size=(32,40000),p=probs);truth=rng.choice(4,size=40000,p=probs)
    pp=np.stack([(draw==k).mean(0) for k in range(4)],axis=-1)
    q=m.merge_reliability([m.reliability(pp,truth,np.zeros(len(truth),int),draws=32)])
    expected=1-float(probs@probs)
    assert abs(q['fair_brier']-expected)<.01
    return dict(passed=True,exact_scalar_joint_oracles=checks,brier_expected=expected,brier_measured=q['fair_brier'],
        reference='nominal via auxiliary-randomization averaging, NOT Stage A continuous central order-interval reference')


def score(pilot=False):
    c.require_compute();tick=time.monotonic()
    out=p.ROOT/('score_pilot' if pilot else 'score');out.mkdir(parents=True,exist_ok=False)
    prepared_path=p.ROOT/'prepare'/'COMPLETE.json'
    prepared=json.loads(prepared_path.read_text());assert prepared['passed'] and prepared['panel_complete']
    replay_path=p.ROOT/'replay'/'COMPLETE.json'
    replay=json.loads(replay_path.read_text());assert replay['passed'] and replay['panel_complete']
    binding=dict(config=c.file_record(p.CONFIG,content_hash=True),source=c.file_record(__file__,content_hash=True),
        metrics=c.file_record(m.__file__,content_hash=True),prepared=c.file_record(prepared_path,content_hash=True),
        replay=c.file_record(replay_path,content_hash=True),job=os.environ['SLURM_JOB_ID'],pilot=pilot)
    assert replay['binding']['prepared']['sha256']==binding['prepared']['sha256']
    assert prepared['binding']['config']['sha256']==binding['config']['sha256']==replay['binding']['config']['sha256']
    c.atomic_json(out/'BINDING.json',binding)
    ctl=controls();c.atomic_json(out/'CONTROLS.json',ctl)
    records=[]
    chosen=replay['cases']
    if pilot:chosen=[next(q for q in chosen if q['phase']==phase) for phase in a.PHASES]
    for row in chosen:
        p.guard(row['phase']);pid=row['pair_id']
        pair=next(v for v in prepared['cases'] if v['pair_id']==pid)
        for q in (pair,row):assert c.sha256(q['output']['path'])==q['output']['sha256']
        with np.load(row['output']['path']) as f:labels=f['labels']
        assert labels.shape==(2,2,32,2,4096) and np.all(labels<=3)
        with np.load(pair['output']['path']) as f:
            counts=f['cell_counts'];truth=f['truth_cell_labels'];flat=f['flat_index']
            ids=f['targetid'];xyz=f['xyz'];z=f['z']
        assert len(np.unique(ids))==len(ids)==counts.sum() and np.all((z>=.15)&(z<.55))
        inside,reconstructed=p.cell_addresses(xyz,pair['origin_mpc_h'])
        assert inside.all() and np.array_equal(flat,reconstructed)
        coreids=np.array([pid+f'/core{side}' for side in flat//4096])
        result={k:row[k] for k in ('phase','cap','shell','support_stratum','pair_id')}
        result.update(galaxies=len(ids),core_galaxies=counts.sum(-1).tolist(),reliability={},regional={},sources=[pair['output'],row['output']])
        for ti,threshold in enumerate(p.THRESHOLDS):
            galaxytruth=truth[ti].reshape(-1)[flat]
            base=np.broadcast_to(prepared['baseline'][str(threshold)]['probabilities'],(len(flat),4))
            result['reliability'][f'baseline/{threshold}']=m.reliability(base,galaxytruth,coreids)
            for mi,method in enumerate(('original','corrected')):
                galaxylabels=labels[mi,ti].reshape(32,-1)[:,flat]
                probability=np.stack([(galaxylabels==k).mean(0) for k in range(4)],axis=-1)
                result['reliability'][f'{method}/{threshold}']=m.reliability(probability,galaxytruth,coreids,draws=32)
                for weightname,weights in [('volume',np.ones((2,4096))),('galaxy',counts)]:
                    result['regional'][f'{method}/{threshold}/{weightname}']=regional(labels[mi,ti],truth[ti],weights)
        c.atomic_json(out/(pid+'.json'),result);records.append(result)
    groups=[]
    for phase in a.PHASES:
        rows=[q for q in records if q['phase']==phase]
        selections=[('all',rows)]
        for field in ('cap','shell','support_stratum'):
            selections.extend((f'{field}={value}',[q for q in rows if q[field]==value]) for value in sorted({q[field] for q in rows}))
        for name,selected in selections:
            entry=dict(phase=phase,group=name,pairs=len(selected),independent_phases=1,
                reliability={key:m.merge_reliability([q['reliability'][key] for q in selected]) for key in selected[0]['reliability']},
                regional={key:summarize_regional([q['regional'][key] for q in selected]) for key in selected[0]['regional']})
            groups.append(entry)
    pooled=dict(weighting='equal-phase for Brier; equal-case/equal-core for regional summaries',independent_phases=2,
        fair_brier={key:float(np.mean([q['reliability'][key]['fair_brier'] for q in groups if q['group']=='all'])) for key in records[0]['reliability']},
        regional={key:summarize_regional([q['regional'][key] for q in records]) for key in records[0]['regional']})
    result=dict(passed=True,panel_complete=not pilot and len(records)==32,cases=records,groups=groups,pooled=pooled,
        binding=binding,controls=ctl,seconds=time.monotonic()-tick,posterior_qualified=False)
    c.atomic_json(out/'COMPLETE.json',result)
    print(json.dumps(dict(stage='b_score_complete',seconds=result['seconds'],pairs=len(records),out=str(out))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--pilot',action='store_true');args=parser.parse_args();score(args.pilot)
