"""Bounded saved-draw mechanism discrimination. No model fits or field sampling."""
import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
import e2e_stage_b_prepare as p
import e2e_stage_b_metrics as b
import e2e_stage_c_mechanism_metrics as m
import numpy as np
c=p.c
CONFIG=Path(__file__).resolve().parents[2]/'configs/e2e_stage_c_mechanism_20261006.json'
ROOT=p.a.ROOT/'c_mechanism'
METHODS=('original','corrected')
WEIGHTS=('galaxy','volume')


def score(x,y):
    marginal=b.fraction_score(x.transpose(1,0,2,3),y)
    result={key:np.asarray(marginal[key]).mean((0,1)).tolist() for key in ('bias','sd','crps')}
    result['coverage']={key:np.asarray(v).mean((0,1)).tolist() for key,v in marginal['coverage'].items()}
    result['width']={key:np.asarray(v).mean((0,1)).tolist() for key,v in marginal['width'].items()}
    for name,draw,truth in [('joint_pair',x[...,:3],y[...,:3]),
            ('joint_difference',x[:,:,1,:3]-x[:,:,0,:3],y[:,1,:3]-y[:,0,:3])]:
        q=[b.joint_fraction_score(xx,yy) for xx,yy in zip(draw,truth)]
        result[name]=dict(coverage={str(v):float(np.mean([r['coverage'][str(v)] for r in q])) for v in b.LEVELS},
            energy=float(np.mean([r['energy'] for r in q])),degenerate=sum(r['degenerate'] for r in q))
    return result


def selections(rows):
    yield 'all',np.arange(len(rows))
    for field in ('cap','shell','support_stratum'):
        for value in sorted({r[field] for r in rows}):
            yield f'{field}={value}',np.array([i for i,r in enumerate(rows) if r[field]==value])


def pairing_probe(x,y):
    pair_scores=[];difference_scores=[];cross=[]
    for offset,z in m.cyclic_pairings(x):
        for side in (0,1):
            np.testing.assert_array_equal(np.sort(z[:,side],axis=0),np.sort(x[:,side],axis=0))
        pair_scores.append(b.joint_fraction_score(z[...,:3],y[:,:3]))
        difference_scores.append(b.joint_fraction_score(z[:,1,:3]-z[:,0,:3],y[1,:3]-y[0,:3]))
        cross.append(np.cov(z.reshape(32,8),rowvar=False)[:4,4:])
    np.testing.assert_allclose(np.mean(cross,0),0,atol=2e-15)
    out={}
    for name,qq in [('joint_pair',pair_scores),('joint_difference',difference_scores)]:
        out[name]=dict(coverage={str(v):[q['coverage'][str(v)] for q in qq] for v in b.LEVELS},
            energy=[q['energy'] for q in qq])
    out['max_abs_mean_cross_covariance']=float(abs(np.mean(cross,0)).max())
    return out


def verified(path,sha=None):
    record=c.file_record(path,content_hash=True)
    if sha is not None:assert record['sha256']==sha
    return json.loads(Path(path).read_text()),record


def run(pilot=False):
    c.require_compute();started=time.monotonic()
    cfg=json.loads(CONFIG.read_text())
    assert tuple(cfg['phases'])==p.a.PHASES and cfg['draws']==32 and not cfg['training'] and not cfg['new_sampling']
    dest=ROOT/('pilot' if pilot else 'full');dest.mkdir(parents=True,exist_ok=False)
    review,rr=verified(p.ROOT/'REVIEW.json','8bfb45b027aad1f30d370dff730886958d7c1da16a973fc11afdce1d8ed5e341')
    assert review['passed'] and review['checks']==1184
    baseline,br=verified(p.ROOT/'score/COMPLETE.json',review['score']['sha256'])
    prep,pr=verified(p.ROOT/'prepare/COMPLETE.json',baseline['binding']['prepared']['sha256'])
    replay,rp=verified(p.ROOT/'replay/COMPLETE.json',baseline['binding']['replay']['sha256'])
    a_review,ar=verified(p.a.ROOT/'REVIEW.json')
    a1,a1r=verified(p.a.ROOT/'a1/COMPLETE.json',a_review['a1']['sha256'])
    assert a_review['passed'] and a1['passed'] and a1['panel_complete']
    binding=dict(config=c.file_record(CONFIG,content_hash=True),job=os.environ['SLURM_JOB_ID'],
        source=[c.file_record(q,content_hash=True) for q in (__file__,m.__file__,b.__file__,p.__file__,p.a.__file__)],
        b_review=rr,b_score=br,prepared=pr,replay=rp,a_review=ar,a1=a1r,pilot=pilot)
    c.atomic_json(dest/'BINDING.json',binding)
    rows=baseline['cases']
    if pilot:rows=[next(r for r in rows if r['phase']==phase) for phase in p.a.PHASES]
    assert len({r['pair_id'] for r in rows})==len(rows)==(2 if pilot else 32)
    values=[];targets=[];metadata=[];a_values={};a_targets={};a_sources=[];checks=0
    for row in rows:
        p.guard(row['phase']);pid=row['pair_id']
        prepared=next(q for q in prep['cases'] if q['pair_id']==pid)
        generated=next(q for q in replay['cases'] if q['pair_id']==pid)
        for q in (prepared,generated):assert c.sha256(q['output']['path'])==q['output']['sha256']
        with np.load(prepared['output']['path']) as f:counts=f['cell_counts'];truth=f['truth_cell_labels']
        with np.load(generated['output']['path']) as f:labels=f['labels']
        assert counts.shape==(2,4096) and (counts.sum(-1)>0).all() and labels.shape==(2,2,32,2,4096)
        xx=np.empty((2,2,2,32,2,4));yy=np.empty((2,2,2,4))
        for mi,method in enumerate(METHODS):
            for ti,t in enumerate(p.THRESHOLDS):
                for wi,weight in enumerate(WEIGHTS):
                    w=counts if weight=='galaxy' else np.ones_like(counts)
                    x,_=b.region_fractions(labels[mi,ti],w);actual,_=b.region_fractions(truth[ti][None],w);y=actual[0]
                    np.testing.assert_allclose(x.sum(-1),1,atol=1e-14);np.testing.assert_allclose(y.sum(-1),1,atol=1e-14)
                    xx[mi,ti,wi]=x;yy[ti,wi]=y
                    old=row['regional'][f'{method}/{t}/{weight}'];new=b.fraction_score(x,y)
                    for key in ('bias','sd','crps','mean','truth'):
                        np.testing.assert_allclose(new[key],old['marginal'][key],atol=1e-14,rtol=0);checks+=1
                    joint=b.joint_fraction_score(x[...,:3],y[:,:3])
                    assert joint['coverage']==old['joint_pair']['coverage'];checks+=1
        values.append(xx);targets.append(yy)
        metadata.append({k:row[k] for k in ('phase','pair_id','cap','shell','support_stratum','core_galaxies')})
        old_a=next(q for q in a1['cases'] if q['pair_id']==pid)
        path=p.a.ROOT/'a1'/(pid+'_SUMMARIES.npz');a_sources.append(c.file_record(path,content_hash=True))
        with np.load(path) as f:
            for arm,factor in [('canvas_coarse_local_fine','coarse'),('canvas_coarse_local_fine','fine'),('oracle_coarse_local_fine','fine')]:
                key=f'{arm}/{factor}';x=f[f'{arm}_{factor}_shear'];y=f[f'truth_{factor}_shear']
                a_values.setdefault(key,[]).append(x);a_targets.setdefault(key,[]).append(y)
                # Reconcile saved arrays against the source-bound Stage A score.
                q=p.a.scalar_scores(x,y);old=old_a['metrics'][f'{arm}/{factor}_projections']
                for field in ('bias','mse','variance','crps','mean','truth'):
                    np.testing.assert_allclose(q[field],old[field],rtol=0,atol=1e-13);checks+=1
        print(json.dumps(dict(stage='c_inputs',pair_id=pid,checks=checks)),flush=True)
    values,targets=np.array(values),np.array(targets)
    arrays=dict(fractions=values,truth=targets,pair_id=np.array([r['pair_id'] for r in metadata]))
    arrays.update({key.replace('/','_')+'_draws':np.array(v) for key,v in a_values.items()})
    arrays.update({key.replace('/','_')+'_truth':np.array(v) for key,v in a_targets.items()})
    np.savez(dest/'PANEL.npz',**arrays)
    phase_indices={phase:np.array([i for i,r in enumerate(metadata) if r['phase']==phase]) for phase in p.a.PHASES}
    results=[];factors=[];shuffle_cases=[]
    for phase,index in phase_indices.items():
        other=next(q for q in p.a.PHASES if q!=phase);opposite=phase_indices[other]
        meta=[metadata[i] for i in index]
        for mi,method in enumerate(METHODS):
            for ti,t in enumerate(p.THRESHOLDS):
                for wi,weight in enumerate(WEIGHTS):
                    x=values[index,mi,ti,wi];y=targets[index,ti,wi]
                    ox=values[opposite,mi,ti,wi];oy=targets[opposite,ti,wi]
                    transfer_bias=(ox.mean(1)-oy).mean((0,1))
                    probes={}
                    for shift in (False,True):
                        for scale in cfg['fixed_spread_multipliers']:
                            z,invalid=m.affine_probe(x,transfer_bias if shift else np.zeros(4),scale)
                            name=f'{"other_phase_bias" if shift else "no_bias"}/scale{scale:.1f}'
                            probes[name]=dict(score=score(z,y),support=invalid)
                    for name,selected in selections(meta):
                        xx,yy=x[selected],y[selected]
                        entry=dict(phase=phase,operator=method,threshold=t,weighting=weight,group=name,pairs=len(selected),
                            classwise=m.core_moments(xx,yy),joint=m.moments(xx[...,:3].reshape(len(xx),32,6),yy[...,:3].reshape(len(yy),6)),
                            cross_core=m.cross_core(xx,yy))
                        if name=='all':
                            entry.update(other_phase=other,transfer_bias=transfer_bias.tolist(),sensitivity=probes)
                            if t==.2:
                                shuffled=[pairing_probe(v,u) for v,u in zip(x,y)]
                                entry['cyclic_pairing']={}
                                for key in ('joint_pair','joint_difference'):
                                    entry['cyclic_pairing'][key]=dict(coverage={str(level):float(np.mean([q[key]['coverage'][str(level)] for q in shuffled])) for level in b.LEVELS},
                                        energy=float(np.mean([q[key]['energy'] for q in shuffled])))
                                for r,q in zip(meta,shuffled):shuffle_cases.append(dict(pair_id=r['pair_id'],operator=method,weighting=weight,result=q))
                        results.append(entry)
                    print(json.dumps(dict(stage='c_fraction_diagnostics',phase=phase,operator=method,threshold=t,weighting=weight)),flush=True)
        for key,draw in a_values.items():
            x=np.array(draw)[index];y=np.array(a_targets[key])[index]
            for name,selected in selections(meta):
                xx,yy=x[selected],y[selected]
                factors.append(dict(phase=phase,factor=key,group=name,pairs=len(selected),
                    projection=m.core_moments(xx,yy),joint=m.moments(xx.reshape(len(xx),32,-1),yy.reshape(len(yy),-1))))
    result=dict(passed=True,panel_complete=not pilot and len(rows)==32,binding=binding,cases=metadata,
        panel=c.file_record(dest/'PANEL.npz',content_hash=True),stage_a_summaries=a_sources,
        source_parity_checks=checks,fraction_results=results,factor_results=factors,cyclic_case_results=shuffle_cases,
        seconds=time.monotonic()-started,max_host_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
        posterior_qualified=False,physical_intervention_tested=False)
    c.atomic_json(dest/'COMPLETE.json',result)
    print(json.dumps(dict(stage='c_complete',pairs=len(rows),seconds=result['seconds'],max_host_rss_gib=result['max_host_rss_gib'],out=str(dest))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--pilot',action='store_true');args=parser.parse_args();run(args.pilot)
