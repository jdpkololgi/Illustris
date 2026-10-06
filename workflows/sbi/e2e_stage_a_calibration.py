"""Stage A saved-draw factor diagnostics. Development phases only, no fitting."""
import argparse
import gc
import json
import os
from pathlib import Path
import resource
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2e_true_density_closure as b
import numpy as np
import h5py
from workflows.sbi.e2e_vdm_context_metrics import calibration, fair_energy, variogram_score
from workflows.sbi.e2e_exterior_metrics import joint_rank

ROOT = b.ROOT.parent / 'calibration_investigation_20261006_v1'
SAVED = b.ROOT / 'verification/sampler_factor_20261005_v1'
PHASES = ('ph012', 'ph013')
LEVELS = (.5, .68, .9, .95)
ARMS = ('canvas_raw', 'canvas_coarse_local_fine', 'local_both', 'oracle_coarse_local_fine')
CROP = (slice(16,32), slice(18,30), slice(18,30))


def guard(phase):
    if phase not in PHASES:
        raise PermissionError('Stage A permits only ph012/ph013')


def shear_coordinates(t):
    """Orthonormal traceless coordinates for xx,xy,xz,yy,yz,zz ordering."""
    t = np.asarray(t, float)
    return np.stack(((t[...,0]-t[...,3])/np.sqrt(2),
        (t[...,0]+t[...,3]-2*t[...,5])/np.sqrt(6),
        np.sqrt(2)*t[...,1], np.sqrt(2)*t[...,2], np.sqrt(2)*t[...,4]), axis=-1)


def scalar_scores(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if not y.size:
        return dict(kind='scalar', status='empty_support', targets=0)
    q = calibration(x, y)
    v = q['std']**2
    return dict(kind='scalar', targets=int(y.size), mean=float(q['mean'].mean()),
        truth=float(y.mean()), bias=float(q['bias'].mean()), mse=float(np.mean(q['bias']**2)),
        variance=float(v.mean()), mse_minus_draw_noise=float(np.mean(q['bias']**2-v/len(x))),
        crps=float(q['crps'].mean()), sd=float(q['std'].mean()),
        coverage={str(p):float(q[f'covered_{p}'].mean()) for p in LEVELS},
        width={str(p):float(q[f'width_{p}'].mean()) for p in LEVELS},
        reference=q['attainable'], rank_histogram=np.histogram(q['rank'],np.linspace(0,1,9))[0].tolist())


def vector_scores(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x.reshape(len(x), -1), y.ravel()
    if not y.size:
        return dict(kind='vector', status='empty_support', dimensions=0)
    if np.max(np.std(x,axis=0)) < 1e-12:
        return dict(kind='vector', status='conditioned_or_degenerate',
                    absolute_error=float(np.max(abs(x-y))), dimensions=len(y))
    rank = joint_rank(x,y)
    return dict(kind='vector', dimensions=len(y), rank=rank['rank'],
        coverage={k:float(v) for k,v in rank['covered'].items()},
        reference=rank['attainable_no_ties'], energy=float(fair_energy(x,y)),
        variogram=variogram_score(x,y) if len(y)>1 else None)


def eig_metrics(x,y):
    x,y=np.asarray(x),np.asarray(y)
    trace=x.sum(-1);ty=y.sum(-1);gaps=np.diff(x,axis=-1);tg=np.diff(y,axis=-1)
    ans={f'lambda{i+1}':scalar_scores(x[...,i],y[...,i]) for i in range(3)}
    ans.update(trace=scalar_scores(trace,ty),gap12=scalar_scores(gaps[...,0],tg[...,0]),
        gap23=scalar_scores(gaps[...,1],tg[...,1]),width=scalar_scores(x[...,-1]-x[...,0],y[...,-1]-y[...,0]),
        joint6=vector_scores(x,y),joint4gap=vector_scores(gaps,tg),joint2mass=vector_scores(trace,ty),
        mass_sum_difference=vector_scores(np.stack((trace.mean(-1),trace[:,1]-trace[:,0]),-1),
                                         np.array([ty.mean(),ty[1]-ty[0]])))
    return ans


def synthetic_controls():
    rng=np.random.default_rng(61006); y=rng.normal(size=20000); x=rng.normal(size=(32,20000))
    scalars={k:scalar_scores(v,y) for k,v in [('oracle',x),('biased',x+1.5),('underdispersed',x*.3)]}
    for p in LEVELS:
        assert abs(scalars['oracle']['coverage'][str(p)]-scalars['oracle']['reference'][str(p)])<.02
    assert all(scalars[k]['coverage']['0.9']<scalars['oracle']['reference']['0.9']-.15
               for k in ('biased','underdispersed'))
    good=[];bad=[];ranks=[];narrow=[]
    for _ in range(512):
        z=rng.normal(size=(33,4)); z[:,2:]=.95*z[:,:2]+np.sqrt(1-.95**2)*z[:,2:]
        truth,draw=z[0],z[1:]; shuffled=draw.copy();shuffled[:,2:]=draw[rng.permutation(32),2:]
        np.testing.assert_allclose(np.sort(draw,axis=0),np.sort(shuffled,axis=0))
        good.append(variogram_score(draw,truth));bad.append(variogram_score(shuffled,truth))
        ranks.append(vector_scores(draw,truth));narrow.append(vector_scores(draw*.2,truth))
    joint={str(p):float(np.mean([q['coverage'][str(p)] for q in ranks])) for p in LEVELS}
    # An initial 512-case stochastic control exceeded a fixed 0.06 band at
    # 50% (0.552734 vs 0.484848). Preserve it as a measured Monte Carlo result.
    # Test exchangeability without a stochastic acceptance band: cycle every
    # member of each fixed ensemble through the truth position.
    permutation_checks=[]
    for sample in (rng.normal(size=(33,4)),rng.standard_t(4,size=(33,4)),
                   rng.normal(size=(33,6)),rng.normal(size=(33,10))):
        got=[joint_rank(np.delete(sample,i,axis=0),sample[i]) for i in range(33)]
        values={str(p):float(np.mean([v['covered'][str(p)] for v in got])) for p in LEVELS}
        for p in LEVELS:np.testing.assert_allclose(values[str(p)],got[0]['attainable_no_ties'][str(p)],atol=1e-14)
        permutation_checks.append(dict(dimensions=sample.shape[1],coverage=values))
    assert np.mean([q['coverage']['0.9'] for q in narrow])<.4
    assert np.mean(bad)>np.mean(good)
    # Discrete oracle: inclusive central intervals need not have the continuous null.
    d=rng.binomial(4,.3,size=(33,20000))/4
    tied=calibration(d[1:],d[0],seed=928)
    tied_cdf={str(p):float(np.mean(tied['rank']<=p)) for p in LEVELS}
    assert all(abs(tied_cdf[str(p)]-p)<.02 for p in LEVELS)
    fixed=vector_scores(np.ones((32,2)),np.ones(2));assert fixed['status']=='conditioned_or_degenerate'
    assert scalar_scores(np.empty((32,0)),np.empty(0))['status']=='empty_support'
    # Direct all-pairs arithmetic is independent of the sorted CRPS formula.
    small=x[:,:7];target=y[:7]
    direct=abs(small-target).mean(0)-abs(small[:,None]-small[None,:]).sum((0,1))/(2*32*31)
    np.testing.assert_allclose(direct,calibration(small,target)['crps'],atol=1e-14)
    return dict(passed=True,synthetic=True,marginal=scalars,joint_coverage=joint,
        joint_reference=ranks[0]['reference'],joint_permutation_checks=permutation_checks,
        stochastic_joint_control_note='Initial fixed 0.06 band failed at 50%; retained raw Monte Carlo result; deterministic all-member exchangeability controls govern scorer integrity.',
        joint_underdispersed90=float(np.mean([q['coverage']['0.9'] for q in narrow])),
        variogram_oracle=float(np.mean(good)),variogram_shuffled=float(np.mean(bad)),
        discrete_randomized_rank_cdf=tied_cdf,conditioned_control=fixed,
        caveat='Oracle repetitions validate scorers, not the CFM population posterior.')


def aggregate(rows):
    groups=[]
    for phase in PHASES:
        rr=[r for r in rows if r['phase']==phase]
        if not rr:continue
        selections=[('all',rr)]
        for field in ('cap','shell','support_stratum'):
            selections.extend((f'{field}={v}',[r for r in rr if r[field]==v]) for v in sorted({r[field] for r in rr}))
        for label,chosen in selections:
            metrics={}
            for key in chosen[0]['metrics']:
                qq=[r['metrics'][key] for r in chosen]
                if any('status' in q for q in qq):continue
                entry=dict(kind=qq[0]['kind'],cases=len(qq),independent_phases=1,
                    coverage={str(p):float(np.mean([q['coverage'][str(p)] for q in qq])) for p in LEVELS},
                    reference=qq[0]['reference'])
                for fld in ('crps','bias','mse','variance','mse_minus_draw_noise','sd','mean','truth','energy','variogram'):
                    if fld in qq[0] and qq[0][fld] is not None:entry[fld]=float(np.mean([q[fld] for q in qq]))
                if 'width' in qq[0]:entry['width']={str(p):float(np.mean([q['width'][str(p)] for q in qq])) for p in LEVELS}
                error={str(p):entry['coverage'][str(p)]-entry['reference'][str(p)] for p in LEVELS}
                entry['coverage_error']=error
                entry['screen_0p05']='pass' if max(abs(v) for v in error.values())<=.05 else 'fail'
                entry['evidential_status']='insufficient_independent_phases_for_population_equivalence'
                if label!='all':
                    entry['split_deficit90_0p10']='insufficient_cores' if len(qq)<2 else ('fail' if error['0.9']<-.10 else 'pass')
                metrics[key]=entry
            groups.append(dict(phase=phase,group=label,pairs=len(chosen),metrics=metrics))
    return groups


def covariance_diagnostic(x,y):
    x,y=np.asarray(x,float),np.asarray(y,float)
    errors=x.mean(1)-y
    cov=np.mean([np.cov(v.T) for v in x],axis=0)
    floor=max(float(np.trace(cov))/cov.shape[0]*1e-8,1e-15)
    eig,vec=np.linalg.eigh(cov);w=(vec/np.sqrt(np.maximum(eig,floor)))@vec.T
    scatter=np.cov(errors.T)-cov/x.shape[1]
    return dict(mean_bias=errors.mean(0).tolist(),
        demeaned_scatter_eigenvalues=np.linalg.eigvalsh(w@scatter@w.T).tolist(),
        covariance_floor=floor,
        caveat='Dependent cases; descriptive de-meaned error scatter minus draw-mean noise; negative eigenvalues retained.')


def run(limit=0,attempt=1):
    b.c.require_compute();tick=time.monotonic()
    name='a1' if not limit else f'a1_pilot{limit}'
    if attempt!=1:name+=f'_attempt{attempt}'
    dest=ROOT/name;dest.mkdir(parents=True,exist_ok=False)
    prepared_path=SAVED/'PREPARED.json';prepared=json.loads(prepared_path.read_text());assert prepared['passed']
    binding=dict(schema='cfm-stage-a1-v1',phases=list(PHASES),draws=32,levels=list(LEVELS),
        source=b.c.file_record(__file__,content_hash=True),freeze=b.c.file_record(b.FREEZE,content_hash=True),
        prepared=b.c.file_record(prepared_path,content_hash=True),job=os.environ['SLURM_JOB_ID'],
        metric_sources=[b.c.file_record(f.__code__.co_filename,content_hash=True) for f in (calibration,joint_rank,fair_energy,variogram_score)],
        limit_per_phase=limit or None,arms=list(ARMS),operator='historical local+wide for factor controls; capwise free-space for full-cap summaries',
        units='two independent reused phases; pairs/cores/draws dependent',screen_tolerance=.05,
        primary=['canvas_raw/coarse_shear_rms','canvas_raw/coarse_joint_shear',
                 'oracle_coarse_local_fine/fine_shear_rms','oracle_coarse_local_fine/fine_joint_shear',
                 'fullcap/physical/joint4gap','fullcap/physical/joint6'])
    b.c.atomic_json(dest/'BINDING.json',binding)
    control=synthetic_controls();b.c.atomic_json(dest/'CONTROLS.json',control)
    cases=[];sources=[];vectors={}
    for phase in PHASES:
        guard(phase);geometry={r['pair_id']:r for r in b.auth.geometry(phase,b.FREEZE)}
        complete_path=SAVED/'results'/f'COMPLETE_{phase}.json';complete=json.loads(complete_path.read_text())
        assert complete['passed'] and complete['draws']==32 and complete['nfe']==128
        assert complete['prepared']['sha256']==b.c.sha256(prepared_path)
        freeze=json.loads(b.FREEZE.read_text())
        assert complete['checkpoints']==freeze['checkpoints'] and complete['normalizer']==freeze['normalizer']
        sources.append(b.c.file_record(complete_path,content_hash=True))
        regional_path=b.ROOT/'verification/trace_shear_20261005_v1'/f'{phase}_REGIONAL_DRAWS.npz'
        regional_record=json.loads(regional_path.with_name(phase+'.json').read_text())
        assert b.c.sha256(regional_path)==regional_record['draws']['sha256']
        with np.load(regional_path) as f:regional={k:f[k] for k in f.files}
        sources.append(b.c.file_record(regional_path,content_hash=True))
        chosen=complete['cases'][:limit] if limit else complete['cases']
        assert len(complete['cases'])==16 and len({r['pair_id'] for r in complete['cases']})==16
        for row in chosen:
            before=time.monotonic();pid=row['pair_id'];meta=geometry[pid];guard(row['phase'])
            assert row['phase']==phase and row['draws']==32
            assert b.c.sha256(row['input']['path'])==row['input']['sha256']
            assert b.c.sha256(row['fields']['path'])==row['fields']['sha256']
            with h5py.File(row['input']['path']) as f:
                rho,wide,start=f['true_rho'][:],f['true_wide'][:],f['core_start'][:]
                joint=f['joint'][:]
            cores=[tuple(slice(int(a),int(a+16)) for a in start+np.array([16*i,0,0])) for i in (0,1)]
            lifted=b.op.lift(wide[CROP]); residual=rho-lifted
            fine_full=b.op.tensor_from_delta(residual,6.766,workers=8)
            ft=np.stack([fine_full[sl] for sl in cores]);del fine_full
            coarse_full=b.op.tensor_from_delta(b.op.lift(wide-1),6.766,workers=8)
            crop=tuple(slice(4*s.start,4*s.stop) for s in CROP)
            ct=np.stack([coarse_full[crop][sl] for sl in cores]);del coarse_full
            matched=b.tidal.eigensystem(ct+ft).mean((1,2,3))
            np.testing.assert_allclose(matched,row['matched'],atol=1e-10,rtol=0)
            np.testing.assert_allclose(b.tidal.eigensystem(ct).mean((1,2,3)),row['truth_scale']['coarse_eigenvalues'],atol=1e-10,rtol=0)
            truthc=shear_coordinates(ct.mean((1,2,3)));truthf=shear_coordinates(ft.mean((1,2,3)))
            metrics={};draw_arrays={}
            for arm in ARMS:
                scale=row['arms'][arm]['scale'];e=np.asarray(row['arms'][arm]['eigenvalues'])
                for target,y in [('matched',matched),('physical',np.asarray(row['fullbox']))]:
                    for name,val in eig_metrics(e,y).items():
                        if arm.startswith('oracle') and name in ('trace','joint2mass','mass_sum_difference'):continue
                        metrics[f'{arm}/{target}/{name}']=val
                cs=shear_coordinates([v['coarse_tensor_mean'] for v in scale])
                fs=shear_coordinates([v['fine_tensor_mean'] for v in scale])
                for factor,s,t in [('coarse',cs,truthc),('fine',fs,truthf)]:
                    if factor=='coarse' and arm.startswith('oracle'):continue
                    metrics[f'{arm}/{factor}_shear_rms']=scalar_scores(
                        np.array([v[f'{factor}_shear_rms'] for v in scale]),np.asarray(row['truth_scale'][f'{factor}_shear_rms']))
                    metrics[f'{arm}/{factor}_projections']=scalar_scores(s,t)
                    metrics[f'{arm}/{factor}_joint_shear']=vector_scores(s,t)
                    draw_arrays[f'{arm}_{factor}_shear']=s;draw_arrays[f'truth_{factor}_shear']=t
                metrics[f'{arm}/fine_residual_variance']=scalar_scores(
                    np.array([v['fine_residual_variance'] for v in scale]),np.asarray(row['truth_scale']['fine_residual_variance']))
                if not arm.startswith('oracle'):
                    combined=np.concatenate((cs.reshape(32,-1),fs.reshape(32,-1)),axis=1)
                    truth=np.r_[truthc.ravel(),truthf.ravel()]
                    metrics[f'{arm}/joint_cross_scale']=vector_scores(combined,truth)
                    vectors.setdefault((phase,arm),[]).append((combined,truth))
            idx=list(regional['pair_id']).index(pid);x=regional['eigenvalues'][idx]
            draw_arrays['fullcap_eigenvalues']=x
            for target,key in [('physical','fullbox'),('matched','matched')]:
                y=regional[key][idx];draw_arrays['fullcap_'+target+'_truth']=y
                for name,val in eig_metrics(x,y).items():metrics[f'fullcap/{target}/{name}']=val
            # Raw count/support channels are bound by the original condition schema.
            support=[float(joint[1][sl].mean()) for sl in cores]
            counts=[float(np.expm1(joint[0][sl]).sum()) for sl in cores]
            rec=dict(phase=phase,pair_id=pid,cap=meta['cap'],shell=meta['shell'],redshift=meta['redshift'],
                support_stratum=meta['support_stratum'],support_fraction=support,observed_counts=counts,
                metrics=metrics,input=row['input'],fields=row['fields'],seconds=time.monotonic()-before)
            np.savez(dest/(pid+'_SUMMARIES.npz'),**draw_arrays)
            b.c.atomic_json(dest/(pid+'.json'),rec);cases.append(rec)
            print(json.dumps(dict(stage='a1',phase=phase,pair_id=pid,seconds=rec['seconds'])),flush=True)
            gc.collect()
    cov={f'{p}/{a}':covariance_diagnostic(np.array([v[0] for v in values]),np.array([v[1] for v in values]))
         for (p,a),values in vectors.items() if len(values)>1}
    result=dict(passed=True,panel_complete=not limit and len(cases)==32,posterior_qualified=False,
        cases=cases,groups=aggregate(cases),cross_scale_covariance=cov,sources=sources,binding=binding,
        seconds=time.monotonic()-tick,max_host_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)
    b.c.atomic_json(dest/'COMPLETE.json',result)
    print(json.dumps(dict(stage='a1_complete',cases=len(cases),seconds=result['seconds'],out=str(dest))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--attempt',type=int,default=1)
    args=parser.parse_args();run(args.limit,args.attempt)
