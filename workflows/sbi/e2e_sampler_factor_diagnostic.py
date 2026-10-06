"""Matched production-window controls, phases012/013 only, frozen weights.

Compare the saved canvas and a single production window using identical input
channels, coarse/fine initial noise, cells and one local+wide tidal operator.
Oracle coarse is privileged conditioning, not a deployable posterior.
"""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2e_true_density_closure as b
import numpy as np
import h5py

OUT = b.ROOT/'verification/sampler_factor_20261005_v1'


def slices(start, shape):
    return tuple(slice(int(a), int(a+n)) for a,n in zip(start,shape))


def prepare():
    b.c.require_compute(); OUT.mkdir(parents=True, exist_ok=False)
    records=[]
    for phase in ('ph012','ph013'):
        b.auth.frozen(phase,b.FREEZE)
        geometry=b.auth.geometry(phase,b.FREEZE)
        for cap in ('NGC','SGC'):
            root=b.ROOT/'fullcap_inputs'/phase/cap
            prepared=json.loads((root/'PREPARED.json').read_text())
            assert b.c.sha256(root/'PLAN.json')==prepared['plan_sha256']
            plan=json.loads((root/'PLAN.json').read_text())
            mids=np.asarray(plan['midpoints'])
            with h5py.File(prepared['outputs'][0]['path']) as cf, h5py.File(b.OUT/phase/(cap+'_TRUE_SOURCE.h5')) as tf:
                origin=np.asarray(tf.attrs['source_origin_mpc_h'])
                for row in (r for r in geometry if r['cap']==cap):
                    pid=row['pair_id']; midpoint=np.asarray(row['center'])+[16,0,0]
                    # Geometry-only choice, nearest production window; deterministic ties.
                    eligible=np.all(abs(mids-midpoint)<=np.array([32,32,32]),axis=1)
                    distances=np.sum((mids-midpoint)**2,axis=1)
                    idx=int(np.argmin(np.where(eligible,distances,np.inf)))
                    assert eligible[idx]
                    mid=mids[idx];fine_start=(mid-[64,48,48])//2
                    core_start=(np.asarray(row['center'])-16)//2-fine_start
                    assert np.all(core_start>=0) and np.all(core_start+[32,16,16]<=[64,48,48])
                    true,prov=b.auth.truth(phase,pid,b.FREEZE)
                    wstart=4+(mid-midpoint)//8
                    assert np.all(wstart>=0) and np.all(wstart+48<=56)
                    truewide=true['coarse_rho_extended'][slices(wstart,(48,48,48))]
                    physical=np.asarray(row['grid']['origin_mpc_h'])+fine_start*6.766
                    tstart=np.rint((physical-origin)/6.766).astype(int)
                    assert np.all(tstart>=0) and np.all(tstart+[64,48,48]<=tf['rho'].shape)
                    truerho=tf['rho'][slices(tstart,(64,48,48))]
                    np.testing.assert_allclose(b.op.mean_pool(truerho),truewide[16:32,18:30,18:30],atol=2e-6,rtol=0)
                    joint,wide=cf['joint'][idx],cf['wide'][idx]
                    import hashlib
                    assert hashlib.sha256(joint.tobytes()+wide.tobytes()).hexdigest()==prepared['condition_sha256'][idx]
                    path=OUT/(pid+'_INPUT.h5')
                    with h5py.File(path,'x') as f:
                        for key,value in dict(joint=joint,wide=wide,true_wide=truewide,true_rho=truerho,
                            fullbox_tensor=true['fullbox_tensor_cores'],midpoint=mid,core_start=core_start).items():
                            f.create_dataset(key,data=value)
                        f.create_dataset('canvas_rho',shape=(32,64,48,48),dtype='f4')
                        f.create_dataset('completed_rho',shape=(32,64,48,48),dtype='f4')
                        f.create_dataset('canvas_z',shape=(32,48,48,48),dtype='f4')
                        f.attrs.update(phase=phase,cap=cap,pair_id=pid,window_index=idx)
                    rec=dict(phase=phase,cap=cap,pair_id=pid,window_index=idx,midpoint=mid.tolist(),
                        registered_midpoint=midpoint.tolist(),core_start=core_start.tolist(),truth=prov,
                        fine_origin_mpc_h=physical.tolist(),input=str(path))
                    records.append(rec)
            selected=[r for r in records if r['phase']==phase and r['cap']==cap]
            for first in range(0,32,4):
                dr=b.ROOT/'fullcap_density'/phase/cap/f'd{first:03d}'/'COMPLETE.json'
                tr=b.ROOT/'fullcap_tides'/phase/cap/f'd{first:03d}'/'COMPLETE.json'
                d=json.loads(dr.read_text());t=json.loads(tr.read_text())
                assert d['passed'] and t['passed'] and d['first']==t['first']==first
                # Full payload verification once per chunk; no confirmation IO.
                assert b.c.sha256(d['field']['path'])==d['field']['sha256']
                assert b.c.sha256(t['output']['path'])==t['output']['sha256']
                with h5py.File(d['field']['path']) as df,h5py.File(t['output']['path']) as ef:
                    origin=np.asarray(df.attrs['origin_mpc_h'])
                    corigin=np.asarray(df.attrs['coarse_origin_raw_cells'])
                    eorigin=np.asarray(ef.attrs['completed_origin_mpc_h'])
                    for r in selected:
                        raw=np.asarray(r['fine_origin_mpc_h']);mid=np.asarray(r['midpoint'])
                        fs=np.rint((raw-origin)/6.766).astype(int)
                        es=np.rint((raw-eorigin)/6.766).astype(int)
                        cs=(mid-192-corigin)//8
                        assert np.all(cs>=0) and np.all(cs+48<=df['coarse_logrho_standardized'].shape[1:])
                        with h5py.File(r['input'],'r+') as f:
                            f['canvas_rho'][first:first+4]=df['rho'][(slice(None),*slices(fs,(64,48,48)))]
                            f['completed_rho'][first:first+4]=ef['rho_completed'][(slice(None),*slices(es,(64,48,48)))]
                            f['canvas_z'][first:first+4]=df['coarse_logrho_standardized'][(slice(None),*slices(cs,(48,48,48)))]
                print(json.dumps(dict(stage='extract',phase=phase,cap=cap,first=first)),flush=True)
            for r in selected:
                with h5py.File(r['input']) as f:
                    valid=np.isfinite(f['canvas_rho'][:])
                    assert np.array_equal(valid, np.broadcast_to(valid[0],valid.shape))
                    core=tuple(slices(np.asarray(r['core_start'])+[16*side,0,0],(16,16,16)) for side in (0,1))
                    assert all(valid[(slice(None),*sl)].all() for sl in core)
                    r['window_uncovered_fraction']=float(1-valid.mean())
                r['file']=b.c.file_record(r['input'],content_hash=True)
    b.c.atomic_json(OUT/'PREPARED.json',dict(cases=records,passed=True,source_sha256=b.c.sha256(__file__),
        freeze_sha256=b.c.sha256(b.FREEZE),job=os.environ['SLURM_JOB_ID']))


def noise_crop(geometry, index, phase, cap, stage, first, count):
    import torch
    from workflows.sbi.e2e_vdm_context_models import project
    values=[]
    for draw in range(first,first+count):
        seed=int(b.c.digest(['global-exterior-cap-v1',phase,cap,stage,draw])[:15],16)
        gen=torch.Generator(device='cuda').manual_seed(seed)
        z=torch.randn((1,1,*geometry.shape),device='cuda',generator=gen)
        if stage=='fine':z=project(z)
        values.append(z[geometry.crop(index)].clone())
    return torch.cat(values)


def run(limit, draws, phase_filter=None):
    b.c.require_compute()
    import torch
    from workflows.sbi.e2e_loa_field_sampling import setup
    from workflows.sbi import e2e_cfm_pilot_evaluate as ev
    from workflows.sbi import e2e_coupled_views as views
    from workflows.sbi.e2e_coupled_benchmark_models import FieldCondition
    from workflows.sbi.e2e_vdm_context_models import replicate,decode_density,block_mean
    from workflows.sbi.e2e_global_canvas import Geometry,integrate
    from e2e_checkpoint_shear import LocalOperator
    from e2e_trace_shear_diagnostic import features,assess as assess32
    def assess(value, target):
        if len(value)==32:
            return assess32(value,target)
        return dict(mean=value.mean(0).tolist(),sd=value.std(0,ddof=1).tolist(),
                    truth=np.asarray(target).tolist(),scope='Four-draw engineering smoke; no coverage claim')
    setup(True)
    prepared=json.loads((OUT/'PREPARED.json').read_text())
    assert prepared['passed'] and prepared['freeze_sha256']==b.c.sha256(b.FREEZE)
    models,hashes,normalizer=ev.load_models(17,13312,26624)
    freeze=json.loads(b.FREEZE.read_text())
    assert hashes==freeze['checkpoints'] and normalizer==freeze['normalizer']
    chart=views.load_chart(normalizer)
    dest=OUT/('results' if limit==0 and draws==32 else f'pilot_v2_{limit}_{draws}')
    dest.mkdir(exist_ok=True)
    chosen=[r for r in prepared['cases'] if phase_filter is None or r['phase']==phase_filter]
    if limit:chosen=chosen[:limit]
    records=[]
    operator=LocalOperator()
    for row in chosen:
        pid=row['pair_id'];path=dest/(pid+'.json')
        if path.exists():
            prior=json.loads(path.read_text())
            assert prior['source_sha256']==b.c.sha256(__file__) and prior['draws']==draws
            records.append(prior);continue
        phase,cap=row['phase'],row['cap'];b.auth.frozen(phase,b.FREEZE)
        assert b.c.sha256(row['input'])==row['file']['sha256']
        with h5py.File(row['input']) as f:arrays={k:f[k][:] for k in f}
        plan=json.loads((b.ROOT/'fullcap_inputs'/phase/cap/'PLAN.json').read_text())
        cg,fg=(Geometry.make(plan['midpoints'],stage) for stage in ('coarse','fine'))
        single={stage:Geometry.make([row['midpoint']],stage) for stage in ('coarse','fine')}
        core=np.asarray(row['core_start'])
        operator.cores=[slices(core+[16*side,0,0],(16,16,16)) for side in (0,1)]
        crop=(slice(16,32),slice(18,30),slice(18,30))
        true_tensor=operator.tensor(arrays['true_rho'],arrays['true_wide'])
        full=b.tidal.eigensystem(arrays['fullbox_tensor']).mean((1,2,3))
        matched=b.tidal.eigensystem(true_tensor).mean((1,2,3))
        valid=np.isfinite(arrays['canvas_rho'][0])
        arms={name:[] for name in ('canvas_raw','canvas_completed','canvas_coarse_local_fine','local_both','oracle_coarse_local_fine')}
        bands={name:[] for name in arms}
        max_mass=0.;tick=time.monotonic()
        targetz=(np.log(arrays['true_wide'])-chart['coarse_logrho']['mean'][0])/chart['coarse_logrho']['std'][0]
        fieldpath=dest/(pid+'_DRAWS.h5')
        with h5py.File(fieldpath,'x') as saved,torch.no_grad():
            for name in arms:
                group=saved.create_group(name)
                group.create_dataset('rho',shape=(draws,64,48,48),dtype='f4',compression='lzf')
                group.create_dataset('wide',shape=(draws,48,48,48),dtype='f4',compression='lzf')
            for first in range(0,draws,4):
                count=min(4,draws-first)
                tensor=lambda x:torch.as_tensor(x,device='cuda',dtype=torch.float32)[None].expand(count,*x.shape)
                base=FieldCondition(tensor(views.scale_channels(arrays['joint'],chart['joint'])),
                    tensor(views.scale_channels(arrays['wide'],chart['wide'])),tensor(np.zeros(3,'f4')),'wide')
                nc=noise_crop(cg,row['window_index'],phase,cap,'coarse',first,count)
                nf=noise_crop(fg,row['window_index'],phase,cap,'fine',first,count)
                localz,_=integrate(models['coarse'],'coarse',[base],nc,single['coarse'],128)
                canvasz=torch.as_tensor(arrays['canvas_z'][first:first+count,None],device='cuda')
                oraclez=tensor(targetz[None])
                generated={}
                for name,z in [('canvas_coarse_local_fine',canvasz),('local_both',localz),
                               ('oracle_coarse_local_fine',oraclez)]:
                    fine=replace(base,region='joint',coarse_joint=replicate(z[(slice(None),slice(None),*crop)]),
                        coarse_wide=z,coarse_source='oracle_diagnostic' if name.startswith('oracle') else 'sampled')
                    u,_=integrate(models['fine'],'fine',[fine],nf,single['fine'],128)
                    w=torch.exp(z.double()*chart['coarse_logrho']['std'][0]+chart['coarse_logrho']['mean'][0])
                    r=decode_density(w[(slice(None),slice(None),*crop)],u.double()*chart['fine_residual']['std'][0])
                    max_mass=max(max_mass,float(((block_mean(r)-w[(slice(None),slice(None),*crop)])/w[(slice(None),slice(None),*crop)]).abs().max()))
                    generated[name]=(r[:,0].cpu().numpy(),w[:,0].cpu().numpy())
                cw=np.exp(arrays['canvas_z'][first:first+count].astype(float)*chart['coarse_logrho']['std'][0]+chart['coarse_logrho']['mean'][0])
                generated['canvas_raw']=(arrays['canvas_rho'][first:first+count].copy(),cw)
                generated['canvas_completed']=(arrays['completed_rho'][first:first+count].copy(),cw)
                for name,(rho,wide) in generated.items():
                    for j,(r,w) in enumerate(zip(rho,wide)):
                        # Common deterministic coarse-only fill if a window contains
                        # cells never covered by the production canvas.
                        lifted=b.op.lift(w[crop]) if hasattr(b.op,'lift') else np.repeat(np.repeat(np.repeat(w[crop],4,0),4,1),4,2)
                        r=np.where(valid,r,lifted)
                        assert np.isfinite(r).all() and r.min()>0
                        t=operator.tensor(r,w)
                        eig=b.tidal.eigensystem(t).mean((1,2,3))
                        expected=np.array([r[sl].mean()-1 for sl in operator.cores])
                        np.testing.assert_allclose(eig.sum(-1),expected,atol=1e-9,rtol=0)
                        arms[name].append(eig)
                        # Coarse-only tidal contribution on the same cells/operator.
                        ct=operator.tensor(lifted,w)
                        ce=b.tidal.eigensystem(ct).mean((1,2,3))
                        bands[name].append(dict(coarse_eigenvalues=ce.tolist(),
                            coarse_tensor_mean=ct.mean((1,2,3)).tolist(),
                            fine_tensor_mean=(t-ct).mean((1,2,3)).tolist(),
                            coarse_shear_rms=shear_rms(ct),fine_shear_rms=shear_rms(t-ct),
                            fine_residual_variance=float(np.mean((r-lifted)**2))))
                        saved[name+'/rho'][first+j]=r
                        saved[name+'/wide'][first+j]=w
                print(json.dumps(dict(pair_id=pid,draws_done=first+count,seconds=time.monotonic()-tick)),flush=True)
        assert max_mass<2e-6
        truth_lift=np.repeat(np.repeat(np.repeat(arrays['true_wide'][crop],4,0),4,1),4,2)
        truthct=operator.tensor(truth_lift,arrays['true_wide'])
        truthscale=dict(coarse_shear_rms=shear_rms(truthct),fine_shear_rms=shear_rms(true_tensor-truthct),
            coarse_eigenvalues=b.tidal.eigensystem(truthct).mean((1,2,3)).tolist(),
            fine_residual_variance=float(np.mean((arrays['true_rho']-truth_lift)**2)))
        rec=dict(phase=phase,cap=cap,pair_id=pid,draws=draws,source_sha256=b.c.sha256(__file__),
            input=row['file'],fullbox=full.tolist(),matched=matched.tolist(),truth_scale=truthscale,
            window_uncovered_fraction=row['window_uncovered_fraction'],max_mass_error=max_mass,
            seconds=time.monotonic()-tick,arms={})
        for name,values in arms.items():
            x=np.asarray(values)
            rec['arms'][name]=dict(eigenvalues=x.tolist(),scale=bands[name],
                fullbox={k:assess(v,features(full)[k]) for k,v in features(x).items()},
                matched={k:assess(v,features(matched)[k]) for k,v in features(x).items()})
        rec['fields']=b.c.file_record(fieldpath,content_hash=True)
        b.c.atomic_json(path,rec);records.append(rec)
    receipt='COMPLETE.json' if phase_filter is None else f'COMPLETE_{phase_filter}.json'
    b.c.atomic_json(dest/receipt,dict(cases=records,passed=True,posterior_qualified=False,
        checkpoints=hashes,normalizer=normalizer,nfe=128,draws=draws,
        source_sha256=b.c.sha256(__file__),prepared=b.c.file_record(OUT/'PREPARED.json',content_hash=True),
        job=os.environ['SLURM_JOB_ID'],scope='Development production windows; common historical local+wide operator. Oracle arms are conditional diagnostics.'))


def shear_rms(tensor):
    value=np.array(tensor,copy=True)
    trace=value[...,[0,3,5]].sum(-1)/3
    value[...,[0,3,5]]-=trace[...,None]
    return float(np.sqrt(np.mean(np.sum(value**2*np.array([1,2,2,1,2,1]),axis=-1))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prepare','run'],required=True)
    p.add_argument('--limit',type=int,default=0);p.add_argument('--draws',type=int,choices=[4,32],default=32)
    p.add_argument('--phase',choices=['ph012','ph013'])
    a=p.parse_args()
    if a.stage=='prepare':prepare()
    else:run(a.limit,a.draws,a.phase)
