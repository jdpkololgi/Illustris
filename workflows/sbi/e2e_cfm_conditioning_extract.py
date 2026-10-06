"""Compact physical R7 core draws under original and corrected cap operators.

New fields use the unchanged learned exterior prior and its original addressed
noise. Float32 completion round-trip matches Stage A/B replay arithmetic.
Only registered development cores are retained; no new physical smoothing.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import resource
import sys
import time

RUN=Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/cfm_conditioning_augmentation_20261006_v3')
BASE=RUN.parent/'coherent_exterior_20261002_v1'
EVAL_SOURCE=BASE/'source_78156ac3417555ef269f'
CONFIG=RUN/'source/configs/e2e_cfm_conditioning_augmentation_v1.json'
sys.path.insert(0,str(EVAL_SOURCE))
sys.path.insert(1,str(RUN/'source/workflows/sbi'))
import h5py
import numpy as np
import torch
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_spatial_exterior as spatial
from workflows.sbi.e2e_exterior_cap_tides import CapTidal
from workflows.sbi.e2e_loa_field_sampling import setup
from workflows.sbi.e2e_coupled_target_products import owned_core_arrays
import e2e_stage_b_prepare as prep
import e2e_density_only_reconstruction as reconstruction

b=prep.b


def read(path):
    return json.loads(Path(path).read_text())


def verify(record):
    if c.sha256(record['path'])!=record['sha256']:
        raise ValueError('Changed payload: '+record['path'])


def run(phase,cap,first,arm,strength,smoke=False,recomplete=False):
    if phase not in ('ph012','ph013') or cap not in ('NGC','SGC'):
        raise PermissionError('Development caps only')
    if first not in range(0,32,4) or arm not in ('parent','control','augmented'):
        raise ValueError('Unregistered draw/arm')
    if strength not in (0.,.025,.05,.1,.2) or (arm!='augmented' and strength!=0):
        raise ValueError('Unregistered strength')
    c.require_compute();setup(True);torch.set_num_threads(4)
    if c.REPO!=EVAL_SOURCE:
        raise PermissionError('Original frozen evaluation snapshot required')
    if recomplete and (not smoke or arm!='parent'):raise ValueError('Recompletion is a parent engineering smoke only')
    scope='extract_recomplete_smoke' if recomplete else ('extract_smoke_v2' if smoke else 'cores')
    dest=RUN/scope/phase/cap/arm/f's{strength:g}'/f'd{first:03d}'
    dest.mkdir(parents=True,exist_ok=False)
    receipt=(BASE/'fullcap_density'/phase/cap/f'd{first:03d}/COMPLETE.json' if arm=='parent'
             else RUN/'fields'/phase/cap/arm/f's{strength:g}'/f'd{first:03d}/COMPLETE.json')
    record=read(receipt)
    if not record['passed'] or record['phase']!=phase or record['cap']!=cap or record['first']!=first:
        raise ValueError('Field address mismatch')
    frozen_sources=read(RUN/'SMOKE.json')['sources']
    for name,sha in frozen_sources.items():
        if c.sha256(RUN/'source/workflows/sbi'/name)!=sha:raise ValueError('Training source changed')
    if arm!='parent' and (record['sources']!=frozen_sources or record['config_sha256']!=c.sha256(CONFIG)
            or record['arm']!=arm or record['strength']!=strength or record['draws']!=4 or record['nfe']!=128):
        raise ValueError('Generation contract mismatch')
    verify(record['field'])
    prepared_path=prep.ROOT/'prepare/COMPLETE.json';prepared=read(prepared_path)
    if not prepared['passed'] or not prepared['panel_complete']:
        raise ValueError('Incomplete fixed galaxy selection')
    rows=[v for v in b.auth.geometry(phase,b.FREEZE) if v['cap']==cap]
    if len(rows)!=8:
        raise ValueError('Eight registered pairs per cap required')
    tick=time.monotonic();binding=dict(source=c.file_record(__file__,content_hash=True),
        field_receipt=c.file_record(receipt,content_hash=True),field=record['field'],
        prepared=c.file_record(prepared_path,content_hash=True),freeze=c.file_record(b.FREEZE,content_hash=True),
        config=c.file_record(CONFIG,content_hash=True),
        dependencies={str(Path(v.__file__).name):c.sha256(v.__file__) for v in (spatial,reconstruction)},
        tidal_source=c.file_record(sys.modules[CapTidal.__module__].__file__,content_hash=True),
        job=os.environ['SLURM_JOB_ID'])
    c.atomic_json(dest/'BINDING.json',binding)
    truth=[];truthrho=[];counts=[];sources=[]
    for row in rows:
        values,provenance=b.auth.truth(phase,row['pair_id'],b.FREEZE);sources.append(provenance)
        truth.append(b.tidal.eigensystem(values['fullbox_tensor_cores']).reshape(2,4096,3))
        truthrho.append(owned_core_arrays(values['rho_joint']).reshape(2,4096))
        selection=next(q for q in prepared['cases'] if q['pair_id']==row['pair_id'])
        verify(selection['output']);sources.append(selection['output'])
        with np.load(selection['output']['path']) as f:
            w=f['cell_counts'];counts.append(w)
            np.testing.assert_array_equal((truth[-1]>.2).sum(-1),f['truth_cell_labels'][0])
        if np.any(w.sum(-1)<=0):raise ValueError('Empty registered galaxy core')
    truth,truthrho,counts=map(np.asarray,(truth,truthrho,counts))
    np.testing.assert_allclose(truth.sum(-1),truthrho-1,rtol=0,atol=2e-6)
    oldfile=None;prior=None;prior_record=None;baseline_record=None
    with h5py.File(record['field']['path']) as f:
        shape=f['rho'].shape[1:];origin=np.asarray(f.attrs['origin_mpc_h']);support=f['support'][:]
        if f['rho'].shape[0]!=4 or f.attrs['phase']!=phase or f.attrs['cap']!=cap or f.attrs['first_draw']!=first:
            raise ValueError('HDF5 field address mismatch')
        if arm=='parent':
            baseline_path=BASE/'fullcap_tides'/phase/cap/f'd{first:03d}/COMPLETE.json'
            baseline=read(baseline_path);verify(baseline['output']);baseline_record=c.file_record(baseline_path,content_hash=True)
            if not baseline['passed'] or baseline['draws']!=4 or baseline['first']!=first:
                raise ValueError('Baseline tides address mismatch')
            oldfile=h5py.File(baseline['output']['path'])
        if arm!='parent' or recomplete:
            prior_path=BASE/'prior/PRIOR.json';fit=read(prior_path)
            if fit['training_phases']!=list(c.TRAIN):raise ValueError('Exterior prior is not train-only')
            prior_record=c.file_record(prior_path,content_hash=True)
            prior=spatial.CudaSpatialPrior(tuple(n+64 for n in shape),fit,workers=4)
        operator=CapTidal(shape,32)
        localcoarse=CapTidal((32,16,16),32);localfine=CapTidal((64,32,32),64)
        cores=[b.cuts(row,origin) for row in rows]
        patches=[tuple(slice(v.start-33,v.stop+33) for v in b.cuts(row,origin-32*6.766)) for row in rows]
        eigen=np.empty((2,8,4,2,4096,3));densitycores=np.empty((8,4,2,4096));diagnostics=[]
        trace=parity=0.
        try:
            for draw in range(4):
                if prior is not None:
                    seed=int(c.digest(['spatial-exterior-cap-v1',phase,cap,first+draw])[:15],16)
                    completed,diagnostic=spatial.complete_density(f['rho'][draw],support,prior,np.random.default_rng(seed),'learned',32)
                    density=completed.astype('f4').astype(float);del completed
                    if oldfile is not None:
                        np.testing.assert_array_equal(density,oldfile['rho_completed'][draw])
                else:
                    density=oldfile['rho_completed'][draw].astype(float);diagnostic={}
                tensor=np.empty((8,2,16,16,16,6))
                for k,component in enumerate(operator.components(density)):
                    for j,core in enumerate(cores):tensor[j,...,k]=component[core].reshape(2,16,16,16)
                for j,(core,patch) in enumerate(zip(cores,patches)):
                    coarse=density[patch];fine=reconstruction.reconstruct(coarse,'linear')[2:-2,2:-2,2:-2]
                    correction=np.empty_like(tensor[j])
                    for k,(tc,tf) in enumerate(zip(localcoarse.components(coarse[1:-1,1:-1,1:-1]),localfine.components(fine))):
                        correction[...,k]=(b.op.mean_pool(tf,2)-tc).reshape(2,16,16,16)
                    trace=max(trace,float(abs(correction[...,[0,3,5]].sum(-1)).max()))
                    for mi,t in enumerate((tensor[j],tensor[j]+correction)):
                        eigen[mi,j,draw]=b.tidal.eigensystem(t).reshape(2,4096,3)
                    densitycores[j,draw]=f['rho'][(draw,*core)].reshape(2,4096)
                    np.testing.assert_allclose(eigen[0,j,draw].sum(-1),densitycores[j,draw]-1,rtol=0,atol=1e-9)
                    if oldfile is not None:
                        saved=oldfile['eigenvalues'][(draw,*core,slice(None))].reshape(2,4096,3)
                        parity=max(parity,float(abs(saved-eigen[0,j,draw]).max()))
                if trace>1e-9 or parity>2e-7:raise ValueError('Tidal replay identity failed')
                diagnostic.update(draw=first+draw,seconds=time.monotonic()-tick);diagnostics.append(diagnostic)
                print(json.dumps(dict(stage='extract',phase=phase,cap=cap,arm=arm,strength=strength,
                    draw=first+draw,seconds=time.monotonic()-tick)),flush=True)
                del density,tensor,component,coarse,fine,correction,tc,tf
                gc.collect()
        finally:
            if oldfile is not None:oldfile.close()
    if not np.isfinite(eigen).all() or not np.isfinite(densitycores).all() or np.any(densitycores<=0):
        raise ValueError('Nonfinite or nonpositive core draws')
    stage_parity=0.
    if arm=='parent':
        replay=read(prep.ROOT/'replay/COMPLETE.json')
        for j,row in enumerate(rows):
            summary=prep.a.ROOT/'a2'/(row['pair_id']+'_SUMMARIES.npz');sources.append(c.file_record(summary,content_hash=True))
            labelrecord=next(q for q in replay['cases'] if q['pair_id']==row['pair_id'])['output'];verify(labelrecord);sources.append(labelrecord)
            with np.load(summary) as f,np.load(labelrecord['path']) as g:
                for mi,method in enumerate(('original','corrected')):
                    err=float(abs(eigen[mi,j].mean(2)-f[method][first:first+4]).max());stage_parity=max(stage_parity,err)
                    np.testing.assert_allclose(eigen[mi,j].mean(2),f[method][first:first+4],rtol=0,atol=1e-10)
                    for ti,threshold in enumerate(prep.THRESHOLDS):
                        np.testing.assert_array_equal((eigen[mi,j]>threshold).sum(-1),g['labels'][mi,ti,first:first+4])
    payload=dest/'CORES.npz'
    with payload.open('xb') as stream:
        np.savez_compressed(stream,eigen=eigen,density=densitycores,truth_eigen=truth,truth_density=truthrho,
            galaxy_counts=counts,pair_id=np.array([r['pair_id'] for r in rows]))
    result=dict(passed=True,phase=phase,cap=cap,first=first,arm=arm,strength=strength,draws=4,smoke=smoke,
        output=c.file_record(payload,content_hash=True),binding=c.file_record(dest/'BINDING.json',content_hash=True),
        cases=[{k:r[k] for k in ('phase','cap','pair_id','shell','support_stratum')} for r in rows],
        truth_sources=sources,prior=prior_record,baseline=baseline_record,diagnostics=diagnostics,
        correction_trace_maxabs=trace,original_parity_maxabs=parity,stage_ab_parity_maxabs=stage_parity,
        stage_b_labels_exact=arm=='parent',exact_exterior_recompletion=recomplete,seconds=time.monotonic()-tick,
        peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,
        max_host_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
        posterior_qualified=False)
    c.atomic_json(dest/'COMPLETE.json',result)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--phase',required=True);ap.add_argument('--cap',required=True)
    ap.add_argument('--first',type=int,default=0);ap.add_argument('--arm',required=True);ap.add_argument('--strength',type=float,default=0.)
    ap.add_argument('--smoke',action='store_true');ap.add_argument('--recomplete',action='store_true');args=ap.parse_args()
    run(args.phase,args.cap,args.first,args.arm,args.strength,args.smoke,args.recomplete)
