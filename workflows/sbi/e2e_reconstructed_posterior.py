"""Apply fixed density-only reconstruction to saved full-cap posterior draws.

Four geometry-selected pairs, 32 unchanged draws each; no learned parameters
or true subcell inputs. Common correction for truth and posterior, scored
against both the original physical target and reconstructed-operator truth.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
import e2e_density_only_reconstruction as r
import e2e_true_density_closure as b
import e2e_operator_split as split
from e2e_trace_shear_diagnostic import assess,features
import numpy as np
import h5py

OUT=b.ROOT/'verification/reconstructed_posterior_20261005_v1'


def run(phase):
    import torch
    from workflows.sbi.e2e_exterior_cap_tides import CapTidal
    from workflows.sbi.e2e_loa_field_sampling import setup
    b.c.require_compute();b.auth.frozen(phase,b.FREEZE);setup(True)
    dest=OUT/phase;dest.mkdir(parents=True,exist_ok=False)
    geometry=b.auth.geometry(phase,b.FREEZE)
    selected=[row for row in geometry if row['pair_id'] in split.SELECTED[phase]]
    recsource=b.ROOT/'verification/density_only_reconstruction_20261005_v1_all64/COMPLETE.json'
    rec=json.loads(recsource.read_text());assert rec['passed']
    rows=[];sources=[]
    localcoarse=CapTidal((32,16,16),32)
    localfine=CapTidal((64,32,32),64)
    for row in selected:
        cap,pid=row['cap'],row['pair_id'];tick=time.monotonic()
        with h5py.File(b.OUT/phase/(cap+'_TRUE_SOURCE.h5')) as f:
            shape=tuple(map(int,f.attrs['target_shape']));origin=np.asarray(f.attrs['origin_mpc_h'])
            source_origin=np.asarray(f.attrs['source_origin_mpc_h'])
        core=b.cuts(row,origin);source_core=b.cuts(row,source_origin)
        patch=tuple(slice(v.start-33,v.stop+33) for v in source_core)
        target=next(v for v in rec['cases'] if v['pair_id']==pid)
        full=np.asarray(target['fullbox'])
        matched=np.asarray(next(v['eigenvalues'] for v in target['results'] if v['method']=='linear'))
        # Independently check GPU local quadrature against the CPU truth-only
        # reconstruction, before processing posterior payloads.
        with h5py.File(b.OUT/phase/(cap+'_TRUE_SOURCE.h5')) as f:
            truthpatch=f['rho'][patch]
        truthfine=r.reconstruct(truthpatch,'linear')[2:-2,2:-2,2:-2]
        with np.load(split.OUT/phase/(cap+'_CORE_TENSORS.npz')) as f:
            truthbase=f['pad32'][list(f['pair_id']).index(pid)]
        for k,(tc,tf) in enumerate(zip(localcoarse.components(truthpatch[1:-1,1:-1,1:-1]),localfine.components(truthfine))):
            truthbase[...,k]+=(b.op.mean_pool(tf,2)-tc).reshape(2,16,16,16)
        truth_parity=float(abs(b.tidal.eigensystem(truthbase).mean((1,2,3))-matched).max())
        assert truth_parity<1e-11
        del truthpatch,truthfine,truthbase,tc,tf
        operator=CapTidal(shape,32)
        eigs=[];base_eigs=[];parity=0.;trace=0.
        for first in range(0,32,4):
            receipt=b.ROOT/'fullcap_tides'/phase/cap/f'd{first:03d}'/'COMPLETE.json'
            data=json.loads(receipt.read_text());assert data['passed'] and data['first']==first
            sources.append(b.c.file_record(receipt,content_hash=True))
            # Payloads were independently hashed by sampler_factor preparation.
            with h5py.File(data['output']['path']) as f:
                for draw in range(4):
                    density=f['rho_completed'][draw].astype(float)
                    tensor=np.empty((2,16,16,16,6))
                    for k,component in enumerate(operator.components(density)):
                        tensor[...,k]=component[core].reshape(2,16,16,16)
                    coarse=density[patch]
                    fine=r.reconstruct(coarse,'linear')[2:-2,2:-2,2:-2]
                    delta=np.empty_like(tensor)
                    for k,(tc,tf) in enumerate(zip(localcoarse.components(coarse[1:-1,1:-1,1:-1]),
                                                   localfine.components(fine))):
                        delta[...,k]=(b.op.mean_pool(tf,2)-tc).reshape(2,16,16,16)
                    trace=max(trace,float(abs(delta[...,[0,3,5]].sum(-1)).max()))
                    original=b.tidal.eigensystem(tensor).mean((1,2,3))
                    saved=f['eigenvalues'][(draw,*core,slice(None))].reshape(2,16,16,16,3).mean((1,2,3),dtype=np.float64)
                    parity=max(parity,float(abs(original-saved).max()))
                    assert parity<2e-7 and trace<1e-9
                    eigs.append(b.tidal.eigensystem(tensor+delta).mean((1,2,3)))
                    base_eigs.append(original)
                    del density,fine,coarse,component,tc,tf
                print(json.dumps(dict(phase=phase,pair_id=pid,draws_done=first+4,seconds=time.monotonic()-tick)),flush=True)
        assert parity<2e-7 and trace<1e-9
        x=np.asarray(eigs);old=np.asarray(base_eigs)
        result=dict(phase=phase,pair_id=pid,fullbox=full.tolist(),matched=matched.tolist(),
            corrected={t:{k:assess(v,features(y)[k]) for k,v in features(x).items()}
                       for t,y in [('fullbox',full),('matched',matched)]},
            original_fullbox={k:assess(v,features(full)[k]) for k,v in features(old).items()},
            corrected_joint6={t:b.joint_rank(x.reshape(32,6),y.ravel()) for t,y in [('fullbox',full),('matched',matched)]},
            original_joint6=b.joint_rank(old.reshape(32,6),full.ravel()),
            baseline_replay_maxabs=parity,correction_trace_maxabs=trace,
            truth_reconstruction_cpu_gpu_maxabs=truth_parity,seconds=time.monotonic()-tick)
        np.savez(dest/(pid+'_DRAWS.npz'),original=old,corrected=x,fullbox=full,matched=matched)
        b.c.atomic_json(dest/(pid+'.json'),result);rows.append(result)
        del operator;gc.collect();torch.cuda.empty_cache()
    b.c.atomic_json(dest/'COMPLETE.json',dict(phase=phase,cases=rows,passed=True,
        source_sha256=b.c.sha256(__file__),reconstruction_sha256=b.c.sha256(r.__file__),
        sources=sources,truth_reconstruction=b.c.file_record(recsource,content_hash=True),
        job=os.environ['SLURM_JOB_ID'],posterior_qualified=False,
        scope='Four preselected development pairs across both phases; local conservative reconstruction using only each draw density.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['ph012','ph013'],required=True);a=p.parse_args();run(a.phase)
