"""Extent and local grid-refinement controls, fixed development geometry."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
import e2e_true_density_closure as b
import h5py
import numpy as np
from scipy import fft

OUT=b.ROOT/'verification/operator_split_20261005_v1'
SELECTED={'ph012':('ph012_NGC_s1_interior_00','ph012_SGC_s2_boundary_00'),
          'ph013':('ph013_NGC_s1_boundary_00','ph013_SGC_s2_interior_00')}


def patch_tensor(rho,target_shape,padding):
    kernels=b.tidal.kernel(rho.shape,target_shape,offset=(padding,)*3)
    return b.tidal.convolve(rho-1,kernels,target_shape,workers=16)


def prepare(phase):
    directory=OUT/phase;directory.mkdir(parents=True,exist_ok=False)
    rows=b.auth.geometry(phase,b.FREEZE)
    native=b.c.verify_receipt(b.c.ROOT/'matter'/phase/'DENSITY_COMPLETE.json')
    spectrum,_=b.op.smoothed_spectrum(native['outputs'][0]['path'],32)
    field=fft.irfftn(spectrum,s=(2048,)*3,workers=32,overwrite_x=True)
    del spectrum;gc.collect()
    sources=[];local=[]
    for cap in ('NGC','SGC'):
        with h5py.File(b.OUT/phase/(cap+'_TRUE_SOURCE.h5')) as f:
            shape=tuple(map(int,f.attrs['target_shape']));origin=np.asarray(f.attrs['origin_mpc_h'])
        row=next(r for r in rows if r['cap']==cap);grid=row['grid'];pad=64
        raw=(origin-pad*6.766-np.asarray(grid['origin_mpc_h']))/3.383
        start=np.rint(raw).astype(int);assert np.max(abs(start-raw))<1e-8
        source_shape=tuple(n+2*pad for n in shape)
        rho=1+b.op.sample_averaged_local(field,[v+2*np.arange(n) for v,n in zip(start,source_shape)],2,grid)
        with h5py.File(b.OUT/phase/(cap+'_TRUE_SOURCE.h5')) as f:
            old=f['rho'][:]
        np.testing.assert_allclose(rho[32:-32,32:-32,32:-32],old,atol=0,rtol=0)
        del old
        path=directory/(cap+'_PAD64.h5')
        with h5py.File(path,'x') as f:
            f.create_dataset('rho',data=rho,chunks=(8,32,32),compression='lzf')
            f.attrs.update(origin_mpc_h=origin,target_shape=shape,padding=pad)
        sources.append(dict(cap=cap,source=b.c.file_record(path,content_hash=True)))
        del rho;print(json.dumps(dict(phase=phase,cap=cap,stage='pad64_ready')),flush=True)
    for row in (r for r in rows if r['pair_id'] in SELECTED[phase]):
        arrays={};checks=[]
        for pad in (16,32):
            shape=np.array([32,16,16]);ss=shape+2*pad
            start=np.asarray(row['center'])-16-2*pad
            raw_starts=[v+np.arange(2*n) for v,n in zip(start,ss)]
            fine=1+b.op.sample_averaged_local(field,raw_starts,1,row['grid'])
            coarse=b.op.mean_pool(fine,2)
            tc=patch_tensor(coarse,tuple(shape),pad)
            tf=patch_tensor(fine,tuple(2*shape),2*pad)
            averaged=np.stack([b.op.mean_pool(tf[...,k],2) for k in range(6)],axis=-1)
            correction=averaged-tc
            trace=float(abs(correction[...,[0,3,5]].sum(-1)).max())
            assert trace<1e-9
            arrays['correction_pad'+str(pad)]=correction.reshape(2,16,16,16,6)
            checks.append(dict(padding=pad,trace_error=trace))
            del fine,coarse,tc,tf,averaged;gc.collect()
            print(json.dumps(dict(phase=phase,pair=row['pair_id'],refined_pad=pad,trace_error=trace)),flush=True)
        path=directory/(row['pair_id']+'_CORRECTION.npz');np.savez(path,**arrays)
        local.append(dict(pair_id=row['pair_id'],source=b.c.file_record(path,content_hash=True),checks=checks))
    b.c.atomic_json(directory/'PREPARED.json',dict(phase=phase,sources=sources,local=local,
        source_sha256=b.c.sha256(__file__),job=os.environ['SLURM_JOB_ID'],passed=True))


def evaluate(phase):
    import torch
    from workflows.sbi.e2e_exterior_cap_tides import CapTidal
    from workflows.sbi.e2e_loa_field_sampling import setup
    setup(True)
    directory=OUT/phase;prepared=json.loads((directory/'PREPARED.json').read_text())
    rows=b.auth.geometry(phase,b.FREEZE);allresults=[]
    closure=json.loads((b.OUT/phase/'CLOSURE.json').read_text())
    old={r['pair_id']:r for r in closure['cases']}
    for cap in ('NGC','SGC'):
        tensors={}
        selected=[r for r in rows if r['cap']==cap]
        for pad in (32,64):
            path=b.OUT/phase/(cap+'_TRUE_SOURCE.h5') if pad==32 else directory/(cap+'_PAD64.h5')
            with h5py.File(path) as f:
                rho=f['rho'][:];shape=tuple(map(int,f.attrs['target_shape']));origin=np.asarray(f.attrs['origin_mpc_h'])
            operator=CapTidal(shape,pad)
            tensors[pad]={r['pair_id']:np.empty((2,16,16,16,6)) for r in selected}
            for k,component in enumerate(operator.components(rho)):
                for row in selected:tensors[pad][row['pair_id']][...,k]=component[b.cuts(row,origin)].reshape(2,16,16,16)
            del component,operator,rho;gc.collect();torch.cuda.empty_cache()
        for row in selected:
            pid=row['pair_id'];t32=tensors[32][pid];t64=tensors[64][pid]
            regional=lambda t:b.tidal.eigensystem(t).mean((1,2,3))
            e32,e64=regional(t32),regional(t64)
            np.testing.assert_allclose(e32.ravel(),old[pid]['operator_target'],atol=1e-12,rtol=0)
            full=np.asarray(old[pid]['fullbox_target']).reshape(2,3)
            record=dict(pair_id=pid,cap=cap,fullbox=full.tolist(),pad32=e32.tolist(),pad64=e64.tolist(),
                        posterior_sd=old[pid]['posterior_sd'],extent_shift=(e64-e32).tolist())
            if pid in SELECTED[phase]:
                corrections=next(v for v in prepared['local'] if v['pair_id']==pid)
                assert b.c.sha256(corrections['source']['path'])==corrections['source']['sha256']
                with np.load(corrections['source']['path']) as f:
                    record['refinement']={str(p):dict(pad32_plus_local=regional(t32+f['correction_pad'+str(p)]).tolist(),
                        pad64_plus_local=regional(t64+f['correction_pad'+str(p)]).tolist()) for p in (16,32)}
            allresults.append(record)
            print(json.dumps(dict(phase=phase,**record)),flush=True)
        np.savez(directory/(cap+'_CORE_TENSORS.npz'),pair_id=np.array([r['pair_id'] for r in selected]),
                 pad32=np.stack([tensors[32][r['pair_id']] for r in selected]),
                 pad64=np.stack([tensors[64][r['pair_id']] for r in selected]))
    b.c.atomic_json(directory/'SPLIT.json',dict(phase=phase,cases=allresults,passed=True,
        job=os.environ['SLURM_JOB_ID'],source_sha256=b.c.sha256(__file__),
        caveat='Local half-cell refinement plus tensor correction, not a complete half-cell cap solve or infinite-domain convergence proof.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['ph012','ph013'],required=True)
    p.add_argument('--stage',choices=['prepare','evaluate'],required=True);a=p.parse_args()
    b.c.require_compute();b.auth.frozen(a.phase,b.FREEZE)
    (prepare if a.stage=='prepare' else evaluate)(a.phase)
