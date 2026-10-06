"""Replay fixed Stage A operators solely to retain paired cell class draws."""
import argparse
import gc
import json
import os
from pathlib import Path
import resource
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
import e2e_stage_b_prepare as p
import e2e_density_only_reconstruction as r
import e2e_stage_a_reconstruction as ar
b=p.b;c=p.c;a=p.a
import h5py
import numpy as np


def replay(pilot=False):
    import torch
    from workflows.sbi.e2e_exterior_cap_tides import CapTidal
    from workflows.sbi.e2e_loa_field_sampling import setup
    c.require_compute();setup(True);torch.set_num_threads(4)
    dest=p.ROOT/('replay_pilot' if pilot else 'replay');dest.mkdir(parents=True,exist_ok=False)
    source=p.ROOT/('prepare_pilot' if pilot else 'prepare')/'COMPLETE.json'
    prepared=json.loads(source.read_text());assert prepared['passed']
    if not pilot:assert prepared['panel_complete']
    binding=dict(config=c.file_record(p.CONFIG,content_hash=True),source=c.file_record(__file__,content_hash=True),
        stage_a_reconstruction=c.file_record(ar.__file__,content_hash=True),
        reconstruction=c.file_record(r.__file__,content_hash=True),
        prepared=c.file_record(source,content_hash=True),job=os.environ['SLURM_JOB_ID'],
        freeze=c.file_record(b.FREEZE,content_hash=True),draws=2 if pilot else 32,pilot=pilot)
    c.atomic_json(dest/'BINDING.json',binding)
    localcoarse=CapTidal((32,16,16),32);localfine=CapTidal((64,32,32),64)
    tick=time.monotonic();rows=[];sources=[];qa=[]
    for phase in a.PHASES:
        p.guard(phase);geometry=b.auth.geometry(phase,b.FREEZE)
        for cap in (('NGC',) if pilot else ('NGC','SGC')):
            selected=[v for v in geometry if v['cap']==cap]
            if pilot:selected=selected[:1]
            tp=b.OUT/phase/(cap+'_TRUE_SOURCE.h5')
            with h5py.File(tp) as f:
                shape=tuple(map(int,f.attrs['target_shape']));origin=np.asarray(f.attrs['origin_mpc_h'])
                source_origin=np.asarray(f.attrs['source_origin_mpc_h'])
            operator=CapTidal(shape,32)
            cores=[b.cuts(row,origin) for row in selected]
            patches=[tuple(slice(v.start-33,v.stop+33) for v in b.cuts(row,source_origin)) for row in selected]
            nd=binding['draws'];labels=np.empty((len(selected),2,2,nd,2,4096),dtype='u1')
            means=np.empty((len(selected),2,nd,2,3));parity=trace=0.;start=time.monotonic()
            for first in range(0,nd,4):
                receipt=b.ROOT/'fullcap_tides'/phase/cap/f'd{first:03d}'/'COMPLETE.json'
                record=json.loads(receipt.read_text())
                assert record['passed'] and record['phase']==phase and record['cap']==cap and record['first']==first
                bp=receipt.parent/'BINDING.json';assert c.sha256(bp)==record['binding']
                assert json.loads(bp.read_text())['freeze_sha256']==c.sha256(b.FREEZE)
                assert c.sha256(record['output']['path'])==record['output']['sha256']
                sources.append(c.file_record(receipt,content_hash=True))
                with h5py.File(record['output']['path']) as f:
                    for d in range(min(4,nd-first)):
                        density=f['rho_completed'][d].astype(float)
                        tensor=np.empty((len(selected),2,16,16,16,6))
                        for k,component in enumerate(operator.components(density)):
                            for j,core in enumerate(cores):tensor[j,...,k]=component[core].reshape(2,16,16,16)
                        for j,(row,core,patch) in enumerate(zip(selected,cores,patches)):
                            coarse=density[patch];fine=r.reconstruct(coarse,'linear')[2:-2,2:-2,2:-2]
                            delta=np.empty_like(tensor[j])
                            for k,(tc,tf) in enumerate(zip(localcoarse.components(coarse[1:-1,1:-1,1:-1]),localfine.components(fine))):
                                delta[...,k]=(b.op.mean_pool(tf,2)-tc).reshape(2,16,16,16)
                            trace=max(trace,float(abs(delta[...,[0,3,5]].sum(-1)).max()))
                            for method,t in enumerate((tensor[j],tensor[j]+delta)):
                                eigen=b.tidal.eigensystem(t)
                                means[j,method,first+d]=eigen.mean((1,2,3))
                                for threshold,th in enumerate(p.THRESHOLDS):
                                    labels[j,method,threshold,first+d]=(eigen>th).sum(-1).reshape(2,4096)
                                if method==0:
                                    saved=f['eigenvalues'][(d,*core,slice(None))].reshape(2,16,16,16,3)
                                    parity=max(parity,float(abs(eigen-saved).max()))
                            assert parity<2e-7 and trace<1e-9
                        del density,tensor,coarse,fine,delta,component,tc,tf
                print(json.dumps(dict(stage='b_replay',phase=phase,cap=cap,draws_done=min(first+4,nd),seconds=time.monotonic()-start)),flush=True)
            for j,row in enumerate(selected):
                pid=row['pair_id']
                old=a.ROOT/'a2'/(pid+'_SUMMARIES.npz')
                with np.load(old) as f:
                    for mi,method in enumerate(('original','corrected')):
                        np.testing.assert_allclose(means[j,mi],f[method][:nd],atol=1e-10,rtol=0)
                        for ti,th in enumerate(p.THRESHOLDS):
                            fractions=np.stack([(labels[j,mi,ti]==k).mean(-1) for k in range(4)],axis=-1)
                            np.testing.assert_array_equal(fractions,f[f'{method}_fractions_{th}'][:nd])
                output=dest/(pid+'.npz');np.savez_compressed(output,labels=labels[j])
                rows.append(dict(phase=phase,cap=cap,pair_id=pid,shell=row['shell'],support_stratum=row['support_stratum'],
                    output=c.file_record(output,content_hash=True),stage_a_source=c.file_record(old,content_hash=True)))
            qa.append(dict(phase=phase,cap=cap,draws=nd,baseline_maxabs=parity,trace_maxabs=trace,
                exact_stage_a_fraction_replay=True,seconds=time.monotonic()-start))
            del operator,labels,means;gc.collect();torch.cuda.empty_cache()
    out=dict(passed=True,panel_complete=not pilot and len(rows)==32,cases=rows,sources=sources,numerical=qa,
        binding=binding,seconds=time.monotonic()-tick,max_host_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
        peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3,posterior_qualified=False)
    c.atomic_json(dest/'COMPLETE.json',out)
    print(json.dumps(dict(stage='b_replay_complete',out=str(dest),seconds=out['seconds'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--pilot',action='store_true');args=parser.parse_args();replay(args.pilot)
