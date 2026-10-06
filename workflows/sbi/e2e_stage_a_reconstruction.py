"""All-core conservative posterior replay, Stage A2, ph012/ph013 only.

One cap FFT per draw serves all eight registered pairs. No sampling or fitting.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import resource
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
import e2e_stage_a_calibration as a
import e2e_density_only_reconstruction as r
import e2e_operator_split as split
b=a.b
import numpy as np
import h5py


def class_summary(draw_eigen, truth_eigen, threshold):
    labels=(draw_eigen>threshold).sum(-1);truth=(truth_eigen>threshold).sum(-1)
    probabilities=np.stack([(labels==k).mean(0) for k in range(4)],axis=-1)
    fractions=np.stack([(labels==k).mean((2,3,4)) for k in range(4)],axis=-1)
    actual=np.stack([(truth==k).mean((1,2,3)) for k in range(4)],axis=-1)
    brier=float(np.mean(np.sum((probabilities-np.eye(4)[truth])**2,-1)))
    correction=float(np.mean(np.sum(probabilities*(1-probabilities),-1))/(len(draw_eigen)-1))
    return dict(threshold=threshold,brier=brier,fair_brier=brier-correction,
        mean_fraction=fractions.mean((0,1)).tolist(),truth_fraction=actual.mean(0).tolist(),
        class_fraction={str(k):a.scalar_scores(fractions[...,k],actual[...,k]) for k in range(4)},
        joint_fraction_status='Deferred to Stage B discrete joint-rank validation; not scored here',
        caveat='Volume-weighted cells, not galaxy probabilities. Fraction ties require discrete-rank checks; continuous central references are descriptive only.'),fractions,actual


def run(pilot=False):
    import torch
    from workflows.sbi.e2e_exterior_cap_tides import CapTidal
    from workflows.sbi.e2e_loa_field_sampling import setup
    b.c.require_compute();setup(True);torch.set_num_threads(4)
    dest=a.ROOT/('a2_pilot' if pilot else 'a2');dest.mkdir(parents=True,exist_ok=False)
    recsource=b.ROOT/'verification/density_only_reconstruction_20261005_v1_all64/COMPLETE.json'
    rec=json.loads(recsource.read_text());assert rec['passed']
    binding=dict(schema='cfm-stage-a2-v1',phases=list(a.PHASES),draws=2 if pilot else 32,
        caps=['NGC'] if pilot else ['NGC','SGC'],collar=32,method='linear',no_new_sampling=True,
        source=b.c.file_record(__file__,content_hash=True),reconstruction=b.c.file_record(r.__file__,content_hash=True),
        metrics=b.c.file_record(a.__file__,content_hash=True),
        freeze=b.c.file_record(b.FREEZE,content_hash=True),truth_reconstruction=b.c.file_record(recsource,content_hash=True),
        job=os.environ['SLURM_JOB_ID'],pilot=pilot)
    b.c.atomic_json(dest/'BINDING.json',binding)
    localcoarse=CapTidal((32,16,16),32);localfine=CapTidal((64,32,32),64)
    allrows=[];sources=[];tick=time.monotonic();numerical=[]
    for phase in a.PHASES:
        a.guard(phase);geometry=b.auth.geometry(phase,b.FREEZE)
        for cap in binding['caps']:
            selected=[row for row in geometry if row['cap']==cap]
            assert len(selected)==8
            tp=b.OUT/phase/(cap+'_TRUE_SOURCE.h5')
            with h5py.File(tp) as f:
                shape=tuple(map(int,f.attrs['target_shape']));origin=np.asarray(f.attrs['origin_mpc_h'])
                source_origin=np.asarray(f.attrs['source_origin_mpc_h'])
            operator=CapTidal(shape,32)
            cores=[b.cuts(row,origin) for row in selected]
            patches=[];truths=[];matched=[];truth_parity=0.
            for row in selected:
                pid=row['pair_id'];source_core=b.cuts(row,source_origin)
                patch=tuple(slice(v.start-33,v.stop+33) for v in source_core);patches.append(patch)
                truth,_=b.auth.truth(phase,pid,b.FREEZE);truths.append(b.tidal.eigensystem(truth['fullbox_tensor_cores']))
                target=next(v for v in rec['cases'] if v['pair_id']==pid)
                with h5py.File(tp) as f:truthpatch=f['rho'][patch]
                with np.load(split.OUT/phase/(cap+'_CORE_TENSORS.npz')) as f:
                    tb=f['pad32'][list(f['pair_id']).index(pid)].copy()
                fine=r.reconstruct(truthpatch,'linear')[2:-2,2:-2,2:-2]
                for k,(tc,tf) in enumerate(zip(localcoarse.components(truthpatch[1:-1,1:-1,1:-1]),localfine.components(fine))):
                    tb[...,k]+=(b.op.mean_pool(tf,2)-tc).reshape(2,16,16,16)
                me=b.tidal.eigensystem(tb);matched.append(me)
                prior=next(v['eigenvalues'] for v in target['results'] if v['method']=='linear')
                truth_parity=max(truth_parity,float(abs(me.mean((1,2,3))-prior).max()))
                assert truth_parity<1e-10
            del fine,truthpatch,tb,tc,tf
            count=binding['draws'];old=np.empty((8,count,2,16,16,16,3));new=np.empty_like(old)
            parity=0.;trace=0.;start=time.monotonic()
            for first in range(0,count,4):
                receipt=b.ROOT/'fullcap_tides'/phase/cap/f'd{first:03d}'/'COMPLETE.json'
                record=json.loads(receipt.read_text());assert record['passed'] and record['phase']==phase and record['cap']==cap and record['first']==first
                bp=receipt.parent/'BINDING.json';assert b.c.sha256(bp)==record['binding']
                assert json.loads(bp.read_text())['freeze_sha256']==b.c.sha256(b.FREEZE)
                assert b.c.sha256(record['output']['path'])==record['output']['sha256']
                sources.append(b.c.file_record(receipt,content_hash=True))
                with h5py.File(record['output']['path']) as f:
                    for draw in range(min(4,count-first)):
                        density=f['rho_completed'][draw].astype(float)
                        tensor=np.empty((8,2,16,16,16,6))
                        for k,component in enumerate(operator.components(density)):
                            for j,core in enumerate(cores):tensor[j,...,k]=component[core].reshape(2,16,16,16)
                        for j,(row,core,patch) in enumerate(zip(selected,cores,patches)):
                            coarse=density[patch];fine=r.reconstruct(coarse,'linear')[2:-2,2:-2,2:-2]
                            delta=np.empty_like(tensor[j])
                            for k,(tc,tf) in enumerate(zip(localcoarse.components(coarse[1:-1,1:-1,1:-1]),localfine.components(fine))):
                                delta[...,k]=(b.op.mean_pool(tf,2)-tc).reshape(2,16,16,16)
                            trace=max(trace,float(abs(delta[...,[0,3,5]].sum(-1)).max()))
                            original=b.tidal.eigensystem(tensor[j]);corrected=b.tidal.eigensystem(tensor[j]+delta)
                            saved=f['eigenvalues'][(draw,*core,slice(None))].reshape(2,16,16,16,3)
                            parity=max(parity,float(abs(original-saved).max()))
                            assert parity<2e-7 and trace<1e-9
                            old[j,first+draw]=original;new[j,first+draw]=corrected
                        del density,fine,coarse,delta,tensor,component,tc,tf
                        print(json.dumps(dict(stage='a2',phase=phase,cap=cap,draws_done=first+draw+1,seconds=time.monotonic()-start)),flush=True)
            numerical.append(dict(phase=phase,cap=cap,baseline_eigen_maxabs=parity,trace_maxabs=trace,truth_cpu_gpu_maxabs=truth_parity,
                seconds=time.monotonic()-start,peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3))
            for j,row in enumerate(selected):
                pid=row['pair_id'];means=dict(original=old[j].mean((2,3,4)),corrected=new[j].mean((2,3,4)),
                    physical_truth=truths[j].mean((1,2,3)),matched_truth=matched[j].mean((1,2,3)))
                metrics={};classes={};extra={}
                if not pilot:
                    for method,x in [('original',old[j]),('corrected',new[j])]:
                        for target in ('physical','matched'):
                            for name,val in a.eig_metrics(means[method],means[target+'_truth']).items():
                                if method=='original' and target=='matched':continue
                                metrics[f'{method}/{target}/{name}']=val
                        for threshold in (0.,.2):
                            result,frac,tf=class_summary(x,truths[j],threshold)
                            classes[f'{method}/{threshold}']=result
                            extra[f'{method}_fractions_{threshold}']=frac;extra[f'truth_fractions_{threshold}']=tf
                        for k in range(3):metrics[f'{method}/physical/cell_lambda{k+1}']=a.scalar_scores(x[...,k],truths[j][...,k])
                path=dest/(pid+'_SUMMARIES.npz');np.savez(path,**means,**extra)
                result=dict(phase=phase,cap=cap,pair_id=pid,shell=row['shell'],support_stratum=row['support_stratum'],
                    redshift=row['redshift'],metrics=metrics,classes=classes,draws=count,summary=b.c.file_record(path,content_hash=True))
                b.c.atomic_json(dest/(pid+'.json'),result);allrows.append(result)
            del operator,old,new;gc.collect();torch.cuda.empty_cache()
    out=dict(passed=True,panel_complete=not pilot and len(allrows)==32,cases=allrows,
        groups=a.aggregate(allrows) if not pilot else [],numerical=numerical,sources=sources,binding=binding,
        posterior_qualified=False,seconds=time.monotonic()-tick,
        max_host_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)
    b.c.atomic_json(dest/'COMPLETE.json',out)
    print(json.dumps(dict(stage='a2_complete',cases=len(allrows),seconds=out['seconds'],out=str(dest))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');args=p.parse_args();run(args.pilot)
