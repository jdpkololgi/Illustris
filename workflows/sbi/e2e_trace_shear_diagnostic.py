"""Exact saved-draw trace/eigengap diagnostics; development phases only."""
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import e2e_true_density_closure as base
import h5py
import numpy as np
from workflows.sbi.e2e_exterior_metrics import joint_rank

OUT=base.ROOT/'verification/trace_shear_20261005_v1'


def features(eigen):
    """Linear transform of regional ordered eigenvalues, NOT tensor components."""
    return dict(trace=eigen.sum(-1),gap12=eigen[...,1]-eigen[...,0],
                gap23=eigen[...,2]-eigen[...,1],width=eigen[...,2]-eigen[...,0])


def assess(draws,truth):
    draws=np.asarray(draws,float);truth=np.asarray(truth,float)
    mean,sd=draws.mean(0),draws.std(0,ddof=1)
    ordered=np.sort(draws,axis=0)
    return dict(mean=mean.tolist(),truth=truth.tolist(),sd=sd.tolist(),
                bias=(mean-truth).tolist(),z=((mean-truth)/sd).tolist(),
                covered90=((truth>=ordered[1])&(truth<=ordered[30])).tolist(),
                joint=joint_rank(draws.reshape(32,-1),truth.ravel()))


def run():
    base.c.require_compute();OUT.mkdir(parents=True,exist_ok=False)
    allrows=[];bindings=[]
    for phase in ('ph012','ph013'):
        closure=json.loads((base.OUT/phase/'CLOSURE.json').read_text())
        byid={r['pair_id']:r for r in closure['cases']}
        records=[];samples=[];targets=[];matched=[];mass=[];ids=[]
        counts={str(t):dict(pred=np.zeros(4),truth=np.zeros(4),cells=0) for t in (0.,.2)}
        for row in base.auth.geometry(phase,base.FREEZE):
            chunks=[];density=[]
            for first in range(0,32,4):
                path=base.ROOT/'fullcap_tides'/phase/row['cap']/f'd{first:03d}'/'COMPLETE.json'
                rec=json.loads(path.read_text());assert rec['passed'] and rec['first']==first
                bindings.append(base.c.file_record(path,content_hash=True))
                with h5py.File(rec['output']['path']) as f:
                    core=base.cuts(row,np.asarray(f.attrs['origin_mpc_h']))
                    pad=int(f.attrs['padding_cells'])
                    wide=tuple(slice(s.start+pad,s.stop+pad) for s in core)
                    chunks.append(f['eigenvalues'][(slice(None),*core,slice(None))].reshape(4,2,16,16,16,3))
                    density.append(f['rho_completed'][(slice(None),*wide)].reshape(4,2,16,16,16)-1)
            eig=np.concatenate(chunks)
            ev=eig.mean((2,3,4),dtype=np.float64)
            rho=np.concatenate(density).mean((2,3,4),dtype=np.float64)
            np.testing.assert_allclose(ev.sum(-1),rho,atol=1e-7,rtol=0)
            actual,_=base.auth.truth(phase,row['pair_id'],base.FREEZE)
            full_eig=base.tidal.eigensystem(actual['fullbox_tensor_cores'])
            full=full_eig.mean((1,2,3))
            oracle=np.asarray(byid[row['pair_id']]['operator_target']).reshape(2,3)
            np.testing.assert_allclose(full.sum(-1),oracle.sum(-1),atol=1e-12,rtol=0)
            f=features(ev);tf=features(full);mf=features(oracle)
            result=dict(phase=phase,pair_id=row['pair_id'],cap=row['cap'],
                fullbox={k:assess(f[k],tf[k]) for k in f},
                matched={k:assess(f[k],mf[k]) for k in f},
                fullbox_joint_gaps=joint_rank(np.diff(ev,axis=-1).reshape(32,4),np.diff(full,axis=-1).ravel()),
                matched_joint_gaps=joint_rank(np.diff(ev,axis=-1).reshape(32,4),np.diff(oracle,axis=-1).ravel()),
                lambda1_lambda3_correlation=[float(np.corrcoef(ev[:,i,0],ev[:,i,2])[0,1]) for i in range(2)],
                approximate_width_sd=np.sqrt(ev[:,:,0].var(0,ddof=1)+ev[:,:,2].var(0,ddof=1)).tolist(),
                trace_density_maxabs=float(abs(ev.sum(-1)-rho).max()))
            for t in (0.,.2):
                key=str(t);label=(eig>t).sum(-1);truth_label=(full_eig>t).sum(-1)
                counts[key]['pred']+=np.bincount(label.ravel(),minlength=4)/32
                counts[key]['truth']+=np.bincount(truth_label.ravel(),minlength=4)
                counts[key]['cells']+=truth_label.size
            records.append(result);samples.append(ev);targets.append(full);matched.append(oracle);mass.append(rho);ids.append(row['pair_id'])
        path=OUT/(phase+'_REGIONAL_DRAWS.npz')
        np.savez(path,pair_id=np.array(ids),eigenvalues=np.stack(samples),fullbox=np.stack(targets),
                 matched=np.stack(matched),density_contrast=np.stack(mass))
        class_offsets={k:dict(predicted_fraction=(v['pred']/v['cells']).tolist(),truth_fraction=(v['truth']/v['cells']).tolist(),
                       probability_minus_frequency=((v['pred']-v['truth'])/v['cells']).tolist(),cells=v['cells']) for k,v in counts.items()}
        base.c.atomic_json(OUT/(phase+'.json'),dict(phase=phase,cases=records,class_offsets=class_offsets,
                            draws=base.c.file_record(path,content_hash=True),passed=True))
        allrows.extend(records)
        print(json.dumps(dict(phase=phase,cases=len(records),class_offsets=class_offsets)),flush=True)
    report=[]
    for phase in ('ph012','ph013','pooled_descriptive'):
        selected=[r for r in allrows if phase=='pooled_descriptive' or r['phase']==phase]
        for target in ('fullbox','matched'):
            item=dict(phase=phase,target=target,pairs=len(selected),metrics={})
            for name in ('trace','gap12','gap23','width'):
                vals=[r[target][name] for r in selected]
                bias=np.concatenate([v['bias'] for v in vals]);sd=np.concatenate([v['sd'] for v in vals])
                mu=np.concatenate([v['mean'] for v in vals]);truth=np.concatenate([v['truth'] for v in vals])
                item['metrics'][name]=dict(mean_bias=float(bias.mean()),rms_bias=float(np.sqrt(np.mean(bias*bias))),
                    mean_posterior_sd=float(sd.mean()),mean_z=float(np.mean(bias/sd)),median_z=float(np.median(bias/sd)),
                    rms_z=float(np.sqrt(np.mean((bias/sd)**2))),below=int((bias<0).sum()),cores=len(bias),
                    mean_prediction=float(mu.mean()),mean_truth=float(truth.mean()),
                    ratio_of_means=None if name=='trace' else float(mu.mean()/truth.mean()),
                    marginal90=float(np.mean([v['covered90'] for v in vals])),
                    joint90=float(np.mean([v['joint']['covered']['0.9'] for v in vals])),
                    error_to_variance_ratio=float(np.mean(bias*bias)/((1+1/32)*np.mean(sd*sd))))
            item['joint_four_gaps90']=float(np.mean([r[target+'_joint_gaps']['covered']['0.9'] for r in selected]))
            report.append(item)
    base.c.atomic_json(OUT/'SUMMARY.json',dict(results=report,inputs=bindings,
        source_sha256=base.c.sha256(__file__),job=os.environ['SLURM_JOB_ID'],independent_phases=2,
        interpretation='Regional mean eigenvalue gaps are not the full five-component oriented shear tensor.',
        posterior_qualified=False,passed=True))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':run()
