"""Small-receipt scientific summaries and independent operator cross-check."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import e2e_true_density_closure as b
import numpy as np
import h5py

ROOT=b.ROOT/'verification'


def width(e):
    e=np.asarray(e);return e[...,2]-e[...,0]


def reconstruction():
    root=ROOT/'density_only_reconstruction_20261005_v1_all64'
    data=json.loads((root/'COMPLETE.json').read_text());assert data['passed']
    summary=[]
    for phase in ('ph012','ph013','pooled'):
        rows=[r for r in data['cases'] if phase=='pooled' or r['phase']==phase]
        truth=np.array([r['fullbox'] for r in rows]);base=np.array([r['baseline'] for r in rows])
        for method in ('baseline','constant','linear','monotone'):
            eig=base if method=='baseline' else np.array([next(v['eigenvalues'] for v in r['results']
                       if v['method']==method and v['padding']==32) for r in rows])
            residual=eig-truth
            entry=dict(phase=phase,method=method,cores=2*len(rows),truth_width=float(width(truth).mean()),
                width=float(width(eig).mean()),width_deficit=float(1-width(eig).mean()/width(truth).mean()),
                recovered_fraction=float((width(eig).mean()-width(base).mean())/(width(truth).mean()-width(base).mean())),
                eigenvalue_rmse=float(np.sqrt(np.mean(residual**2))),
                improved_width_cores=int(np.sum(abs(width(eig)-width(truth))<abs(width(base)-width(truth)))),
                min_recovered=float(np.min((width(eig)-width(base))/(width(truth)-width(base)))),
                max_recovered=float(np.max((width(eig)-width(base))/(width(truth)-width(base)))))
            summary.append(entry)
    result=dict(results=summary,independent_phases=2,passed=True,
        source=b.c.file_record(root/'COMPLETE.json',content_hash=True),
        caveat='True coarse density only; diagnostic correction on fixed local collars, not a production cap implementation or posterior qualification.')
    b.c.atomic_json(root/'SUMMARY.json',result);print(json.dumps(result,indent=2))


def parity():
    b.c.require_compute()
    root=ROOT/'sampler_factor_20261005_v1'
    prepared=json.loads((root/'PREPARED.json').read_text());row=prepared['cases'][0]
    pilot=json.loads((root/'pilot_v2_1_4'/'COMPLETE.json').read_text())['cases'][0]
    assert row['pair_id']==pilot['pair_id']
    with h5py.File(row['input']) as f:
        rho=f['true_rho'][:];wide=f['true_wide'][:];start=f['core_start'][:]
        full=f['fullbox_tensor'][:]
    tensor=b.op.consistent_tensor(rho-1,wide-1,b.op.layout()['joint_coarse_crop_in_wide'],workers=16)
    cuts=[tuple(slice(int(a),int(a+16)) for a in start+[16*side,0,0]) for side in (0,1)]
    expected=b.tidal.eigensystem(np.stack([tensor[sl] for sl in cuts])).mean((1,2,3))
    error=float(abs(expected-np.asarray(pilot['matched'])).max())
    assert error<1e-11
    trace_error=float(abs(np.array([rho[sl].mean()-1 for sl in cuts])-full[...,[0,3,5]].sum(-1).mean((1,2,3))).max())
    assert trace_error<1e-10
    result=dict(passed=True,cpu_gpu_regional_eigenvalue_maxabs=error,
        shifted_window_fullbox_trace_maxabs=trace_error,pair_id=row['pair_id'],
        scope='One representative shifted production window; independent scipy/CPU spectral implementation.')
    b.c.atomic_json(root/'NUMERICAL_VALIDATION.json',result);print(json.dumps(result))


def sampler():
    root=ROOT/'sampler_factor_20261005_v1/results'
    complete=[json.loads((root/f'COMPLETE_{p}.json').read_text()) for p in ('ph012','ph013')]
    assert all(d['passed'] and len(d['cases'])==16 for d in complete)
    assert complete[0]['source_sha256']==complete[1]['source_sha256']
    rows=sum([d['cases'] for d in complete],[]);summaries=[]
    for phase in ('ph012','ph013','pooled'):
        selected=[r for r in rows if phase=='pooled' or r['phase']==phase]
        for arm in selected[0]['arms']:
            values=[r['arms'][arm] for r in selected]
            record=dict(phase=phase,arm=arm,cores=2*len(selected),targets={})
            for target in ('fullbox','matched'):
                metrics={}
                for feature in ('width','trace','gap12','gap23'):
                    v=[r[target][feature] for r in values]
                    pred=np.array([a['mean'] for a in v]);truth=np.array([a['truth'] for a in v])
                    metrics[feature]=dict(mean=float(pred.mean()),truth=float(truth.mean()),
                        bias=float((pred-truth).mean()),sd=float(np.mean([a['sd'] for a in v])),
                        mean_z=float(np.mean([a['z'] for a in v])),marginal90=float(np.mean([a['covered90'] for a in v])),
                        joint90=float(np.mean([a['joint']['covered']['0.9'] for a in v])))
                    if arm=='oracle_coarse_local_fine' and feature=='trace':
                        # Regional block mass is conditioned on, not uncertain.
                        # FP32 log-density conditioning leaves O(1e-9) offsets
                        # divided by O(1e-15) spread: raw z/ranks have no meaning.
                        assert max(np.max(a['sd']) for a in v)<1e-10
                        assert np.max(abs(pred-truth))<2e-6
                        metrics[feature].update(mean_z=None,marginal90=None,joint90=None,
                            interpretation='Conditioned deterministic mass; quantization residual, not a coverage diagnostic',
                            max_absolute_residual=float(np.max(abs(pred-truth))))
                    if feature!='trace':metrics[feature]['ratio']=float(pred.mean()/truth.mean())
                record['targets'][target]=metrics
            for feature in ('coarse_shear_rms','fine_shear_rms','fine_residual_variance'):
                record[feature]=dict(predicted=float(np.mean([d[feature] for r in values for d in r['scale']])),
                                    truth=float(np.mean([r['truth_scale'][feature] for r in selected])))
            if phase!='pooled':
                x=np.diff(np.array([v['eigenvalues'] for v in values]),axis=-1).reshape(len(values),32,4)
                y=np.diff(np.array([r['matched'] for r in selected]),axis=-1).reshape(len(values),4)
                cov=np.mean([np.cov(v.T) for v in x],axis=0)
                residual=x.mean(1)-y
                scatter=np.cov(residual.T)-cov/32
                chol=np.linalg.cholesky(cov)
                whitened=np.linalg.solve(chol,np.linalg.solve(chol,scatter).T).T
                record['matched_four_gaps']=dict(
                    joint90=float(np.mean([b.joint_rank(v,t)['covered']['0.9'] for v,t in zip(x,y)])),
                    scatter_eigenvalues=np.linalg.eigvalsh(whitened).tolist(),
                    mean_bias_squared_norm=float(np.sum(np.linalg.solve(chol,residual.mean(0))**2)),
                    caveat='Descriptive correlated-pair residual scatter, phase mean removed and marginal draw-mean noise subtracted; not a calibration proof.')
            summaries.append(record)
    result=dict(results=summaries,independent_phases=2,passed=True,posterior_qualified=False,
                sources=[b.c.file_record(root/f'COMPLETE_{p}.json',content_hash=True) for p in ('ph012','ph013')])
    b.c.atomic_json(root/'SUMMARY.json',result);print(json.dumps(result,indent=2))


def reconstructed():
    root=ROOT/'reconstructed_posterior_20261005_v1'
    complete=[json.loads((root/p/'COMPLETE.json').read_text()) for p in ('ph012','ph013')]
    assert all(d['passed'] and len(d['cases'])==2 for d in complete)
    assert complete[0]['source_sha256']==complete[1]['source_sha256']
    rows=sum([d['cases'] for d in complete],[]);summary=[]
    for phase in ('ph012','ph013','pooled'):
        selected=[r for r in rows if phase=='pooled' or r['phase']==phase]
        for arm,target in [('original','fullbox'),('corrected','fullbox'),('corrected','matched')]:
            values=[r['original_fullbox'] if arm=='original' else r['corrected'][target] for r in selected]
            metrics={}
            for feature in ('width','trace','gap12','gap23'):
                v=[r[feature] for r in values]
                pred=np.array([a['mean'] for a in v]);truth=np.array([a['truth'] for a in v])
                metrics[feature]=dict(mean=float(pred.mean()),truth=float(truth.mean()),
                    bias=float((pred-truth).mean()),sd=float(np.mean([a['sd'] for a in v])),
                    mean_z=float(np.mean([a['z'] for a in v])),marginal90=float(np.mean([a['covered90'] for a in v])),
                    joint90=float(np.mean([a['joint']['covered']['0.9'] for a in v])))
                if feature!='trace':metrics[feature]['ratio']=float(pred.mean()/truth.mean())
            summary.append(dict(phase=phase,arm=arm,target=target,cores=2*len(selected),metrics=metrics))
    result=dict(results=summary,passed=True,posterior_qualified=False,
        max_baseline_replay_error=max(r['baseline_replay_maxabs'] for r in rows),
        max_trace_error=max(r['correction_trace_maxabs'] for r in rows),
        max_cpu_gpu_truth_error=max(r['truth_reconstruction_cpu_gpu_maxabs'] for r in rows),
        sources=[b.c.file_record(root/p/'COMPLETE.json',content_hash=True) for p in ('ph012','ph013')],
        caveat='Four geometry-selected pairs, eight correlated cores, two phases; exploratory draw replay, not full-gate qualification.')
    b.c.atomic_json(root/'SUMMARY.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['reconstruction','parity','sampler','reconstructed']);a=p.parse_args()
    globals()[a.stage]()
