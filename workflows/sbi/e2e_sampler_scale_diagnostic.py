"""Scale-resolved replay of frozen diagnostic fields, without new sampling.

Rectangular, unwindowed pseudo-spectra: bins are not independent modes of a
survey-wide posterior. Coarse/fine is the model's density decomposition.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import e2e_true_density_closure as b
import e2e_sampler_factor_diagnostic as s
import numpy as np
import h5py
from scipy import fft


def bands(draws, target, cell, edges):
    axes=np.meshgrid(*[2*np.pi*fft.fftfreq(n,d=cell) for n in target.shape],indexing='ij',sparse=True)
    k=np.sqrt(sum(x*x for x in axes))
    f=fft.fftn(draws,axes=(-3,-2,-1),norm='ortho',workers=8)
    truth=fft.fftn(target,norm='ortho',workers=8)
    mean=f.mean(0)
    variance=np.mean(abs(f-mean)**2,axis=0)*len(draws)/(len(draws)-1)
    result=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        mask=(k>lo)&(k<=hi)
        if not mask.any():continue
        result.append(dict(k_min=lo,k_max=None if np.isinf(hi) else hi,modes=int(mask.sum()),
            truth_power=float(np.mean(abs(truth[mask])**2)),
            mean_power=float(np.mean(abs(mean[mask])**2)),
            sample_power=float(np.mean(abs(f[:,mask])**2)),
            posterior_variance_power=float(variance[mask].mean()),
            mean_error_power=float(np.mean(abs(mean[mask]-truth[mask])**2)),
            mean_error_minus_draw_noise=float(np.mean(abs(mean[mask]-truth[mask])**2-variance[mask]/len(draws)))))
    return result


def run(phase):
    b.c.require_compute();b.auth.frozen(phase,b.FREEZE)
    root=s.OUT/'results';receipt=root/f'COMPLETE_{phase}.json'
    complete=json.loads(receipt.read_text());assert complete['passed']
    output=s.OUT/f'SCALES_{phase}.json';rows=[]
    for row in complete['cases']:
        with h5py.File(row['input']['path']) as f:
            truthw=f['true_wide'][:];truthrho=f['true_rho'][:]
        crop=(slice(16,32),slice(18,30),slice(18,30))
        truthfine=truthrho-b.op.lift(truthw[crop])
        assert b.c.sha256(row['fields']['path'])==row['fields']['sha256']
        arms={}
        with h5py.File(row['fields']['path']) as f:
            for name in f:
                wide=f[name+'/wide'][:].astype(float)
                rho=f[name+'/rho'][:].astype(float)
                residual=rho-np.stack([b.op.lift(w[crop]) for w in wide])
                arms[name]=dict(coarse=bands(wide-1,truthw-1,27.064,(0,.01,.02,.04,.08,.16,np.inf)),
                    fine_residual=bands(residual,truthfine,6.766,(0,.04,.08,.16,.32,np.inf)))
        rows.append(dict(pair_id=row['pair_id'],arms=arms))
        print(json.dumps(dict(phase=phase,scale_pair=row['pair_id'])),flush=True)
    summary=[]
    for arm in rows[0]['arms']:
        for factor in ('coarse','fine_residual'):
            for i in range(len(rows[0]['arms'][arm][factor])):
                vals=[r['arms'][arm][factor][i] for r in rows]
                entry=dict(arm=arm,factor=factor,k_min=vals[0]['k_min'],k_max=vals[0]['k_max'],
                    modes_per_window=vals[0]['modes'])
                for key in ('truth_power','mean_power','sample_power','posterior_variance_power','mean_error_power','mean_error_minus_draw_noise'):
                    entry[key]=float(np.mean([v[key] for v in vals]))
                entry['sample_truth_power_ratio']=entry['sample_power']/entry['truth_power']
                entry['error_variance_ratio']=None if arm=='oracle_coarse_local_fine' and factor=='coarse' else entry['mean_error_minus_draw_noise']/entry['posterior_variance_power']
                summary.append(entry)
    b.c.atomic_json(output,dict(phase=phase,cases=rows,summary=summary,passed=True,
        input=b.c.file_record(receipt,content_hash=True),source_sha256=b.c.sha256(__file__),
        caveat='Two reused phases; correlated overlapping windows; rectangular pseudo-spectra, not uniquely global Fourier-mode posterior errors. Oracle coarse spread is deterministic.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['ph012','ph013'],required=True);a=p.parse_args();run(a.phase)
