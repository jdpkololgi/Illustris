"""Independent arithmetic review of the eight-draw target-specific screen."""
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from workflows.sbi.e2e_cfm_mixture_futility import OUT, record, read, verify, write
from workflows.sbi.e2e_coupled_contract import require_compute


def run():
    require_compute()
    result=read(OUT/'COMPLETE.json');cfg=read(OUT/'CONFIG.json')
    for name in ('config','panel','moments'):verify(result[name])
    for src in cfg['sources']:verify(src)
    checks=0;rows=[]
    with np.load(result['panel']['path']) as p, np.load(result['moments']['path']) as m:
        assert p['fractions'].shape==(2,16,8,6) and p['gaps'].shape==(2,16,8,4)
        assert np.all(p['fractions']>=0) and np.all(p['fractions'].reshape(2,16,8,2,3).sum(-1)<=1+1e-12)
        for g in result['groups']:
            mask=np.array([r['phase']==g['phase'] and (g['stratum']=='all' or r['support_stratum']==g['stratum']) for r in cfg['cases']])
            bank=p[g['target']][:,mask];truth=p['truth_'+g['target']][mask]
            key=g['phase']+'_'+g['target']+'_'+g['stratum']
            sx=np.array([np.cov(v.T,ddof=1) for v in bank[0]])
            sy=np.array([np.cov(v.T,ddof=1) for v in bank[1]])
            delta=bank[1].mean(1)-bank[0].mean(1)
            between=np.array([np.outer(v,v)/4 for v in delta])
            db=between-np.array([np.cov(v.T,ddof=1) for v in bank[1]-bank[0]])/32
            expected=dict(covariance17=sx,covariance29=sy,between=between,
                          between_debiased=db,covariance_mix=(sx+sy)/2+between)
            for name,arr in expected.items():
                np.testing.assert_allclose(arr,m[key+'_'+name],atol=1e-14,rtol=1e-12);checks+=arr.size
            direct=[]
            for a,b,t in zip(bank[0],bank[1],truth):
                # Deliberately scalar loops, independent of the vectorized scorer.
                first=np.array([sum(np.linalg.norm(v-t) for v in arm)/8 for arm in (a,b)])
                dxx=dyy=dxy=0.
                for i in range(8):
                    for j in range(8):
                        if i==j:continue
                        dxx+=np.linalg.norm(a[i]-a[j])/56
                        dyy+=np.linalg.norm(b[i]-b[j])/56
                        dxy+=np.linalg.norm(a[i]-b[j])/56
                direct.append([first[0]-dxx/2,first[1]-dyy/2,
                               first.mean()-(dxx+dyy)/8-dxy/4])
            es=np.mean(direct,axis=0)
            np.testing.assert_allclose(es,g['energy'],atol=1e-14,rtol=1e-12);checks+=3
            gain=(es[0]-es[2])/es[0]
            np.testing.assert_allclose(gain,g['point']['energy_gain'],atol=1e-14);checks+=1
            for band in g['bands'].values():assert np.all(np.diff(band)>=0);checks+=1
            denom=np.trace(sx,axis1=-2,axis2=-1).mean()
            ratio=lambda v:float(np.trace(v,axis1=-2,axis2=-1).mean()/denom)
            row=dict(phase=g['phase'],target=g['target'],stratum=g['stratum'],pairs=g['pairs'],
                between_trace_fraction=ratio(between),between_debiased_trace_fraction=ratio(db),
                within_average_trace_change=ratio((sy-sx)/2),
                total_trace_change=ratio((sy-sx)/2+between),
                energy_gain=gain,energy_gain_mc_band=g['bands']['energy_gain'],
                mean_shift_sd=g['point']['mean_shift_sd'],
                residual_shift_dot=g['residual_shift_dot'],halves=g['halves'])
            rows.append(row)
        assert result['baseline_labels_exact'] and result['baseline_gap_maxabs']<2e-7
    primary=[r for r in rows if (r['target'],r['stratum']) in [('gaps','all'),('fractions','interior')]]
    final=dict(passed=True,checks=checks,source=record(Path(__file__)),result=record(OUT/'COMPLETE.json'),
        primary=primary,groups=rows,job=os.environ['SLURM_JOB_ID'],
        scope='Arithmetic validation; no population calibration or full-panel futility claim')
    write(OUT/'REVIEW.json',final)
    print(json.dumps(dict(passed=True,checks=checks,primary=primary)),flush=True)


if __name__=='__main__':run()
