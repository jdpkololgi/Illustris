"""Audit the unusable classical control without tuning against held-out data."""
import argparse
import json
from pathlib import Path
import numpy as np
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_views as views
from workflows.sbi.e2e_coarse_controls import BASE,panel,target,regions,observation

def run(root):
    c.require_compute();old=BASE/'coarse_controls_20260926_v1'
    complete=json.loads((old/'COMPLETE.json').read_text());fit=json.loads((old/'FIT.json').read_text())
    result=dict(source=c.sha256(old/'COMPLETE.json'),fit=c.sha256(old/'FIT.json'),summaries=[],training_closure=[])
    for side in ('train','development'):
        selected=[r for r in complete['rows'] if r['model']=='classical' and (r['phase'] in c.TRAIN)==(side=='train')]
        for label,sl in (('core',slice(0,2)),('block',slice(2,10))):
            truth=np.asarray([r['truth'] for r in selected])[:,sl].ravel();pred=np.asarray([r['mean'] for r in selected])[:,sl].ravel()
            rmse=float(np.sqrt(np.mean((pred-truth)**2)));null=float(np.sqrt(np.mean(truth**2)));corr=float(np.corrcoef(pred,truth)[0,1])
            result['summaries'].append(dict(panel=side,group=label,rmse=rmse,zero_predictor_rmse=null,correlation=corr,
                functional_gate=rmse<null and corr>0))
    chart=views.load_chart(fit['normalizer']);coef=np.asarray(fit['coefficients'])
    for phase,pid in panel():
        if phase not in c.TRAIN:continue
        rho,truth,sha=target(phase,pid);closure=float(np.max(abs(regions(rho,phase)-truth)))
        if closure>2e-6:raise ValueError('real-anchor regional alignment failed')
        ob=views.load_observations(phase,pid,fit['normalizer']);y,kind,bins,valid=observation(ob,chart,phase);x=np.log(rho)
        for k in (0,1):
            mask=valid&(kind==k);error=y[mask]-(coef[k,0]+coef[k,1]*x[mask])
            result['training_closure'].append(dict(phase=phase,pair=pid,kind=k,region_closure=closure,target_sha256=sha,
                observation_truth_correlation=float(np.corrcoef(y[mask],x[mask])[0,1]),
                likelihood_mean_residual=float(error.mean()),likelihood_rmse=float(np.sqrt(np.mean(error**2))),
                support_fraction=float(valid.mean())))
    result['interpretation']='Unusable control; regional alignment tested. Solver validity does not validate empirical likelihood; no repair or general classical-method verdict.'
    c.atomic_json(Path(root)/'CLASSICAL_AUDIT.json',result,replace=True)
    print('CLASSICAL_AUDIT_COMPLETE',result['summaries'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);run(p.parse_args().root)
