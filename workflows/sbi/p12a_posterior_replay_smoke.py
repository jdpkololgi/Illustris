"""Frozen P12-A sampling reproducibility on eight existing training contexts."""
import argparse,json,os
from pathlib import Path
import numpy as np
import torch
from workflows.sbi.p12a_blind_inference import reconstruct_fmpe
from workflows.sbi.p12_train_base_response_fmpe import sample_posterior,theta_to_eigenvalues
from workflows.abacus_tweb.p12a_coordinate_sample_audit import record
BASE=Path('/pscratch/sd/d/dkololgi/abacus/p10_multiphase/p12a_base_response_v1')

def main(output):
    if not torch.cuda.is_available() or not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('GPU allocation required')
    source=record(__file__,small=True)
    data=BASE/'training_oof_sample.npz';ckpath=BASE/'fmpe_seed42/fmpe_estimator.pt'
    with np.load(data) as f:
        x=f['context'];rows=[]
        for cap in [0,1]:
            for lo,hi in [(.15,.25),(.25,.35),(.35,.45),(.45,.55)]:
                candidates=np.flatnonzero((x[:,5]==cap)&(x[:,3]>=lo)&(x[:,3]<hi))
                if not len(candidates):raise ValueError('missing stratum')
                rows.append(int(candidates[0]))
        context=x[rows].copy()
    posterior,ck=reconstruct_fmpe(ckpath,'cuda')
    x=((context-np.asarray(ck['context_mean'],dtype='f4'))/np.asarray(ck['context_std'],dtype='f4')).astype('f4')
    samples=[]
    for _ in range(2):
        torch.manual_seed(20260924);torch.cuda.manual_seed_all(20260924)
        samples.append(sample_posterior(posterior,x,32,8,'cuda'))
    theta=samples[0]*np.asarray(ck['theta_std'],dtype='f4')+np.asarray(ck['theta_mean'],dtype='f4')
    eig=theta_to_eigenvalues(theta)
    checks=dict(repeat_draws_equal=bool(np.array_equal(*samples)),finite_ordered=bool(np.isfinite(eig).all() and (np.diff(eig,axis=-1)>=0).all()))
    r=dict(schema='p12a-posterior-replay-smoke-v1',source=source,checkpoint=record(ckpath,small=True),training_archive=record(data),rows=rows,contexts=context.tolist(),draws_per_context=32,seed=20260924,checks=checks,pass_checks=all(checks.values()),ready_for_desi_canary=False,scope='sampling repeatability only; no recalibration or golden-adapter qualification')
    if source!=record(__file__,small=True):raise ValueError('source changed')
    with output.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
    print(checks,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    main(a.output)
