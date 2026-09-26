"""Apply inherited coverage criteria to saved ph006 candidate draws; no fitting."""
import argparse,json,hashlib,os
from pathlib import Path
import numpy as np
import torch
from workflows.sbi.p12a_evaluate_blind import _conditional_coverage, interval_coverage
from workflows.sbi.p12a_blind_evaluation_contract import GATES
from workflows.sbi.p12_train_base_response_fmpe import theta_to_eigenvalues

def main(root,output):
    if output.exists():raise FileExistsError(output)
    a=json.loads((root/'posterior/calibration_audit/P12A_CALIBRATION_AUDIT.json').read_text())
    ready=json.loads((root/'dataset/P12A_DATASET_READY.json').read_text())
    ck=torch.load(root/'posterior/fmpe_estimator.pt',map_location='cpu',weights_only=False)
    index=np.load(a['provenance']['evaluation_index']); samples=np.load(a['provenance']['samples'],mmap_mode='r')
    with np.load(ready['validation']['path']) as d:
        print('validation fields',d.files,flush=True)
        truth=d['truth_eigenvalues'][index];x=d['context'][index];base=d['base_prediction_eigenvalues'][index]
        shell=d['shell'][index]
    q=np.empty((len(index),4,3));mean=np.empty((len(index),3))
    for start in range(0,len(index),512):
        stop=min(start+512,len(index));e=theta_to_eigenvalues(np.asarray(samples[start:stop])*np.asarray(ck['theta_std'])+np.asarray(ck['theta_mean']))
        q[start:stop]=np.quantile(e,[.05,.16,.84,.95],axis=1).transpose(1,0,2);mean[start:stop]=e.mean(axis=1)
    summary=dict(shell=shell,ntilde_mpc3=np.exp(x[:,4]),distance_to_support_boundary_mpc=np.expm1(x[:,6]),eigenvalue_q05=q[:,0],eigenvalue_q16=q[:,1],eigenvalue_q84=q[:,2],eigenvalue_q95=q[:,3])
    conditional=_conditional_coverage(summary,truth)
    c68=interval_coverage(truth,q[:,1],q[:,2]);c90=interval_coverage(truth,q[:,0],q[:,3]);err=max(abs(c68-.68).max(),abs(c90-.9).max())
    gates=dict(global_coverage=bool(err<=GATES['global_coverage_absolute_error_maximum']),nonsparse_conditional_coverage=bool(conditional['nonsparse_maximum_absolute_error']<=GATES['nonsparse_conditional_coverage_absolute_error_maximum']),sparse_shell_release=bool(conditional['sparse_shell_maximum_absolute_error']<=GATES['sparse_shell_release_absolute_error_maximum']))
    report=dict(schema='p12a-halo48-conditional-review-v1',phase='ph006',phase_already_exposed=True,rows=len(index),draws=samples.shape[1],weighting='unweighted per-row, inherited evaluator; ph006 sample differs from natural full-phase population',coverage68=c68.tolist(),coverage90=c90.tolist(),maximum_global_error=float(err),conditional=conditional,gates=gates,coverage_gates_pass=all(gates.values()),thresholds=GATES,not_independent_confirmation=True,ready_for_desi_canary=False,job=os.environ.get('SLURM_JOB_ID'),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ['coverage68','coverage90','maximum_global_error','gates']}),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.root,a.output)
