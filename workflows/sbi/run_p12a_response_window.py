"""Bounded, deterministic matched-response queue inside an approved allocation.

No scheduler submission, retries, fits, role changes, or readiness promotion.
This can continue independently of the unresolved P12 spatial-fold contract.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
from workflows.sbi import prepare_p12a_lowz_inputs as p12
from workflows.sbi.prepare_cfm_lowz_inputs import require_compute, save_json, sha256


def run(seconds):
    require_compute()
    if not 300 <= seconds <= 9300:raise ValueError('Bounded existing-allocation window required')
    plan,_,policy=p12.load_plan();root=Path(plan['output_root'])
    directory=root/'response_windows'/os.environ['SLURM_JOB_ID']
    directory.mkdir(parents=True,exist_ok=False)
    source=Path(__file__).with_name('build_p12a_matched_response.py');source_hash=sha256(source)
    deadline=time.monotonic()+seconds;records=[];stop=None;estimate=900.
    phases=['ph006']+[p for p in policy['training']+policy['development_and_calibration'] if p not in ('ph001','ph006')]
    # ph001 has already completed; it is not retried by this continuation.
    if not all((root/'ph001/matched_cic_response_v1'/(cap+'_COMPLETE.json')).exists() for cap in ('SGC','NGC')):
        raise ValueError('Complete ph001 response products first')
    for phase in phases:
        remaining=deadline-time.monotonic()
        if remaining<estimate+120:
            stop=dict(reason='budget pause',next_phase=phase,remaining_seconds=remaining);break
        if sha256(source)!=source_hash:raise ValueError('Builder changed during queue')
        log=directory/(phase+'.log');started=time.monotonic()
        print('P12_RESPONSE_START',phase,flush=True)
        with log.open('x') as stream:
            try:
                result=subprocess.run([sys.executable,'-u','-m','workflows.sbi.build_p12a_matched_response',
                    '--phase',phase,'--workers','16'],cwd=p12.REPO,stdout=stream,stderr=subprocess.STDOUT,
                    timeout=remaining-60,check=False)
                code=result.returncode
            except subprocess.TimeoutExpired:code=124
        elapsed=time.monotonic()-started
        record=dict(phase=phase,exitcode=code,elapsed_seconds=elapsed,log=str(log),builder_sha256=source_hash)
        records.append(record);save_json(directory/(phase+('_COMPLETE.json' if code==0 else '_FAILED.json')),record)
        print('P12_RESPONSE_END',phase,code,round(elapsed,1),flush=True)
        if code:
            stop=dict(reason='failure; no retry',phase=phase);break
        estimate=max(300.,elapsed*1.25)
    result=dict(job=os.environ['SLURM_JOB_ID'],stages=records,stop=stop,
        host_folds_unresolved=True,model_inputs_ready=False,learned_model_training_started=False)
    save_json(directory/'WINDOW_COMPLETE.json',result)
    print('P12_RESPONSE_WINDOW_COMPLETE',result,flush=True)
    return 0 if not stop or stop['reason']=='budget pause' else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--seconds',type=int,required=True)
    raise SystemExit(run(parser.parse_args().seconds))
