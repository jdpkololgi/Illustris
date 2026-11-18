"""Bounded approved CPU window: independent identity checks and CFM responses.

No automatic retry. A P12 failure is recorded and does not prevent independent
CFM preparation. Cap starts are limited by measured, conservatively padded cost.
"""
import argparse
import os
from pathlib import Path
import time
import traceback

from workflows.sbi import cfm_lowz_selection_response as lowz
from workflows.sbi.build_cfm_matched_response import build
from workflows.sbi.prepare_p12a_galaxy_identity import run as identity
from workflows.sbi.prepare_cfm_lowz_inputs import require_compute, save_json


def run(seconds):
    require_compute()
    if not 300 <= seconds <= 7000:
        raise ValueError('Window must stay inside approved two hours')
    started=time.monotonic();plan,policy,_=lowz.load_plan()
    output=Path(plan['output_root'])/('repair_window_'+os.environ['SLURM_JOB_ID'])
    output.mkdir(exist_ok=False)
    stages=[];cost_per_chunk=None
    for phase in ('ph001','ph006'):
        then=time.monotonic()
        try:identity(phase);status='complete';error=None
        except Exception:
            status='failed';error=traceback.format_exc();print(error,flush=True)
        stages.append(dict(stage='p12_identity',phase=phase,status=status,error=error,elapsed=time.monotonic()-then))
        save_json(output/('identity_'+phase+'.json'),stages[-1])
    phases=['ph001','ph006']+[p for p in policy['training']+policy['development_and_calibration'] if p not in ('ph001','ph006')]
    stop=None
    for phase in phases:
        record=lowz.read_json(Path(plan['output_root'])/'responses_full'/phase/'RESPONSE_PREPARED.json')
        for cap in lowz.CAPS:
            shape=record['diagnostics'][cap]['grid']['shape']
            chunks=1
            for n in shape:chunks*=((n+31)//32)
            remaining=seconds-(time.monotonic()-started)
            estimate=900 if cost_per_chunk is None else 1.5*cost_per_chunk*chunks+60
            if remaining<estimate+120:
                stop=dict(reason='remaining budget below padded next-cap estimate',phase=phase,cap=cap,
                          remaining_seconds=remaining,estimated_seconds=estimate)
                break
            then=time.monotonic()
            try:build(phase,[cap],16);status='complete';error=None
            except Exception:
                status='failed';error=traceback.format_exc();print(error,flush=True)
            elapsed=time.monotonic()-then
            stages.append(dict(stage='cfm_response',phase=phase,cap=cap,status=status,error=error,elapsed=elapsed))
            save_json(output/(phase+'_'+cap+'.json'),stages[-1])
            if status=='failed':
                stop=dict(reason='response failed; no automatic retry',phase=phase,cap=cap);break
            cost_per_chunk=max(cost_per_chunk or 0.,elapsed/chunks)
        if stop:break
    save_json(output/'WINDOW_COMPLETE.json',dict(job=os.environ['SLURM_JOB_ID'],stages=stages,
        stop=stop,elapsed_seconds=time.monotonic()-started,model_inputs_ready=False,
        phase_roles_changed=False,training_started=False))
    print('REPAIR_WINDOW_FINISHED',output,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seconds',type=int,default=7000)
    run(p.parse_args().seconds)
