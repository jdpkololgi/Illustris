"""Bounded sequential FFT queue; at most one full native FFT per worker/node."""
import argparse
import json
import time
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_target_products as targets
from workflows.sbi import prepare_shared_wholefield as whole


def run(coupled,wholefield,seconds):
    c.require_compute()
    if not c.SHARED_POLICY or not 0<seconds<=4800:raise ValueError('Explicit bounded campaign required')
    deadline=time.monotonic()+seconds
    for phase in coupled:
        c.phase_guard(phase)
        remaining=deadline-time.monotonic()
        if remaining<900:return 75
        value=targets.build(phase,workers=32,stop_after_seconds=min(1800,remaining-300))
        if value.get('paused'):return 75
        print(json.dumps(dict(phase=phase,coupled_targets_complete=True)),flush=True)
    for phase in wholefield:
        rows,_=whole.rows_for(phase)
        while not all((whole.data.ROOT/'data'/phase/r['pair_id']/'OBSERVATIONS.json').exists() for r in rows):
            if time.monotonic()>deadline-900:return 75
            time.sleep(30)
        if time.monotonic()>deadline-900:return 75
        whole.fresh(phase,'targets',32)
    return 0

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--coupled-phases',nargs='+',required=True);p.add_argument('--wholefield-phases',nargs='+',required=True);p.add_argument('--seconds',type=float,default=4500);a=p.parse_args();raise SystemExit(run(a.coupled_phases,a.wholefield_phases,a.seconds))
