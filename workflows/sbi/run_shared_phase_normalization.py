"""Collect audited full-20 moments incrementally; never publish a partial fit."""
import argparse
import json
import time
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_normalization as norm
from workflows.sbi import e2e_coupled_audit_worker as audits


def run(seconds):
    c.require_compute()
    if not c.SHARED_POLICY or len(c.TRAIN)!=20 or not 0<seconds<=5400:raise ValueError('Bounded full-20 campaign required')
    paths=[c.CONFIG,norm.__file__,audits.__file__,c.REPO/'workflows/sbi/audit_shared_phase_sources.py']
    frozen={str(p):c.sha256(p) for p in paths};deadline=time.monotonic()+seconds
    while time.monotonic()<deadline-600:
        if any(c.sha256(p)!=digest for p,digest in frozen.items()):raise ValueError('Normalization worker source drift')
        result=norm.fit(collect_ready=True)
        if 'fit_phases' in result:
            print(json.dumps(dict(normalization_complete=True,fit_phases=result['fit_phases'],phase_weight=result['phase_weight'])),flush=True)
            return 0
        print(json.dumps(result),flush=True);time.sleep(60)
    return 75

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seconds',type=float,default=4800);a=p.parse_args();raise SystemExit(run(a.seconds))
