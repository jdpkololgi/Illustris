"""Bounded deterministic preparation worker; no allocations or model fitting."""
import argparse
import json
import time

from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_coordinates as coord

ORDER=tuple(f'ph{i:03d}' for i in [7,8,9,10,11,20,21,22,23,24,12,13,14,15,16,17,18,19,0,2,3])


def run(kind,seconds):
    c.require_compute()
    source=c.REPO/'SOURCE.json'
    frozen=json.loads(source.read_text())
    for record in frozen['files']:
        if c.sha256(c.REPO/record['relative'])!=record['sha256']:
            raise ValueError('frozen source changed')
    end=time.monotonic()+seconds
    while time.monotonic()<end:
        remaining=[]
        did_work=False
        for phase in ORDER:
            if kind=='pairing':
                done=c.ROOT/'observations'/phase/'OBSERVED_COMPLETE.json'
            elif kind=='matter':
                if phase in ('ph000','ph002','ph003'):
                    continue  # Separate explicit provenance-adoption step.
                done=c.ROOT/'matter'/phase/'DENSITY_COMPLETE.json'
            else:
                done=coord.ROOT/'observations'/phase/'SGC_COMPLETE.json'
            if done.exists():
                c.verify_receipt(done,payload=False)
                if kind=='observations':
                    coord.verify_receipt(done,payload=False)
                    coord.verify_receipt(coord.ROOT/'observations'/phase/'NGC_COMPLETE.json',payload=False)
                continue
            remaining.append(phase)
            left=end-time.monotonic()
            reserve={'pairing':600,'matter':1200,'observations':3600}[kind]
            if left<reserve:
                print(json.dumps(dict(kind=kind,paused=True,remaining=remaining)),flush=True)
                return 75
            if kind!='pairing' and not (c.ROOT/'observations'/phase/'OBSERVED_COMPLETE.json').exists():
                continue
            if kind=='matter' and phase!='ph007' and not (c.ROOT/'particle_b'/phase/'TRANSFER_COMPLETE.json').exists():
                continue
            print(json.dumps(dict(kind=kind,phase=phase,starting=True,seconds_left=left)),flush=True)
            if kind=='pairing':
                from workflows.sbi.e2e_coupled_pairing import prepare_phase
                prepare_phase(phase)
            elif kind=='matter':
                from workflows.sbi.e2e_coupled_matter import build
                result=build(phase,max(60,left-900),threads=32)
                if result.get('paused'):
                    return 75
            else:
                from workflows.sbi.e2e_coupled_observations import build
                build(phase)
            did_work=True
        if not remaining:
            print(json.dumps(dict(kind=kind,stage_complete=True)),flush=True)
            return 0
        if not did_work:
            print(json.dumps(dict(kind=kind,waiting_for_dependencies=remaining)),flush=True)
            time.sleep(60)
    return 75


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind',required=True,choices=['pairing','matter','observations'])
    p.add_argument('--seconds',type=float,default=13800)
    a=p.parse_args(); raise SystemExit(run(a.kind,a.seconds))
