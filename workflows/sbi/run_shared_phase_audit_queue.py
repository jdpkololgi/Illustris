"""Bounded disjoint current-campaign audit queue inside an approved allocation."""
import argparse
import json
from pathlib import Path
import time
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_coordinates as coord
from workflows.sbi import e2e_coupled_audit_worker as audits


def run(phases,seconds):
    c.require_compute()
    if not c.SHARED_POLICY or not 0<seconds<=5400:raise ValueError('Explicit bounded shared campaign required')
    for phase in phases:c.phase_guard(phase)
    deadline=time.monotonic()+seconds
    for phase in phases:
        if audits.qualified(phase):continue
        paths=audits.ready_paths(phase)+[c.ROOT/'matter'/phase/'PARTICLES_VERIFIED.json']
        while not all(p.exists() for p in paths):
            if time.monotonic()>deadline-600:return 75
            time.sleep(30)
        if time.monotonic()>deadline-600:return 75
        audits.qualify_phase(phase)
    return 0

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phases',nargs='+',required=True);p.add_argument('--seconds',type=float,default=4800);a=p.parse_args();raise SystemExit(run(a.phases,a.seconds))
