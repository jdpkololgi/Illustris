"""A bounded, explicitly assigned native-phase lane inside an existing CPU job."""
import argparse
import json
import time
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_matter as matter


def warm_particle_reader(phase):
    """Resolve ASDF/Astropy lazy extensions before parallel header readers.

ASDF's first extension-manager construction can import astropy.coordinates
concurrently in different threads and observe partially initialized modules.
This is initialization only: the actual inventory still checks every CRC/header.
Do not edit the checkpoint-bound matter kernel to change startup ordering.
"""
    import asdf
    path=c.guarded(c.paths(phase)['snapshot_root']/'field_rv_A/field_rv_A_000.asdf',phase)
    with asdf.open(path,lazy_load=True) as saved:
        saved.extension_manager
        if saved.tree['header']['SimName']!=f'AbacusSummit_base_c000_{phase}':
            raise ValueError('reader initialization phase mismatch')


def run(phases,seconds,threads):
    c.require_compute()
    if len(set(phases))!=len(phases): raise ValueError('duplicate assigned native phase')
    for phase in phases:
        c.phase_guard(phase)
        if phase in ('ph000','ph002','ph003'): raise PermissionError('legacy sources have a separate qualification route')
    for item in json.loads((c.REPO/'SOURCE.json').read_text())['files']:
        if c.sha256(c.REPO/item['relative'])!=item['sha256']:
            raise ValueError('native-lane source snapshot changed')
    deadline=time.monotonic()+seconds
    remaining=list(phases)
    while remaining and time.monotonic()<deadline-1200:
        did_work=False
        for phase in tuple(remaining):
            marker=c.ROOT/'matter'/phase/'DENSITY_COMPLETE.json'
            if marker.exists():
                c.verify_receipt(marker,payload=False); remaining.remove(phase); continue
            if not (c.ROOT/'observations'/phase/'OBSERVED_COMPLETE.json').exists(): continue
            if phase!='ph007' and not (c.ROOT/'particle_b'/phase/'TRANSFER_COMPLETE.json').exists(): continue
            left=deadline-time.monotonic()
            if left<1200: return 75
            try:
                warm_particle_reader(phase)
                result=matter.build(phase,max(60,left-900),threads=threads)
            except BlockingIOError:
                print(json.dumps(dict(phase=phase,owned_by_another_worker=True)),flush=True); continue
            if result.get('paused'): return 75
            remaining.remove(phase); did_work=True
            print(json.dumps(dict(phase=phase,native_lane_complete=True)),flush=True)
        if remaining and not did_work: time.sleep(60)
    return 75 if remaining else 0


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--phases',nargs='+',required=True)
    p.add_argument('--seconds',type=float,default=12500); p.add_argument('--threads',type=int,default=16)
    a=p.parse_args(); raise SystemExit(run(a.phases,a.seconds,a.threads))
