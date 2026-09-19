"""Bounded preparation-only workers for paired geometry, targets and old phases."""
import argparse
import json
import time
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_coordinates as coord


def run(kind,phases,seconds):
    c.require_compute()
    for phase in phases: c.phase_guard(phase)
    for item in json.loads((c.REPO/'SOURCE.json').read_text())['files']:
        if c.sha256(c.REPO/item['relative'])!=item['sha256']:
            raise ValueError('postprocessing snapshot changed')
    deadline=time.monotonic()+seconds
    while time.monotonic()<deadline-900:
        remaining=[]; worked=False
        for phase in phases:
            if kind=='conditions':
                done=coord.ROOT/'conditions'/phase/'CONDITIONS_COMPLETE.json'
                ready=all((coord.ROOT/'observations'/phase/f'{cap}_COMPLETE.json').exists() for cap in ('NGC','SGC'))
            elif kind=='targets':
                done=coord.ROOT/'targets'/phase/'TARGETS_COMPLETE.json'
                ready=((c.ROOT/'matter'/phase/'DENSITY_COMPLETE.json').exists() and
                       (coord.ROOT/'geometry'/phase/'GEOMETRY_COMPLETE.json').exists())
            elif kind=='observations':
                done=coord.ROOT/'observations'/phase/'SGC_COMPLETE.json'
                ready=(c.ROOT/'observations'/phase/'OBSERVED_COMPLETE.json').exists()
            elif kind=='legacy_adopt':
                if phase not in ('ph002','ph003'): raise PermissionError('only counted legacy arrays can be adopted')
                done=c.ROOT/'matter'/phase/'DENSITY_COMPLETE.json'
                ready=((c.ROOT/'observations'/phase/'OBSERVED_COMPLETE.json').exists() and
                       (c.ROOT/'matter'/phase/'PARTICLES_VERIFIED.json').exists())
            else: raise ValueError('unregistered preparation kind')
            if done.exists():
                (c if kind=='legacy_adopt' else coord).verify_receipt(done,payload=False)
                continue
            remaining.append(phase)
            if not ready: continue
            left=deadline-time.monotonic()
            if left<({'targets':1800,'observations':3600}.get(kind,900)): return 75
            print(json.dumps(dict(kind=kind,phase=phase,starting=True,seconds_left=left)),flush=True)
            try:
                if kind=='conditions':
                    from workflows.sbi.e2e_coupled_geometry_repair import build as geometry
                    from workflows.sbi.e2e_coupled_condition_products import build
                    from workflows.sbi.e2e_coupled_condition_smoke import qualify
                    geometry(phase)
                    build(phase,limit=2)
                    qualify(phase)
                    build(phase)
                elif kind=='targets':
                    from workflows.sbi.e2e_coupled_target_products import build
                    result=build(phase,workers=32,stop_after_seconds=max(60,left-900))
                    if result.get('paused'): return 75
                elif kind=='observations':
                    from workflows.sbi.e2e_coupled_observations import build
                    build(phase)
                else:
                    from workflows.sbi.e2e_coupled_legacy_adopt import run as adopt
                    adopt(phase)
            except BlockingIOError:
                print(json.dumps(dict(kind=kind,phase=phase,owned_by_other_preparation_worker=True)),flush=True)
                continue
            worked=True
            print(json.dumps(dict(kind=kind,phase=phase,complete=True)),flush=True)
        if not remaining: return 0
        if not worked:
            print(json.dumps(dict(kind=kind,waiting_for_dependencies=remaining)),flush=True)
            time.sleep(60)
    return 75


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind',required=True,choices=['conditions','targets','observations','legacy_adopt'])
    p.add_argument('--phases',nargs='+',default=['ph007','ph008','ph000','ph002','ph003']+
        [v for v in c.ROLES if v not in ('ph007','ph008','ph000','ph002','ph003')])
    p.add_argument('--seconds',type=float,default=13800)
    a=p.parse_args(); raise SystemExit(run(a.kind,a.phases,a.seconds))
