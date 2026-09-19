"""Bounded phase-audit queue in an existing allocation; no fits or new jobs."""
import argparse
import json
from pathlib import Path
import time
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_coordinates as coord
from workflows.sbi import e2e_coupled_product_audit as auditor


def ready_paths(phase):
    c.phase_guard(phase)
    return [coord.ROOT/'conditions'/phase/'CONDITIONS_COMPLETE.json',
            coord.ROOT/'targets'/phase/'TARGETS_COMPLETE.json']


def qualified(phase):
    pointer=coord.ROOT/'product_audit'/phase/'LATEST_AUDIT.json'
    if not pointer.exists(): return False
    record=coord.verify_receipt(pointer,payload=False)
    path=c.guarded(record['audit']['path'],phase)
    if path.parent!=pointer.parent.resolve() or c.sha256(path)!=record['audit']['sha256']:
        raise ValueError('phase audit pointer/hash mismatch')
    report=coord.verify_receipt(path,payload=False)
    if report['source_code_sha256']!=c.sha256(auditor.__file__):
        raise ValueError('phase audit code differs from continuation; review required')
    for item in record['completion_sources']:
        if c.sha256(item['path'])!=item['sha256']: raise ValueError('phase products changed after audit')
    return True


def qualify_phase(phase):
    """Publish one audit; callers must serialize the audit queue itself."""
    result=auditor.run(phase)
    record=dict(**coord.provenance(),phase=phase,audit=result,
        completion_sources=[c.file_record(p,content_hash=True) for p in ready_paths(phase)],
        global_training_readiness=None,outputs=[],**{'pass':True})
    c.atomic_json(coord.ROOT/'product_audit'/phase/'LATEST_AUDIT.json',record)
    print(json.dumps(dict(phase=phase,phase_audit=result)),flush=True)


def run(seconds):
    c.require_compute()
    for item in json.loads((c.REPO/'SOURCE.json').read_text())['files']:
        if c.sha256(c.REPO/item['relative'])!=item['sha256']: raise ValueError('audit worker source drift')
    deadline=time.monotonic()+seconds
    while time.monotonic()<deadline-1200:
        remaining=[]; worked=False
        for phase in c.ROLES:
            if qualified(phase): continue
            remaining.append(phase)
            if not all(path.exists() for path in ready_paths(phase)): continue
            if time.monotonic()>deadline-1200: return 75
            try:
                qualify_phase(phase)
            except BlockingIOError:
                continue
            worked=True
        if not remaining: return 0
        if not worked:
            print(json.dumps(dict(audit_waiting_for_products=remaining)),flush=True); time.sleep(60)
    return 75


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--seconds',type=float,default=5400)
    a=p.parse_args(); raise SystemExit(run(a.seconds))
