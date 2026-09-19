"""Serial audit/statistics handoff after explicitly named old steps terminate.

No new allocation, no scientific fitting, and no concurrent audit publishers.
The eight-GiB lane alternates audit and normalization rather than running both
memory footprints together. Unexpected predecessor failures stop the handoff.
"""
import argparse
import json
import re
import subprocess
import time

from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_audit_worker as audits
from workflows.sbi import e2e_coupled_normalization as norm


def predecessors_ready(output, expected):
    rows = {row[0]: row[1:] for row in
            (line.split('|', 3) for line in output.splitlines() if line.strip())}
    ready = True
    for step, module in expected.items():
        if not re.fullmatch(r'\d+\.\d+', step):
            raise ValueError('explicit numerical predecessor step required')
        row = rows.get(step)
        if row is None:
            ready = False
            continue
        state, code, command = row
        if f'-m workflows.sbi.{module} ' not in command + ' ':
            raise ValueError('predecessor step is not the expected qualification worker')
        if state in ('RUNNING', 'PENDING', 'COMPLETING', 'CONFIGURING'):
            ready = False
        elif (state, code) not in (('COMPLETED', '0:0'), ('FAILED', '75:0')):
            raise RuntimeError('unexpected qualification predecessor failure: ' + str(row))
    return ready


def run(seconds, audit_step, normalization_step):
    c.require_compute()
    expected = {audit_step: 'e2e_coupled_audit_worker',
                normalization_step: 'e2e_coupled_normalization_worker'}
    if len(expected) != 2:
        raise ValueError('two distinct predecessor steps required')
    for step in expected:
        if not re.fullmatch(r'\d+\.\d+', step):
            raise ValueError('explicit numerical predecessor step required')
    for item in json.loads((c.REPO / 'SOURCE.json').read_text())['files']:
        if c.sha256(c.REPO / item['relative']) != item['sha256']:
            raise ValueError('qualification tail source snapshot changed')
    deadline = time.monotonic() + seconds
    jobs = ','.join(sorted({step.split('.')[0] for step in expected}))
    while time.monotonic() < deadline - 1200:
        result = subprocess.run(
            ['sacct', '-n', '-P', '-j', jobs, '-o', 'JobIDRaw,State,ExitCode,SubmitLine%1000'],
            check=True, text=True, capture_output=True, timeout=30)
        if predecessors_ready(result.stdout, expected):
            break
        print(json.dumps(dict(qualification_waiting_for=list(expected))), flush=True)
        time.sleep(60)
    else:
        return 75
    print(json.dumps(dict(qualification_predecessors_terminal=list(expected))), flush=True)
    while time.monotonic() < deadline - 1200:
        # Only one ready phase at a time, then update all newly available train
        # moments. This makes normalization available promptly to the IO lane.
        remaining = [phase for phase in c.ROLES if not audits.qualified(phase)]
        worked = False
        for phase in remaining:
            if not all(path.exists() for path in audits.ready_paths(phase)):
                continue
            audits.qualify_phase(phase)
            worked = True
            break
        if time.monotonic() >= deadline - 1200:
            return 75
        result = norm.fit(collect_ready=True)
        normalized = 'fit_phases' in result
        print(json.dumps(dict(audits_remaining=len(remaining) - int(worked),
                              normalizer_complete=normalized)), flush=True)
        if not remaining and normalized:
            return 0
        if not worked:
            time.sleep(60)
    return 75


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=13800)
    parser.add_argument('--audit-step', required=True)
    parser.add_argument('--normalization-step', required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.seconds, args.audit_step, args.normalization_step))
