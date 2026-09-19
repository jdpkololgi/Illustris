"""One reviewed engineering recovery for target OOM in58552205.0.

Not a generic failed-job retry policy. Other predecessor steps may be healthy
and running at launch, but must end normally/planned before publisher handoff.
Scientific kernels remain unchanged.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time

from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi.e2e_coupled_prepare_ops import accounting

PREDECESSOR = '58552205'
EXPECTED = {
    '0': 'e2e_coupled_post_worker --kind targets',
    '1': 'e2e_coupled_interface_worker',
    '2': 'e2e_coupled_native_pool',
    '3': 'e2e_coupled_qualification_tail',
    '4': 'e2e_coupled_post_worker --kind observations',
    '5': 'e2e_coupled_post_worker --kind conditions',
}


def review(output, allow_running=False):
    rows = {parts[0]: parts[1:] for parts in
            (line.split('|', 3) for line in output.splitlines() if line.strip())}
    parent = rows.get(PREDECESSOR)
    parents = {('FAILED', '1:0')}
    if allow_running:
        parents.add(('RUNNING', '0:0'))
    if parent is None or tuple(parent[:2]) not in parents:
        raise RuntimeError('reviewed OOM parent has an unexpected state')
    numerical = {key for key in rows if key.startswith(PREDECESSOR + '.')
                 and key.split('.')[-1].isdigit()}
    if numerical != {PREDECESSOR + '.' + key for key in EXPECTED}:
        raise RuntimeError('unexpected predecessor step set')
    for suffix, module in EXPECTED.items():
        state, code, command = rows[PREDECESSOR + '.' + suffix]
        if '-m workflows.sbi.' + module + ' ' not in command + ' ':
            raise RuntimeError('predecessor worker identity mismatch')
        allowed = {('OUT_OF_MEMORY', '0:125')} if suffix == '0' else {
            ('COMPLETED', '0:0'), ('FAILED', '75:0')}
        if allow_running and suffix != '0':
            allowed.add(('RUNNING', '0:0'))
        if (state, code) not in allowed:
            raise RuntimeError('additional unreviewed predecessor failure')
    return dict(predecessor=PREDECESSOR, reviewed_target_oom=True,
                predecessor_terminal=parent[0] == 'FAILED',
                other_steps_normal_or_planned_or_explicitly_running=True)


def verify_predecessor(allow_running=False):
    result = subprocess.run(['sacct', '-n', '-P', '-j', PREDECESSOR,
                             '-o', 'JobIDRaw,State,ExitCode,SubmitLine%1000'],
                            check=True, text=True, capture_output=True, timeout=30)
    return dict(**review(result.stdout, allow_running), accounting_output=result.stdout)


def wait_for_writers(seconds=6000):
    """No replacement audit/interface publisher before the OLD parent is gone."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        record = verify_predecessor(allow_running=True)
        if record['predecessor_terminal']:
            verify_predecessor()
            return
        print(json.dumps(dict(waiting_for_old_publishers=PREDECESSOR)), flush=True)
        time.sleep(60)
    raise TimeoutError('bounded old-writer handoff expired')


def command(snapshot):
    return ['salloc', '--nodes=1', '--ntasks=1', '--cpus-per-task=128', '--mem=0',
            '--constraint=cpu', '--qos=interactive', '--time=02:00:00',
            '--account=desi', '--licenses=scratch', '--immediate=600',
            '--job-name=coupled-target-recovery', '/bin/bash',
            str(snapshot / 'workflows/sbi/e2e_coupled_target_recovery_step.sh'), str(snapshot)]


def run(snapshot):
    snapshot = snapshot.resolve()
    if snapshot.parent != (c.ROOT / 'source_snapshots').resolve():
        raise PermissionError('approved preparation source snapshot required')
    for item in json.loads((snapshot / 'SOURCE.json').read_text())['files']:
        if c.sha256(snapshot / item['relative']) != item['sha256']:
            raise ValueError('recovery snapshot source drift')
    deadline = time.monotonic() + 10800
    while time.monotonic() < deadline:
        result = subprocess.run(['squeue', '-h', '-u', 'dkololgi', '-o', '%i|%T'],
                                check=True, text=True, capture_output=True, timeout=30)
        rows = [line.split('|') for line in result.stdout.splitlines() if line.strip()]
        if len(rows) < 2:
            break
        print(json.dumps(dict(waiting_for=PREDECESSOR, jobs=rows)), flush=True)
        time.sleep(60)
    else:
        raise RuntimeError('bounded recovery handoff expired; no request')
    evidence = verify_predecessor(allow_running=True)
    usage = accounting()
    charged = sum(row['hours_cap'] if row['state'] in
                  ('RUNNING', 'PENDING', 'COMPLETING', 'CONFIGURING') else
                  row['elapsed_seconds'] / 3600 for row in usage['allocations']
                  if row['kind'] == 'cpu')
    if charged + 2 > c.config()['approval']['cpu_node_hours']:
        raise RuntimeError('two-hour recovery exceeds original CPU allowance')
    # Exclusive publication prevents any second request, including after failure.
    c.atomic_json(c.ROOT / 'ops/TARGET_OOM_RECOVERY_REQUEST.json',
                  dict(**c.provenance(), **evidence, command=command(snapshot),
                       cpu_hours_reserved_before=charged, retry_count=0,
                       numerical_kernels_unchanged=True, scientific_fit=False))
    return subprocess.run(command(snapshot), check=False).returncode


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', required=True, type=Path)
    args = parser.parse_args()
    with c.single_writer(c.ROOT / 'ops/target_oom_recovery_controller'):
        raise SystemExit(run(args.snapshot))
