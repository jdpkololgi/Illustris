"""One approved technical GPU hour after a verified preparation handoff.

Frozen deterministic launcher only: no automatic retry, cancellation, batch
fallback, scientific fitting or new resource decisions after disconnect.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import time

from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi.e2e_coupled_post_allocation import require_planned_terminal
from workflows.sbi.e2e_coupled_prepare_ops import accounting


def can_request(rows, predecessor):
    return len(rows) < 2 and all(row[0] != predecessor for row in rows)


def reserved_gpu_hours(usage):
    return sum(row.get('gpus', 0) * (row['hours_cap'] if row['state'] in
               ('RUNNING', 'PENDING', 'COMPLETING', 'CONFIGURING') else
               row['elapsed_seconds'] / 3600) for row in usage['allocations']
               if row['kind'] == 'gpu')


def command(snapshot):
    return ['salloc', '--nodes=1', '--ntasks=1', '--cpus-per-task=32',
            '--constraint=gpu&hbm80g', '--gpus=1', '--qos=shared_interactive',
            '--time=01:00:00', '--account=desi_g', '--licenses=scratch',
            '--immediate=600', '--job-name=coupled-gpu-technical', '/bin/bash',
            str(snapshot / 'workflows/sbi/e2e_coupled_gpu_benchmark_step.sh'), str(snapshot)]


def run(predecessor, snapshot):
    if not re.fullmatch(r'\d+', predecessor):
        raise ValueError('explicit numerical predecessor allocation required')
    snapshot = snapshot.resolve()
    if snapshot.parent != (c.ROOT / 'source_snapshots').resolve():
        raise PermissionError('approved preparation source snapshot required')
    files = json.loads((snapshot / 'SOURCE.json').read_text())['files']
    for item in files:
        if c.sha256(snapshot / item['relative']) != item['sha256']:
            raise ValueError('technical GPU snapshot source drift')
    if not (snapshot / 'workflows/sbi/e2e_coupled_gpu_benchmark_step.sh').is_file():
        raise FileNotFoundError('technical GPU step missing')
    deadline = time.monotonic() + 14400
    while time.monotonic() < deadline:
        result = subprocess.run(['squeue', '-h', '-u', 'dkololgi', '-o', '%i|%T'],
                                check=True, text=True, capture_output=True, timeout=30)
        rows = [line.split('|') for line in result.stdout.splitlines() if line.strip()]
        if can_request(rows, predecessor):
            break
        print(json.dumps(dict(waiting_for=predecessor, jobs=rows, requested_gpu=False)), flush=True)
        time.sleep(60)
    else:
        raise RuntimeError('four-hour bounded handoff wait expired; no GPU requested')
    require_planned_terminal(predecessor)
    usage = accounting()
    if reserved_gpu_hours(usage) + 1 > c.config()['approval']['gpu_hours']:
        raise RuntimeError('one technical GPU hour would exceed the original allowance')
    launch = command(snapshot)
    print(json.dumps(dict(command=launch, reserved_gpu_hours_before=reserved_gpu_hours(usage),
                          scientific_training_authorized=False)), flush=True)
    return subprocess.run(launch, check=False).returncode


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--after-job', required=True)
    parser.add_argument('--snapshot', type=Path, required=True)
    args = parser.parse_args()
    with c.single_writer(c.ROOT / 'ops/technical_gpu_controller'):
        raise SystemExit(run(args.after_job, args.snapshot))
