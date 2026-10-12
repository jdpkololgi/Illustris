"""Fixed, authorized CPU screen owner; survives SSH disconnect, never retries."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time

REPO = Path(__file__).resolve().parents[2]
ROOT = Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/wholefield_comparison_20261009_v1')
OUT = ROOT/'matched_gate2_20261013_v1'
PY = '/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python'


def owner():
    if not os.environ.get('TMUX'):
        raise RuntimeError('Approved detached tmux owner required')
    jobs = subprocess.check_output(['squeue', '-u', os.environ['USER'], '-h', '-o', '%q'], text=True)
    if sum('interactive' in q for q in jobs.splitlines()) >= 2:
        raise RuntimeError('Two interactive slots already occupied')
    paths = [Path(__file__), REPO/'workflows/sbi/e2e_wholefield_matched_screen.py',
             REPO/'workflows/sbi/e2e_wholefield_matched_screen_verify.py',
             REPO/'configs/e2e_wholefield_matched_gate2_20261013.json']
    source = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    command = ['salloc', '-N1', '-n1', '-c64', '--constraint=cpu', '--qos=interactive',
               '--account=desi', '--licenses=scratch', '--time=01:00:00', '--immediate=600',
               '--job-name=wf_matched_gate2', 'srun', '-N1', '-n1', '-c64', '--cpu-bind=cores',
               PY, '-u', str(Path(__file__).resolve()), 'compute']
    code = subprocess.run(command, cwd=REPO).returncode
    receipt = dict(command=command, exit_code=code, passed=False, sources=source)
    if (OUT/'BINDING.json').exists():
        job = json.loads((OUT/'BINDING.json').read_text())['job']
        for _ in range(12):
            account = subprocess.check_output(['sacct', '-X', '-j', job, '-n', '-P',
                    '-o', 'JobID,State,ElapsedRaw,AllocCPUS,ExitCode'], text=True).strip()
            if account and '|RUNNING|' not in account and '|COMPLETING|' not in account:
                break
            time.sleep(5)
        receipt.update(job=job, accounting=account)
        verified = json.loads((OUT/'VERIFIED.json').read_text()) if (OUT/'VERIFIED.json').exists() else {}
        fields = account.rstrip('|').split('|')
        receipt['passed'] = (code == 0 and len(fields) == 5 and fields[0] == job
                             and fields[1] == 'COMPLETED' and fields[4] == '0:0'
                             and verified.get('passed', False))
    if any(hashlib.sha256(Path(p).read_bytes()).hexdigest() != h for p, h in source.items()):
        receipt['passed'] = False
        receipt['source_changed'] = True
    with (ROOT/'matched_gate2_20261013_OWNER_EXIT.json').open('x') as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps(receipt), flush=True)
    if not receipt['passed']:
        raise RuntimeError('Gate execution or independent verification failed; inspect without retry')


def compute():
    if not socket.gethostname().startswith('nid') or not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Compute node required')
    for command in ([PY, '-u', '-m', 'workflows.sbi.e2e_wholefield_matched_screen', 'prepare'],
                    [PY, '-u', '-m', 'workflows.sbi.e2e_wholefield_matched_screen', 'run', '--workers', '8'],
                    [PY, '-u', '-m', 'workflows.sbi.e2e_wholefield_matched_screen_verify']):
        subprocess.run(command, cwd=REPO, check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['owner', 'compute'])
    args = parser.parse_args()
    owner() if args.action == 'owner' else compute()
