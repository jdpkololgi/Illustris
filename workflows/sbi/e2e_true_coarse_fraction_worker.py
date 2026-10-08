"""Bounded foreground interactive worker for explicitly listed draw batches.

No scheduler submission, background process, retries, or automatic task selection.
Every batch is generated and extracted before the next listed batch starts.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

import e2e_true_coarse_fraction_saved as io


def run(addresses):
    io.compute_guard()
    if not 1 <= len(addresses) <= 2 or len(set(addresses)) != len(addresses):
        raise ValueError('One or two unique explicitly assigned batches required')
    batches = []
    for address in addresses:
        phase, cap, first = address.split(':'); first = int(first)
        if phase not in io.PHASES or cap not in ('NGC', 'SGC') or first not in range(0, 32, 4):
            raise ValueError('Unregistered draw address')
        for kind in ('fields', 'cores'):
            if (io.OUT/kind/phase/cap/f'd{first:03d}').exists():
                raise FileExistsError('Existing output: '+address+' '+kind)
        batches.append((phase, cap, first))
    here = Path(__file__).resolve().parent
    sources = [io.record(here/name) for name in
        ('e2e_true_coarse_fraction_worker.py', 'e2e_true_coarse_fraction_sample.py',
         'e2e_true_coarse_fraction_extract.py', 'e2e_true_coarse_fraction_saved.py')]
    started = time.monotonic(); completed = []
    for phase, cap, first in batches:
        for stage in ('sample', 'extract'):
            for rec in sources:
                if io.sha(rec['path']) != rec['sha256']:
                    raise ValueError('Worker source changed')
            command = [sys.executable, str(here/f'e2e_true_coarse_fraction_{stage}.py'),
                '--phase', phase, '--cap', cap, '--first', str(first)]
            print(io.json.dumps(dict(stage='worker_launch', command=command)), flush=True)
            subprocess.run(command, check=True, cwd=io.REPO)
        path = io.OUT/'cores'/phase/cap/f'd{first:03d}/COMPLETE.json'
        rec = io.read(path)
        if not rec['passed']:
            raise ValueError('Extraction did not complete')
        io.verify(rec['output']); completed.append(io.record(path))
    out = io.OUT/'workers'; out.mkdir(exist_ok=True)
    tag = '__'.join(addresses).replace(':', '_')
    io.write(out/(tag+'.json'), dict(passed=True, addresses=addresses, completed=completed,
        sources=sources, seconds=time.monotonic()-started, job=os.environ['SLURM_JOB_ID'],
        scope='one parent seed, two reused phases, diagnostic only'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('addresses', nargs='+')
    run(p.parse_args().addresses)
