"""Bounded foreground extraction of currently verified full-cap products."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/wholefield_comparison_20261009_v1/fullcap_v1')
sys.path.insert(0, str(ROOT/'source'))
from workflows.sbi import e2e_wholefield_fullcap_run as gen


def run(segment):
    gen.compute(); manifest = gen.read(ROOT/'MANIFEST.json'); gen.verify_sources(manifest)
    gen.start(segment)
    reservation = gen.read(ROOT/'allocations'/segment/'RESERVED.json')
    if reservation['gpus'] != 1:
        raise ValueError('One GPU only')
    start = gen.read(ROOT/'allocations'/segment/'STARTED.json')
    deadline = start['started_epoch']+reservation['wall_seconds']-180
    source = gen.record(__file__)
    reuse = gen.read(ROOT/'REUSE_VERIFIED.json'); gen.verify_record(reuse['manifest'])
    if not reuse['passed']:
        raise ValueError('Unverified reuse')
    extraction = ROOT/'extraction_v1'
    binding = gen.read(extraction/'MANIFEST.json'); gen.verify_record(binding['source'])
    gen.write(ROOT/'allocations'/segment/'EXTRACTION_STARTED.json', dict(source=source,
              extraction=gen.record(extraction/'MANIFEST.json'), job=os.environ['SLURM_JOB_ID']))
    completed = []; skipped = []
    for task in manifest['tasks']:
        key = task['key']; target = extraction/'cores'/key
        if (target/'COMPLETE.json').exists():
            rec = gen.read(target/'COMPLETE.json')
            gen.verify_record(rec['output']); gen.verify_record(rec['binding'])
            if not rec['passed'] or rec['key'] != key:
                raise ValueError('Invalid prior extraction')
            skipped.append(key); continue
        if key not in reuse['tasks'] and not (ROOT/'tasks'/key/'COMPLETE.json').exists():
            continue
        if time.time()+120 > deadline:
            break
        if target.exists():
            raise ValueError('Partial extraction preserved; explicit repair needed: '+key)
        gen.verify_record(source); gen.verify_record(binding['source'])
        log = ROOT/'logs'/segment/(key.replace('/', '_')+'.log')
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('x') as stream:
            subprocess.run([sys.executable, '-u', str(extraction/'extract.py'), 'run',
                            '--phase', task['phase'], '--cap', task['cap'], '--seed', str(task['seed']),
                            '--first', str(task['first'])], check=True, stdout=stream, stderr=subprocess.STDOUT)
        completed.append(key)
        print('EXTRACTED '+key, flush=True)
    gen.verify_record(source)
    gen.write(ROOT/'allocations'/segment/'EXTRACTION_COMPLETE.json', dict(passed=True,
              completed=completed, skipped_verified=skipped, source=source,
              job=os.environ['SLURM_JOB_ID'], stopped_epoch=time.time(), panel_complete=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--segment', required=True)
    run(parser.parse_args().segment)
