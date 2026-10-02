"""Persist focused compute-node verification before freezing production."""
import argparse
import json
import os
import sys
import time
from pathlib import Path
import unittest
from workflows.sbi import e2e_coupled_contract as c


def run(out):
    c.require_compute()
    if Path(sys.prefix).resolve() != Path('/pscratch/sd/d/dkololgi/conda/envs/cosmic_env').resolve():
        raise RuntimeError('Expected cosmic_env Python environment')
    patterns = ['test_e2e_exterior*.py', 'test_e2e_spatial_exterior.py',
                'test_e2e_global_overlap.py', 'test_e2e_boundary_mock.py']
    suite = unittest.TestSuite()
    for pattern in patterns:
        suite.addTests(unittest.defaultTestLoader.discover(str(c.REPO/'tests/phase4'), pattern=pattern))
    source_paths = list((c.REPO/'workflows/sbi').glob('e2e_exterior*.py'))
    source_paths += [c.REPO/'workflows/sbi'/name for name in
        ('e2e_spatial_exterior.py', 'e2e_footprint_tidal_operator.py', 'e2e_global_canvas.py', 'e2e_global_sampler.py')]
    for pattern in patterns:
        source_paths.extend((c.REPO/'tests/phase4').glob(pattern))
    hashes = {str(p.relative_to(c.REPO)): c.sha256(p) for p in source_paths}
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if any(c.sha256(c.REPO/name) != sha for name, sha in hashes.items()):
        raise ValueError('Code changed while tests ran')
    record = dict(passed=result.wasSuccessful() and not result.skipped, tests=result.testsRun,
                  skipped=len(result.skipped), failures=len(result.failures), errors=len(result.errors),
                  sources=hashes, job=os.environ['SLURM_JOB_ID'],
                  python_executable=sys.executable, python_prefix=sys.prefix)
    if not record['passed']:
        c.atomic_json(out.with_stem(out.stem+'_FAILED_'+str(time.time_ns())), record)
        raise RuntimeError('Focused pre-freeze suite did not pass completely')
    c.atomic_json(out, record)
    print(json.dumps({k: v for k, v in record.items() if k != 'sources'}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); run(a.out)
