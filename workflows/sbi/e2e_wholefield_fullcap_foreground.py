"""Single-GPU foreground coordinator, run inside one salloc/srun step.

No scheduler calls, background workers, automatic retries or allocation chains.
Numerical generation remains in the unchanged frozen source snapshot.
"""
import argparse
from pathlib import Path
import sys

ROOT = Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/wholefield_comparison_20261009_v1/fullcap_v1')


def execute(gen, segment, source):
    gen.compute()
    reservation = gen.read(ROOT/'allocations'/segment/'RESERVED.json')
    if reservation['gpus'] != 1:
        raise ValueError('Foreground validation requires one GPU')
    gen.start(segment)
    gen.write(ROOT/'allocations'/segment/'FOREGROUND_STARTED.json', dict(source=source,
              topology='single coordinator and worker in the same foreground compute step'))
    try:
        gen.worker(segment, 0)
    except BaseException as error:
        gen.write(ROOT/'allocations'/segment/'FOREGROUND_FAILED.json',
                  dict(source=source, exception=type(error).__name__, message=str(error)))
        raise
    gen.verify_record(source)
    gen.write(ROOT/'allocations'/segment/'FOREGROUND_COMPLETE.json', dict(passed=True, source=source))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--segment', required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT/'source'))
    from workflows.sbi import e2e_wholefield_fullcap_run as gen
    if gen.REPO != ROOT/'source':
        raise RuntimeError('Unchanged frozen generation snapshot required')
    execute(gen, args.segment, gen.record(__file__))
