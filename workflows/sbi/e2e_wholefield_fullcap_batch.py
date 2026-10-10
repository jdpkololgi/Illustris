"""Source-bound batch owner for the unchanged frozen single-GPU worker.

No scheduler submissions, retries, claim changes or numerical modifications.
Explicit smoke modes exercise CUDA and failure propagation without task claims.
"""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/wholefield_comparison_20261009_v1/fullcap_v1')
MODES = ('production', 'smoke-pass', 'smoke-fail')


def prepare(gen, segment, mode):
    if mode not in MODES or not segment.isalnum():
        raise ValueError('Invalid launcher mode or segment')
    reservation = gen.read(ROOT/'allocations'/segment/'RESERVED.json')
    if reservation['gpus'] != 1:
        raise ValueError('Single-GPU validation only')
    gen.verify_record(reservation['manifest'])
    gen.write(ROOT/'allocations'/segment/'LAUNCHER.json', dict(
        mode=mode, sources=[gen.record(__file__), gen.record(Path(__file__).with_suffix('.sh'))],
        reservation=gen.record(ROOT/'allocations'/segment/'RESERVED.json')))


def execute(gen, segment, cuda_check):
    gen.compute()
    root = ROOT/'allocations'/segment
    launch = gen.read(root/'LAUNCHER.json')
    mode = launch['mode']
    if mode not in MODES:
        raise ValueError('Invalid mode')
    for source in launch['sources']:
        gen.verify_record(source)
    gen.verify_record(launch['reservation'])
    if gen.read(root/'RESERVED.json')['gpus'] != 1:
        raise ValueError('Single-GPU validation only')
    if mode == 'production' and not os.environ.get('SLURM_BATCH_HOST'):
        raise RuntimeError('Production requires a scheduler-owned batch job')
    if int(os.environ.get('SLURM_NTASKS', '0')) != 1:
        raise RuntimeError('Exactly one Slurm task required')
    gen.start(segment)
    gen.write(root/'BATCH_STARTED.json', dict(mode=mode, job=os.environ['SLURM_JOB_ID'],
              batch_host=os.environ.get('SLURM_BATCH_HOST'), launcher=gen.record(root/'LAUNCHER.json')))
    try:
        cuda_check()
        if mode == 'smoke-fail':
            raise RuntimeError('Deliberate lifecycle smoke failure; no generation attempted')
        if mode == 'production':
            gen.worker(segment, 0)
            worker = gen.read(root/'WORKER_0.json')
            if not worker['passed'] or not worker['completed']:
                raise RuntimeError('No completed generation tasks; not production success')
        for source in launch['sources']:
            gen.verify_record(source)
    except BaseException as error:
        gen.write(root/'BATCH_FAILED.json', dict(mode=mode, exception=type(error).__name__, message=str(error)))
        raise
    gen.write(root/'BATCH_COMPLETE.json', dict(passed=True, mode=mode,
              scientific_completion=mode == 'production'))


def cuda_check():
    import torch
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one visible GPU required')
    value = torch.eye(4, device='cuda')
    if not torch.equal(value @ value, value):
        raise RuntimeError('CUDA smoke mismatch')
    torch.cuda.synchronize()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run'))
    parser.add_argument('--segment', required=True)
    parser.add_argument('--mode', choices=MODES, default='production')
    args = parser.parse_args()
    if not args.segment.isalnum():
        parser.error('Alphanumeric segment required')
    sys.path.insert(0, str(ROOT/'source'))
    from workflows.sbi import e2e_wholefield_fullcap_run as gen
    if gen.REPO != ROOT/'source':
        raise RuntimeError('Frozen numerical snapshot required')
    if args.action == 'prepare':
        prepare(gen, args.segment, args.mode)
    else:
        execute(gen, args.segment, cuda_check)
