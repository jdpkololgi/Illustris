"""Independent GPU workers in one foreground multi-task Slurm step.

No distributed numerical rewrite, scheduler submissions, allocation chains or
retries. The unchanged frozen worker claims distinct tasks atomically.
"""
import argparse
import os
from pathlib import Path
import sys
import time

ROOT = Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/wholefield_comparison_20261009_v1/fullcap_v1')


def validate_topology(gpus, tasks, rank, visible):
    if gpus not in (2, 4) or tasks != gpus or not 0 <= rank < tasks or visible != 1:
        raise ValueError('One visible GPU per independent Slurm rank required')


def execute(gen, segment, rank, tasks, visible):
    gen.compute()
    root = ROOT/'allocations'/segment
    reserved = gen.read(root/'RESERVED.json')
    validate_topology(reserved['gpus'], tasks, rank, visible)
    source = gen.record(__file__)
    if rank == 0:
        gen.start(segment)
        # Release other ranks only after STARTED.json has been closed.
        gen.write(root/'PARALLEL_READY_WRITTEN.json', dict(job=os.environ['SLURM_JOB_ID'], source=source))
        os.link(root/'PARALLEL_READY_WRITTEN.json', root/'PARALLEL_READY.json')
    else:
        deadline = time.monotonic()+120
        while not (root/'PARALLEL_READY.json').exists():
            if time.monotonic() >= deadline:
                raise TimeoutError('Rank-zero startup did not finish')
            time.sleep(1)
    ready = gen.read(root/'PARALLEL_READY.json')
    if ready['job'] != os.environ['SLURM_JOB_ID'] or ready['source'] != source:
        raise ValueError('Parallel startup source/job mismatch')
    gen.write(root/f'RANK_{rank}_STARTED.json', dict(source=source, tasks=tasks, rank=rank,
              cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), visible_gpu_count=visible))
    try:
        gen.worker(segment, rank)
    except BaseException as error:
        gen.write(root/f'RANK_{rank}_FAILED.json', dict(exception=type(error).__name__, message=str(error)))
        raise
    gen.verify_record(source)
    gen.write(root/f'RANK_{rank}_COMPLETE.json', dict(passed=True, source=source))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--segment', required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT/'source'))
    from workflows.sbi import e2e_wholefield_fullcap_run as gen
    if gen.REPO != ROOT/'source':
        raise RuntimeError('Frozen numerical generation snapshot required')
    gen.compute()
    import torch
    execute(gen, args.segment, int(os.environ['SLURM_PROCID']),
            int(os.environ['SLURM_NTASKS']), torch.cuda.device_count())
