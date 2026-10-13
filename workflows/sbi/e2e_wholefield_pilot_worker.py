"""One bounded allocated GPU worker; never launches or retries scheduler jobs."""
import argparse
import json
import time

from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi.e2e_wholefield_pilot_data import ROOT
from workflows.sbi.e2e_wholefield_pilot_train import run, sources, STEPS
from workflows.sbi.e2e_wholefield_pilot_smoke import run as smoke
from workflows.sbi.e2e_wholefield_pilot_exterior import prepare


def worker(architecture, seconds, engineering=False):
    c.require_compute()
    started = time.monotonic()
    if engineering:
        receipt = ROOT/'smoke/COMPLETE.json'
        if not receipt.exists():
            smoke()
        if json.loads(receipt.read_text())['sources'] != sources():
            raise ValueError('Engineering check source changed')
    selected = json.loads((ROOT/'gaussian/COMPLETE.json').read_text())['selection']
    prediction = selected[architecture]['selected_prediction']
    # Productive, bounded reuse while the CPU prepares training data. This
    # common reference cache supports testing, never permission to fit.
    if not all((ROOT/'data'/p/'COMPLETE.json').exists() for p in c.TRAIN):
        for phase in ('ph012', 'ph013'):
            for cap in ('NGC', 'SGC'):
                for seed in (17, 29):
                    for first in range(0, 32, 4):
                        if all((ROOT/'data'/p/'COMPLETE.json').exists() for p in c.TRAIN):
                            break
                        if time.monotonic()-started > seconds-600:
                            print('BOUNDED_WORKER_SAVED_BEFORE_TRAINING', flush=True)
                            return
                        prepare(phase, cap, seed, first)
    if not all((ROOT/'data'/p/'COMPLETE.json').exists() for p in c.TRAIN):
        print('REFERENCE_CACHE_COMPLETE_TRAINING_DATA_NOT_READY', flush=True)
        return
    for seed in (17, 29):
        remaining = seconds-(time.monotonic()-started)-120
        if remaining < 120:
            break
        run(architecture, prediction, seed, remaining)
        status = json.loads((ROOT/'fits'/f'{architecture}_{prediction}_{seed}'/'STATUS.json').read_text())
        if status['step'] != STEPS:
            break
    print('BOUNDED_WORKER_COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--architecture', choices=('unet', 'transformer'), required=True)
    parser.add_argument('--seconds', type=int, default=10200)
    parser.add_argument('--engineering', action='store_true')
    worker(**vars(parser.parse_args()))
