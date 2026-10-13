"""Bounded fit-and-evaluate continuation; no scheduler submissions or decisions.

This worker runs the already selected configuration, both original seeds,
fixed final checkpoint, and the exposed six-phase panel. Existing complete
artifacts are reused. A time-limited segment leaves remaining work explicit.
"""
import argparse
import inspect
import json
import os
import time

from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi.e2e_wholefield_pilot_data import ROOT, EVAL
from workflows.sbi.e2e_wholefield_pilot_train import STEPS
from workflows.sbi.e2e_wholefield_pilot_worker import worker
from workflows.sbi.e2e_wholefield_pilot_evaluate import run as evaluate
from workflows.sbi.e2e_wholefield_pilot_exterior import prepare


def tasks():
    result = []
    for phase in EVAL:
        rows = json.loads((ROOT/'data'/phase/'COMPLETE.json').read_text())['pairs']
        caps = [[r for r in rows if r['cap'] == cap] for cap in ('NGC', 'SGC')]
        if [len(v) for v in caps] != [8, 8]:
            raise ValueError('Incomplete sixteen-pair evaluation panel')
        # Interleave caps, then model seeds, to expose both early in the readout.
        ordered = [row for pair in zip(*caps) for row in pair]
        for ordinal, row in enumerate(ordered):
            for seed in (17, 29):
                for coarse in ('sampled', 'oracle'):
                    nfes = (128, 256) if phase == 'ph012' and ordinal == 0 and seed == 17 else (128,)
                    for nfe in nfes:
                        result.append(dict(phase=phase, cap=row['cap'], pair=row['pair_id'],
                                           seed=seed, coarse=coarse, nfe=nfe))
    return result


def run(architecture, seconds, train=False):
    c.require_compute()
    start = time.monotonic(); deadline = start+seconds
    if train:
        worker(architecture, max(1, seconds-120))
    selection = json.loads((ROOT/'gaussian/COMPLETE.json').read_text())['selection']
    prediction = selection[architecture]['selected_prediction']
    for seed in (17, 29):
        checkpoint = ROOT/'fits'/f'{architecture}_{prediction}_{seed}'/f'CHECKPOINT_{STEPS:06d}.pt'
        if not checkpoint.exists():
            print('TRAINING_SEGMENT_SAVED_EVALUATION_NOT_STARTED', flush=True)
            return
    expected = tasks(); completed = 0
    job = os.environ['SLURM_JOB_ID']
    root = ROOT/'campaign'/architecture; root.mkdir(parents=True, exist_ok=True)
    binding = dict(architecture=architecture, prediction=prediction, checkpoint=STEPS,
                   draws=32, tasks=expected, source=c.sha256(__file__), job=job,
                   scope='fixed final checkpoint exploratory readout; no statistical stop rule')
    c.atomic_json(root/f'BINDING_{job}.json', binding)
    for task in expected:
        output = (ROOT/'evaluation'/f'{architecture}_{prediction}_{task["seed"]}'/f'step{STEPS}'/
                  task['phase']/task['pair']/task['coarse']/'fullcap'/f'nfe{task["nfe"]}'/'COMPLETE.json')
        if output.exists():
            finished = json.loads(output.read_text())
            bound_path = output.parent/'BINDING.json'
            bound = json.loads(bound_path.read_text())
            if (not finished['passed'] or finished['binding'] != c.sha256(bound_path)
                    or bound['source'] != c.sha256(inspect.getfile(evaluate))):
                raise ValueError('Completed evaluation binding changed')
            completed += 1
            continue
        # Cache work is bounded chunk by chunk and reused across all models.
        for first in range(0, 32, 4):
            if time.monotonic() > deadline-600:
                break
            prepare(task['phase'], task['cap'], task['seed'], first)
        else:
            # Reserve at least fifteen minutes for an indivisible 32-draw case.
            if time.monotonic() < deadline-900:
                evaluate(architecture, prediction, task['seed'], STEPS, task['phase'], task['pair'],
                         task['coarse'], 32, task['nfe'], 'fullcap')
                completed += 1
                c.atomic_json(root/f'STATUS_{job}.json', dict(completed=completed, expected=len(expected),
                    last=task, seconds=time.monotonic()-start, full_panel_complete=completed == len(expected)), replace=True)
                continue
        break
    c.atomic_json(root/f'SEGMENT_{job}.json', dict(completed=completed, expected=len(expected),
                  seconds=time.monotonic()-start, full_panel_complete=completed == len(expected),
                  posterior_qualified=False, job=job))
    print(json.dumps(dict(stage='evaluation_segment_saved', completed=completed, expected=len(expected))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--architecture', choices=('unet', 'transformer'), required=True)
    parser.add_argument('--seconds', type=int, default=10200)
    parser.add_argument('--train', action='store_true')
    run(**vars(parser.parse_args()))
