"""Bounded, development-only epoch-24/32 posterior comparison; never fit models.

The subset is chosen from geometry, not scores. At each development phase/cap/
shell retain one boundary/interior pair, alternating that choice across phases
and caps. Both seeds and coarse modes are retained. Epoch 32 is reused unchanged.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/wholefield_pilot_20261013_v1')
MODELS = {'unet': 'x', 'transformer': 'v'}
EARLY, FINAL = 7488, 9984


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_pairs(phase, rows):
    if phase not in ('ph012', 'ph013'):
        raise PermissionError('Stability uses development phases only')
    selected = []
    for cap_index, cap in enumerate(('NGC', 'SGC')):
        cap_rows = [r for r in rows if r['cap'] == cap]
        shells = sorted({r['shell'] for r in cap_rows})
        if shells != [0, 1, 2, 3]:
            raise ValueError('Expected four registered shells')
        for shell in shells:
            boundary = (int(phase[-1])+cap_index+shell) % 2 == 0
            stratum = 'boundary' if boundary else 'interior'
            choices = [r for r in cap_rows if r['shell'] == shell and r['support_stratum'] == stratum]
            if len(choices) != 1:
                raise ValueError('Expected one registered pair per geometry stratum')
            selected.append(choices[0])
    return selected


def tasks(root=ROOT):
    output = []
    for phase in ('ph012', 'ph013'):
        rows = read(root/'data'/phase/'COMPLETE.json')['pairs']
        for row in select_pairs(phase, rows):
            for seed in (17, 29):
                for coarse in ('sampled', 'oracle'):
                    output.append(dict(phase=phase, pair=row['pair_id'], cap=row['cap'],
                                       shell=row['shell'], stratum=row['support_stratum'],
                                       seed=seed, coarse=coarse, nfe=128))
    if len(output) != 64:
        raise ValueError('Changed stability panel')
    return output


def result_path(root, architecture, task, step):
    return (root/'evaluation'/f'{architecture}_{MODELS[architecture]}_{task["seed"]}'/
            f'step{step}'/task['phase']/task['pair']/task['coarse']/'fullcap'/'nfe128'/'COMPLETE.json')


def verify_result(path, task, evaluation_sha, step):
    result = read(path)
    bound_path = path.parent/'BINDING.json'
    bound = read(bound_path)
    if (not result['passed'] or result['binding'] != digest(bound_path)
            or bound['source'] != evaluation_sha or bound['draws'] != 32
            or bound['operator'] != 'fullcap' or bound['truth_is_input'] != (task['coarse'] == 'oracle')
            or any(bound[k] != task[k] for k in ('phase', 'pair', 'coarse', 'nfe'))
            or Path(bound['checkpoint']['path']).name != f'CHECKPOINT_{step:06d}.pt'):
        raise ValueError('Stability result does not match fixed task: '+str(path))


def run(architecture, seconds):
    from workflows.sbi import e2e_coupled_contract as c
    from workflows.sbi.e2e_wholefield_pilot_evaluate import run as evaluate
    c.require_compute()
    if architecture not in MODELS or not 1800 <= seconds <= 10200:
        raise ValueError('Invalid bounded stability request')
    start = time.monotonic()
    panel = tasks()
    out = ROOT/'stability_v1'/architecture
    out.mkdir(parents=True, exist_ok=True)
    evaluation_sha = digest(Path(__file__).with_name('e2e_wholefield_pilot_evaluate.py'))
    bound = dict(schema='wholefield-checkpoint-stability-v1', architecture=architecture,
                 early=EARLY, final=FINAL, tasks=panel, draws=32, source=digest(__file__),
                 evaluation_source=evaluation_sha, selection='geometry-only alternating strata',
                 no_training=True, no_checkpoint_selection=True, sealed_access=False)
    if (out/'BINDING.json').exists():
        if read(out/'BINDING.json') != bound:
            raise ValueError('Changed stability binding')
    else:
        c.atomic_json(out/'BINDING.json', bound)
    # Require complete matching final-checkpoint results and pre-existing caches.
    for task in panel:
        verify_result(result_path(ROOT, architecture, task, FINAL), task, evaluation_sha, FINAL)
        for first in range(0, 32, 4):
            cache = ROOT/'exterior'/task['phase']/task['cap']/f'seed{task["seed"]}'/f'd{first:03d}'/'COMPLETE.json'
            if not cache.exists() or not read(cache)['passed']:
                raise ValueError('Required frozen exterior cache absent')
    done = 0
    for task in panel:
        path = result_path(ROOT, architecture, task, EARLY)
        if not path.exists():
            if time.monotonic()-start > seconds-900:
                break
            evaluate(architecture, MODELS[architecture], task['seed'], EARLY,
                     task['phase'], task['pair'], task['coarse'], 32, 128, 'fullcap')
        verify_result(path, task, evaluation_sha, EARLY)
        done += 1
        c.atomic_json(out/'STATUS.json', dict(completed=done, expected=len(panel), last=task,
                      job=os.environ['SLURM_JOB_ID'], seconds=time.monotonic()-start), replace=True)
    receipt = dict(completed=done, expected=len(panel), complete=done == len(panel),
                   job=os.environ['SLURM_JOB_ID'], seconds=time.monotonic()-start,
                   binding=digest(out/'BINDING.json'), posterior_qualified=False)
    c.atomic_json(out/f'SEGMENT_{os.environ["SLURM_JOB_ID"]}.json', receipt)
    if done == len(panel):
        c.atomic_json(out/'COMPLETE.json', receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--architecture', required=True, choices=MODELS)
    parser.add_argument('--seconds', type=int, default=10200)
    args = parser.parse_args()
    run(args.architecture, args.seconds)
