"""Frozen, bounded interactive segments for the registered two-model comparison.

Generation only: no targets, model selection, training, or calibration claim.
Each cap must pass full-length seed17 replay before seed29 is generated there.
Incomplete/failed tasks are preserved and never automatically retried.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from workflows.sbi.e2e_cfm_mixture_contract import PHASES, component_seed

BASE = Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/coherent_exterior_20261002_v1')
OUT = BASE.parent/'cfm_model_mixture_20261006_v2'
FROZEN = BASE/'source_78156ac3417555ef269f'
CAPS = [(p, c) for p in PHASES for c in ('NGC', 'SGC')]
CHANGED = ['workflows/sbi/'+n+'.py' for n in
           ('e2e_exterior_fullcap', 'e2e_cfm_mixture_contract', 'e2e_cfm_mixture_generate')]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(16 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    # Exclusive creation is intentional: preserve failed and prior evidence.
    with Path(path).open('x') as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')


def freeze():
    OUT.mkdir(parents=True, exist_ok=False)
    snapshot = OUT/'source'
    shutil.copytree(FROZEN, snapshot, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git'))
    for name in CHANGED:
        if (snapshot/name).exists():
            (snapshot/name).chmod(0o600)
        shutil.copy2(REPO/name, snapshot/name)
    sources = {str(p.relative_to(snapshot)): sha(p) for p in snapshot.rglob('*') if p.is_file()}
    pilot = BASE.parent/'cfm_mixture_pilot_20261006_v1'
    b17 = read(BASE/'fullcap_density/ph012/SGC/d000/BINDING.json')
    b29 = read(pilot/'ph012/seed29/BINDING.json')
    if b17['normalizer'] != b29['normalizer']:
        raise ValueError('Component normalizers differ')
    prepared = {}; baselines = {}; tides = {}; timings = {}
    for phase, cap in CAPS:
        key = phase+'/'+cap
        prepared[key] = sha(BASE/'fullcap_inputs'/key/'PREPARED.json')
        for first in range(0, 32, 4):
            address = key+f'/d{first:03d}'
            for group, dest in [('fullcap_density', baselines), ('fullcap_tides', tides)]:
                path = BASE/group/address/'COMPLETE.json'
                dest[address] = dict(path=str(path), sha256=sha(path))
        old = read(BASE/'fullcap_density'/key/'d000/COMPLETE.json')
        timings[key] = old['sampling_seconds']+old['loading_seconds']
    prior = BASE/'prior/PRIOR.json'
    write(OUT/'MANIFEST.json', dict(schema='cfm-model-mixture-generation-v1',
        phases=list(PHASES), seeds=[17, 29], nfe=128, draws=32,
        checkpoints={'17': b17['checkpoints'], '29': b29['checkpoints']},
        normalizer=b17['normalizer'], sources=sources, prepared=prepared,
        baselines=baselines, baseline_tides=tides, timings=timings,
        prior=dict(path=str(prior), sha256=sha(prior)), method='learned', padding=32,
        assignments={p: [component_seed(p, d) for d in range(32)] for p in PHASES},
        protocol=dict(path=str(REPO/'docs/e2e_cfm_model_mixture_20261006.md'),
                      sha256=sha(REPO/'docs/e2e_cfm_model_mixture_20261006.md')),
        authorization='User 2026-10-06: interactive jobs to complete next comparison',
        scientific_scoring_complete=False, posterior_qualified=False))
    (OUT/'logs').mkdir()
    print(json.dumps(dict(manifest=str(OUT/'MANIFEST.json'), sha256=sha(OUT/'MANIFEST.json'))))


def verify_sources(m):
    if REPO != OUT/'source':
        raise ValueError('Run only the immutable comparison snapshot')
    for name, digest in m['sources'].items():
        if sha(REPO/name) != digest:
            raise ValueError('Source changed: '+name)


def parity(m, phase, cap, destination):
    import h5py
    import numpy as np
    key = phase+'/'+cap+'/d000'
    original = m['baselines'][key]
    if sha(original['path']) != original['sha256']:
        raise ValueError('Baseline receipt changed')
    a, b = read(original['path']), read(destination/'density/COMPLETE.json')
    for r in (a, b):
        if not r['passed'] or sha(r['field']['path']) != r['field']['sha256']:
            raise ValueError('Invalid baseline field receipt')
    checks = 0
    with h5py.File(a['field']['path']) as f, h5py.File(b['field']['path']) as g:
        for name in ('rho', 'coarse_logrho_standardized', 'support', 'coverage'):
            if f[name].shape != g[name].shape:
                raise ValueError('Baseline shape changed: '+name)
            for index in range(f[name].shape[0]):
                if not np.array_equal(f[name][index], g[name][index], equal_nan=True):
                    raise ValueError('Full-NFE baseline replay mismatch: '+name)
                checks += 1
        for name in ('origin_mpc_h', 'cell_mpc_h', 'phase', 'cap', 'first_draw',
                     'shared_state', 'coarse_origin_raw_cells'):
            if not np.array_equal(f.attrs[name], g.attrs[name]):
                raise ValueError('Baseline metadata changed: '+name)
    write(destination/'PARITY.json', dict(passed=True, exact=True, checks=checks,
          original=original, replay_sha256=sha(destination/'density/COMPLETE.json'),
          meaning='Matched full-NFE implementation, not posterior calibration'))


def task(phase, cap, seed, first):
    from workflows.sbi import e2e_coupled_contract as c
    from workflows.sbi import e2e_exterior_fullcap as sampler
    c.require_compute()
    m = read(OUT/'MANIFEST.json'); verify_sources(m)
    if (phase, cap) not in CAPS or seed not in (17, 29) or first not in range(0, 32, 4):
        raise ValueError('Unregistered task')
    if seed == 17 and first != 0:
        raise ValueError('Only d000 seed17 replay is authorized')
    parent = OUT/phase/cap/f'seed{seed}'/f'd{first:03d}'
    if seed == 29:
        p = read(OUT/phase/cap/'seed17/d000/PARITY.json')
        if not p['passed'] or not p['exact']:
            raise ValueError('Baseline parity prerequisite failed')
    parent.mkdir(parents=True, exist_ok=False)
    write(parent/'START.json', dict(job=os.environ['SLURM_JOB_ID'],
          manifest_sha256=sha(OUT/'MANIFEST.json'), phase=phase, cap=cap, seed=seed, first=first))
    sampler.run(BASE/'fullcap_inputs'/phase/cap, parent/'density', first=first,
                model_seed=seed, comparison=OUT/'MANIFEST.json')
    verify_sources(m)
    if seed == 17:
        parity(m, phase, cap, parent)
    else:
        # Release all sampler models/cached conditions before the FFT stage.
        import gc
        import torch
        gc.collect(); torch.cuda.empty_cache()
        if sha(m['prior']['path']) != m['prior']['sha256']:
            raise ValueError('Completion prior changed')
        subprocess.run([sys.executable, '-m', 'workflows.sbi.e2e_exterior_cap_tides',
            '--field-receipt', str(parent/'density/COMPLETE.json'),
            '--prior', m['prior']['path'], '--out', str(parent/'tides'),
            '--method', 'learned', '--padding', '32'], check=True, cwd=REPO)
    verify_sources(m)
    write(parent/'COMPLETE.json', dict(passed=True, phase=phase, cap=cap, seed=seed,
          first=first, manifest_sha256=sha(OUT/'MANIFEST.json'),
          density_sha256=sha(parent/'density/COMPLETE.json'),
          auxiliary_sha256=sha(parent/('PARITY.json' if seed == 17 else 'tides/COMPLETE.json')),
          posterior_qualified=False))


def segment(worker, seconds):
    """Shared queue of independent GPU tasks; no waiting on failed prerequisites."""
    from workflows.sbi import e2e_coupled_contract as c
    c.require_compute()
    m = read(OUT/'MANIFEST.json'); verify_sources(m)
    started = time.monotonic(); done = []
    tasks = [(p, cap, 17, 0) for p, cap in CAPS]
    tasks += [(p, cap, 29, d) for d in range(0, 32, 4) for p, cap in CAPS]
    claims = OUT/'claims'; claims.mkdir(exist_ok=True)
    while True:
        selected = None
        for phase, cap, seed, first in tasks:
            parent = OUT/phase/cap/f'seed{seed}'/f'd{first:03d}'
            if (parent/'COMPLETE.json').exists():
                old = read(parent/'COMPLETE.json')
                if not old['passed'] or old['manifest_sha256'] != sha(OUT/'MANIFEST.json'):
                    raise ValueError('Existing completion changed')
                for name, key in [('density/COMPLETE.json', 'density_sha256'),
                                  ('PARITY.json' if seed == 17 else 'tides/COMPLETE.json', 'auxiliary_sha256')]:
                    if sha(parent/name) != old[key]:
                        raise ValueError('Completed task receipt changed')
                continue
            if seed == 29 and not (OUT/phase/cap/'seed17/d000/COMPLETE.json').exists():
                continue
            estimate = m['timings'][phase+'/'+cap]
            if time.monotonic()-started+estimate*1.12+(420 if seed == 29 else 180) > seconds:
                continue
            claim = claims/f'{phase}_{cap}_s{seed}_d{first:03d}'
            try:
                claim.mkdir()
            except FileExistsError:
                continue  # A claimed task is never retried, including after failure.
            if parent.exists():
                raise RuntimeError('Partial task exists without claim; stop')
            write(claim/'OWNER.json', dict(job=os.environ['SLURM_JOB_ID'], worker=worker))
            selected = (phase, cap, seed, first)
            break
        if selected is None:
            break
        phase, cap, seed, first = selected
        log = OUT/'logs'/f'{phase}_{cap}_s{seed}_d{first:03d}.log'
        with log.open('x') as stream:
            subprocess.run([sys.executable, str(Path(__file__)), 'task', '--phase', phase,
                '--cap', cap, '--seed', str(seed), '--first', str(first)],
                stdout=stream, stderr=subprocess.STDOUT, check=True, cwd=REPO)
        done.append(list(selected))
    write(OUT/'logs'/f'segment_{os.environ["SLURM_JOB_ID"]}_{worker}.json',
          dict(completed=done, seconds=time.monotonic()-started,
               all_generation_complete=all((OUT/p/cap/'seed29'/f'd{d:03d}/COMPLETE.json').exists()
                                           for p, cap in CAPS for d in range(0, 32, 4)),
               scoring_complete=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['freeze', 'task', 'segment'])
    p.add_argument('--phase'); p.add_argument('--cap'); p.add_argument('--seed', type=int)
    p.add_argument('--first', type=int); p.add_argument('--worker', type=int, choices=range(4))
    p.add_argument('--seconds', type=int, default=13800)
    args = p.parse_args()
    if args.action == 'freeze': freeze()
    elif args.action == 'task': task(args.phase, args.cap, args.seed, args.first)
    else: segment(args.worker, args.seconds)
