"""Frozen full-cap regeneration, verified reuse, and bounded GPU segments.

No scheduler submissions. A separate explicitly approved salloc invocation owns
each segment; foreground workers exit when no complete task fits the time left.
All old products and incomplete attempts are read-only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

from workflows.sbi import e2e_wholefield_fullcap_contract as contract

REPO = Path(__file__).resolve().parents[2]
BASE = Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/coherent_exterior_20261002_v1')
MIX = BASE.parent/'cfm_model_mixture_20261006_v2'
ROOT = BASE.parent/'wholefield_comparison_20261009_v1/fullcap_v1'
CONFIG = REPO/'configs/e2e_wholefield_fullcap_v1.json'
OVERLAYS = ['configs/e2e_wholefield_fullcap_v1.json',
            'workflows/sbi/e2e_cfm_mixture_contract.py',
            'workflows/sbi/e2e_wholefield_fullcap_contract.py',
            'workflows/sbi/e2e_wholefield_fullcap_run.py',
            'workflows/sbi/e2e_wholefield_fullcap_segment.sh',
            'tests/test_e2e_wholefield_fullcap_contract.py',
            'tests/phase4/test_e2e_cfm_mixture_contract.py']


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(16 << 20), b''):
            result.update(block)
    return result.hexdigest()


def record(path):
    return dict(path=str(path), bytes=Path(path).stat().st_size, sha256=sha(path))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def compute():
    if not os.environ.get('SLURM_JOB_ID') or not socket.gethostname().startswith('nid'):
        raise RuntimeError('Interactive compute allocation required')
    if any(os.environ.get(k) for k in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONUSERBASE', 'LD_PRELOAD')):
        raise RuntimeError('Isolated environment required')
    if os.environ.get('PYTHONNOUSERSITE') != '1':
        raise RuntimeError('PYTHONNOUSERSITE=1 required')


def verify_record(item):
    if sha(item['path']) != item['sha256']:
        raise ValueError('Changed payload: '+item['path'])


def verify_sources(m):
    if REPO != ROOT/'source':
        raise ValueError('Run generation only from the frozen source snapshot')
    contract.validate(m)
    for name, digest in m['sources'].items():
        if sha(REPO/name) != digest:
            raise ValueError('Frozen source changed: '+name)


def stage():
    compute()
    cfg = read(CONFIG); contract.validate(cfg)
    preflight_path = ROOT.parent/'fullcap_resource_preflight_v2.json'
    if sha(preflight_path) != cfg['preflight_sha256']:
        raise ValueError('Resource preflight changed')
    preflight = read(preflight_path)
    old = read(MIX/'MANIFEST.json')
    for name, digest in old['sources'].items():
        if sha(MIX/'source'/name) != digest:
            raise ValueError('Previous exact-replay source changed: '+name)
    if sha(REPO/'workflows/sbi/e2e_exterior_fullcap.py') != old['sources']['workflows/sbi/e2e_exterior_fullcap.py']:
        raise ValueError('Numerical sampler differs from the exact-replay source')
    ROOT.mkdir(parents=True, exist_ok=False)
    snapshot = ROOT/'source'
    shutil.copytree(MIX/'source', snapshot, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git'))
    for name in OVERLAYS:
        dst = snapshot/name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            dst.chmod(0o600)  # Only the new private copy, never the source.
        shutil.copy2(REPO/name, dst)
    sources = {str(p.relative_to(snapshot)): sha(p) for p in snapshot.rglob('*') if p.is_file()}
    for name, digest in old['sources'].items():
        if name not in OVERLAYS and sources[name] != digest:
            raise ValueError('Unintended numerical snapshot change')
    prepared = {}; prepared_receipts = {}; replay = []
    for row in preflight['inputs']:
        key = row['phase']+'/'+row['cap']
        p = Path(row['prepared']); r = read(p)
        prepared[key] = sha(p); prepared_receipts[key] = record(p)
        if prepared[key] != preflight['source_hashes'][str(p)]:
            raise ValueError('Prepared input changed after preflight')
        verify_record(r['outputs'][0])
        print(json.dumps(dict(stage='input_verified', cap=key)), flush=True)
    for phase in ('ph012', 'ph013'):
        for cap in ('NGC', 'SGC'):
            path = MIX/phase/cap/'seed17/d000/PARITY.json'
            r = read(path)
            if not r['passed'] or not r['exact']:
                raise ValueError('Historical full-NFE replay not qualified')
            verify_record(r['original']); replay.append(record(path))
    checkpoints = old['checkpoints']
    for seed in (17, 29):
        for kind, step in (('coarse', 13312), ('fine', 26624)):
            path = BASE.parent/'cfm_pilot_20260924_v1'/f'{kind}_{seed}'/f'CHECKPOINT_{step:06d}.pt'
            if sha(path) != checkpoints[str(seed)][kind]:
                raise ValueError('Frozen checkpoint changed')
    prior = old['prior']; verify_record(prior)
    if read(prior['path'])['training_phases'] != cfg['training_phases']:
        raise ValueError('Exterior prior training phases changed')
    tasks = []
    for row in preflight['tasks']:
        key = contract.address(row['phase'], row['cap'], row['seed'], row['first'])
        task = dict(row, key=key)
        for product in ('density', 'tides'):
            path = Path(row[product+'_receipt'])
            if row[product+'_complete_candidate']:
                if sha(path) != preflight['source_hashes'][str(path)]:
                    raise ValueError('Reuse receipt changed')
                task[product+'_bound_receipt'] = record(path)
        # Historical timings exclude verification and output overhead. Reserve
        # 20% plus ten minutes; refreshed measured costs may only raise estimates.
        task['admission_seconds'] = 600+1.2*sum(
            row['windows']*preflight['rates'][row['cap']+'/'+product]['max_seconds_per_window']
            for product in ('density', 'tides') if not row[product+'_complete_candidate'])
        tasks.append(task)
    manifest = dict(cfg, checkpoints=checkpoints, normalizer=old['normalizer'],
                    sources=sources, prepared=prepared, prepared_receipts=prepared_receipts,
                    tasks=tasks, prior=prior, replay=replay, numerical_parent=record(MIX/'MANIFEST.json'),
                    preflight=record(preflight_path), authorization=record(CONFIG),
                    source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
                    job=os.environ['SLURM_JOB_ID'], created_utc=datetime.now(timezone.utc).isoformat())
    write(ROOT/'MANIFEST.json', manifest)
    print(json.dumps(dict(stage='staged', manifest=record(ROOT/'MANIFEST.json'))), flush=True)


def reuse_verify():
    compute(); m = read(ROOT/'MANIFEST.json'); verify_sources(m)
    verified = []
    for task in m['tasks']:
        if not task['density_complete_candidate'] or not task['tides_complete_candidate']:
            continue
        for product, payload in (('density', 'field'), ('tides', 'output')):
            item = task[product+'_bound_receipt']; verify_record(item)
            r = read(item['path']); verify_record(r[payload])
            binding_path = Path(item['path']).parent/'BINDING.json'
            if sha(binding_path) != r['binding']:
                raise ValueError('Reuse binding changed')
            b = read(binding_path)
            if product == 'density':
                if b['checkpoints'] != m['checkpoints'][str(task['seed'])] or b['normalizer'] != m['normalizer']:
                    raise ValueError('Reuse model identity changed')
                if b['prepared_sha256'] != m['prepared'][task['phase']+'/'+task['cap']]:
                    raise ValueError('Reuse observation cache changed')
                if b['nfe'] != 128:
                    raise ValueError('Reuse integration setting changed')
            else:
                if b['method'] != 'learned' or b['padding'] != 32:
                    raise ValueError('Reuse exterior operator changed')
                verify_record(b['input_receipt']); verify_record(b['prior'])
                if b['input_receipt']['sha256'] != task['density_bound_receipt']['sha256']:
                    raise ValueError('Tidal/density source mismatch')
                if b['prior']['sha256'] != m['prior']['sha256']:
                    raise ValueError('Reuse exterior prior changed')
        verified.append(task['key'])
        print(json.dumps(dict(stage='reuse_verified', key=task['key'], count=len(verified))), flush=True)
    write(ROOT/'REUSE_VERIFIED.json', dict(passed=True, tasks=verified, count=len(verified),
          manifest=record(ROOT/'MANIFEST.json'), job=os.environ['SLURM_JOB_ID'],
          sources_verified=True, payload_hashes_verified=True))


def reserve(segment, gpus, seconds):
    m = read(ROOT/'MANIFEST.json'); contract.validate(m)
    if not segment.isalnum():
        raise ValueError('Alphanumeric segment label required')
    booked = 0.
    for path in (ROOT/'allocations').glob('*/RESERVED.json'):
        closed = path.with_name('ACCOUNTED.json')
        booked += read(closed)['allocated_gpu_hours'] if closed.exists() else read(path)['reserved_gpu_hours']
    contract.admit(m, booked, gpus, seconds)
    write(ROOT/'allocations'/segment/'RESERVED.json', dict(segment=segment, gpus=gpus,
          wall_seconds=seconds, reserved_gpu_hours=gpus*seconds/3600, previously_booked_gpu_hours=booked,
          manifest=record(ROOT/'MANIFEST.json'), created_utc=datetime.now(timezone.utc).isoformat()))
    print(json.dumps(dict(segment=segment, booked_before=booked, reserved=gpus*seconds/3600)), flush=True)


def one(task):
    compute(); m = read(ROOT/'MANIFEST.json'); verify_sources(m)
    key = contract.address(task['phase'], task['cap'], task['seed'], task['first'])
    parent = ROOT/'tasks'/key
    parent.mkdir(parents=True, exist_ok=False)
    write(parent/'START.json', dict(key=key, job=os.environ['SLURM_JOB_ID'], manifest=record(ROOT/'MANIFEST.json')))
    if task['density_complete_candidate']:
        item = task['density_bound_receipt']; verify_record(item)
        density = Path(item['path']); verify_record(read(density)['field'])
    else:
        # Process isolation releases the models/conditions before exterior FFTs.
        command = [sys.executable, '-u', '-m', 'workflows.sbi.e2e_wholefield_fullcap_run',
                   'density', '--key', key]
        subprocess.run(command, check=True, cwd=REPO)
        density = parent/'density/COMPLETE.json'
    if task['tides_complete_candidate']:
        item = task['tides_bound_receipt']; verify_record(item)
        tides = Path(item['path']); verify_record(read(tides)['output'])
    else:
        verify_record(m['prior'])
        subprocess.run([sys.executable, '-u', '-m', 'workflows.sbi.e2e_exterior_cap_tides',
                        '--field-receipt', str(density), '--prior', m['prior']['path'],
                        '--out', str(parent/'tides'), '--method', 'learned', '--padding', '32'],
                       check=True, cwd=REPO)
        tides = parent/'tides/COMPLETE.json'
        subprocess.run([sys.executable, '-u', '-m', 'workflows.sbi.e2e_exterior_cap_verify',
                        '--root', str(parent/'tides')], check=True, cwd=REPO)
    verify_sources(m)
    write(parent/'COMPLETE.json', dict(passed=True, key=key, phase=task['phase'], cap=task['cap'],
          seed=task['seed'], first=task['first'], density=record(density), tides=record(tides),
          manifest=record(ROOT/'MANIFEST.json'), job=os.environ['SLURM_JOB_ID'], posterior_qualified=False))


def density(key):
    compute(); m = read(ROOT/'MANIFEST.json'); verify_sources(m)
    task = next(t for t in m['tasks'] if t['key'] == key)
    contract.address(task['phase'], task['cap'], task['seed'], task['first'])
    from workflows.sbi.e2e_exterior_fullcap import run
    run(BASE/'fullcap_inputs'/task['phase']/task['cap'], ROOT/'tasks'/key/'density',
        first=task['first'], model_seed=task['seed'], comparison=ROOT/'MANIFEST.json')


def worker(segment, worker_id):
    compute(); m = read(ROOT/'MANIFEST.json'); verify_sources(m)
    reservation = read(ROOT/'allocations'/segment/'RESERVED.json')
    verify_record(reservation['manifest'])
    started = read(ROOT/'allocations'/segment/'STARTED.json')
    if started['job'] != os.environ['SLURM_JOB_ID']:
        raise ValueError('Worker outside reserved allocation')
    reuse = read(ROOT/'REUSE_VERIFIED.json'); verify_record(reuse['manifest'])
    if not reuse['passed']:
        raise ValueError('Reuse verification incomplete')
    deadline = min(started['started_epoch']+reservation['wall_seconds']-180,
                   datetime.fromisoformat(m['deadline_utc']).timestamp())
    completed = []
    # Long tasks first; choose shorter tasks only if a long one no longer fits.
    candidates = sorted(m['tasks'], key=lambda t: (-t['admission_seconds'], t['key']))
    for task in candidates:
        key = task['key']
        if key in reuse['tasks']:
            continue
        dest = ROOT/'tasks'/key
        if (dest/'COMPLETE.json').exists():
            old = read(dest/'COMPLETE.json'); verify_record(old['manifest'])
            if not old['passed']:
                raise ValueError('Failed task cannot be reused')
            continue
        if time.time()+task['admission_seconds'] > deadline:
            continue
        claim = ROOT/'claims'/key
        claim.parent.mkdir(parents=True, exist_ok=True)
        try:
            claim.mkdir()
        except FileExistsError:
            continue  # Preserve active/failed claims; no automatic retry.
        if dest.exists():
            raise RuntimeError('Partial task without prior claim: '+key)
        write(claim/'OWNER.json', dict(segment=segment, worker=worker_id, job=os.environ['SLURM_JOB_ID']))
        log = ROOT/'logs'/segment/(key.replace('/', '_')+'.log')
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('x') as stream:
            subprocess.run([sys.executable, '-u', '-m', 'workflows.sbi.e2e_wholefield_fullcap_run',
                            'task', '--key', key], stdout=stream, stderr=subprocess.STDOUT, check=True, cwd=REPO)
        completed.append(key)
        print(json.dumps(dict(stage='task_complete', key=key, log=str(log))), flush=True)
    write(ROOT/'allocations'/segment/f'WORKER_{worker_id}.json', dict(passed=True, completed=completed,
          job=os.environ['SLURM_JOB_ID'], stopped_epoch=time.time()))


def start(segment):
    # salloc may execute its foreground coordinator on the submitting host;
    # only metadata is handled here. All numerical work goes through srun.
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Reserved allocation required')
    m = read(ROOT/'MANIFEST.json'); verify_sources(m)
    r = read(ROOT/'allocations'/segment/'RESERVED.json'); verify_record(r['manifest'])
    write(ROOT/'allocations'/segment/'STARTED.json', dict(job=os.environ['SLURM_JOB_ID'],
          started_epoch=time.time(), host=socket.gethostname(), reservation=record(ROOT/'allocations'/segment/'RESERVED.json')))


def account(segment):
    root = ROOT/'allocations'/segment
    job = read(root/'STARTED.json')['job']
    if not str(job).isdigit():
        raise ValueError('Invalid Slurm identifier')
    env = dict(os.environ, TZ='UTC')
    output = subprocess.check_output(['sacct', '-X', '-j', str(job), '-n', '-P',
              '--format=JobID,State,ElapsedRaw,AllocTRES,Start,End,ExitCode'], text=True, env=env)
    rows = [line.split('|') for line in output.strip().splitlines() if line.split('|')[0] == str(job)]
    if len(rows) != 1 or rows[0][1] not in ('COMPLETED', 'FAILED', 'TIMEOUT', 'CANCELLED', 'OUT_OF_MEMORY'):
        raise ValueError('Allocation not confirmed terminal: '+output)
    row = rows[0]; r = read(root/'RESERVED.json')
    tres = dict(pair.split('=', 1) for pair in row[3].split(',') if '=' in pair)
    if int(tres.get('gres/gpu', 0)) != r['gpus']:
        raise ValueError('Allocated GPU count differs from reservation')
    write(root/'ACCOUNTED.json', dict(job=job, state=row[1], seconds=int(row[2]), gpus=r['gpus'],
          allocated_gpu_hours=r['gpus']*int(row[2])/3600, start_utc=row[4], end_utc=row[5],
          exit_code=row[6], sacct=output))
    print((root/'ACCOUNTED.json').read_text())


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['stage', 'reuse', 'reserve', 'start', 'worker', 'task', 'density', 'account'])
    p.add_argument('--segment'); p.add_argument('--gpus', type=int); p.add_argument('--seconds', type=int)
    p.add_argument('--worker', type=int); p.add_argument('--key')
    a = p.parse_args()
    if a.action == 'stage': stage()
    elif a.action == 'reuse': reuse_verify()
    elif a.action == 'reserve': reserve(a.segment, a.gpus, a.seconds)
    elif a.action == 'start': start(a.segment)
    elif a.action == 'worker': worker(a.segment, a.worker)
    elif a.action == 'density': density(a.key)
    elif a.action == 'account': account(a.segment)
    elif a.action == 'task': one(next(t for t in read(ROOT/'MANIFEST.json')['tasks'] if t['key'] == a.key))
