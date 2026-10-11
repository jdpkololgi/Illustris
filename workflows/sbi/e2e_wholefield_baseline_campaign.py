"""Finite, user-authorized baseline-only interactive campaign (11 October).

Login-node work is metadata/scheduler orchestration only. Numerical work uses
unchanged frozen generation/extraction sources on compute nodes. No scientific
gate, fitting, batch submission, automatic failure retry, or sealed-phase access.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path('/global/u2/d/dkololgi/TNG/Illustris')
ROOT = Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/wholefield_comparison_20261009_v1/fullcap_v1')
CAMPAIGN = ROOT/'baseline_campaign_20261011'
SCRIPT = REPO/'workflows/sbi/e2e_wholefield_baseline_campaign.py'
PY = '/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python'
TERMINAL = {'COMPLETED', 'FAILED', 'TIMEOUT', 'CANCELLED', 'OUT_OF_MEMORY'}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen():
    sys.path.insert(0, str(ROOT/'source'))
    from workflows.sbi import e2e_wholefield_fullcap_run as gen
    if gen.REPO != ROOT/'source':
        raise RuntimeError('Frozen generation dependency required')
    return gen


def environment():
    env = dict(os.environ)
    for key in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONUSERBASE', 'LD_PRELOAD'):
        env.pop(key, None)
    env.update(PYTHONNOUSERSITE='1', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', CUBLAS_WORKSPACE_CONFIG=':4096:8', TZ='UTC')
    return env


def interactive_jobs():
    output = subprocess.check_output(['squeue', '-h', '-u', os.environ['USER'],
                                      '-o', '%i|%q|%T'], text=True)
    return [line.split('|') for line in output.splitlines()
            if line.split('|')[1] in ('interactive', 'shared_interactive')]


def verify_sources():
    binding = read(CAMPAIGN/'MANIFEST.json')
    for name, expected in binding['sources'].items():
        if digest(name) != expected:
            raise RuntimeError('Campaign source changed: '+name)
    if digest(ROOT/'MANIFEST.json') != binding['generation_sha256']:
        raise RuntimeError('Generation manifest changed')
    return binding


def inventory():
    manifest = read(ROOT/'MANIFEST.json')
    gen = frozen(); gen.contract.validate(manifest)
    reused = read(ROOT/'REUSE_VERIFIED.json')
    if not reused['passed'] or reused['manifest']['sha256'] != digest(ROOT/'MANIFEST.json'):
        raise RuntimeError('Unverified reuse')
    complete = set(reused['tasks']); extracted = set(); ready = []
    for task in manifest['tasks']:
        key = task['key']; gen.contract.address(task['phase'], task['cap'], task['seed'], task['first'])
        p = ROOT/'tasks'/key/'COMPLETE.json'
        if p.exists():
            r = read(p)
            if not r['passed'] or r['key'] != key or r['manifest']['sha256'] != digest(ROOT/'MANIFEST.json'):
                raise RuntimeError('Invalid generation receipt: '+key)
            complete.add(key)
        e = ROOT/'extraction_v1/cores'/key/'COMPLETE.json'
        if e.exists():
            r = read(e)
            if not r['passed'] or r['key'] != key or key not in complete:
                raise RuntimeError('Invalid extraction receipt: '+key)
            extracted.add(key)
        if key not in complete and not (ROOT/'claims'/key).exists():
            if (ROOT/'tasks'/key).exists():
                raise RuntimeError('Unclaimed partial destination: '+key)
            ready.append(key)
    if len(manifest['tasks']) != 192 or len({t['key'] for t in manifest['tasks']}) != 192:
        raise RuntimeError('Wrong baseline inventory')
    return complete, extracted, ready


def retry_candidates():
    """Only previously accounted interrupted tasks with metadata, never payloads."""
    found = []
    for task in read(ROOT/'MANIFEST.json')['tasks']:
        key = task['key']; dest = ROOT/'tasks'/key; claim = ROOT/'claims'/key
        if not claim.exists() or (dest/'COMPLETE.json').exists():
            continue
        owner = read(claim/'OWNER.json')
        accounting = read(ROOT/'allocations'/owner['segment']/'ACCOUNTED.json')
        if accounting['state'] not in TERMINAL or accounting['job'] != owner['job']:
            raise RuntimeError('Attempt not terminal: '+key)
        files = {}
        for base, allowed in ((dest, {'START.json', 'density/BINDING.json'}),
                              (claim, {'OWNER.json'})):
            actual = {str(p.relative_to(base)) for p in base.rglob('*') if p.is_file()}
            if actual != allowed or any(p.is_symlink() for p in base.rglob('*')):
                raise RuntimeError('Unexpected interrupted-task contents: '+key)
            files.update({str(base/p): digest(base/p) for p in actual})
        found.append(dict(key=key, owner=owner, original_files=files))
    return found


def prepare():
    if interactive_jobs():
        raise RuntimeError('Prepare requires idle interactive slots')
    inventory()
    retries = retry_candidates()
    if len(retries) != 14:
        raise RuntimeError('Expected exactly the 14 audited interrupted tasks')
    for path in (ROOT/'allocations').glob('*/RESERVED.json'):
        if not path.with_name('ACCOUNTED.json').exists():
            raise RuntimeError('Unreconciled prior reservation: '+str(path))
    names = [SCRIPT, REPO/'workflows/sbi/e2e_wholefield_fullcap_parallel.py',
             ROOT/'extraction_v1/extract.py', ROOT/'extraction_v1/MANIFEST.json']
    names += [REPO/'workflows/sbi'/name for name in (
        'e2e_wholefield_fullcap_bank.py', 'e2e_wholefield_fullcap_extract.py',
        'e2e_wholefield_fullcap_run.py', 'e2e_wholefield_fullcap_contract.py',
        'e2e_wholefield_sensitivity.py', 'e2e_vdm_context_metrics.py')]
    binding = dict(scope='baseline generation, corrected extraction, complete-bank verification only',
                   sources={str(p): digest(p) for p in names}, retries=retries,
                   generation_sha256=digest(ROOT/'MANIFEST.json'), max_allocations=2,
                   gpu_hour_limit=250, automatic_new_failure_retries=0,
                   git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
                   created_utc=datetime.now(timezone.utc).isoformat())
    CAMPAIGN.mkdir(exist_ok=False)
    write(CAMPAIGN/'MANIFEST.json', binding)
    # Save the mapping before moving anything; a interrupted preparation fails
    # closed and requires inspection. Originals are never deleted or overwritten.
    for row in retries:
        key = row['key']; archive = CAMPAIGN/'preserved_attempts'/key
        archive.mkdir(parents=True, exist_ok=False)
        for old, new in ((ROOT/'tasks'/key, archive/'task'), (ROOT/'claims'/key, archive/'claim')):
            old.rename(new)
        for original, expected in row['original_files'].items():
            original = Path(original)
            base = ROOT/('claims' if original.is_relative_to(ROOT/'claims') else 'tasks')/key
            kind = 'claim' if base.is_relative_to(ROOT/'claims') else 'task'
            if digest(archive/kind/original.relative_to(base)) != expected:
                raise RuntimeError('Preserved metadata mismatch')
    write(CAMPAIGN/'PREPARED.json', dict(passed=True, preserved_attempts=len(retries),
          new_attempt_policy='one fresh attempt per missing task; stop on any new failure'))
    print(json.dumps(dict(prepared=True, preserved_attempts=len(retries))), flush=True)


def resource_plan(mode, ready_count):
    if mode == 'generate':
        if ready_count <= 0:
            raise ValueError('No generation tasks available')
        return (4 if ready_count >= 4 else 2 if ready_count >= 2 else 1), 10800
    if mode in ('extract', 'bank', 'smoke'):
        return 1, (600 if mode == 'smoke' else 1800)
    raise ValueError('Baseline-only mode required')


def choose_work(smoke_passed, active_modes, backlog, ready_count, generated, extracted):
    if not smoke_passed:
        if active_modes:
            return None
        if not backlog:
            raise RuntimeError('Smoke requires completed unextracted product')
        return 'smoke', backlog[:1], 0
    extracting = any(mode in ('extract', 'smoke', 'bank') for mode in active_modes)
    if not extracting and (len(backlog) >= 8 or (backlog and not ready_count)):
        return 'extract', backlog[:16], 0
    if ready_count:
        return 'generate', [], ready_count
    if not active_modes:
        if generated != 192 or extracted != 192:
            raise RuntimeError('No runnable tasks but baseline incomplete')
        return 'bank', [], 0
    return None


def allocation_command(segment, plan):
    gpus = plan['gpus']; mode = plan['mode']; seconds = plan['seconds']
    qos = 'interactive' if gpus == 4 else 'shared_interactive'
    wall = f'{seconds//3600:02d}:{seconds%3600//60:02d}:{seconds%60:02d}'
    payload = ([PY, '-u', str(REPO/'workflows/sbi/e2e_wholefield_fullcap_parallel.py'), '--segment', segment]
               if mode == 'generate' and gpus > 1 else
               [PY, '-u', str(SCRIPT), 'compute', '--segment', segment])
    return ['salloc', '-N1', f'-n{gpus}', '-c32', f'--gpus={gpus}', '--constraint=gpu&hbm80g',
            f'--qos={qos}', '--account=desi_g', '--licenses=scratch', f'--time={wall}',
            '--immediate=600', f'--job-name=wf_base_{segment}',
            'srun', '--mpi=none', '--resv-ports=0', '--kill-on-bad-exit=1', '-N1',
            f'-n{gpus}', '-c32', '--gpus-per-task=1', '--gpu-bind=single:1', '--cpu-bind=cores', *payload]


def owner(segment):
    if not os.environ.get('TMUX'):
        raise RuntimeError('Explicitly approved tmux ownership required')
    verify_sources(); gen = frozen(); directory = ROOT/'allocations'/segment
    plan = read(directory/'BASELINE_PLAN.json')
    if plan['campaign_sha256'] != digest(CAMPAIGN/'MANIFEST.json'):
        raise RuntimeError('Campaign binding mismatch')
    reservation = read(directory/'RESERVED.json')
    if (plan['gpus'], plan['seconds']) != (reservation['gpus'], reservation['wall_seconds']):
        raise RuntimeError('Reservation mismatch')
    result = subprocess.run(allocation_command(segment, plan), env=environment(), cwd=ROOT/'source').returncode
    write(directory/'BASELINE_OWNER_EXIT.json', dict(allocation_command_exit_code=result))
    if (directory/'STARTED.json').exists():
        for attempt in range(12):
            try:
                gen.account(segment)
                break
            except ValueError:
                if attempt == 11:
                    raise
                time.sleep(5)
    verify_sources()
    if result != 0:
        raise RuntimeError(f'Allocation command exited {result}')
    account = read(directory/'ACCOUNTED.json')
    if account['state'] != 'COMPLETED' or account['exit_code'] != '0:0':
        raise RuntimeError('Unsuccessful allocation accounting')
    if plan['mode'] == 'generate':
        keys = []
        for rank in range(plan['gpus']):
            worker = read(directory/f'WORKER_{rank}.json')
            if not worker['passed'] or worker['job'] != account['job']:
                raise RuntimeError('Missing/failed worker')
            if plan['gpus'] > 1 and not read(directory/f'RANK_{rank}_COMPLETE.json')['passed']:
                raise RuntimeError('Missing rank completion')
            keys.extend(worker['completed'])
        if not keys or len(set(keys)) != len(keys):
            raise RuntimeError('Empty or duplicate generation completion')
    elif not read(directory/'BASELINE_COMPUTE_COMPLETE.json')['passed']:
        raise RuntimeError('Missing extraction/bank verification')
    write(directory/'BASELINE_OWNER_COMPLETE.json', dict(passed=True, mode=plan['mode'], job=account['job']))


def compute(segment):
    gen = frozen(); gen.compute(); verify_sources(); gen.start(segment)
    directory = ROOT/'allocations'/segment; plan = read(directory/'BASELINE_PLAN.json')
    if plan['mode'] == 'generate':
        gen.worker(segment, 0)
    elif plan['mode'] in ('extract', 'smoke'):
        for key in plan['keys']:
            task = next(t for t in gen.read(ROOT/'MANIFEST.json')['tasks'] if t['key'] == key)
            gen.contract.address(task['phase'], task['cap'], task['seed'], task['first'])
            subprocess.run([PY, '-u', str(ROOT/'extraction_v1/extract.py'), 'run',
                            '--phase', task['phase'], '--cap', task['cap'], '--seed', str(task['seed']),
                            '--first', str(task['first'])], check=True, cwd=ROOT/'source', env=environment())
    elif plan['mode'] == 'bank':
        subprocess.run([PY, '-u', '-m', 'workflows.sbi.e2e_wholefield_fullcap_bank'],
                       check=True, cwd=REPO, env=environment())
        result = gen.read(ROOT/'baseline_v1/COMPLETE.json')
        if not result['passed'] or result['cases'] != 192 or result['gate2_passed'] is not None:
            raise RuntimeError('Invalid complete-bank result')
    else:
        raise RuntimeError('Forbidden mode')
    verify_sources()
    write(directory/'BASELINE_COMPUTE_COMPLETE.json', dict(passed=True, mode=plan['mode'], keys=plan['keys']))


def reserve_launch(mode, keys, count):
    verify_sources(); gen = frozen(); gpus, seconds = resource_plan(mode, count)
    manifest = read(ROOT/'MANIFEST.json')
    if time.time()+600+seconds > datetime.fromisoformat(manifest['deadline_utc']).timestamp():
        raise RuntimeError('Queue plus walltime exceeds campaign deadline')
    existing = [int(p.name[1:]) for p in (ROOT/'allocations').iterdir()
                if p.name.startswith('s') and p.name[1:].isdigit()]
    segment = f's{max(existing)+1:03d}'
    gen.reserve(segment, gpus, seconds)  # Called by only the locked controller.
    directory = ROOT/'allocations'/segment
    write(directory/'BASELINE_PLAN.json', dict(mode=mode, keys=keys, gpus=gpus, seconds=seconds,
          campaign_sha256=digest(CAMPAIGN/'MANIFEST.json')))
    with (directory/'BASELINE_OWNER.log').open('x') as log:
        process = subprocess.Popen([PY, '-u', str(SCRIPT), 'owner', '--segment', segment],
                                   stdout=log, stderr=subprocess.STDOUT, env=environment(), cwd=REPO)
    print(json.dumps(dict(stage='launch', segment=segment, mode=mode, gpus=gpus, keys=keys)), flush=True)
    return segment, process, mode


def run():
    if not os.environ.get('TMUX') or not read(CAMPAIGN/'PREPARED.json')['passed']:
        raise RuntimeError('Prepared, tmux-owned campaign required')
    with (CAMPAIGN/'CONTROLLER.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(CAMPAIGN/'STARTED.json', dict(pid=os.getpid(), host=os.uname().nodename,
              started_utc=datetime.now(timezone.utc).isoformat()))
        active = []; stop = None; smoke_passed = False; bank_launched = False
        # A finite inventory plus one smoke and one bank validation. No restart
        # policy, retries after new failures, or model/diagnostic decision loop.
        launches = 0
        while True:
            try:
                for item in active[:]:
                    segment, process, mode = item
                    if process.poll() is None:
                        continue
                    active.remove(item)
                    if process.returncode != 0 or not (ROOT/'allocations'/segment/'BASELINE_OWNER_COMPLETE.json').exists():
                        raise RuntimeError('Segment failed; no replacements: '+segment)
                    if mode == 'smoke':
                        smoke_passed = True
                    if mode == 'bank':
                        complete, extracted, _ = inventory()
                        if len(complete) != 192 or len(extracted) != 192:
                            raise RuntimeError('Final inventory incomplete')
                        write(CAMPAIGN/'COMPLETE.json', dict(passed=True, generated=192, extracted=192,
                              baseline=str(ROOT/'baseline_v1/COMPLETE.json'), gate2_run=False, training_run=False))
                        print('BASELINE_CAMPAIGN_COMPLETE', flush=True)
                        return
                if stop is None:
                    verify_sources()
                    complete, extracted, ready = inventory()
                    print(json.dumps(dict(stage='status', generated=len(complete), extracted=len(extracted),
                          active=[a[0] for a in active], ready=len(ready))), flush=True)
                    # Count all user interactive jobs plus not-yet-visible owners.
                    # This can overcount a pending owner but never creates slot 3.
                    slots_used = len(interactive_jobs())+sum(
                        not (ROOT/'allocations'/s/'STARTED.json').exists() for s, _, _ in active)
                    if slots_used < 2 and len(active) < 2:
                        backlog = sorted(complete-extracted)
                        choice = choose_work(smoke_passed, [a[2] for a in active], backlog,
                                             len(ready), len(complete), len(extracted))
                        if choice is not None:
                            if choice[0] == 'bank':
                                if bank_launched:
                                    raise RuntimeError('Bank verification cannot be retried')
                                bank_launched = True
                            if launches >= 160:
                                raise RuntimeError('Finite allocation-count safety limit')
                            active.append(reserve_launch(*choice)); launches += 1
            except Exception as error:
                if stop is None:
                    stop = dict(error=type(error).__name__, message=str(error),
                                active=[a[0] for a in active], utc=datetime.now(timezone.utc).isoformat())
                    write(CAMPAIGN/'STOP_REQUESTED.json', stop)
                    print(json.dumps(stop), flush=True)
            if stop is not None and not active:
                write(CAMPAIGN/'STOPPED.json', stop)
                raise RuntimeError(stop['message'])
            time.sleep(60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run', 'owner', 'compute'])
    parser.add_argument('--segment')
    args = parser.parse_args()
    if args.action == 'prepare': prepare()
    elif args.action == 'run': run()
    elif args.action == 'owner': owner(args.segment)
    elif args.action == 'compute': compute(args.segment)
