"""Authorized baseline recovery and resource-only ceiling waiver, 12 October.

Preserve the original scientific manifest and frozen numerical sources. A new
resource-authority sidecar supersedes its historical 250-GPU-hour ceiling only.
Recover verified density products without rerunning their expensive sampler.
Fixed tmux ownership, two interactive slots, deadline and scientific scope stay.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import importlib.util
import os
from pathlib import Path
import time

ADAPTER = Path('/global/u2/d/dkololgi/TNG/Illustris/workflows/sbi/e2e_wholefield_baseline_campaign_v2.py')
spec = importlib.util.spec_from_file_location('baseline_campaign_v2', ADAPTER)
v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2)
c = v2.c
PREVIOUS = c.CAMPAIGN
c.CAMPAIGN = c.ROOT/'baseline_campaign_20261012_v3'
c.SCRIPT = Path(__file__).resolve()
original_choose = c.choose_work
original_compute = c.compute
original_resources = c.resource_plan


def validate_density_identity(task, record, binding, manifest):
    expected = dict(phase=task['phase'], cap=task['cap'], first=task['first'], draws=4)
    if not record['passed'] or any(record.get(k) != value for k, value in expected.items()):
        raise ValueError('Density receipt address mismatch')
    if (any(binding.get(k) != value for k, value in expected.items())
            or binding['model_seed'] != task['seed'] or binding['nfe'] != 128
            or binding['checkpoints'] != manifest['checkpoints'][str(task['seed'])]
            or binding['normalizer'] != manifest['normalizer']
            or binding['prepared_sha256'] != manifest['prepared'][task['phase']+'/'+task['cap']]
            or binding['comparison_sha256'] != c.digest(c.ROOT/'MANIFEST.json')
            or binding['benchmark'] or binding['freeze_sha256'] is not None):
        raise ValueError('Density scientific provenance mismatch')


def prepare():
    if c.interactive_jobs():
        raise RuntimeError('Recovery preparation requires idle interactive slots')
    if not (PREVIOUS/'STOPPED.json').exists():
        raise RuntimeError('Predecessor campaign must be stopped')
    gen = c.frozen(); manifest = gen.read(c.ROOT/'MANIFEST.json'); gen.verify_sources(manifest)
    complete, extracted, ready = c.inventory()
    if (len(complete), len(extracted), len(ready)) != (105, 105, 79):
        raise RuntimeError('Baseline inventory changed since recovery audit')
    old = c.read(PREVIOUS/'MANIFEST.json')
    for path, expected in old['sources'].items():
        if c.digest(path) != expected:
            raise RuntimeError('Predecessor source changed')
    recoveries = []
    for task in manifest['tasks']:
        key = task['key']; dest = c.ROOT/'tasks'/key
        if key in complete or not (dest/'density/COMPLETE.json').exists():
            continue
        owner_path = c.ROOT/'claims'/key/'OWNER.json'; owner = gen.read(owner_path)
        account_path = c.ROOT/'allocations'/owner['segment']/'ACCOUNTED.json'
        account = gen.read(account_path)
        if owner['segment'] not in ('s032', 's034') or account['state'] != 'TIMEOUT' or account['job'] != owner['job']:
            raise RuntimeError('Recovery outside the two explicitly audited timeouts')
        if (dest/'tides').exists() or (dest/'tides_recovery_20261012').exists():
            raise RuntimeError('Unexpected prior tidal attempt; preserve and inspect')
        density = dest/'density/COMPLETE.json'; rec = gen.read(density)
        binding_path = dest/'density/BINDING.json'
        if gen.sha(binding_path) != rec['binding']:
            raise RuntimeError('Density binding hash mismatch')
        validate_density_identity(task, rec, gen.read(binding_path), manifest)
        payload = Path(rec['field']['path'])
        if payload != dest/'density/FIELD.h5' or payload.stat().st_size != rec['field']['bytes']:
            raise RuntimeError('Missing/truncated density payload')
        recoveries.append(dict(key=key, density=gen.record(density), binding=gen.record(binding_path),
                               old_owner=gen.record(owner_path), old_account=gen.record(account_path)))
    if len(recoveries) != 8:
        raise RuntimeError('Exactly eight density-only recoveries expected')
    if any(not p.with_name('ACCOUNTED.json').exists() for p in (c.ROOT/'allocations').glob('*/RESERVED.json')):
        raise RuntimeError('Unreconciled allocation')
    c.CAMPAIGN.mkdir(exist_ok=False)
    authority = dict(scope='baseline generation, corrected extraction and complete-bank verification only',
                     user_instruction='No need to worry about budgets! Just complete the work',
                     allocated_gpu_hour_limit=None, supersedes_allocated_gpu_hour_limit=250,
                     maximum_simultaneous_allocations=2, maximum_segment_seconds=10800,
                     deadline_utc=manifest['deadline_utc'], interactive_only=True,
                     preserve_all_accounting=True, no_automatic_new_failure_retry=True,
                     ssh_disconnect_survival='fixed tmux server independent of SSH/app terminal')
    c.write(c.CAMPAIGN/'RESOURCE_AUTHORITY.json', authority)
    new = dict(old, scope=authority['scope'], recoveries=recoveries,
               gpu_hour_limit=None, resources=gen.record(c.CAMPAIGN/'RESOURCE_AUTHORITY.json'),
               predecessor=gen.record(PREVIOUS/'MANIFEST.json'),
               sources={**old['sources'], str(c.SCRIPT): c.digest(c.SCRIPT),
                        str(c.CAMPAIGN/'RESOURCE_AUTHORITY.json'): c.digest(c.CAMPAIGN/'RESOURCE_AUTHORITY.json')},
               git_commit=c.subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=c.REPO, text=True).strip(),
               created_utc=datetime.now(timezone.utc).isoformat())
    c.write(c.CAMPAIGN/'MANIFEST.json', new)
    c.write(c.CAMPAIGN/'PREPARED.json', dict(passed=True, complete=105, extracted=105,
            recovery_density_payloads=8, untouched_generation=79, payload_hash_verification_pending=True))
    print('V3_PREPARED_NO_SCHEDULER_SUBMISSION', flush=True)


def resources(mode, count):
    if mode == 'smoke': return 1, 3600
    if mode == 'recover': return 1, 3600
    return original_resources(mode, count)


def resource_admit(authority, gpus, seconds, now):
    if (authority['allocated_gpu_hour_limit'] is not None
            or authority['supersedes_allocated_gpu_hour_limit'] != 250
            or authority['maximum_simultaneous_allocations'] != 2
            or not authority['interactive_only'] or gpus not in (1, 2, 4)
            or not 0 < seconds <= authority['maximum_segment_seconds']):
        raise PermissionError('Outside resource-only amendment')
    if now+600+seconds > datetime.fromisoformat(authority['deadline_utc']).timestamp():
        raise PermissionError('Campaign deadline, including queue allowance, would be exceeded')


def reserve_launch(mode, keys, count):
    binding = c.verify_sources(); gen = c.frozen(); gpus, seconds = resources(mode, count)
    authority = c.read(binding['resources']['path']); gen.verify_record(binding['resources'])
    resource_admit(authority, gpus, seconds, time.time())
    if len(c.interactive_jobs()) >= 2:
        raise RuntimeError('Both interactive slots are occupied')
    existing = [int(p.name[1:]) for p in (c.ROOT/'allocations').iterdir()
                if p.name.startswith('s') and p.name[1:].isdigit()]
    segment = f's{max(existing)+1:03d}'; directory = c.ROOT/'allocations'/segment
    booked = sum(c.read(p.with_name('ACCOUNTED.json'))['allocated_gpu_hours']
                 if p.with_name('ACCOUNTED.json').exists() else c.read(p)['reserved_gpu_hours']
                 for p in (c.ROOT/'allocations').glob('*/RESERVED.json'))
    # Do not alter the historical manifest or monkey-patch its frozen scientific
    # validator. This reservation explicitly binds the new resource authority.
    c.write(directory/'RESERVED.json', dict(segment=segment, gpus=gpus, wall_seconds=seconds,
            reserved_gpu_hours=gpus*seconds/3600, previously_booked_gpu_hours=booked,
            manifest=gen.record(c.ROOT/'MANIFEST.json'), resource_amendment=binding['resources'],
            created_utc=datetime.now(timezone.utc).isoformat()))
    c.write(directory/'BASELINE_PLAN.json', dict(mode=mode, keys=keys, gpus=gpus, seconds=seconds,
            campaign_sha256=c.digest(c.CAMPAIGN/'MANIFEST.json')))
    with (directory/'BASELINE_OWNER.log').open('x') as log:
        process = c.subprocess.Popen([c.PY, '-u', str(c.SCRIPT), 'owner', '--segment', segment],
                                     stdout=log, stderr=c.subprocess.STDOUT, env=c.environment(), cwd=c.REPO)
    print(c.json.dumps(dict(stage='launch', segment=segment, mode=mode, gpus=gpus, keys=keys,
                           booked_before=booked, reserved_gpu_hours=gpus*seconds/3600)), flush=True)
    return segment, process, mode


def choose_work(smoke_passed, active_modes, backlog, ready_count, generated, extracted):
    binding = c.read(c.CAMPAIGN/'MANIFEST.json')
    pending = [r['key'] for r in binding['recoveries']
               if not (c.ROOT/'tasks'/r['key']/'COMPLETE.json').exists()]
    if not smoke_passed:
        if active_modes: return None
        if not pending: raise RuntimeError('Recovery smoke has no pending density product')
        return 'smoke', pending[:1], 0
    if pending and not any(m in ('recover', 'smoke') for m in active_modes):
        return 'recover', pending[:4], 0
    # Recovery also performs extraction. Treat it as an extraction owner to
    # prevent another allocation claiming a just-completed recovery's core.
    normalized = ['extract' if m == 'recover' else m for m in active_modes]
    return original_choose(True, normalized, backlog, ready_count, generated, extracted)


def stage_command(label, command, timeout):
    print(c.json.dumps(dict(stage=label, event='begin', utc=datetime.now(timezone.utc).isoformat())), flush=True)
    before = time.monotonic()
    c.subprocess.run(command, cwd=c.ROOT/'source', env=c.environment(), check=True, timeout=timeout)
    print(c.json.dumps(dict(stage=label, event='complete', seconds=time.monotonic()-before)), flush=True)


def recover(key):
    gen = c.frozen(); gen.compute(); binding = c.verify_sources()
    row = next(r for r in binding['recoveries'] if r['key'] == key)
    m = gen.read(c.ROOT/'MANIFEST.json'); gen.verify_sources(m)
    task = next(t for t in m['tasks'] if t['key'] == key)
    gen.contract.address(task['phase'], task['cap'], task['seed'], task['first'])
    for field in ('density', 'binding', 'old_owner', 'old_account'):
        gen.verify_record(row[field])
    density_path = Path(row['density']['path']); density = gen.read(density_path)
    validate_density_identity(task, density, gen.read(row['binding']['path']), m)
    print(c.json.dumps(dict(stage='verify_saved_density', key=key, event='begin')), flush=True)
    gen.verify_record(density['field']); gen.verify_record(m['prior'])
    dest = c.ROOT/'tasks'/key; tides = dest/'tides_recovery_20261012'
    gen.write(dest/'RECOVERY_20261012.json', dict(key=key, density=row['density'],
              payload_hash_verified=True, old_owner=row['old_owner'], old_account=row['old_account'],
              campaign=gen.record(c.CAMPAIGN/'MANIFEST.json'), job=os.environ['SLURM_JOB_ID']))
    stage_command('recover_tides', [c.PY, '-u', '-m', 'workflows.sbi.e2e_exterior_cap_tides',
                  '--field-receipt', str(density_path), '--prior', m['prior']['path'],
                  '--out', str(tides), '--method', 'learned', '--padding', '32'], 1200)
    stage_command('verify_recovered_tides', [c.PY, '-u', '-m', 'workflows.sbi.e2e_exterior_cap_verify',
                  '--root', str(tides)], 300)
    verified = gen.read(tides/'VERIFIED.json')
    if not verified['passed']:
        raise RuntimeError('Recovered tidal verification failed')
    gen.verify_sources(m)
    gen.write(dest/'COMPLETE.json', dict(passed=True, key=key, phase=task['phase'], cap=task['cap'],
              seed=task['seed'], first=task['first'], density=gen.record(density_path),
              tides=gen.record(tides/'COMPLETE.json'), manifest=gen.record(c.ROOT/'MANIFEST.json'),
              recovery=gen.record(dest/'RECOVERY_20261012.json'), job=os.environ['SLURM_JOB_ID'],
              posterior_qualified=False))
    stage_command('extract_recovered_core', [c.PY, '-u', str(c.ROOT/'extraction_v1/extract.py'), 'run',
                  '--phase', task['phase'], '--cap', task['cap'], '--seed', str(task['seed']),
                  '--first', str(task['first'])], 300)


def compute(segment):
    plan = c.read(c.ROOT/'allocations'/segment/'BASELINE_PLAN.json')
    if plan['mode'] not in ('smoke', 'recover'):
        return original_compute(segment)
    gen = c.frozen(); gen.compute(); c.verify_sources(); gen.start(segment)
    started = c.read(c.ROOT/'allocations'/segment/'STARTED.json')
    deadline = started['started_epoch']+plan['seconds']-180
    completed = []
    for key in plan['keys']:
        # Never start another recovery if its three bounded commands could
        # overrun this allocation. Unstarted keys remain eligible next time.
        if time.time()+1800 > deadline:
            break
        recover(key)
        completed.append(key)
    if not completed:
        raise RuntimeError('No complete recovery fits the remaining allocation')
    c.verify_sources()
    c.write(c.ROOT/'allocations'/segment/'BASELINE_COMPUTE_COMPLETE.json',
            dict(passed=True, mode=plan['mode'], keys=completed,
                 deferred=[k for k in plan['keys'] if k not in completed]))


c.resource_plan = resources
c.reserve_launch = reserve_launch
c.choose_work = choose_work
c.compute = compute


def run():
    # Keep every predecessor controller locked out; no global directory-mode
    # workaround or running-job handoff is needed after their verified stop.
    with (PREVIOUS/'CONTROLLER.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        c.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run', 'owner', 'compute'])
    parser.add_argument('--segment')
    args = parser.parse_args()
    if args.action == 'prepare': prepare()
    elif args.action == 'run': run()
    elif args.action == 'owner': c.owner(args.segment)
    elif args.action == 'compute': compute(args.segment)
