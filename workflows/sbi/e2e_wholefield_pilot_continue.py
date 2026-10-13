"""Fixed two-slot evaluation continuation after successful bounded segments.

No new fits, statistical decisions, cancellations, failed-job retries or batch
submissions. Only the already registered 386 addresses per architecture.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

REPO = Path('/global/u2/d/dkololgi/TNG/Illustris')
ROOT = Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/wholefield_pilot_20261013_v1')
STATUS = Path('/global/u2/d/dkololgi/.codex/skills/nersc-interactive-allocation/scripts/allocation_status.py')
LAUNCHER = REPO/'workflows/sbi/e2e_wholefield_pilot_campaign_gpu.sh'
INITIAL = {'unet': 'u002', 'transformer': 't001'}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, payload):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def expected(architecture):
    prediction = {'unet': 'x', 'transformer': 'v'}[architecture]
    paths = []
    for phase in (f'ph{i:03d}' for i in range(12, 18)):
        rows = read(ROOT/'data'/phase/'COMPLETE.json')['pairs']
        if len(rows) != 16 or len({r['pair_id'] for r in rows}) != 16:
            raise ValueError('Changed evaluation geometry')
        first = next(r['pair_id'] for r in rows if r['cap'] == 'NGC')
        for row in rows:
            for seed in (17, 29):
                for coarse in ('sampled', 'oracle'):
                    nfes = (128, 256) if phase == 'ph012' and row['pair_id'] == first and seed == 17 else (128,)
                    for nfe in nfes:
                        paths.append(ROOT/'evaluation'/f'{architecture}_{prediction}_{seed}'/'step9984'/phase/
                                     row['pair_id']/coarse/'fullcap'/f'nfe{nfe}'/'COMPLETE.json')
    if len(paths) != 386:
        raise ValueError('Changed case count')
    return paths


def completed(paths, evaluation_source):
    count = 0
    for path in paths:
        if not path.exists():
            continue
        result = read(path)
        bound_path = path.parent/'BINDING.json'
        bound = read(bound_path)
        if (not result['passed'] or result['binding'] != sha(bound_path)
                or bound['source'] != evaluation_source or bound['draws'] != 32
                or bound['operator'] != 'fullcap'):
            raise ValueError('Invalid completed evaluation: '+str(path))
        count += 1
    return count


def terminal_owner(text):
    jobs = re.findall(r'(?:Pending|Granted) job allocation (\d+)', text)
    if not jobs:
        return None, None
    if 'owner_end=' not in text:
        return jobs[-1], None
    exits = re.findall(r'allocation_exit_code=(\d+) source_check_exit_code=(\d+)', text)
    return jobs[-1], bool(exits and exits[-1] == ('0', '0'))


def successor_allowed(owner_ok, scheduler_state, exit_code, done, expected_count):
    return owner_ok is True and scheduler_state == 'COMPLETED' and exit_code == '0:0' and done < expected_count


def main():
    os.chdir(REPO)
    directory = ROOT/'continuation_v1'; directory.mkdir(exist_ok=True)
    with (directory/'OWNER.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = directory/'STATE.json'
        # Deliberately no restart/retry mode: an unexpected controller exit
        # requires inspection rather than an automatic second scheduler owner.
        if state_path.exists():
            raise RuntimeError('Existing continuation state; inspect before any restart')
        names = ('models', 'train', 'smoke', 'worker', 'exterior', 'evaluate', 'campaign', 'continue')
        sources = {str(REPO/f'workflows/sbi/e2e_wholefield_pilot_{name}.py'):
                   sha(REPO/f'workflows/sbi/e2e_wholefield_pilot_{name}.py') for name in names}
        sources[str(LAUNCHER)] = sha(LAUNCHER)
        evaluate_sha = sources[str(REPO/'workflows/sbi/e2e_wholefield_pilot_evaluate.py')]
        panels = {a: expected(a) for a in INITIAL}
        state = dict(started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                     sources=sources, architectures={a: dict(segment=s, status='watching', completed=0)
                                                    for a, s in INITIAL.items()},
                     scope='Fixed existing evaluation panel only; no new fits or failed-job retries',
                     posterior_qualified=False)
        write(state_path, state)
        children = {}
        while True:
            for path, checksum in sources.items():
                if sha(path) != checksum:
                    raise ValueError('Frozen continuation source changed: '+path)
            check = subprocess.run([sys.executable, str(STATUS)], capture_output=True, text=True)
            if check.returncode not in (0, 2):
                raise RuntimeError('Allocation status unavailable: '+check.stderr)
            slots = json.loads(check.stdout)
            active = {j['job_id'] for j in slots['interactive_jobs']}
            state['interactive_count'] = slots['interactive_count']
            state['updated_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
            for architecture, item in state['architectures'].items():
                if item['status'] in ('complete', 'failed'):
                    continue
                owner = ROOT/f'{architecture}_{item["segment"]}_OWNER.log'
                job, owner_ok = terminal_owner(owner.read_text() if owner.exists() else '')
                item['job'] = job
                if job is None or job in active or owner_ok is None:
                    if architecture in children and children[architecture].poll() not in (None, 0):
                        item.update(status='failed', reason='Owner process failed; no retry')
                    continue
                record = subprocess.run(['sacct', '-X', '-n', '-P', '-j', job,
                                         '-o', 'JobIDRaw,State,ExitCode'], text=True, capture_output=True, check=True)
                entries = [line.split('|') for line in record.stdout.splitlines() if line.startswith(job+'|')]
                if len(entries) != 1:
                    continue  # Accounting propagation; never infer success.
                _, status, exit_code = entries[0][:3]
                if status in ('RUNNING', 'PENDING', 'COMPLETING'):
                    continue
                item.update(scheduler_state=status, exit_code=exit_code)
                if owner_ok is not True or status != 'COMPLETED' or exit_code != '0:0':
                    item.update(status='failed', reason='Failed segment; no automatic retry')
                    continue
                done = completed(panels[architecture], evaluate_sha)
                item['completed'] = done
                if done == 386:
                    item['status'] = 'complete'
                    continue
                if slots['interactive_count'] >= 2:
                    item['status'] = 'waiting_for_slot'
                    continue
                if not successor_allowed(owner_ok, status, exit_code, done, 386):
                    raise RuntimeError('Unapproved continuation state')
                old = item['segment']; segment = old[0]+f'{int(old[1:])+1:03d}'
                if (ROOT/f'{architecture}_{segment}_OWNER.log').exists():
                    raise RuntimeError('Refuse reuse of an existing segment address')
                item.update(segment=segment, status='launching', job=None,
                            previous_segment=old, previous_completed=done)
                write(state_path, state)
                children[architecture] = subprocess.Popen(['bash', str(LAUNCHER), architecture, segment, 'evaluate'])
                # One submission per minute; do not race scheduler visibility.
                break
            write(state_path, state)
            if all(i['status'] in ('complete', 'failed') for i in state['architectures'].values()):
                state['all_evaluation_complete'] = all(i['status'] == 'complete' for i in state['architectures'].values())
                write(directory/'COMPLETE.json', state)
                return
            time.sleep(60)


if __name__ == '__main__':
    main()
