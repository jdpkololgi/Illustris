"""Non-disruptive QOS-name correction for the fixed baseline controller.

The v1 file stays byte-identical while its two existing allocation owners run.
This adapter counts NERSC's expanded GPU/CPU QOS names, waits for the original
owners to finish and the original controller to stop, then continues the same
baseline-only campaign. No running Slurm job is cancelled or modified.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import importlib.util
import os
from pathlib import Path
import time

BASE_SCRIPT = Path('/global/u2/d/dkololgi/TNG/Illustris/workflows/sbi/e2e_wholefield_baseline_campaign.py')
spec = importlib.util.spec_from_file_location('baseline_campaign_v1', BASE_SCRIPT)
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
PREVIOUS = c.CAMPAIGN
c.CAMPAIGN = c.ROOT/'baseline_campaign_20261011_v2'
c.SCRIPT = Path(__file__).resolve()


def parse_interactive_jobs(output):
    rows = []
    for line in output.splitlines():
        fields = line.strip().split('|')
        if len(fields) != 3:
            raise ValueError('Unexpected scheduler response')
        if fields[1] == 'interactive' or fields[1].endswith('_interactive'):
            rows.append(fields)
    return rows


def interactive_jobs():
    return parse_interactive_jobs(c.subprocess.check_output(
        ['squeue', '-h', '-u', os.environ['USER'], '-o', '%i|%q|%T'], text=True))


c.interactive_jobs = interactive_jobs


def prepare():
    old = c.read(PREVIOUS/'MANIFEST.json')
    if not c.read(PREVIOUS/'PREPARED.json')['passed']:
        raise RuntimeError('Original archive preparation not verified')
    for path, expected in old['sources'].items():
        if c.digest(path) != expected:
            raise RuntimeError('Original running sources must remain unchanged')
    if len(old['retries']) != 14:
        raise RuntimeError('Unexpected preserved-attempt inventory')
    for row in old['retries']:
        key = row['key']
        for original, expected in row['original_files'].items():
            original = Path(original)
            kind = 'claim' if original.is_relative_to(c.ROOT/'claims') else 'task'
            prefix = c.ROOT/('claims' if kind == 'claim' else 'tasks')/key
            saved = PREVIOUS/'preserved_attempts'/key/kind/original.relative_to(prefix)
            if c.digest(saved) != expected:
                raise RuntimeError('Preserved interrupted metadata changed')
    adopted = []
    for segment in ('s020', 's021'):
        directory = c.ROOT/'allocations'/segment
        plan = c.read(directory/'BASELINE_PLAN.json')
        if plan['mode'] != 'generate' or plan['campaign_sha256'] != c.digest(PREVIOUS/'MANIFEST.json'):
            raise RuntimeError('Unexpected predecessor segment')
        adopted.append(dict(segment=segment, job=c.read(directory/'STARTED.json')['job']))
    if (c.ROOT/'allocations').stat().st_mode & 0o777 != 0o500:
        raise RuntimeError('New-reservation hold is not in place')
    for path in (c.ROOT/'allocations').glob('*/RESERVED.json'):
        if not path.with_name('ACCOUNTED.json').exists() and path.parent.name not in ('s020', 's021'):
            raise RuntimeError('Unexpected unresolved reservation')
    c.CAMPAIGN.mkdir(exist_ok=False)
    new = dict(old, sources={**old['sources'], str(c.SCRIPT): c.digest(c.SCRIPT)},
               predecessor=dict(path=str(PREVIOUS/'MANIFEST.json'), sha256=c.digest(PREVIOUS/'MANIFEST.json')),
               adopted_segments=adopted, restore_allocation_directory_mode='0700',
               git_commit=c.subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=c.REPO, text=True).strip(),
               created_utc=datetime.now(timezone.utc).isoformat())
    c.write(c.CAMPAIGN/'MANIFEST.json', new)
    c.write(c.CAMPAIGN/'PREPARED.json', dict(passed=True, preserved_attempts=14,
            archive_location=str(PREVIOUS/'preserved_attempts'), additional_retries=0))
    print('V2_PREPARED_NO_SCHEDULER_SUBMISSION', flush=True)


def adopted_complete(binding):
    for row in binding['adopted_segments']:
        directory = c.ROOT/'allocations'/row['segment']
        if list(directory.glob('RANK_*_FAILED.json')):
            raise RuntimeError('Predecessor generation failed; no continuation')
        exit_path = directory/'BASELINE_OWNER_EXIT.json'
        if exit_path.exists() and c.read(exit_path)['allocation_command_exit_code'] != 0:
            raise RuntimeError('Predecessor allocation failed; no continuation')
        account = directory/'ACCOUNTED.json'
        if not account.exists():
            return False
        result = c.read(account)
        if (result['state'], result['exit_code'], result['job']) != ('COMPLETED', '0:0', row['job']):
            raise RuntimeError('Predecessor accounting failed; no continuation')
        success = directory/'BASELINE_OWNER_COMPLETE.json'
        if not success.exists():
            return False
        if not c.read(success)['passed']:
            raise RuntimeError('Predecessor owner verification failed')
    return True


def takeover():
    if not os.environ.get('TMUX'):
        raise RuntimeError('Approved tmux owner required')
    binding = c.verify_sources()
    c.write(c.CAMPAIGN/'HANDOFF_WAITING.json', dict(pid=os.getpid(), adopted=binding['adopted_segments']))
    deadline = time.monotonic()+4*3600
    # The held parent directory blocks new reservations, but existing allocation
    # directories remain writable for completion/accounting. v1 fails closed on
    # its first attempted replacement and drains its other successful owner.
    while time.monotonic() < deadline:
        c.verify_sources()
        finished = adopted_complete(binding)
        stopped = PREVIOUS/'STOPPED.json'
        if finished and stopped.exists():
            reason = c.read(stopped)
            if reason['error'] != 'PermissionError' or str(c.ROOT/'allocations') not in reason['message']:
                raise RuntimeError('Unexpected predecessor stop; manual inspection required')
            with (PREVIOUS/'CONTROLLER.lock').open('a') as prior_lock:
                try:
                    fcntl.flock(prior_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    time.sleep(60)
                    continue
                if (c.ROOT/'allocations').stat().st_mode & 0o777 != 0o500:
                    raise RuntimeError('Reservation hold changed unexpectedly')
                (c.ROOT/'allocations').chmod(0o700)
                c.write(c.CAMPAIGN/'HANDOFF_COMPLETE.json', dict(passed=True, prior_stop=reason,
                        restored_mode='0700', prior_controller_lock_held=True))
                print('V2_HANDOFF_COMPLETE', flush=True)
                c.run()
                return
        time.sleep(60)
    raise RuntimeError('Bounded four-hour handoff expired; reservation hold retained')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'takeover', 'owner', 'compute'])
    parser.add_argument('--segment')
    args = parser.parse_args()
    try:
        if args.action == 'prepare': prepare()
        elif args.action == 'takeover': takeover()
        elif args.action == 'owner': c.owner(args.segment)
        elif args.action == 'compute': c.compute(args.segment)
    except Exception as error:
        if args.action == 'takeover' and not (c.CAMPAIGN/'STOPPED.json').exists():
            c.write(c.CAMPAIGN/'HANDOFF_FAILED.json', dict(error=type(error).__name__, message=str(error)))
        raise
