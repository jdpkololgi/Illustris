"""One-off bounded lifetime repair for the original58544227 launcher shell.

Only the verified, user-owned allocation wrapper is temporarily stopped. Its
salloc process, Slurm allocation and all compute workers keep running. Resume
it after its externally attached workers terminate, on any ordinary error, or
after25minutes. No job cancellation, extension, new allocation or worker signal.
The replacement product wrapper owns/waits for its children and needs no guard.
"""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

from workflows.sbi import e2e_coupled_contract as c

JOB = '58544227'
PID = 91332
SNAPSHOT = c.ROOT / 'source_snapshots/post_prepare_v1'
SCRIPT = SNAPSHOT / 'workflows/sbi/e2e_coupled_post_step.sh'
STEPS = {JOB + suffix for suffix in ('.8', '.9', '.10')}
TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED'}


def identity(pid):
    root = Path('/proc') / str(pid)
    if root.stat().st_uid != os.getuid():
        raise PermissionError('launcher process belongs to another user')
    stat = (root / 'stat').read_text().rsplit(') ', 1)[1].split()
    argv = (root / 'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
    return dict(pid=pid, ppid=int(stat[1]), start_ticks=int(stat[19]), argv=argv)


def validate_launcher(record, parent):
    if record['pid'] != PID or record['argv'] != ['/bin/bash', str(SCRIPT), str(SNAPSHOT)]:
        raise PermissionError('not the exact original preparation wrapper')
    if (parent['pid'] != record['ppid'] or Path(parent['argv'][0]).name != 'salloc'
            or '--job-name=coupled-products-01' not in parent['argv']):
        raise PermissionError('wrapper is not owned by the expected allocation launcher')


def step_states():
    output = subprocess.run(['sacct', '-n', '-P', '-j', JOB,
                             '-o', 'JobIDRaw,State,ExitCode'],
                            check=True, text=True, capture_output=True).stdout
    rows = {parts[0]: parts[1:] for line in output.splitlines()
            if len(parts := line.split('|')) >= 3 and parts[0] in STEPS}
    if set(rows) != STEPS:
        raise RuntimeError('attached step accounting is incomplete')
    return rows


def all_terminal(states):
    return set(states) == STEPS and all(row[0].split()[0] in TERMINAL for row in states.values())


def hold(record, seconds=1500):
    if not 0 < seconds <= 1500:
        raise ValueError('one-off guard is capped at25minutes')
    stopped = False
    try:
        if identity(PID) != record:
            raise RuntimeError('launcher identity changed before stop')
        # Mark before the syscall so an asynchronous interruption immediately
        # after SIGSTOP still enters the matching resume path.
        stopped = True
        os.kill(PID, signal.SIGSTOP)
        print(json.dumps(dict(event='wrapper_stopped_workers_continue', launcher=record)), flush=True)
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            states = step_states()
            print(json.dumps(dict(attached_steps=states)), flush=True)
            if all_terminal(states):
                return
            time.sleep(min(60, max(0, deadline-time.monotonic())))
        print(json.dumps(dict(event='bounded_wait_expired_resuming_wrapper')), flush=True)
    finally:
        if stopped:
            try:
                current = identity(PID)
            except FileNotFoundError:
                current = None
            if current == record:
                os.kill(PID, signal.SIGCONT)
                print(json.dumps(dict(event='wrapper_resumed', pid=PID)), flush=True)


def run():
    if socket.gethostname().split('.')[0] not in ('login30', 'x3115c0s21b0n0'):
        raise RuntimeError('this one-off repair must run on login30')
    record = identity(PID); validate_launcher(record, identity(record['ppid']))
    states = step_states()
    if all_terminal(states):
        print(json.dumps(dict(event='workers_already_terminal_no_signal')), flush=True)
        return
    def interrupted(signum, frame):
        raise SystemExit('guard interrupted; resuming wrapper')
    for signum in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(signum, interrupted)
    with c.single_writer(c.ROOT / 'ops/products_01_lifetime_guard'):
        hold(record)


if __name__ == '__main__':
    run()
