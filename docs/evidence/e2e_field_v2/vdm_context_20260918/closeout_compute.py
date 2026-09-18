"""One bounded CPU metadata-verification allocation after the scientific report.

The original report allocation ended normally and terminated its waiting
supplemental verifier. No fit, field draw, target read, or metric is repeated.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v2/wide_pipeline_v1/vdm_context_20260917_v1')
FOLDER = ROOT/'analysis/closeout_resources'
PYTHON = '/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python'
HERE = Path(__file__).resolve().parent
FILES = ('case_report_audit.py', 'final_report_audit.py', 'closeout_compute.py')


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def allocation():
    request = read(FOLDER/'REQUEST.json')
    for name, expected in request['source_sha256'].items():
        assert digest(HERE/name) == expected
    job = os.environ['SLURM_JOB_ID']
    publish(ROOT/'resources/CASE_AUDIT_START.json', dict(job=job, gpus=0,
        stage='case_audit', started_unix=time.time(), host=socket.gethostname(),
        reason='JSON-only closeout after original report allocation ended normally',
        source_request=str((FOLDER/'REQUEST.json').relative_to(ROOT))))
    codes = []
    for name in FILES[:2]:
        command = ['srun', '--nodes=1', '--ntasks=1', '--cpus-per-task=1',
                   '--exclusive', '--exact', '--cpu-bind=cores', '--mem=1G',
                   PYTHON, '-u', str(HERE/name)]
        code = subprocess.run(command).returncode
        codes.append(code)
        if code:
            break
    publish(FOLDER/'END.json', dict(job=job, worker_codes=codes, ended_unix=time.time()))
    return 0 if codes == [0, 0] else 1


def launch():
    complete = read(ROOT/'EXPERIMENT_COMPLETE.json')
    assert complete['complete'] and not complete['production_ready']
    assert digest(ROOT/'analysis/RESULTS.json') == complete['results_sha256']
    assert complete['resource_accounting']['cpu_node_hours']+.25 <= 8
    assert not (ROOT/'analysis/CASE_REPORT_AUDIT.json').exists()
    assert not (ROOT/'analysis/CLOSEOUT_AUDIT.json').exists()
    tasks = read(ROOT/'DRAW_LEDGER.json')['tasks']
    assert len(tasks) == 688 and all((ROOT/'analysis/cases'/(t['task_id']+'.json')).is_file() for t in tasks)
    listing = subprocess.check_output(['squeue', '--me', '-h', '-o', '%i|%q'], text=True)
    assert sum('interactive' in row for row in listing.splitlines()) < 2
    FOLDER.mkdir()
    command = ['salloc', '--nodes=1', '--ntasks=1', '--cpus-per-task=1', '--constraint=cpu',
        '--qos=interactive', '--account=desi', '--licenses=scratch', '--immediate=600',
        '--time=00:15:00', '--job-name=vdm-closeout-audit', PYTHON, '-u', str(Path(__file__).resolve()),
        '--mode', 'allocation']
    publish(FOLDER/'REQUEST.json', dict(command=command, maximum_cpu_node_hours=.25,
        prior_complete_sha256=digest(ROOT/'EXPERIMENT_COMPLETE.json'),
        source_sha256={name:digest(HERE/name) for name in FILES},
        scope='Read completed JSON metrics; verify/reconcile and aggregate only',
        previous_step='58524609.4 cancelled on normal parent-allocation completion; no output published',
        requested_unix=time.time(), automatic_retry=False))
    with (FOLDER/'job.log').open('x') as stream:
        code = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT).returncode
    publish(FOLDER/'RETURN.json', dict(exit_code=code, ended_unix=time.time()))
    assert code == 0, 'audit failed; no automatic retry'
    sys.path.insert(0, str(ROOT/'source'))
    from workflows.sbi.e2e_vdm_context_interactive import accounting
    from workflows.sbi.e2e_vdm_context_queue import disk_bytes
    for attempt in range(6):
        try:
            usage = accounting(ROOT)
            break
        except RuntimeError:
            if attempt == 5:
                raise
            time.sleep(10)
    size = disk_bytes(ROOT)
    assert usage['cpu_node_hours'] <= 8 and usage['gpu_hours'] <= 112 and size <= 300*1024**3
    publish(FOLDER/'ACCOUNTING.json', dict(resource_accounting=usage, scratch_bytes=size,
        recorded_unix=time.time(), original_complete_sha256=digest(ROOT/'EXPERIMENT_COMPLETE.json'),
        original_complete_unchanged=True, includes_supplemental_audit=True))
    print('CLOSEOUT_COMPUTE_COMPLETE', usage['cpu_node_hours'], usage['gpu_hours'], size, flush=True)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', required=True, choices=('launch', 'allocation'))
    args = parser.parse_args()
    raise SystemExit(launch() if args.mode == 'launch' else allocation())
