"""User-authorized finite four-GPU interactive assessment chain, not an agent.

Two 75-minute allocations maximum; only clean chunk-boundary exit 75 resumes.
Unexpected failures and unavailable allocations stop, without batch fallback.
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

PYTHON = '/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python'
BASE = Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v2/wide_pipeline_v1')
BRANCHES = ('fixed_vlb_seed0', 'learned_vlb_seed0', 'fixed_vlb_seed1', 'learned_vlb_seed1')
SEGMENTS = 2


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def publish(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())


def verify(root):
    if root.parent != BASE or not root.name.startswith('vdm_assessment_'):
        raise ValueError('registered assessment root required')
    digest = sha(root/'MANIFEST.json')
    manifest = json.loads((root/'MANIFEST.json').read_text())
    for name, expected in manifest['source_sha256'].items():
        if sha(root/'source'/name) != expected:
            raise ValueError('frozen source drift: '+name)
    for name in ('SMOKE.json', 'RESTART_TEST.json'):
        proof = json.loads((root/name).read_text())
        if not proof['passed'] or proof['manifest_sha256'] != digest:
            raise ValueError('matching technical gate missing: '+name)
    return digest


def clean_env():
    env = os.environ.copy()
    for key in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONUSERBASE', 'LD_PRELOAD'):
        env.pop(key, None)
    env.update(PYTHONNOUSERSITE='1', OMP_NUM_THREADS='4',
               CUBLAS_WORKSPACE_CONFIG=':4096:8', SLURM_EXPORT_ENV='ALL')
    return env


def allocation_command(root, launcher, segment):
    return ['salloc', '--nodes=1', '--ntasks=4', '--cpus-per-task=32',
            '--constraint=gpu', '--gpus=4', '--qos=interactive',
            '--time=01:15:00', '--account=desi_g', '--licenses=scratch',
            '--immediate=600', '--job-name=vdm-assess-4gpu',
            PYTHON, str(launcher), '--root', str(root), '--mode=allocation',
            '--segment', str(segment)]


def worker_command(root, branch):
    if branch not in BRANCHES:
        raise ValueError('unknown branch')
    return ['srun', '--nodes=1', '--ntasks=1', '--cpus-per-task=32',
            '--gpus=1', '--exclusive', '--exact', '--cpu-bind=cores',
            '--chdir='+str(root/'source'),
            'timeout', '--preserve-status', '--signal=USR1', '--kill-after=120',
            '4200', PYTHON, '-u', '-m', 'workflows.sbi.e2e_vdm_assessment',
            'run', '--root', str(root), '--branch', branch, '--panel=all']


def disposition(codes):
    if len(codes) != 4:
        raise ValueError('four worker statuses required')
    if any(code not in (0, 75) for code in codes):
        return 1
    return 75 if 75 in codes else 0


def allocation(root, segment):
    digest = verify(root)
    intent = json.loads((root/'INTERACTIVE_INTENT.json').read_text())
    if sha(__file__) != intent['launcher_sha256'] or digest != intent['manifest_sha256']:
        raise ValueError('launcher/manifest drift')
    if not os.environ.get('SLURM_JOB_ID') or segment not in range(SEGMENTS):
        raise ValueError('authorized allocation and segment required')
    logs = root/'logs'
    job = os.environ['SLURM_JOB_ID']
    publish(root/f'INTERACTIVE_SEGMENT_{segment}_START.json',
            dict(job=job, segment=segment, manifest_sha256=digest,
                 launcher_sha256=sha(__file__), started_unix=time.time(),
                 workers={b:worker_command(root,b) for b in BRANCHES}))
    processes, handles = [], []
    try:
        for branch in BRANCHES:
            handle = (logs/f'interactive_{segment}_{job}_{branch}.log').open('x')
            handles.append(handle)
            processes.append(subprocess.Popen(worker_command(root, branch),
                             env=clean_env(), stdout=handle, stderr=subprocess.STDOUT))
        codes = [process.wait() for process in processes]
    finally:
        for handle in handles:
            handle.close()
    status = disposition(codes)
    publish(root/f'INTERACTIVE_SEGMENT_{segment}_END.json',
            dict(job=job, worker_codes=dict(zip(BRANCHES,codes)), status=status,
                 ended_unix=time.time(), manifest_sha256=digest))
    if status:
        return status
    # All four branch completion receipts are required before aggregation.
    for branch in BRANCHES:
        proof = json.loads((root/branch/'ALL_COMPLETE.json').read_text())
        if not proof['complete'] or proof['manifest_sha256'] != digest:
            raise ValueError('branch completion drift')
    cmd = ['srun', '--nodes=1', '--ntasks=1', '--cpus-per-task=32', '--gpus=1',
           '--exclusive', '--exact', '--cpu-bind=cores', '--chdir='+str(root/'source'),
           'timeout', '240', PYTHON, '-u', '-m',
           'workflows.sbi.e2e_vdm_assessment_report', '--root', str(root)]
    with (logs/f'report_{segment}_{job}.log').open('x') as handle:
        result = subprocess.run(cmd, env=clean_env(), stdout=handle, stderr=subprocess.STDOUT)
    publish(root/'INTERACTIVE_REPORT.json',dict(job=job, returncode=result.returncode,
            command=cmd, manifest_sha256=digest))
    return result.returncode


def controller(root):
    digest = verify(root)
    intent = json.loads((root/'INTERACTIVE_INTENT.json').read_text())
    if sha(__file__) != intent['launcher_sha256'] or digest != intent['manifest_sha256']:
        raise ValueError('launcher/manifest drift')
    # Exclusive marker rejects duplicate launchers. Never automatically remove it.
    publish(root/'INTERACTIVE_CONTROLLER.json',dict(pid=os.getpid(), started_unix=time.time()))
    for segment in range(SEGMENTS):
        # Count both pending and running allocations before EACH request.
        result = subprocess.run(['squeue','--noheader','--user','dkololgi',
                                 '--states=PENDING,RUNNING,CONFIGURING,COMPLETING',
                                 '--format=%q'],capture_output=True,text=True,check=True)
        active = sum('interactive' in line.lower() for line in result.stdout.splitlines())
        if active >= 2:
            raise RuntimeError('two-allocation limit reached; stopping without submitting')
        cmd = allocation_command(root, Path(__file__).resolve(), segment)
        publish(root/f'INTERACTIVE_REQUEST_{segment}.json',dict(command=cmd,
                active_interactive_allocations=active, manifest_sha256=digest))
        print('REQUEST',segment,' '.join(cmd),flush=True)
        result = subprocess.run(cmd,env=clean_env())
        publish(root/f'INTERACTIVE_RETURN_{segment}.json',dict(returncode=result.returncode))
        # salloc propagates the command status; a matching clean-pause receipt
        # must also exist. Queue failure, signals, crashes and timeouts do NOT retry.
        if result.returncode == 0:
            proof = json.loads((root/'INTERACTIVE_REPORT.json').read_text())
            if proof['returncode'] != 0 or not (root/'analysis/RESULTS.json').exists():
                raise RuntimeError('missing scientific report')
            publish(root/'INTERACTIVE_COMPLETE.json',dict(complete=True,
                    manifest_sha256=digest, report_sha256=sha(root/'analysis/RESULTS.json')))
            print('ASSESSMENT COMPLETE',root/'analysis/RESULTS.json',flush=True)
            return 0
        end = root/f'INTERACTIVE_SEGMENT_{segment}_END.json'
        if result.returncode != 75 or not end.exists() or json.loads(end.read_text())['status'] != 75:
            raise RuntimeError('unexpected allocation/worker failure; no retry')
        print('CLEAN PAUSE; resume only committed-safe work',flush=True)
    print('AUTHORIZED BUDGET EXHAUSTED; receipts retained; no further requests',flush=True)
    return 75


def stage(root):
    digest = verify(root)
    target = root/'interactive_launcher.py'
    # Exclusive creation prevents replacement of a staged launch contract.
    with Path(__file__).open('rb') as source, target.open('xb') as dest:
        shutil.copyfileobj(source,dest)
    publish(root/'INTERACTIVE_INTENT.json',dict(manifest_sha256=digest,
            launcher_sha256=sha(target), segments=SEGMENTS, minutes_per_segment=75,
            gpus_per_segment=4, maximum_gpu_hours=10, branch_order=BRANCHES,
            authorization='User requested chained interactive jobs and all four node GPUs',
            no_training=True, unexpected_failure_policy='stop, do not retry'))
    print(target,flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--mode',choices=['stage','controller','allocation'],required=True)
    parser.add_argument('--segment',type=int,default=0)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.mode == 'stage':
        stage(root)
    else:
        sys.exit(controller(root) if args.mode == 'controller' else allocation(root,args.segment))
