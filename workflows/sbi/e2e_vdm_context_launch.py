"""Finite launch/staging helpers for the approved context/diversity experiment.

No agent loop, scientific decisions, automatic resource expansion, or batch fallback.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

REPO = Path(__file__).resolve().parents[2]
SCRATCH = Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v2/wide_pipeline_v1')
PYTHON = '/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def publish(path, value):
    # Exclusive small metadata receipt, fsynced; no mutable completion claims.
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def check_root(root):
    root = Path(root).resolve()
    if root.parent != SCRATCH or not root.name.startswith('vdm_context_'):
        raise ValueError('separate vdm_context_ Scratch child required')
    return root


def stage(root, build=False):
    root = check_root(root)
    names = subprocess.check_output(['git', 'ls-files', '-z'], cwd=REPO).decode().split('\0')
    names = [n for n in names if n.startswith(('workflows/', 'shared/', 'configs/'))
             and n.endswith(('.py', '.json', '.sh', '.slurm'))]
    names += ['workflows/sbi/e2e_vdm_context_data.py', 'workflows/sbi/e2e_vdm_context_launch.py',
              'configs/e2e_vdm_context_diversity_v1.json', 'docs/e2e_vdm_context_diversity_v1.md']
    if build:
        names += ['workflows/sbi/e2e_vdm_context_products.py', 'workflows/sbi/e2e_vdm_context_models.py',
                  'workflows/sbi/e2e_vdm_context_metrics.py', 'tests/test_e2e_vdm_context.py']
    names = sorted(set(names))
    if not build:
        root.mkdir(exist_ok=False)
        (root/'logs').mkdir()
    elif not (root/'SCREEN_RETURN.json').is_file():
        raise ValueError('screen return required before product staging')
    source = root/('source_products' if build else 'source_geometry')
    source.mkdir()
    hashes = {}
    for name in names:
        src, dst = REPO/name, source/name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        hashes[name] = digest(dst)
        if digest(src) != hashes[name]:
            raise ValueError('source changed during snapshot')
    publish(root/('BUILD_SOURCE.json' if build else 'GEOMETRY_SOURCE.json'), dict(source=str(source), source_sha256=hashes,
        git_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        status=subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True),
        created=datetime.now(timezone.utc).isoformat(), purpose='data only; no training authority from stage'))
    print('STAGED', source, flush=True)


def cpu_screen(root, build=False):
    root = check_root(root)
    receipt = json.loads((root/('BUILD_SOURCE.json' if build else 'GEOMETRY_SOURCE.json')).read_text())
    source = Path(receipt['source'])
    if source.resolve() != REPO:
        raise ValueError('launch from frozen geometry snapshot')
    for name, expected in receipt['source_sha256'].items():
        if digest(source/name) != expected:
            raise ValueError('source drift: '+name)
    # User-wide limit includes all running and pending interactive allocations.
    listing = subprocess.check_output(['squeue', '--me', '-h', '-o', '%i|%q'], text=True)
    if sum('interactive' in row for row in listing.splitlines()) >= 2:
        raise RuntimeError('two interactive allocations already pending/running')
    cmd = ['salloc', '--nodes=1', '--ntasks=1', '--cpus-per-task=64', '--constraint=cpu',
           '--qos=interactive', '--account=desi', '--time=02:00:00' if build else '--time=01:00:00', '--licenses=scratch',
           '--immediate=600', '--job-name=vdm-context-products' if build else '--job-name=vdm-context-geometry',
           'srun', '--nodes=1', '--ntasks=1', '--cpus-per-task=64', '--cpu-bind=cores',
           PYTHON, '-u', '-m']
    cmd += (['workflows.sbi.e2e_vdm_context_products', '--root', str(root), '--phases', 'ph000','ph002'] if build
            else ['workflows.sbi.e2e_vdm_context_data', 'screen', '--root', str(root)])
    now = time.time()
    deadline = (json.loads((root/'SCREEN_REQUEST.json').read_text())['deadline_epoch'] if build else now+48*3600)
    if now >= deadline:
        raise RuntimeError('experiment elapsed-time ceiling reached')
    publish(root/('BUILD_REQUEST.json' if build else 'SCREEN_REQUEST.json'), dict(command=cmd, epoch=now,
        deadline_epoch=deadline, cpu_node_hours_limit=2 if build else 1, experiment_cpu_node_hours_limit=8,
        explicit_user_authorization='approved context/diversity goal 2026-09-17'))
    env = dict(os.environ)
    for key in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONUSERBASE', 'LD_PRELOAD'):
        env.pop(key, None)
    env.update(PYTHONNOUSERSITE='1', OMP_NUM_THREADS='1')
    with (root/('logs/products.log' if build else 'logs/geometry.log')).open('x') as log:
        status = subprocess.run(cmd, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    publish(root/('BUILD_RETURN.json' if build else 'SCREEN_RETURN.json'), dict(exit_code=status, elapsed_seconds=time.time()-now,
                                          no_automatic_retry=True))
    raise SystemExit(status)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['stage', 'cpu-screen', 'stage-build', 'cpu-build'])
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    (stage if args.mode.startswith('stage') else cpu_screen)(args.root, build='build' in args.mode)


if __name__ == '__main__':
    main()
