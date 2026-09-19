"""Freeze small source bundles and reconcile approved preparation resources."""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess

from workflows.sbi import e2e_coupled_contract as c


def freeze(name):
    if not name.replace('_', '').isalnum():
        raise ValueError('simple snapshot name required')
    c.bind_run()
    destination = c.ROOT/'source_snapshots'/name
    destination.mkdir(parents=True, exist_ok=False)
    sources = [c.CONFIG, c.REPO/'configs/p10_phase_registry_v1.json',
               c.REPO/'workflows/abacus_tweb/p10_build_bright_parent.py',
               c.REPO/'workflows/abacus_tweb/p10_phase_assets.py']
    sources += sorted((c.REPO/'configs').glob('e2e_coupled*.json'))
    sources += [c.REPO/'configs/data'/name for name in
                ('desi_distance_701f498.dat','README.md','cosmoprimo-LICENSE')]
    sources += sorted((c.REPO/'workflows/sbi').glob('e2e_coupled*.sh'))
    sources += sorted((c.REPO/'workflows/sbi').glob('e2e_coupled*.slurm'))
    # Freeze Python dependencies as well as entrypoints. Source only, no arrays,
    # credentials or data trees; avoids accidental live imports during long jobs.
    for directory in ('workflows','shared'):
        sources += sorted((c.REPO/directory).rglob('*.py'))
    sources += sorted(c.REPO.glob('*.py'))
    sources = sorted(set(sources))
    records = []
    for path in sources:
        relative = path.relative_to(c.REPO)
        target = destination/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        records.append(dict(relative=str(relative), sha256=c.sha256(target)))
    c.atomic_json(destination/'SOURCE.json', dict(**c.provenance(), files=records))
    return str(destination)


def accounting():
    jobs = c.ROOT/'ops/jobs.json'
    registry = json.loads(jobs.read_text()) if jobs.exists() else []
    total = {'cpu_node_hours': 0.0, 'gpu_hours': 0.0}
    records = []
    for registered in registry:
        result = subprocess.run(['sacct','-X','-n','-P','-j',str(registered['job_id']),
                                 '-o','JobIDRaw,State,ElapsedRaw,AllocNodes,AllocTRES'],
                                check=True, text=True, capture_output=True)
        row = next((line.split('|') for line in result.stdout.splitlines()
                    if line.split('|')[0] == str(registered['job_id'])), None)
        if row is None:
            raise ValueError('registered allocation missing from accounting')
        hours = int(row[2])*int(row[3])/3600
        total['cpu_node_hours'] += hours if registered['kind'] == 'cpu' else 0
        total['gpu_hours'] += int(registered.get('gpus', 0))*int(row[2])/3600
        records.append(dict(**registered, state=row[1], elapsed_seconds=int(row[2])))
    return dict(allocations=records, **total)


def register(job_id, kind, hours, gpus):
    directory = c.ROOT/'ops'
    with c.single_writer(directory):
        path = directory/'jobs.json'
        jobs = json.loads(path.read_text()) if path.exists() else []
        if any(str(row['job_id']) == str(job_id) for row in jobs):
            raise ValueError('already registered allocation')
        jobs.append(dict(job_id=str(job_id), kind=kind, hours_cap=hours, gpus=gpus,
                         **c.provenance()))
        c.atomic_json(path, jobs, replace=path.exists())
    return accounting()


def validate_step_budget(info, memory_gib, cpus):
    """Check aggregate simultaneous steps against allocated, not node, RAM."""
    fields = dict(value.split('=', 1) for value in info.split() if '=' in value)
    if fields.get('JobState') != 'RUNNING' or fields.get('NumNodes') != '1':
        raise ValueError('one running node required for the packed preparation steps')
    tres = dict(value.split('=', 1) for value in fields['AllocTRES'].split(','))
    match = re.fullmatch(r'(\d+(?:\.\d+)?)([KMGT]?)', tres['mem'])
    if not match or memory_gib <= 0 or cpus <= 0:
        raise ValueError('invalid step/allocation memory or CPU count')
    scale = {'K': 1 / 1024, '': 1, 'M': 1, 'G': 1024, 'T': 1024**2}
    allocated_mib = float(match[1]) * scale[match[2]]
    if memory_gib * 1024 > allocated_mib or cpus > int(tres['cpu']):
        raise RuntimeError('aggregate steps exceed the actual Slurm allocation')
    return dict(allocated_mib=allocated_mib, requested_step_mib=memory_gib * 1024,
                allocated_cpus=int(tres['cpu']), requested_step_cpus=cpus)


def step_budget(job_id, memory_gib, cpus):
    info = subprocess.run(['scontrol', 'show', 'job', str(job_id), '-o'],
                          check=True, text=True, capture_output=True).stdout
    return validate_step_budget(info, memory_gib, cpus)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['freeze','register','accounting','step-budget'])
    p.add_argument('--name')
    p.add_argument('--job-id')
    p.add_argument('--kind', choices=['cpu','gpu','xfer'], default='cpu')
    p.add_argument('--hours', type=float, default=1)
    p.add_argument('--gpus', type=int, default=0)
    p.add_argument('--memory-gib', type=float)
    p.add_argument('--cpus', type=int)
    a = p.parse_args()
    result = step_budget(a.job_id, a.memory_gib, a.cpus) if a.action == 'step-budget' else (
        freeze(a.name) if a.action == 'freeze' else (
        register(a.job_id, a.kind, a.hours, a.gpus) if a.action == 'register' else accounting())
    )
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
