"""Bounded early execution of registered density metrics; never a final release.

Imports scientific functions from the immutable run snapshot, not the live repo.
This supplemental CPU job is accounted separately from the frozen controller;
its cost must be added at final closeout. No model, draw, or final report writes.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time

ROOT = Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v2/wide_pipeline_v1/vdm_context_20260917_v1')
PYTHON = '/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python'
FOLDER = ROOT/'analysis/density_preview'


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


def selected():
    tasks = [t for t in read(ROOT/'DRAW_LEDGER.json')['tasks']
             if t['arm'] in 'ABC' and t['purpose'] == 'main' and t['steps'] == 250]
    cells = {}
    for task in tasks:
        phase = task['anchor'].split('_')[0]
        key = (task['arm'], task['replica'], task['checkpoint'], phase)
        cells.setdefault(key, []).append(task)
        done = read(ROOT/'draws'/task['task_id']/'COMPLETE.json')
        if done['task'] != task or done['manifest_sha256'] != digest(ROOT/'MANIFEST.json'):
            raise ValueError('incomplete or drifted preview input')
    expected = {(a, s, k, p) for a in 'ABC' for s in (0, 1)
                for k, p in ((5120, 'ph004'), (10240, 'ph004'),
                             (20480, 'ph004'), (20480, 'ph005'))}
    if set(cells) != expected or len(tasks) != 384 or any(len(v) != 16 for v in cells.values()):
        raise ValueError('preview must include the full registered A/B/C main panel')
    return tasks


def init_worker():
    global DATA
    sys.path.insert(0, str(ROOT/'source'))
    from workflows.sbi.e2e_vdm_context_dataset import Products
    DATA = Products(ROOT, ['ph004', 'ph005'], targets=True,
                    confirmation_receipt=ROOT/'MODELS_FROZEN.json', verify=False)


def case(task):
    import numpy as np
    from workflows.sbi.e2e_vdm_context_analysis import ensemble, density_spectra, CORE
    from workflows.sbi.e2e_vdm_context_metrics import calibration
    fields, receipts = ensemble(ROOT, task)  # Includes all actual array hashes.
    truth = DATA.raw_targets(task['anchor'])['rho']-1
    spectra, _ = density_spectra(fields, truth)
    values = fields[(slice(None), *CORE)].reshape(len(fields), -1)
    c = calibration(values, truth[CORE].reshape(-1))
    scale = DATA.chart['physical_probe_scales']['std'][0]
    record = dict(task=task, phase=task['anchor'].split('_')[0], draw_receipts=receipts,
                  density_crps=float(c['crps'].mean()/scale),
                  density_rmse=float(np.sqrt(np.mean(c['bias']**2))),
                  density_bias=float(c['bias'].mean()),
                  rms_spread=float(np.sqrt(np.mean(c['std']**2))),
                  coverage={k:float(c['covered_'+k].mean()) for k in c['attainable']},
                  width={k:float(c['width_'+k].mean()) for k in c['attainable']},
                  attainable=c['attainable'], spectra=spectra)
    publish(FOLDER/'cases'/(task['task_id']+'.json'), record)
    print('PREVIEW_CASE', task['task_id'], flush=True)
    return record


def compute():
    sys.path.insert(0, str(ROOT/'source'))
    from workflows.sbi.e2e_vdm_context_train import verify_launch
    from workflows.sbi.e2e_vdm_context_control import verify_models_frozen
    from workflows.sbi.e2e_vdm_context_dataset import Products
    from workflows.sbi.e2e_field_build_products import require_compute
    import numpy as np
    require_compute()
    publish(FOLDER/'START.json', dict(job=os.environ['SLURM_JOB_ID'],
                                    node=socket.gethostname(), started_unix=time.time()))
    request = read(FOLDER/'REQUEST.json')
    if digest(__file__) != request['wrapper_sha256']:
        raise ValueError('preview wrapper drift')
    verify_launch(ROOT)
    verify_models_frozen(ROOT)
    if not read(ROOT/'analysis/SAMPLER_GATE.json')['passed']:
        raise ValueError('sampler gate required')
    tasks = selected()
    if tasks != request['tasks']:
        raise ValueError('preview task selection drift')
    data = Products(ROOT, ['ph004', 'ph005'], targets=True,
                    confirmation_receipt=ROOT/'MODELS_FROZEN.json')  # Verify raw product payloads.
    (FOLDER/'cases').mkdir()
    # Compare one predetermined case with the complete frozen report calculation.
    # Suppress its case-file write; final report outputs remain untouched.
    from unittest.mock import patch
    from workflows.sbi.e2e_vdm_context_report import summarize_case
    init_worker()
    first = case(tasks[0])
    with patch('workflows.sbi.e2e_vdm_context_report.durable.publish_json'):
        reference = summarize_case(ROOT, tasks[0], data)
    density = reference['closures']['periodic']['physical']['all']
    np.testing.assert_allclose([first['density_crps'], first['density_rmse'], first['density_bias'], first['rms_spread']],
        [density['standardized_crps'][0], density['rmse_mean'][0], density['bias'][0], density['rms_spread'][0]],
        rtol=1e-10, atol=1e-12)
    for level in first['coverage']:
        np.testing.assert_allclose([first['coverage'][level], first['width'][level], first['attainable'][level]],
            [density['coverage'][level][0], density['width'][level][0], density['attainable_coverage'][level]],
            rtol=1e-10, atol=1e-12)
    if first['spectra'] != reference['spectra']:
        raise ValueError('preview/frozen report spectral parity failure')
    publish(FOLDER/'PARITY.json', dict(passed=True, task=tasks[0], rtol=1e-10, atol=1e-12,
        reference_source_sha256=digest(ROOT/'source/workflows/sbi/e2e_vdm_context_report.py')))
    with ProcessPoolExecutor(max_workers=8, initializer=init_worker) as pool:
        records = [first, *pool.map(case, tasks[1:])]
    grouped = {}
    for row in records:
        t = row['task']
        grouped.setdefault((t['arm'], t['replica'], t['checkpoint'], row['phase']), []).append(row)
    progression = []
    for (arm, seed, checkpoint, phase), rows in sorted(grouped.items()):
        avg = lambda name:float(np.mean([r[name] for r in rows]))
        sample = np.mean([r['spectra']['mean_sample_power'] for r in rows], axis=0)
        truth = np.mean([r['spectra']['truth_power'] for r in rows], axis=0)
        progression.append(dict(arm=arm, replica=seed, checkpoint=checkpoint, phase=phase, anchors=len(rows),
            density_crps=avg('density_crps'), density_bias=avg('density_bias'),
            density_rmse=float(np.sqrt(np.mean([r['density_rmse']**2 for r in rows]))),
            rms_spread=float(np.sqrt(np.mean([r['rms_spread']**2 for r in rows]))),
            coverage={k:float(np.mean([r['coverage'][k] for r in rows])) for k in rows[0]['coverage']},
            attainable={k:float(np.mean([r['attainable'][k] for r in rows])) for k in rows[0]['attainable']},
            width={k:float(np.mean([r['width'][k] for r in rows])) for k in rows[0]['width']},
            sample_power_ratio=(sample/np.maximum(truth,1e-30)).tolist(),
            power_discrepancy=float(np.mean(np.abs(np.log(np.maximum(sample,1e-30)/np.maximum(truth,1e-30))))),
            mean_field_power_ratio=(np.mean([r['spectra']['posterior_mean_power'] for r in rows],axis=0)/np.maximum(truth,1e-30)).tolist(),
            mean_field_correlation=np.mean([r['spectra']['correlation_posterior_mean'] for r in rows],axis=0).tolist()))
    publish(FOLDER/'PRELIMINARY.json', dict(preliminary=True, final_decision=False,
        scope='All registered A/B/C main density panels; D, tides, pairs and final decisions remain pending',
        manifest_sha256=digest(ROOT/'MANIFEST.json'), models_sha256=digest(ROOT/'MODELS_FROZEN.json'),
        sampler_sha256=digest(ROOT/'analysis/SAMPLER_GATE.json'), wrapper_sha256=digest(__file__),
        case_receipts={t['task_id']:digest(FOLDER/'cases'/(t['task_id']+'.json')) for t in tasks},
        progression=progression, cases=len(tasks), independent_evaluation_phases=2,
        caveat='Equal-anchor summaries; correlated voxels and only two programme-exposed evaluation phases'))
    print('PREVIEW_COMPLETE', len(tasks), flush=True)


def launch():
    tasks = selected()
    if (ROOT/'analysis/RESULTS.json').exists():
        raise RuntimeError('full report already available; preview unnecessary')
    usage = read(ROOT/'resources/07_ACCOUNTING.json')
    # Hard upper bound: reserve this 0.5 CPU-nodeh plus the sole remaining 2h report.
    if usage['cpu_node_hours']+0.5+2 > 8:
        raise RuntimeError('CPU preview plus final report would exceed original cap')
    if time.time()+1800 >= read(ROOT/'MANIFEST.json')['deadline_epoch']:
        raise RuntimeError('preview would exceed original elapsed deadline')
    listing = subprocess.check_output(['squeue','--me','-h','-o','%i|%q'],text=True)
    if sum('interactive' in line for line in listing.splitlines()) >= 2:
        raise RuntimeError('two interactive allocations already active')
    cmd = ['salloc','--nodes=1','--ntasks=1','--cpus-per-task=64','--constraint=cpu',
           '--qos=interactive','--account=desi','--licenses=scratch','--immediate=600',
           '--time=00:30:00','--job-name=vdm-density-preview',
           'srun','--nodes=1','--ntasks=1','--cpus-per-task=64','--cpu-bind=cores',
           PYTHON,'-u',str(Path(__file__).resolve()),'compute']
    FOLDER.mkdir(exist_ok=False)
    publish(FOLDER/'REQUEST.json',dict(command=cmd,tasks=tasks,wrapper_sha256=digest(__file__),
        max_cpu_node_hours=0.5,original_cpu_cap=8,prior_cpu_node_hours=usage['cpu_node_hours'],
        final_report_cpu_reserved=2,manifest_sha256=digest(ROOT/'MANIFEST.json'),
        accounting='Supplemental job; add actual sacct time to final controller accounting',
        no_retry=True,no_training=True))
    env = dict(os.environ)
    for key in ('PYTHONPATH','PYTHONHOME','PYTHONUSERBASE','LD_PRELOAD'):
        env.pop(key,None)
    env.update(PYTHONNOUSERSITE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    with (FOLDER/'job.log').open('x') as log:
        status = subprocess.run(cmd,cwd=ROOT/'source',env=env,stdout=log,stderr=subprocess.STDOUT).returncode
    publish(FOLDER/'RETURN.json',dict(exit_code=status,ended_unix=time.time()))
    jobs = re.findall(r'Granted job allocation (\d+)', (FOLDER/'job.log').read_text())
    if jobs:
        if len(set(jobs)) != 1:
            raise ValueError('preview must use exactly one CPU allocation')
        job = jobs[0]
        if (FOLDER/'START.json').exists() and read(FOLDER/'START.json')['job'] != job:
            raise ValueError('preview allocation binding mismatch')
        for attempt in range(6):
            text = subprocess.check_output(['sacct','-X','-n','-P','-j',job,
                '--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocTRES'],text=True)
            rows = [line.split('|') for line in text.splitlines() if line.startswith(job+'|')]
            if rows and rows[0][1].split()[0] in {'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL'}:
                row = rows[0]
                publish(FOLDER/'ACCOUNTING.json',dict(job=job,state=row[1],exit_code=row[2],
                    elapsed_seconds=int(row[3]),cpu_node_hours=int(row[3])/3600,gpu_hours=0))
                break
            if attempt<5:time.sleep(10)
        else:
            raise RuntimeError('supplemental terminal accounting unavailable; manual reconciliation required')
    raise SystemExit(status)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['plan','launch','compute'])
    mode = parser.parse_args().mode
    if mode == 'plan':
        print(json.dumps(dict(cases=len(selected()),max_cpu_node_hours=0.5,wrapper_sha256=digest(__file__))))
    elif mode == 'launch':
        launch()
    else:
        compute()
