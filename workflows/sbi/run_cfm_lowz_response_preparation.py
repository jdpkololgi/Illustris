"""Bounded full/OOF selection and full-response preparation; no readiness promotion."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time

from workflows.sbi import cfm_lowz_selection_response as lowz
from workflows.sbi.prepare_cfm_lowz_inputs import require_compute, save_json, sha256


def stages(policy):
    phases = policy['training'] + policy['development_and_calibration']
    return phases, [None] + policy['training']


def run(plan_path=lowz.DEFAULT_PLAN, seconds=5400, workers=4):
    require_compute()
    if not 0 < seconds <= 6900 or not 1 <= workers <= 4:
        raise ValueError('Outside bounded preparation resources')
    plan, policy, _ = lowz.load_plan(plan_path)
    phases, omissions = stages(policy)
    root = Path(plan['output_root'])
    logs = root / ('run_' + os.environ['SLURM_JOB_ID'])
    logs.mkdir(parents=True, exist_ok=False)
    bindings = lowz.code_bindings()
    binding = dict(plan_sha256=sha256(plan_path), code_bindings=bindings,
                   runner_sha256=sha256(__file__), phases=phases,
                   job=os.environ['SLURM_JOB_ID'], host=socket.gethostname(),
                   seconds=seconds, workers=workers, model_inputs_ready=False)
    save_json(logs / 'RUN_BINDING.json', binding)
    deadline = time.monotonic() + seconds
    stop = threading.Event()

    def command(name, arguments):
        if stop.is_set():
            raise RuntimeError('Stopped after another worker failed; no retries')
        left = deadline - time.monotonic()
        if left < 60:
            raise TimeoutError('Preparation budget reserve reached')
        argv = [sys.executable, '-u', '-m', 'workflows.sbi.cfm_lowz_selection_response',
                '--plan', str(plan_path)] + arguments
        try:
            with (logs / (name + '.log')).open('x') as handle:
                subprocess.run(argv, stdout=handle, stderr=subprocess.STDOUT,
                               check=True, timeout=left, cwd=lowz.REPO)
        except Exception:
            stop.set()
            raise
        print(json.dumps(dict(stage=name, pass_stage=True)), flush=True)

    def parallel(items):
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(command, name, args) for name, args in items]
            for future in futures:
                future.result()

    try:
        pending, reused = [], {}
        for phase in phases:
            path = root / 'statistics_r1' / phase / 'STATISTICS.json'
            if path.exists():
                row = lowz.read_json(path)
                if (row['phase'] != phase or row['plan_sha256'] != binding['plan_sha256']
                        or row['code_bindings'] != bindings or row['refinement'] != 1
                        or row['schema'] != 'cfm-lowz-selection-statistics-v1'):
                    raise ValueError('Existing statistics binding differs: ' + phase)
                reused[phase] = dict(path=str(path), sha256=sha256(path))
            else:
                pending.append(('statistics_' + phase, ['statistics', '--phase', phase]))
        parallel(pending)
        for omitted in omissions:
            command('fit_' + (omitted or 'full'),
                    ['fit'] + ([] if omitted is None else ['--omitted-phase', omitted]))
        selection = root / 'full/SELECTION.json'
        parallel([('response_' + phase, ['response', '--phase', phase, '--selection', str(selection)])
                  for phase in phases])
        if (sha256(plan_path) != binding['plan_sha256'] or lowz.code_bindings() != bindings
                or sha256(__file__) != binding['runner_sha256']):
            raise ValueError('Code or plan changed during preparation')
        outputs = {}
        for phase in phases:
            for kind, name in [('statistics_r1', 'STATISTICS.json'),
                               ('responses_full', 'RESPONSE_PREPARED.json')]:
                path = root / kind / phase / name
                outputs[str(path)] = sha256(path)
        for omitted in omissions:
            path = root / ('full' if omitted is None else 'oof_' + omitted) / 'SELECTION.json'
            outputs[str(path)] = sha256(path)
        save_json(logs / 'PREPARATION_COMPLETE.json', dict(binding=binding, outputs=outputs,
            reused_statistics=reused, model_inputs_ready=False, independent_validation_complete=False,
            remaining=['volume convergence', 'selection sensitivity and independent closure',
                       'domain/fold geometry', 'matched targets', 'normalizers and loaders']))
    except Exception as exc:
        save_json(logs / 'PREPARATION_FAILED.json', dict(binding=binding,
            error=f'{type(exc).__name__}: {exc}', model_inputs_ready=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=lowz.DEFAULT_PLAN)
    parser.add_argument('--seconds', type=int, default=5400)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    run(args.plan, args.seconds, args.workers)
