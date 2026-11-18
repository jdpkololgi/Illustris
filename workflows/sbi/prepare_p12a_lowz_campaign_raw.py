"""Extend verified low-z raw grids to 23 permitted phases, not training readiness.

Existing canary payloads are hash-verified and preserved; partial phases fail
closed. Run once under the explicitly approved two-hour CPU preparation budget.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import socket
import time

from workflows.sbi import prepare_p12a_lowz_inputs as raw
from workflows.sbi.prepare_cfm_lowz_inputs import require_compute, save_json


def verify_existing(root, phase, plan):
    receipt = root / phase / 'RAW_PREPARATION_COMPLETE.json'
    row = raw.read(receipt)
    if (row['phase'] != phase or row['pass_raw_preparation'] is not True
            or row['model_inputs_ready'] is not False or row['selection_fitted'] is not False):
        raise ValueError('Invalid existing raw receipt: ' + phase)
    raw.require_matching_domains(row, plan)
    manifest = root / phase / 'p3_fields/field_manifest.json'
    if raw.sha256(manifest) != row['fields_sha256']:
        raise ValueError('Existing field manifest drift: ' + phase)
    fields = raw.read(manifest)
    raw.require_matching_domains(fields, plan)
    checks = [(root / phase / 'p1_canonical/manifest.json', fields['p1_manifest_sha256']),
              (Path(fields['canonical_index']), fields['canonical_index_sha256'])]
    checks += [(Path(c['file']), c['file_sha256']) for c in fields['components'].values()]
    for path, expected in checks:
        if raw.sha256(path) != expected:
            raise ValueError('Existing raw payload drift: ' + str(path))
    return dict(phase=phase, reused=True, receipt=str(receipt), sha256=raw.sha256(receipt))


def run(plan_path=raw.PLAN, seconds=6000, workers=2):
    require_compute()
    if not 0 < seconds <= 6900 or workers not in (1, 2):
        raise ValueError('Outside bounded preparation resources')
    plan, source, policy = raw.load_plan(plan_path)
    phases = policy['training'] + policy['development_and_calibration']
    root = Path(plan['output_root'])
    if raw.read(root / 'registry.json') != raw.make_registry(plan, source):
        raise ValueError('Existing registry differs')
    for name, expected in zip(('p3', 'p4'), raw.make_schemas(plan)):
        if raw.read(root / (name + '_base.json')) != expected:
            raise ValueError('Existing schema differs: ' + name)
    sources = [Path(__file__), Path(raw.__file__),
               raw.REPO / 'workflows/abacus_tweb/p10_build_phase_index.py',
               raw.REPO / 'workflows/abacus_tweb/p10_redshift_contract.py',
               raw.REPO / 'workflows/abacus_tweb/p10_materialize_phase_schemas.py',
               raw.REPO / 'workflows/abacus_tweb/p3a_build_canonical_fields.py',
               raw.REPO / 'workflows/sbi/p12a_parent_response.py']
    binding = dict(plan_sha256=raw.sha256(plan_path),
                   sources={str(p): raw.sha256(p) for p in sources},
                   phases=phases, job=os.environ['SLURM_JOB_ID'], host=socket.gethostname(),
                   seconds=seconds, workers=workers, model_inputs_ready=False)
    save_json(root / 'CAMPAIGN_RAW_RUN_BINDING.json', binding)
    deadline = time.monotonic() + seconds
    try:
        existing, pending = [], []
        for phase in phases:
            if (root / phase).exists():
                existing.append(verify_existing(root, phase, plan))
            else:
                pending.append(phase)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(raw.prepare_phase, phase, plan, source, root, deadline,
                                   campaign=True) for phase in pending]
            created = [future.result() for future in futures]
        if raw.sha256(plan_path) != binding['plan_sha256'] or any(
                raw.sha256(p) != binding['sources'][str(p)] for p in sources):
            raise ValueError('Source changed during raw preparation')
        save_json(root / 'CAMPAIGN_RAW_COMPLETE.json', dict(binding=binding,
            reused=existing, created=created, phases=phases, pass_raw_preparation=True,
            model_inputs_ready=False, selection_fitted=False, host_key_policy_qualified=False))
    except Exception as exc:
        save_json(root / 'CAMPAIGN_RAW_FAILED.json', dict(binding=binding,
            error=f'{type(exc).__name__}: {exc}', model_inputs_ready=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=raw.PLAN)
    parser.add_argument('--seconds', type=int, default=6000)
    parser.add_argument('--workers', type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    run(args.plan, args.seconds, args.workers)
