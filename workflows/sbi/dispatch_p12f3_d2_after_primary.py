#!/usr/bin/env python3
"""One-shot operational dispatch of existing D2 licences; no scientific choices.

Run after the frozen primary decision job. All submitted workers retain their
own deep contract/licence guards. An exclusive claim forbids duplicate or
automatic partial-submission retries. This module is not a model/evaluator.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

SOURCE = Path('/global/u2/d/dkololgi/TNG/Illustris_d2_467f442')
REVISION = '467f442c5c54864658fdfaf948335d6e11a647fe'
CONFIG_HASH = '3143ce1dfdb9546d3eb40413feab91bafa11b70718a2e4bb0ecb451080793533'
ROOT = Path('/pscratch/sd/d/dkololgi/abacus/p10_multiphase/p12f3_d2_diffusion_v1/official_467f442_seed42_v1')
TRAINING_WRAPPER = Path(__file__).with_name('submit_p12f3_d2_training_fullwall.slurm')
TRAINING_WRAPPER_HASH = 'cc2054003c3202125a56f532637a33488e9e3e33f0af13759a76737737cde16c'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def actions_for(decision, replication, stochastic, decision_hash, contract_digest):
    if (decision.get('schema_version') != 'p12f3-d2-ph006-seed-decision-v1'
            or decision.get('pass') is not True or decision.get('ph001_opened') is not False
            or decision.get('seed') != 42 or decision.get('seed_role') != 'primary'
            or decision.get('frozen_digest') != digest(decision.get('frozen_inputs', {}))
            or decision.get('frozen_inputs', {}).get('contract_digest') != contract_digest):
        raise ValueError('Unsafe primary decision')
    arm = decision.get('selected_arm')
    # The official capacity funnel has already selected this one arm.
    if arm != 'modern_base4':
        raise ValueError('Unexpected frozen arm; dispatch cannot select a model')
    for licence, schema, binding in (
        (replication, 'p12f3-d2-second-seed-license-v1', 'seed42_decision_sha256'),
        (stochastic, 'p12f3-d2-stochastic-control-license-v1', 'seed_decision_sha256'),
    ):
        if (licence.get('schema_version') != schema or licence.get('pass') is not True
                or licence.get('ph001_opened') is not False
                or licence.get(binding) != decision_hash
                or licence.get('contract_digest') != contract_digest
                or any(licence.get(k) != decision.get(k) for k in
                       ('selected_arm', 'selected_presentations', 'selected_weights'))):
            raise ValueError('Stale or mismatched licence')
    passed = decision.get('seed_pass')
    convergence = decision.get('sampler_convergence_nfe50_nfe100', {}).get('pass')
    if type(passed) is not bool or type(convergence) is not bool:
        raise ValueError('Missing scientific decision')
    if replication.get('licensed') is not passed or stochastic.get('licensed') is not (not convergence):
        raise ValueError('Licence contradicts frozen primary decision')
    if passed and not convergence:
        raise ValueError('A passing seed must have converged sampling')
    if passed:
        return [f'replicate {arm} {ROOT / "D2_SECOND_SEED_LICENSE.json"}',
                f'export replication 50 {arm}', f'export replication 100 {arm}',
                f'evaluate replication 50 {arm}', f'evaluate replication 100 {arm}',
                f'decide-seed replication {arm}', 'decide-combined']
    return [f'stochastic-control {arm}'] if not convergence else []


def exclusive_json(path, payload):
    with Path(path).open('x') as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    decision_path = ROOT / 'D2_SEED42_PH006_DECISION.json'
    decision = json.loads(decision_path.read_text())
    contract = json.loads((ROOT / 'D2_CONTRACT_FROZEN.json').read_text())
    if contract.get('frozen_digest') != digest(contract.get('frozen', {})):
        raise ValueError('Contract digest changed')
    actions = actions_for(decision,
        json.loads((ROOT / 'D2_SECOND_SEED_LICENSE.json').read_text()),
        json.loads((ROOT / 'D2_STOCHASTIC_CONTROL_LICENSE.json').read_text()),
        sha(decision_path), contract['frozen_digest'])
    if subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip() != REVISION:
        raise ValueError('Pinned D2 source revision changed')
    if subprocess.check_output(['git', '-C', str(SOURCE), 'status', '--porcelain'], text=True).strip():
        raise ValueError('Pinned D2 worktree is dirty')
    if sha(SOURCE / 'configs/p12f3_d2_diffusion_v1.json') != CONFIG_HASH:
        raise ValueError('Pinned D2 configuration changed')
    if sha(TRAINING_WRAPPER) != TRAINING_WRAPPER_HASH:
        raise ValueError('Operational training wrapper changed')
    plan = {'schema_version': 'd2-licensed-dispatch-v1', 'actions': actions,
            'seed42_decision_sha256': sha(decision_path), 'automatic_retry': False,
            'dispatch_only': True, 'created_utc': datetime.now(timezone.utc).isoformat(),
            'dispatcher_sha256': sha(__file__), 'training_wrapper_sha256': TRAINING_WRAPPER_HASH}
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    parent = os.environ.get('SLURM_JOB_ID', '')
    if not parent.isdigit():
        raise ValueError('Dispatch must run as a scheduler job')
    claim = ROOT / 'D2_LICENSED_DISPATCH_CLAIM.json'
    exclusive_json(claim, plan)
    records = []
    for index, action in enumerate(actions):
        script = (TRAINING_WRAPPER if action.startswith('replicate ') else
                  SOURCE / 'workflows/sbi/submit_p12f3_d2_stage.slurm')
        export = ','.join(['ALL', f'D2_ACTION={action}', f'D2_SOURCE_ROOT={SOURCE}',
            f'D2_OUTPUT_ROOT={ROOT}', f'EXPECTED_GIT_REVISION={REVISION}',
            f'EXPECTED_CONFIG_SHA256={CONFIG_HASH}'])
        job = subprocess.check_output(['sbatch', '--parsable',
            f'--job-name=d2_licensed_{index}', f'--dependency=afterok:{parent}',
            f'--export={export}', str(script)],
            text=True).strip().split(';')[0]
        if not job.isdigit():
            raise RuntimeError('Unrecognized sbatch result; reconcile claim manually')
        row = {'job_id': job, 'action': action, 'dependency': parent}
        exclusive_json(ROOT / f'D2_LICENSED_DISPATCH_JOB_{index}.json', row)
        records.append(row)
        parent = job
        print(json.dumps(row), flush=True)
    exclusive_json(ROOT / 'D2_LICENSED_DISPATCH_COMPLETE.json', {**plan, 'jobs': records})


if __name__ == '__main__':
    main()
