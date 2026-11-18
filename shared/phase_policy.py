"""Validate an explicitly selected simulation-phase policy and its bindings."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
POLICY_PATH = REPO / 'configs/shared_phase_policy_20261027.json'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_policy(path=POLICY_PATH):
    value = json.loads(Path(path).read_text())
    if value.get('schema_version') != 'graphweb-shared-phase-policy-v1':
        raise ValueError('Unsupported shared phase policy')
    groups = [value[k] for k in ('training', 'development_and_calibration', 'final_test')]
    flat = [p for group in groups for p in group]
    if ([len(g) for g in groups] != [20, 3, 2] or len(set(flat)) != 25
            or set(flat) != {f'ph{i:03d}' for i in range(25)}):
        raise ValueError('Expected a disjoint, exhaustive 20/3/2 phase partition')
    if (value.get('gaussian_smoothing_mpc_h') != 7.
            or value.get('phase_is_model_input') is not False
            or value.get('fresh_initialization_required') is not True):
        raise ValueError('Shared physical/model policy changed')
    crossfit = value['p12a_crossfit']
    if (crossfit['folds'], crossfit['training_phases_per_fold'],
            crossfit['full_training_encoders'], crossfit['context_halo_voxels']) != (20, 19, 1, 48):
        raise ValueError('Invalid shared cross-fit policy')
    return value


def binding(path=POLICY_PATH):
    path = Path(path).resolve()
    policy = load_policy(path)
    return {'path': str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path),
            'sha256': sha256(path), 'policy_id': policy['policy_id']}


def bound_policy(record):
    pointer = record['phase_policy']
    path = Path(pointer['path'])
    if not path.is_absolute():
        path = REPO / path
    if sha256(path) != pointer['sha256']:
        raise ValueError('Phase policy hash changed')
    policy = load_policy(path)
    if policy['policy_id'] != pointer['policy_id']:
        raise ValueError('Phase policy identity mismatch')
    return policy


def p12_roles(policy):
    return {'training': policy['training'],
            'validation_and_selection': policy['development_and_calibration'],
            'sealed_blind_test': policy['final_test'], 'reserved_confirmation': []}


def check_roles(record, roles, *, omitted=None):
    policy = bound_policy(record)
    expected = p12_roles(policy)
    if omitted is not None:
        if omitted not in expected['training']:
            raise ValueError('OOF phase is not registered for training')
        expected['training'] = [p for p in expected['training'] if p != omitted]
        expected['validation_and_selection'] = [omitted]
    if tuple(roles['training']) != tuple(expected['training']):
        raise ValueError('Training roles disagree with bound phase policy')
    for key, value in expected.items():
        actual = roles.get(key, [])
        actual = [actual] if isinstance(actual, str) else list(actual)
        if actual != value:
            raise ValueError('Roles disagree with bound phase policy: ' + key)
    return policy


def check_preparation_phase(record, phase):
    policy = bound_policy(record)
    if phase not in policy['training'] + policy['development_and_calibration']:
        raise ValueError('Phase is outside the bound preparation policy: ' + phase)
    return policy
