"""Verify reusable low-z selection fits and paired local input/target canaries.

This is a payload-reading engineering check, not a new scientific threshold,
normalizer fit, full training loader, or permission to open the sealed phases.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from workflows.sbi import cfm_lowz_selection_response as lowz
from workflows.sbi import prepare_p12a_lowz_inputs as p12
from workflows.sbi import e2e_coupled_condition_products as condition
from workflows.sbi import e2e_coupled_operators as op
from workflows.sbi import e2e_coupled_target_products as target
from workflows.sbi.prepare_cfm_lowz_inputs import require_compute, save_json, sha256


def checked_json(path, expected=None):
    path = Path(path)
    if expected is not None and sha256(path) != expected:
        raise ValueError('Receipt hash changed: ' + str(path))
    return lowz.read_json(path)


def checked_payload(record):
    path = Path(record['path'])
    if sha256(path) != record['sha256']:
        raise ValueError('Payload hash changed: ' + str(path))
    return path


def validate_fit(fit, policy, omitted, unit):
    expected = [p for p in policy['training'] if p != omitted]
    if (fit['schema'] != 'lowz-point-selection-candidate-v1'
            or fit['fit_phases'] != expected or fit['omitted_phase'] != omitted
            or fit['density_unit'] != unit or set(fit['statistics_sha256']) != set(expected)):
        raise ValueError('Selection phase/units/binding mismatch')
    counts, volume = np.asarray(fit['counts']), np.asarray(fit['volume'])
    density, valid = np.asarray(fit['density']), np.asarray(fit['valid'], bool)
    if (counts.shape != density.shape or volume.shape != density.shape
            or valid.shape != density.shape or density.shape != (2, len(fit['bin_edges'])-1)
            or not all(np.isfinite(x).all() for x in (counts, volume, density))
            or any((x < 0).any() for x in (counts, volume, density))):
        raise ValueError('Invalid selection arrays')
    if not np.array_equal(valid, (volume > 0) & (counts >= 20)):
        raise ValueError('Selection validity changed')
    expected_density = np.divide(counts, volume, out=np.zeros_like(density), where=valid)
    np.testing.assert_allclose(density, expected_density, rtol=1e-13, atol=0)


def selection_inventory(root, policy, unit):
    directory = Path(root)/'point_selection_candidates_v1'
    marker = directory/'CANDIDATES_COMPLETE.json'
    record = checked_json(marker)
    phases = policy['training'] + policy['development_and_calibration']
    if (record['phases'] != phases or record['policy'] != policy
            or record['density_unit'] != unit or record['development_used_for_fitting']):
        raise ValueError('Selection campaign authority mismatch')
    statistics = {phase: checked_json(directory/(phase+'.json'), record['statistics'][phase])
                  for phase in phases}
    for phase, row in statistics.items():
        if row['phase'] != phase or not row['label_free'] or row['density_unit'] != unit:
            raise ValueError('Invalid point-statistics identity')
    fits = {}
    for omitted in [None] + policy['training']:
        name = 'full' if omitted is None else 'oof_' + omitted
        fit = checked_json(directory/(name+'.json'), record['fits'][name])
        validate_fit(fit, policy, omitted, unit)
        for phase, expected in fit['statistics_sha256'].items():
            if expected != record['statistics'][phase]:
                raise ValueError('Fit/statistics provenance mismatch')
        for key in ('counts', 'volume'):
            pooled = sum(np.asarray(statistics[p][key]) for p in fit['fit_phases'])
            np.testing.assert_allclose(fit[key], pooled, rtol=1e-13, atol=0)
        fits[name] = record['fits'][name]
    return dict(marker=str(marker), sha256=sha256(marker), statistics_phases=len(statistics),
                full_fits=1, oof_fits=len(fits)-1, density_unit=unit,
                development_used_for_fitting=False, fit_hashes=fits), statistics


def validate_local(arrays):
    joint = np.asarray(arrays['joint'])
    shape = tuple(op.layout()['joint_density_shape'])
    if joint.shape != (len(condition.LOCAL_CHANNELS), *shape):
        raise ValueError('Local input shape mismatch')
    for key in ('support', 'selection_valid_fraction'):
        value = np.asarray(arrays[key])
        if value.shape != shape or not np.isfinite(value).all() or (value < 0).any() or (value > 1).any():
            raise ValueError('Invalid input support: ' + key)
    expected = np.asarray(arrays['native_expected_counts'])
    if (expected.shape != tuple(2*n for n in shape) or not np.isfinite(expected).all()
            or (expected < 0).any() or not np.isfinite(joint).all()):
        raise ValueError('Invalid local input payload')
    if np.any(arrays['selection_valid_fraction'] > arrays['support'] + 2e-7):
        raise ValueError('Selection-valid support exceeds observation support')
    np.testing.assert_allclose(joint[condition.LOCAL_CHANNELS.index('support_random')],
                               arrays['support'], rtol=0, atol=0)
    expected_channel = op.mean_pool(condition.transformed('expected_counts_random', expected), 2)
    np.testing.assert_allclose(joint[condition.LOCAL_CHANNELS.index('expected_counts_random')],
                               expected_channel, rtol=2e-6, atol=1e-7)


def load_local(path, phase, pair_id):
    """Observation-only read: no target or native matter path is followed."""
    if phase not in ('ph001', 'ph006'):
        raise PermissionError('Canary phase guard precedes payload IO')
    import h5py
    with h5py.File(path, 'r') as handle:
        if (handle.attrs['schema'] != 'cfm-lowz-local-condition-canary-v1'
                or handle.attrs['phase'] != phase or handle.attrs['pair_id'] != pair_id
                or handle.attrs['contains_matter'] or handle.attrs['model_inputs_ready']):
            raise ValueError('Invalid observation-only canary identity')
        keys = {'joint', 'support', 'selection_valid_fraction', 'native_expected_counts'}
        if set(handle) != keys:
            raise ValueError('Unexpected observation payload; target leakage not permitted')
        arrays = {key: handle[key][:] for key in keys}
    validate_local(arrays)
    return arrays


def paired_inventory(root, phase):
    if phase not in ('ph001', 'ph006'):
        raise PermissionError('Canary phase guard precedes payload IO')
    import h5py
    root = Path(root)
    observed_path = root/'local_condition_canaries'/phase/'LOCAL_CONDITION_CANARY_COMPLETE.json'
    targets_path = root/'target_canaries'/phase/'TARGET_CANARY_COMPLETE.json'
    observed, truth = checked_json(observed_path), checked_json(targets_path)
    geometry_path = root/('GEOMETRY_CANDIDATES_'+phase+'.json')
    geometry_hash = sha256(geometry_path)
    if (observed['phase'] != phase or truth['phase'] != phase
            or observed['geometry_sha256'] != geometry_hash
            or truth['binding']['geometry_receipt_sha256'] != geometry_hash
            or observed['local_channels'] != list(condition.LOCAL_CHANNELS)
            or observed['contains_matter'] or not truth['target_canary_pass']
            or truth['gaussian_radius_mpc_h'] != 7 or truth['smoothing_count'] != 1):
        raise ValueError('Input/target geometry or physical binding mismatch')
    if observed['selection_sha256'] != sha256(root/'point_selection_candidates_v1/full.json'):
        raise ValueError('Local input selection changed')
    pairs = truth['binding']['canary_pairs']
    ids = [row['pair_id'] for row in pairs]
    if ([row['pair_id'] for row in observed['outputs']] != ids or len(set(ids)) != 2
            or {row['cap'] for row in pairs} != {'NGC', 'SGC'}):
        raise ValueError('Input/target chart ordering mismatch')
    density_path = checked_payload(truth['density']['outputs'][0])
    components = []
    for component, record in zip(op.COMPONENTS, truth['tensors']):
        if record['component'] != list(component) or record['pair_ids'] != ids:
            raise ValueError('Tensor component/chart ordering mismatch')
        components.append(np.load(checked_payload(record['outputs'][0]), allow_pickle=False))
    if len(components) != 6:
        raise ValueError('Six independent tidal components required')
    qa = []
    with h5py.File(density_path, 'r') as density:
        if density.attrs['phase'] != phase or density.attrs['contains_observations']:
            raise ValueError('Target identity mismatch')
        for ordinal, row in enumerate(observed['outputs']):
            arrays = load_local(checked_payload(row), phase, row['pair_id'])
            rho = density[row['pair_id']]['rho_joint'][:]
            wide = density[row['pair_id']]['coarse_rho_extended'][:]
            tensor = np.stack([a[ordinal] for a in components], axis=-1)
            checks = dict(**target.density_qa(rho, wide), **target.fullbox_qa(rho, tensor))
            z = arrays['joint'][condition.LOCAL_CHANNELS.index('observer_redshift')]
            lowz = (z >= .01) & (z < .03) & (arrays['selection_valid_fraction'] > 0)
            checks.update(pair_id=row['pair_id'], supported_local_cells_z0p01_0p03=int(lowz.sum()),
                          half_quarter_expected_l1_relative=row['half_quarter_expected_l1_relative'])
            qa.append(checks)
    return dict(phase=phase, paired_local_readback_pass=True, pairs=qa,
                receipts={str(p):sha256(p) for p in (observed_path, targets_path, geometry_path)},
                wide_observation_ready=False, full_training_loader_ready=False)


def finalize(output):
    require_compute()
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    cplan, policy, _ = lowz.load_plan()
    pplan, _, ppolicy = p12.load_plan()
    if policy != ppolicy:
        raise ValueError('P12/CFM phase policy mismatch')
    inventories, records = {}, {}
    for name, plan, unit in (('p12', pplan, 'Mpc^-3'), ('cfm', cplan, '(Mpc/h)^-3')):
        inventories[name], records[name] = selection_inventory(plan['output_root'], policy, unit)
    for phase in policy['training']+policy['development_and_calibration']:
        np.testing.assert_array_equal(records['p12'][phase]['counts'], records['cfm'][phase]['counts'])
    paired = [paired_inventory(cplan['output_root'], phase) for phase in ('ph001','ph006')]
    save_json(output, dict(schema='lowz-input-canary-closeout-v1', selections=inventories,
        cross_pipeline_point_counts_identical=True, paired_local_canaries=paired,
        checks_pass=True, model_inputs_ready=False, scientific_training_started=False,
        source_sha256=sha256(__file__),
        remaining=['Full matched-CIC responses and wide observation channels',
                   'Final full-range geometry and role assignment, then all-phase matched targets',
                   'Full-training and OOF transforms and actual model-loader qualification',
                   'P12 distinct-galaxy identity versus host grouping and development split contract']))
    print('LOWZ_INPUT_CANARY_CLOSEOUT_PASS', output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    finalize(parser.parse_args().output)
