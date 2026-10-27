"""Readiness of the corrected-label U-PATCH encoder feeding P12-A.

P10-named files are shared encoder/data infrastructure. This audit never claims
that encoder readiness supplies new out-of-fold predictions or a refit FMPE.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

CONTRACT = Path('/pscratch/sd/d/dkololgi/abacus/p10_galaxy_v2_20261020/training_contract_20261021')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def metadata():
    read = lambda p: json.loads(p.read_text())
    ready = read(CONTRACT / 'TRAINING_LOADER_READY.json')
    require(ready['pass'] and all(ready['gates'].values()), 'Preparation receipt failed')
    roles = ready['roles']; train = roles['training']; validation = roles['validation_and_selection']
    require(len(train) == 15 and validation == ['ph006', 'ph012', 'ph013'], 'Unexpected expanded roles')
    require(not set(train).intersection(validation), 'Training/selection overlap')
    for key in ('registry', 'adapter_inventory', 'transform_manifest'):
        require(sha(ready[key]) == ready[key + '_sha256'], 'Preparation binding changed: ' + key)
    for rel in ('field/field_transform.json', 'field/selection_manifest.json', 'target_scaler.json'):
        spec = read(CONTRACT / 'transforms' / rel)
        require(spec['pass'] and spec['fit_phases'] == train, 'Transform fitting phase mismatch: ' + rel)
    visible = {}
    for phase in train + validation:
        record_path = CONTRACT / 'phases' / phase / 'phase_contract.json'
        record = read(record_path); lineage = record['truth_lineage']
        require(record['pass'] and lineage['label_policy_version'] == 'galaxy-realspace-all-populations-v2',
                'Missing corrected own-position label lineage: ' + phase)
        require(sha(lineage['observed_receipt']) == lineage['observed_receipt_sha256'],
                'Corrected catalogue receipt changed: ' + phase)
        for kind in ('p1_manifest',):
            require(sha(record['inputs'][kind]) == record['inputs'][kind + '_sha256'],
                    'Input manifest changed: ' + phase)
        visible[phase] = {'phase_contract_sha256': sha(record_path), 'eligible_cores': record['eligible_cores'],
                          'authoritative_rows': record['authoritative_rows'], 'truth_lineage': lineage}
    reserved = roles['reserved_confirmation'] + roles['sealed_blind_test']
    require(all(not (CONTRACT / 'phases' / p / 'parent_eigenvalues.npy').exists() for p in reserved),
            'Reserved truth is present in training contract')
    return {'schema': 'p12a-corrected-field-encoder-readiness-v1',
            'created_utc': datetime.now(timezone.utc).isoformat(), 'roles': roles,
            'contract': str(CONTRACT), 'contract_sha256': sha(CONTRACT / 'TRAINING_LOADER_READY.json'),
            'visible_phase_metadata': visible, 'metadata_pass': True, 'encoder_training_ready': False,
            'p12a_posterior_training_ready': False,
            'posterior_prerequisites': [
                'Versioned cross-fit contracts for the expanded roles; existing builder defaults to five old phases',
                'Fresh corrected-label full and out-of-fold U-PATCH fits',
                'New halo48 per-galaxy base predictions and response-feature dataset with corrected targets',
                'Versioned multi-phase posterior calibration/selection contract before FMPE fitting'],
            'existing_models_unchanged': True}


def compute(output, report):
    import numpy as np
    import torch
    from workflows.sbi import e2e_coupled_contract as c
    from workflows.abacus_tweb.p10_training_contract import P10PhaseBalancedLoader
    from workflows.abacus_tweb.p10_train_arm_a import validation_core_groups
    from workflows.abacus_tweb import p8_train_unet_patch as u
    from workflows.abacus_tweb.p8_deterministic_common import unscale_increments, increments_to_eigenvalues
    c.require_compute(); require(torch.cuda.device_count() == 1, 'One allocated GPU required')
    torch.set_num_threads(4)
    # Use the real trainer for uninterrupted versus serialized/restarted updates.
    base = [sys.executable, '-m', 'workflows.abacus_tweb.p10_train_arm_a', '--model', 'unet',
            '--epochs', '1', '--min-epochs', '1', '--disable-early-stopping', '--seed', '42',
            '--contract-root', str(CONTRACT), '--output-root', str(output / 'canary'),
            '--validation-reserve-seconds', '0', '--lr', '0.0002', '--scheduler-total-updates', '100',
            '--checkpoint-every', '1', '--loss-log-every', '1']
    for name, updates, resume in [('restart', 1, False), ('restart', 2, True), ('uninterrupted', 2, False)]:
        command = base + ['--run-name', name, '--stop-after-updates', str(updates)]
        if resume:
            command.append('--auto-resume')
        with (output / f'{name}_{updates}.log').open('x') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
    checkpoints = [torch.load(output / 'canary' / name / 'unet/seed_42/arm_a_checkpoint.pt',
                              map_location='cpu', weights_only=False) for name in ('restart', 'uninterrupted')]
    maximum = max(float((value - checkpoints[1]['model_state'][key]).abs().max())
                  for key, value in checkpoints[0]['model_state'].items())
    replay = {'model_maxabs': maximum, 'global_steps': [x['global_step'] for x in checkpoints],
              'scheduler_equal': checkpoints[0]['scheduler_state'] == checkpoints[1]['scheduler_state'],
              'pass': maximum <= 2e-6 and all(x['global_step'] == 2 for x in checkpoints)}
    with (output / 'ENCODER_REPLAY.json').open('x') as f:
        json.dump(replay, f, indent=2)
    require(replay['pass'], 'Encoder restart parity; see ENCODER_REPLAY.json')
    require(checkpoints[0]['scheduler_state'] == checkpoints[1]['scheduler_state'], 'Scheduler restart mismatch')
    model = u.UPatch().cuda().eval(); model.load_state_dict(checkpoints[0]['model_state'])
    del checkpoints
    panels = []
    with P10PhaseBalancedLoader(CONTRACT, include_blind=False) as loader:
        def forbidden_graph(*args, **kwargs):
            raise AssertionError('Field encoder requested a graph')
        loader.graph_adapter = forbidden_graph
        groups = validation_core_groups(loader)
        require(tuple(groups) == tuple(report['roles']['validation_and_selection']), 'Selection phases omitted')
        for phase in loader.training_phases + loader.validation_phases:
            name = 'training_core_id.npy' if phase in loader.training_phases else 'validation_core_id.npy'
            ids = np.load(CONTRACT / 'phases' / phase / name, mmap_mode='r')
            core = int(ids[len(ids) // 2])
            patch = loader.field_adapter(phase).extract(core, u.HALO_VOXELS, u.CHANNELS,
                                                       alignment_voxels=u.ALIGNMENT_VOXELS)
            parent = patch.authoritative_parent_id
            with np.load(loader.phase_records[phase]['inputs']['assignment']) as assignment:
                require(sha(loader.phase_records[phase]['inputs']['assignment']) ==
                        loader.phase_records[phase]['inputs']['assignment_sha256'], 'Assignment hash mismatch')
                expected = assignment['parent_node_id'][(assignment['core_id'] == core)
                                                        & assignment['supervised_eligible']]
                require(np.array_equal(np.sort(parent), np.sort(expected)), 'Field/target parent-row alignment')
            target_record = loader.phase_records[phase]['target']
            require(sha(target_record['path']) == target_record['sha256'], 'Target cache hash mismatch')
            truth = np.asarray(loader.targets_by_parent(phase)[parent])
            require(len(parent) > 0 and np.isfinite(truth).all() and (np.diff(truth, axis=1) >= 0).all(),
                    'Selected own-position targets invalid')
            values, points = u.model_inputs(patch, loader.field_normalization, 'cuda')
            with torch.no_grad():
                prediction = model(values, points).cpu().numpy()
            eigen = increments_to_eigenvalues(unscale_increments(prediction, loader.target_scaler))
            require(eigen.shape == truth.shape and np.isfinite(eigen).all(),
                    'Interpolated base-prediction head failed')
            # The existing deterministic encoder uses linear increments. P12-A's
            # FMPE enforces ordered draws with softplus; do not confuse the two.
            panels.append({'phase': phase, 'core': core, 'galaxies': len(parent),
                           'base_prediction_ordering_violation_fraction': float(np.mean(
                               np.any(np.diff(eigen, axis=1) < 0, axis=1))), 'pass': True})
            del values, points, patch
        require(not loader._graph, 'Graph products were opened')
    return {'pass': True, 'restart_model_maxabs': maximum, 'phase_panels': panels,
            'selected_phase_core_counts': {p: len(ids) for p, ids in groups.items()},
            'scope': 'encoder numerical readiness; one geometry-selected core per visible phase; no posterior fit'}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--mode', choices=('metadata', 'compute'), required=True)
    parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True); report = metadata()
    if args.mode == 'compute':
        report['checks'] = compute(args.output, report)
        require(metadata()['contract_sha256'] == report['contract_sha256'], 'Contract changed during audit')
        report['encoder_training_ready'] = True
    name = 'P12A_ENCODER_READY.json' if report['encoder_training_ready'] else 'P12A_METADATA_AUDIT.json'
    with (args.output / name).open('x') as f:
        f.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'report': str(args.output / name), 'metadata_pass': True,
                      'encoder_training_ready': report['encoder_training_ready'],
                      'p12a_posterior_training_ready': False}), flush=True)


if __name__ == '__main__':
    main()
