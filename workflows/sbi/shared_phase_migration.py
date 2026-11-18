"""Inventory the shared R7 migration and verify candidate reuse on compute.

Inventory reads small receipts only. A verification receipt certifies the listed
files, not scientific compatibility, a complete data release, or training readiness.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from shared.phase_policy import REPO, bound_policy, check_roles

DEFAULT = REPO/'configs/phase_migration_20261027.json'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def receipt(path):
    path = Path(path)
    result = {'path': str(path), 'exists': path.is_file()}
    if path.is_file():
        if path.stat().st_size > 32 << 20:
            raise ValueError('Inventory accepts bounded metadata only')
        result.update(bytes=path.stat().st_size, sha256=sha256(path))
    return result


def inventory(config):
    policy = bound_policy(config)
    old = Path(config['old_p12a_root']); field = Path(config['old_cfm_root'])
    rows = {}
    for phase in policy['training'] + policy['development_and_calibration']:
        p12 = receipt(old/phase/'PHASE_COMPLETE.json')
        rows[phase] = {
            'role': 'training' if phase in policy['training'] else 'development',
            'p12_phase': p12,
            'p12_action': 'verify_and_reuse_corrected_products' if p12['exists'] else 'qualify_own_position_labels_then_rebuild_label_products',
            'p12_geometry': receipt(old/phase/'p4_patches/PATCH_MANIFEST_COMPLETE'),
            'cfm_targets': receipt(field/'cartesian_v2/targets'/phase/'TARGETS_COMPLETE.json'),
            'cfm_native_density': receipt(field/'matter'/phase/'DENSITY_COMPLETE.json'),
            'cfm_action': 'prepare_missing_observations_and_targets' if phase in config['cfm_missing_phases'] else
                          'expand_to_128_pairs_and_48_wholefield_domains' if phase in config['cfm_expand_geometry_phases'] else 'verify_reuse_physical_products',
            'response_action': 'verify_angular_maps; rebuild_parent_response_against_new_selection',
            'payload_verified': False,
        }
        if phase == 'ph001':
            root = Path(config['ph001_truth_root'])/'stages'
            rows[phase]['historical_truth'] = {stage: receipt(root/(stage+'_COMPLETE.json'))
                                               for stage in ('DENSITY', 'TWEB', 'ANNOTATION')}
    return dict(schema_version='shared-phase-reuse-inventory-v1',
                created_utc=datetime.now(timezone.utc).isoformat(), phase_policy=config['phase_policy'],
                phases=rows, final_test_payloads_opened=False, training_ready=False,
                refit=['full20 selection and target/field transforms', '20 leave-one-phase-out transforms (19 each)',
                       'CFM equal-phase normalization over 20 training phases'],
                later_fits=['21 freshly initialized P12 encoders and halo48 exports',
                            'fresh CFM models and coarse/exterior interfaces bound to new normalization'],
                retained='Historical models, receipts and comparison results; R7; CFM five-phase selection function',
                scope='Metadata existence and hashes only; reuse candidates are not a readiness claim')


def records(value):
    """Yield explicitly hash-bound files, including legacy artifact dictionaries."""
    if isinstance(value, dict):
        if {'path', 'sha256'} <= value.keys():
            yield value
        artifacts, hashes = value.get('artifacts'), value.get('artifact_sha256')
        if isinstance(artifacts, dict) and isinstance(hashes, dict):
            for key, path in artifacts.items():
                if isinstance(path, str) and key in hashes:
                    yield {'path': path, 'sha256': hashes[key]}
        for child in value.values():
            yield from records(child)
    elif isinstance(value, list):
        for child in value:
            yield from records(child)



def verify_file(record, verified):
    source = Path(record['path']).resolve()
    before = source.stat()
    if 'bytes' in record and before.st_size != record['bytes']:
        raise ValueError('Reuse size failed: '+str(source))
    saved = verified.get(str(source))
    if saved is not None:
        if (saved['sha256'] != record['sha256'] or
                (saved['bytes'],saved['mtime_ns']) != (before.st_size,before.st_mtime_ns)):
            raise ValueError('Conflicting or changed repeated source: '+str(source))
        return
    if sha256(source) != record['sha256']:
        raise ValueError('Reuse checksum failed: '+str(source))
    after = source.stat()
    if (before.st_size,before.st_mtime_ns) != (after.st_size,after.st_mtime_ns):
        raise ValueError('Source changed during verification')
    verified[str(source)] = dict(sha256=record['sha256'], bytes=after.st_size, mtime_ns=after.st_mtime_ns)


def verify(config, phase, kind):
    from workflows.sbi.e2e_coupled_contract import require_compute
    require_compute()
    policy = bound_policy(config)
    if phase not in policy['training'] + policy['development_and_calibration']:
        raise PermissionError('Phase is outside preparation scope')
    if kind == 'ph001-truth':
        if phase != 'ph001':
            raise ValueError('Historical truth receipt is ph001 only')
        paths = [Path(config['ph001_truth_root'])/'stages'/(s+'_COMPLETE.json') for s in ('DENSITY','TWEB')]
    elif kind == 'cfm-density':
        paths = [Path(config['old_cfm_root'])/'matter'/phase/'DENSITY_COMPLETE.json']
    else:
        paths = [Path(config['old_p12a_root'])/phase/'PHASE_COMPLETE.json']
    verified = {}
    for path in paths:
        value = json.loads(path.read_text())
        if value.get('phase') != phase or value.get('pass') is not True:
            raise ValueError('Wrong phase or failing source receipt')
        for record in [dict(path=str(path), sha256=sha256(path)), *records(value)]:
            source = Path(record['path'])
            # Never touch final-test payloads through an incorrect source pointer.
            if any(p in str(source.resolve()) for p in policy['final_test']):
                raise PermissionError('Final-test source pointer in reuse receipt')
            verify_file(record, verified)
    return dict(schema_version='shared-phase-reuse-verification-v1', phase=phase, kind=kind,
                phase_policy=config['phase_policy'], pass_files=True, files=verified,
                training_ready=False, scope='Only explicitly listed files verified; no copied readiness or role receipts')


def bind_p12(config):
    path = REPO/config['p12a_config']; value = json.loads(path.read_text())
    marker = Path(value['full_contract'])/'TRAINING_LOADER_READY.json'
    ready = json.loads(marker.read_text())
    if ready.get('pass') is not True or ready.get('phase_policy') != config['phase_policy']:
        raise ValueError('Completed full contract with matching policy required')
    check_roles(ready, ready['roles'])
    value.update(full_contract_sha256=sha256(marker), status='full_contract_bound; encoders_and_exports_pending')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['inventory','verify','bind-p12'])
    parser.add_argument('--config', type=Path, default=DEFAULT)
    parser.add_argument('--phase')
    parser.add_argument('--kind', choices=['p12-phase','cfm-density','ph001-truth'], default='p12-phase')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); config = json.loads(args.config.read_text())
    value = inventory(config) if args.action == 'inventory' else bind_p12(config) if args.action == 'bind-p12' else verify(config,args.phase,args.kind)
    value['migration_config_sha256'] = sha256(args.config)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False); stream.write('\n')
    print(args.output)

if __name__ == '__main__':
    main()
