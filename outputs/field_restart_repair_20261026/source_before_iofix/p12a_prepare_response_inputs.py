"""Build corrected visible-phase P12-A random response products, without fitting.

The registered 18-realisation protocol is preserved. Existing angular maps may
be copied only with matching source registry/phase identity and payload hash.
Raw P3 fields, historical response products and reserved phases are read-only.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

from workflows.abacus_tweb import p3br_build_random_response as build
from workflows.abacus_tweb.p10_training_contract import atomic_json, sha256
from workflows.sbi.e2e_coupled_contract import require_compute

REPO = Path(__file__).resolve().parents[2]
ROOT = Path('/pscratch/sd/d/dkololgi/abacus/p10_galaxy_v2_20261020')
CONTRACT = ROOT / 'training_contract_20261021'
LEGACY = Path('/pscratch/sd/d/dkololgi/abacus/p10_multiphase')
DECISION = LEGACY / 'training_contract/P3BR_RANDOM_DENSITY_DECISION.json'


def visible_roles(contract):
    ready = json.loads((contract / 'TRAINING_LOADER_READY.json').read_text())
    if not ready.get('pass'):
        raise ValueError('Base contract failed')
    roles = ready['roles']
    visible = roles['training'] + roles['validation_and_selection']
    sealed = roles['sealed_blind_test'] + roles.get('reserved_confirmation', [])
    if len(set(visible)) != len(visible) or set(visible) & set(sealed):
        raise ValueError('Invalid phase-role partition')
    return ready, visible


def prepare(phase: str, root: Path = ROOT, contract: Path = CONTRACT):
    require_compute()
    ready, visible = visible_roles(contract)
    if phase not in visible:
        raise PermissionError('Response preparation is restricted to visible phases')
    record = json.loads((contract / 'phases' / phase / 'phase_contract.json').read_text())
    p1 = root / phase / 'p1_canonical/manifest.json'
    if sha256(p1) != record['inputs']['p1_manifest_sha256']:
        raise ValueError('Phase source and corrected contract differ')
    if record.get('truth_lineage', {}).get('label_policy_version') != 'galaxy-realspace-all-populations-v2':
        raise ValueError('Corrected phase lineage required')
    output = root / phase / 'p3b_random_response_v1'
    terminal = output / 'P12A_RESPONSE_INPUT_READY.json'
    selection = contract / 'transforms/field/selection_manifest.json'
    bindings = {'full_contract':sha256(contract/'TRAINING_LOADER_READY.json'),
                'selection':sha256(selection),'decision':sha256(DECISION),
                'p3_manifest':sha256(root/phase/'p3_fields/field_manifest.json'),
                'builder_source':sha256(Path(build.__file__)), 'driver_source':sha256(Path(__file__))}
    if terminal.exists():
        saved = json.loads(terminal.read_text())
        if saved['bindings'] != bindings or sha256(saved['manifest']) != saved['manifest_sha256']:
            raise ValueError('Stale completed response product')
        return saved
    output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    decision = json.loads(DECISION.read_text())
    if not decision.get('pass') or decision['selected_random_ids'] != list(range(18)):
        raise ValueError('Expected frozen 18-realisation protocol')
    old_map = LEGACY / phase / 'p3b_random_response_v1/angular/randoms_n18.npz'
    angular = output / 'angular'
    angular.mkdir(exist_ok=True)
    if old_map.is_file():
        meta = json.loads(old_map.with_suffix('.json').read_text())
        # Copy small immutable angular products; never normalize a symlink into
        # an older run. Their response geometry is independent of galaxy labels.
        if meta['phase'] != phase or meta['random_ids'] != list(range(18)):
            raise ValueError('Wrong existing angular product')
        if sha256(meta['source_registry']) != meta['source_registry_sha256']:
            raise ValueError('Historical angular source registry changed')
        for source in (old_map, old_map.with_suffix('.json')):
            destination = angular/source.name
            if destination.exists():
                if sha256(source) != sha256(destination):
                    raise ValueError('Partial angular copy differs')
            else:
                shutil.copy2(source,destination)
        angular_origin = {'reused_from':str(old_map),'sha256':sha256(old_map)}
    else:
        registry_path = output/'SOURCE_REGISTRY.json'
        if not registry_path.exists():
            registry = json.loads((REPO/'configs/p10_phase_registry_v2.json').read_text())
            mock = registry['phases'][phase]['mock']
            lss = Path(registry['path_templates']['lss'].format(mock=mock))
            records = []
            for i in range(18):
                path = lss/f'BGS_BRIGHT_{i}_full_HPmapcut.ran.fits'
                before = path.stat()
                digest = sha256(path)
                after = path.stat()
                if (before.st_size,before.st_mtime_ns) != (after.st_size,after.st_mtime_ns):
                    raise ValueError('Source changed during hashing')
                records.append({'random_id':i,'path':str(path),'bytes':after.st_size,'mtime_ns':after.st_mtime_ns,'sha256':digest})
                print(json.dumps({'phase':phase,'registered_random':i}),flush=True)
            atomic_json(registry_path,{'schema_version':'p12a-visible-response-sources-v1',
                'visible_phases':visible,'mock_phases':{phase:{'full_random':records}},
                'full_contract_sha256':bindings['full_contract']})
        registry = json.loads(registry_path.read_text())
        if registry.get('full_contract_sha256') != bindings['full_contract']:
            raise ValueError('Response registry has stale base contract')
        sources = registry['mock_phases'][phase]['full_random']
        for source in sources:
            stat = Path(source['path']).stat()
            if (stat.st_size,stat.st_mtime_ns) != (source['bytes'],source['mtime_ns']):
                raise ValueError('Registered random source changed')
        build.build_maps(root,registry_path,phase,(1,4,18))
        # detect in-place input updates during the maps' streaming reads
        for source in sources:
            stat = Path(source['path']).stat()
            if (stat.st_size,stat.st_mtime_ns) != (source['bytes'],source['mtime_ns']):
                raise ValueError('Random source changed during map construction')
        angular_origin = {'source_registry':str(registry_path),'sha256':sha256(registry_path)}
    manifest = build.build_overlay(root,selection,phase,DECISION)
    if not manifest.get('pass'):
        raise RuntimeError('Random response overlay failed QA')
    manifest_path = output/'manifest.json'
    result = {'schema_version':'p12a-corrected-response-input-v1','phase':phase,'pass':True,
              'created_utc':datetime.now(timezone.utc).isoformat(),'bindings':bindings,
              'manifest':str(manifest_path),'manifest_sha256':sha256(manifest_path),
              'angular_origin':angular_origin,'elapsed_seconds':time.monotonic()-start,
              'reserved_opened':False,'scientific_training':False}
    atomic_json(terminal,result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',required=True)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--contract',type=Path,default=CONTRACT)
    args=parser.parse_args()
    print(json.dumps(prepare(args.phase,args.root,args.contract),indent=2),flush=True)

if __name__ == '__main__': main()
