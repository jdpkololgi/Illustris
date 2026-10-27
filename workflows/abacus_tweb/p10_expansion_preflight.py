#!/usr/bin/env python3
"""Bounded metadata inventory, or compute-only verification of an E2E density.

Inventory never reads catalogue rows or reserved target payloads. Adapting a
native count grid reuses its bytes only after a complete SHA256 verification;
no normalized E2E patch may be substituted for the native TSC count grid.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import socket
from p10_phase_assets import load_registry, expand_phase, sha256_file

REPO = Path(__file__).resolve().parents[2]
DEFAULT = REPO / 'configs/p10_phase_registry_v2.json'


def stat_record(path):
    p = Path(path)
    return dict(path=str(p), exists=p.exists(), bytes=p.stat().st_size if p.is_file() else None)


def density_contract(receipt, phase):
    expected = dict(phase=phase, ngrid=2048, box_mpc_h=2000., dtype='float32',
                    processed_files=136, subsample_fraction=.1, **{'pass': True})
    if any(receipt.get(k) != v for k, v in expected.items()):
        raise ValueError('Density receipt differs from the frozen native grid contract')
    error = float(receipt['relative_count_error'])
    if not 0 <= error <= 2e-6 or int(receipt['particle_count']) <= 0:
        raise ValueError('Density count conservation failed')
    if len(receipt['outputs']) != 1:
        raise ValueError('Expected exactly one native density output')
    output = receipt['outputs'][0]
    if Path(output['path']).name != 'counts_tsc_2048.npy' or len(output['sha256']) != 64:
        raise ValueError('Unexpected density output identity')
    return output


def inventory(registry):
    rows = {}
    for phase, cfg in registry['phases'].items():
        paths = expand_phase(registry, phase)['assets']
        paths.update(cfg.get('observation_path_overrides', {}))
        assets = {key: stat_record(paths[key]) for key in registry['required_observation_assets']}
        assets['lss_full'] = stat_record(Path(paths['lss'])/'BGS_BRIGHT_full_HPmapcut.dat.fits')
        name = f'AbacusSummit_base_c000_{phase}'
        assets['cubic'] = stat_record(f'/global/cfs/cdirs/desi/cosmosim/SecondGenMocks/AbacusSummit/CubicBox/BGS/v0.1/z0.200/{name}/BGS_box_{phase}.fits')
        root = Path(paths['phase_output'])
        result = dict(role=cfg['role'], truth_access=cfg['truth_access'], assets=assets,
                      new_campaign_products={key: stat_record(root/rel) for key, rel in {
                          'tweb':'targets/tweb/backend_optimized_ngrid_2048_rsmooth_7/TWEB_COMPLETE.json',
                          'catalogues':'catalogues/GALAXY_TRUTH_MIGRATION_COMPLETE.json',
                          'p1':'p1_canonical/CATALOGUE_COMPLETE.json',
                          'p2':'p2_graph/P2_COMPLETE.json',
                          'p3':'p3_fields/FIELD_COMPLETE',
                          'p4':'p4_patches/PATCH_MANIFEST_COMPLETE',
                      }.items()}, training_ready=False)
        if cfg.get('density_reuse_receipt'):
            receipt_path = Path(cfg['density_reuse_receipt'])
            result['density_receipt'] = stat_record(receipt_path)
            # Reserved phases: existence only; no target statistics or labels.
            if cfg['truth_access'] in ('development','selection') and receipt_path.is_file():
                receipt = json.loads(receipt_path.read_text())
                output = density_contract(receipt, phase)
                result['native_density'] = stat_record(output['path'])
                result['native_density']['receipt_size_matches'] = result['native_density']['bytes'] == output['bytes']
                result['native_density']['payload_hash_verified'] = False
        rows[phase] = result
    return dict(schema_version='p10-expansion-inventory-v1', created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                phases=rows, training_ready=False, scope='Metadata only; no payload validation or readiness claim')


def adapt(registry, phase, destination):
    if phase not in registry['phases'] or registry['phases'][phase]['truth_access'] not in ('development','selection'):
        raise ValueError('Reserved or unregistered phase; no target access')
    if not (os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_STEP_ID') and socket.gethostname().startswith('nid')):
        raise RuntimeError('Density hashing requires an approved compute step')
    if destination.exists():
        raise FileExistsError(destination)
    receipt_path = Path(registry['phases'][phase]['density_reuse_receipt'])
    receipt = json.loads(receipt_path.read_text())
    output = density_contract(receipt, phase)
    path = Path(output['path'])
    before = path.stat()
    if before.st_size != output['bytes'] or sha256_file(path) != output['sha256']:
        raise ValueError('Native density payload checksum/size failed')
    after = path.stat()
    if (before.st_size,before.st_mtime_ns) != (after.st_size,after.st_mtime_ns):
        raise ValueError('Density changed during verification')
    import numpy as np
    array = np.load(path, mmap_mode='r')
    if array.shape != (2048,2048,2048) or array.dtype != np.float32:
        raise ValueError('Density array shape/dtype differs')
    payload = dict(schema_version='p10-e2e-native-density-adapter-v1', phase=phase,
                   source_receipt=str(receipt_path),source_receipt_sha256=sha256_file(receipt_path),
                   source_payload_sha256=output['sha256'],slurm_job_id=os.environ['SLURM_JOB_ID'],
                   target_contract=registry['target_contract'],
                   build=dict(output=str(path),processed_file_count=136,max_files_per_directory=None,
                              ngrid=2048,boxsize_mpc_h=2000.,dtype='float32',
                              deposited_count=receipt['deposited_count'],wall_seconds=receipt['elapsed_seconds_this_segment'],
                              particle_count=receipt['particle_count'],relative_count_error=receipt['relative_count_error']))
    write_new(destination,payload)


def write_new(path, payload):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as handle:
        json.dump(payload,handle,indent=2,allow_nan=False)
        handle.write('\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registry',type=Path,default=DEFAULT)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--adapt-density',metavar='PHASE')
    a=p.parse_args();registry=load_registry(a.registry)
    if a.adapt_density:
        adapt(registry,a.adapt_density,a.output)
    else:
        payload=inventory(registry);payload['registry_sha256']=sha256_file(a.registry)
        write_new(a.output,payload)
    print(a.output)

if __name__=='__main__':main()
