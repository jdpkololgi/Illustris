"""Adopt existing physical sources with new provenance, without refitting.

Native grids are revalidated for shape, finiteness, non-negativity and mass;
new hashes bind their present bytes. This does not claim a fresh particle-CRC
scan or validate any historical galaxy labels. Original products stay read-only.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
from shared.phase_policy import REPO, bound_policy
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi.shared_phase_migration import sha256

CONFIG = REPO/'configs/phase_migration_20261027.json'
PROOF = Path('/pscratch/sd/d/dkololgi/abacus/p12a_r7_20phase_20261027_v1/preparation_smoke_60354736/ph001_truth_verified.json')


def validate_native_manifest(value, phase):
    build = value['build']; target = value['target_contract']
    required = {'phase':phase, 'preflight_only':False}
    if any(value.get(k)!=v for k,v in required.items()):
        raise ValueError('Native density phase/build status mismatch')
    if (build['processed_file_count'] != 136 or build.get('max_files_per_directory') is not None
            or build['ngrid'] != 2048 or build['boxsize_mpc_h'] != 2000.
            or build['dtype'] != 'float32' or target['mass_assignment'] != 'TSC'
            or target['redshift'] != .2 or target['cosmology'] != 'c000'
            or target['particle_subsamples']['total_fraction'] != .1
            or build['particle_count'] <= 0 or not 0 <= build['relative_count_error'] <= 2e-6):
        raise ValueError('Existing native grid differs from the physical contract')
    return build


def verified_ph001(config):
    proof = json.loads(PROOF.read_text())
    if (proof.get('phase') != 'ph001' or proof.get('pass_files') is not True
            or proof.get('phase_policy') != config['phase_policy']):
        raise ValueError('Wrong ph001 reuse verification')
    for path, record in proof['files'].items():
        stat = Path(path).stat()
        if (stat.st_size,stat.st_mtime_ns) != (record['bytes'],record['mtime_ns']):
            raise ValueError('Verified ph001 source changed: '+path)
    return proof


def checked_json(path, expected):
    if sha256(path) != expected:
        raise ValueError('Source metadata checksum changed: '+str(path))
    return json.loads(Path(path).read_text())


def source_manifest(config, phase):
    if phase == 'ph001':
        proof = verified_ph001(config)
        stage_path = Path(config['ph001_truth_root'])/'stages/DENSITY_COMPLETE.json'
        stage = checked_json(stage_path,proof['files'][str(stage_path)]['sha256'])
        record = stage['artifacts']['p10_density_manifest']
        path = Path(record['path']); value = checked_json(path,record['sha256'])
    else:
        phase_path = Path(config['old_p12a_root'])/phase/'PHASE_COMPLETE.json'
        phase_record = json.loads(phase_path.read_text())
        if phase_record.get('phase') != phase or phase_record.get('pass') is not True:
            raise ValueError('Corrected phase proof does not pass')
        path = Path(phase_record['artifacts']['density'])
        value = checked_json(path,phase_record['artifact_sha256']['density'])
    return path, value


def adopt_native(phase):
    c.require_compute()
    if not c.SHARED_POLICY or phase not in ('ph001','ph004','ph005','ph006'):
        raise ValueError('Explicit new campaign and missing native phase required')
    config = json.loads(CONFIG.read_text()); bound_policy(config)
    if c.config()['phase_policy'] != config['phase_policy'] or str(c.ROOT) != config['new_cfm_root']:
        raise ValueError('Wrong destination campaign')
    c.bind_run(); c.phase_guard(phase)
    directory = c.ROOT/'matter'/phase
    destination = directory/'DENSITY_COMPLETE.json'
    if destination.exists():
        raise FileExistsError(destination)
    started = time.monotonic()
    manifest_path, manifest = source_manifest(config,phase)
    build = validate_native_manifest(manifest,phase)
    source = Path(build['output']); before = source.stat()
    array = np.load(source,mmap_mode='r')
    if array.shape != (2048,)*3 or array.dtype != np.float32:
        raise ValueError('Native shape/dtype mismatch')
    for start in range(0,2048,8):
        slab = array[start:start+8]
        if not np.isfinite(slab).all() or np.any(slab<0):
            raise ValueError('Nonfinite/negative native counts')
    # Match the independent auditor's exact reduction convention. Summing
    # slab totals can differ by more than its receipt-consistency ULP bound.
    total = float(array.sum(dtype=np.float64))
    error = abs(total-build['particle_count'])/build['particle_count']
    if error > 2e-6:
        raise ValueError('Revalidated native mass conservation failed')
    digest = sha256(source); after = source.stat()
    if (before.st_size,before.st_mtime_ns) != (after.st_size,after.st_mtime_ns):
        raise ValueError('Native source changed during verification')
    if phase == 'ph001':
        proof = verified_ph001(config)
        if proof['files'][str(source)]['sha256'] != digest:
            raise ValueError('ph001 differs from previously verified source bytes')
    record = dict(**c.provenance(),phase=phase,role=c.ROLES[phase],ngrid=2048,box_mpc_h=2000.,
                  dtype='float32',processed_files=136,subsample_fraction=.1,
                  particle_count=build['particle_count'],deposited_count=total,relative_count_error=error,
                  elapsed_seconds_this_segment=time.monotonic()-started,phase_policy=config['phase_policy'],
                  source_manifest={'path':str(manifest_path),'sha256':sha256(manifest_path)},
                  source_contract=manifest['target_contract'],
                  reuse_scope='Existing P10 build lineage plus current full-grid numerical validation and SHA256; no fresh particle CRC scan',
                  outputs=[dict(path=str(source.resolve()),bytes=after.st_size,mtime_ns=after.st_mtime_ns,sha256=digest)],
                  **{'pass':True})
    with c.single_writer(directory): c.atomic_json(destination,record)
    print(json.dumps(dict(phase=phase,adopted_native=str(destination),seconds=record['elapsed_seconds_this_segment'])),flush=True)
    return record


def adopt_ph001_field():
    c.require_compute()
    config = json.loads(CONFIG.read_text()); policy = bound_policy(config)
    proof = verified_ph001(config)
    stage_path = Path(config['ph001_truth_root'])/'stages/TWEB_COMPLETE.json'
    stage = checked_json(stage_path,proof['files'][str(stage_path)]['sha256'])
    record = stage['artifacts']['p10_tweb_complete']; source_marker = Path(record['path'])
    source = checked_json(source_marker,record['sha256'])
    registry = json.loads((REPO/config['registry']).read_text())
    if source['phase'] != 'ph001' or source['target_contract'] != registry['target_contract']:
        raise ValueError('Historical field physical contract differs')
    if source['outputs']['rank_count'] != 16 or source['mpi_size'] != 16:
        raise ValueError('Historical field layout differs')
    destination = Path(config['new_p12a_root'])/'ph001/targets/tweb/backend_optimized_ngrid_2048_rsmooth_7'
    marker = destination.parent/'FIELD_REUSE_COMPLETE.json'
    if destination.exists() or destination.is_symlink() or marker.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.symlink_to(source_marker.parent,target_is_directory=True)
    result = dict(phase='ph001',role='training',phase_policy=config['phase_policy'],
                  source_marker=str(source_marker),source_marker_sha256=sha256(source_marker),
                  verification=str(PROOF),verification_sha256=sha256(PROOF),
                  field_path=str(destination),physical_contract=registry['target_contract'],
                  historical_role=source['role'],scope='Read-only field reference; no inherited label or phase readiness',
                  **{'pass':True})
    c.atomic_json(marker,result)
    print(marker,flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['native','ph001-field'])
    p.add_argument('--phase',choices=['ph001','ph004','ph005','ph006'])
    args=p.parse_args()
    if args.action=='native': adopt_native(args.phase)
    else: adopt_ph001_field()

if __name__=='__main__':main()
