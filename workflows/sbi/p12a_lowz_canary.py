"""Bounded, read-only source audit and point/support canary for P12-A low-z.

Not a full data build, selection fit, independent tidal recomputation or training.
Only ph001/ph006 are allowed. Existing angular randoms are rehashed; their raw
random-file audit is inherited, not repeated. Output is a new private root.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import time

import numpy as np

REPO = Path(__file__).resolve().parents[2]
INPUT = REPO / 'configs/p12a_20phase_inputs_bound_20261027.json'
EDGES = np.array([0., .01, .03, .05, .075, .10, .15, .25, .35, .45, .55, .60])
PHASES = ('ph001', 'ph006')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def check_phases(phases):
    if tuple(phases) != PHASES:
        raise ValueError('Only approved ph001/ph006 canary is allowed')


def masks(z):
    z = np.asarray(z)
    return {'requested_science': np.isfinite(z) & (z >= .01) & (z < .55),
            'lowz_science': np.isfinite(z) & (z >= .01) & (z < .15),
            'candidate_context': np.isfinite(z) & (z > 0) & (z < .60)
                                 & ~((z >= .585) & (z < .595)),
            'old_context': np.isfinite(z) & (z >= .10) & (z < .60)
                           & ~((z >= .585) & (z < .595))}


def selected_rows(z, cap, per_bin_cap=16):
    """Deterministic observed-only sampling; never select using truth validity."""
    chunks = []
    for lo, hi in zip(EDGES[1:6], EDGES[2:7]):
        for c in (0, 1):
            rows = np.flatnonzero((z >= lo) & (z < hi) & (cap == c))
            if len(rows):
                chunks.append(rows[np.linspace(0, len(rows)-1, min(per_bin_cap, len(rows)), dtype=int)])
    return np.concatenate(chunks) if chunks else np.array([], dtype=int)


def audit_phase(phase, cfg, output):
    import fitsio
    import healpy as hp
    from workflows.abacus_tweb.galaxy_truth_contract import verify_observed_truth, label_validity
    from workflows.abacus_tweb import p3br_build_random_response as response
    from workflows.sbi.p12a_parent_response import sample_centres

    start = time.monotonic()
    root = Path(cfg['source_root'])
    full = Path(cfg['full_contract'])
    phase_record = read(full / 'phases' / phase / 'phase_contract.json')
    p1path = root / phase / 'p1_canonical/manifest.json'
    p1 = read(p1path)
    if digest(p1path) != phase_record['inputs']['p1_manifest_sha256']:
        raise ValueError('P1 input binding changed')
    source = Path(p1['parent'])
    before = source.stat()
    lineage = verify_observed_truth(source, phase)
    if lineage != p1['truth_lineage']:
        raise ValueError('Corrected label lineage mismatch')
    print(json.dumps({'phase': phase, 'stage': 'source_hash_verified'}), flush=True)
    columns = ['TARGETID', 'RA', 'DEC', 'Z', 'HAS_LABEL', 'TWEB_STATUS',
               'LAMBDA1', 'LAMBDA2', 'LAMBDA3', 'INDEP_POS', 'BOX_INDEX']
    with fitsio.FITS(source) as f:
        available = f[1].get_colnames()
        optional = [k for k in ('R_MAG_APP', 'R_MAG_ABS', 'LOGMSTAR', 'MSTAR', 'STELLAR_MASS') if k in available]
        table = f[1].read(columns=[k for k in columns if k in available] + optional)
    if len(table) != p1['counts']['total'] or len(np.unique(table['TARGETID'])) != len(table):
        raise ValueError('Canonical row identity failed')
    valid = label_validity(table)
    z = np.asarray(table['Z'], dtype=float)
    if not np.isfinite(z).all():
        raise ValueError('Nonfinite observed redshift')
    cap = response.galactic_cap(table['RA'], table['DEC'])
    apath = root / phase / 'p3b_random_response_v1/angular/randoms_n18.npz'
    ameta = read(apath.with_suffix('.json'))
    if ameta['phase'] != phase or digest(apath) != ameta['sha256']:
        raise ValueError('Angular random map binding changed')
    angular = response.load_map(apath)
    pix = hp.ang2pix(ameta['nside'], table['RA'], table['DEC'], lonlat=True, nest=False)
    supported = np.asarray(angular['support'], bool)[pix] & ((angular['domain'][pix] // 2) == cap)
    p3path = root / phase / 'p3_fields/field_manifest.json'
    p3 = read(p3path)
    points_path = Path(p3['points'])
    completion = Path(p1['source_completion_marker'])
    if digest(completion) != p1['source_completion_marker_sha256']:
        raise ValueError('P1 completion receipt changed')
    if digest(points_path) != read(completion)['artifacts']['points_sha256']:
        raise ValueError('Existing observed geometry hash changed')
    points = np.load(points_path, mmap_mode='r')
    if points.shape != (len(table), 4) or not np.array_equal(points[:, 3], cap):
        raise ValueError('Geometry/cap alignment failed')
    inventory = read(Path(cfg['response_contract']) / 'adapter_inventory.json')
    r = inventory['phases'][phase]
    if digest(r['parent_response']) != r['parent_response_sha256']:
        raise ValueError('Response marker changed')
    old = read(r['parent_response'])
    spath = Path(old['arrays']['support'])
    if digest(spath) != old['array_sha256']['support']:
        raise ValueError('Old response support changed')
    old_support = np.load(spath, mmap_mode='r')
    rows = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        for c, name in ((0, 'SGC'), (1, 'NGC')):
            chosen = (z >= lo) & (z < hi) & (cap == c)
            item = {'z': [float(lo), float(hi)], 'cap': name, 'rows': int(chosen.sum()),
                    'valid_labels': int((chosen & valid).sum()),
                    'angular_supported': int((chosen & supported).sum()),
                    'old_response_supported': int(np.count_nonzero(old_support[chosen]))}
            if 'BOX_INDEX' in available:
                item['legacy_negative_box_index_with_valid_label'] = int((chosen & valid & (table['BOX_INDEX'] < 0)).sum())
            item['property_quantiles_05_50_95'] = {}
            for k in optional:
                values = table[k][chosen]; values = values[np.isfinite(values)]
                item['property_quantiles_05_50_95'][k] = np.quantile(values, [.05,.5,.95]).tolist() if len(values) else None
            rows.append(item)
    chosen = selected_rows(z, cap)
    xyz = np.asarray(points[chosen, :3]); radius = np.linalg.norm(xyz, axis=1)
    angle_distance = np.zeros(len(chosen))
    old_voxel_support = np.zeros(len(chosen), dtype=bool)
    inside_grid = np.zeros(len(chosen), dtype=bool)
    selection = read(full / 'transforms/field/selection_manifest.json')
    for c, name in ((0, 'SGC'), (1, 'NGC')):
        support = np.asarray(angular['support'], bool) & ((angular['domain'] // 2) == c)
        angle = response.angular_boundary_distance(support, nside=ameta['nside'])
        use = cap[chosen] == c
        angle_distance[use] = angle[pix[chosen[use]]] * radius[use]
        _, old_voxel_support[use] = sample_centres(points[chosen[use]], p3['components'][name], support, angle, selection)
        grid = response.grid_from_component(p3['components'][name])
        index = np.floor((xyz[use] - grid.origin) / grid.cell_mpc).astype(int)
        inside_grid[use] = np.all((index >= 0) & (index < np.asarray(grid.shape)), axis=1)
    folder = output / phase; folder.mkdir()
    np.savez_compressed(folder/'observed_points.npz', TARGETID=table['TARGETID'][chosen],
                        RA=table['RA'][chosen], DEC=table['DEC'][chosen], Z=z[chosen],
                        CAP=cap[chosen], xyz_observed_mpc=xyz,
                        angular_supported=supported[chosen], angular_boundary_distance_mpc=angle_distance,
                        inside_existing_grid=inside_grid, old_voxel_support=old_voxel_support)
    np.savez_compressed(folder/'evaluation_only_labels.npz', TARGETID=table['TARGETID'][chosen],
                        valid=valid[chosen], eigenvalues=np.column_stack([table[k][chosen] for k in ('LAMBDA1','LAMBDA2','LAMBDA3')]))
    after = source.stat()
    if (before.st_size,before.st_mtime_ns) != (after.st_size,after.st_mtime_ns):
        raise ValueError('Source changed during audit')
    result = {'phase': phase, 'source': str(source), 'lineage': lineage,
              'p1_sha256': digest(p1path), 'p3_sha256': digest(p3path), 'angular_sha256': ameta['sha256'],
              'rows': len(table), 'z_min': float(z.min()), 'z_max': float(z.max()), 'bins': rows,
              'source_columns': available, 'optional_property_columns': optional,
              'mask_counts': {k:int(v.sum()) for k,v in masks(z).items()},
              'sample_rows': len(chosen), 'sample_inside_existing_grid': int(inside_grid.sum()),
              'sample_old_voxel_supported': int(old_voxel_support.sum()),
              'sample_angular_supported': int(supported[chosen].sum()),
              'source_and_label_checks_pass': True, 'elapsed_seconds': time.monotonic()-start,
              'outputs': {p.name: {'path':str(p),'sha256':digest(p)} for p in folder.iterdir()}}
    write(folder/'AUDIT.json',result)
    print(json.dumps({'phase':phase,'stage':'complete','elapsed_seconds':result['elapsed_seconds']}),flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not socket.gethostname().startswith('nid'):
        raise RuntimeError('Approved compute-node Slurm step required')
    check_phases(PHASES)
    cfg = read(INPUT)
    from shared.phase_policy import check_roles
    marker = Path(cfg['full_contract'])/'TRAINING_LOADER_READY.json'
    ready = read(marker)
    if digest(marker) != cfg['full_contract_sha256'] or not ready['pass']:
        raise ValueError('Input contract changed')
    policy = check_roles(cfg,ready['roles'])
    if 'ph001' not in policy['training'] or 'ph006' not in policy['development_and_calibration']:
        raise ValueError('Canary roles changed')
    args.output.mkdir(parents=True,exist_ok=False)
    sources = [Path(__file__), REPO/'configs/p12a_campaign_plan_20261029.json', INPUT,
               REPO/'shared/phase_policy.py', REPO/'workflows/abacus_tweb/galaxy_truth_contract.py',
               REPO/'workflows/abacus_tweb/p3br_build_random_response.py', REPO/'workflows/sbi/p12a_parent_response.py']
    source_hashes = {str(p):digest(p) for p in sources}
    report = {'created_utc':datetime.now(timezone.utc).isoformat(), 'job':os.environ['SLURM_JOB_ID'],
              'host':socket.gethostname(), 'phase_policy':cfg['phase_policy'], 'sources':source_hashes,
              'sealed_access':False, 'scientific_training':False, 'lowz_training_inputs_ready':False,
              'limitations':['label availability is not an independent tidal-field resampling check',
                            'angular support is not qualified low-z 3D response or selection',
                            'no stellar-mass completeness or real-DESI calibration claim'], 'phases':{}}
    write(args.output/'START.json',report)
    try:
        for phase in PHASES:
            report['phases'][phase] = audit_phase(phase,cfg,args.output)
        if any(digest(p)!=v for p,v in source_hashes.items()):
            raise ValueError('Source changed during canary')
        report['technical_complete'] = True
        write(args.output/'AUDIT.json',report)
    except Exception as exc:
        report['error'] = repr(exc); report['technical_complete'] = False
        write(args.output/'FAILED.json',report)
        raise


if __name__ == '__main__':
    main()
