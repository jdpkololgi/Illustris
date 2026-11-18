"""Frozen-model low-z audit and point staging; NOT response/target preparation.

Only --metadata-only may run on a login node. Payload audits and FITS scans
require an explicitly authorized compute allocation. Output must be new.
The observation NPZ deliberately excludes Z_COSMO and all real-space truth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import socket

REPO = Path(__file__).resolve().parents[2]
DEFAULT_PLAN = REPO / 'configs/e2e_cfm_lowz_input_canary_20261028.json'
SEALED = {'ph018', 'ph019'}
MODELS = tuple(f'{model}_{seed}' for seed in (17, 29)
               for model in ('coarse', 'fine', 'unet_x', 'transformer_v'))


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def check(condition, message):
    if not condition:
        raise ValueError(message)


def validate_plan(plan):
    check(plan['schema'] == 'cfm-lowz-input-canary-v1', 'Wrong plan schema')
    train, dev = plan['frozen_training_phases'], plan['development_phases']
    check(len(train) == len(set(train)) == 13, 'Expected 13 frozen training phases')
    check(set(dev) == {'ph012', 'ph013'} and len(dev) == 2,
          'This canary permits only ph012/ph013 development payloads')
    check(not (set(train) & set(dev)), 'Training/development overlap')
    check(not ((set(train) | set(dev)) & SEALED), 'Final-test access forbidden')
    check(plan['candidate_science_interval'] == [0.01, 0.1], 'Unexpected low-z scope')
    check(plan['bridge_interval'] == [0.1, 0.15], 'Unexpected bridge scope')
    edges = plan['catalogue_audit_bin_edges']
    check(all(math.isfinite(x) for x in edges)
          and all(a < b for a, b in zip(edges, edges[1:])), 'Invalid bins')


def box_min_radius(row, shape, cell):
    """Continuous Cartesian box bounds, not a voxel-level exposure count."""
    midpoint = [origin + (index + (16 if axis == 0 else 0)) * row['grid']['cell_mpc_h']
                for axis, (origin, index) in enumerate(zip(
                    row['grid']['origin_mpc_h'], row['center']))]
    return math.sqrt(sum(max(0., abs(value) - size * cell / 2.) ** 2
                         for value, size in zip(midpoint, shape)))


def metadata_audit(plan):
    """Read bounded JSON metadata only; report payload verification separately."""
    validate_plan(plan)
    config = read_json(REPO / plan['historical_config'])
    train = set(plan['frozen_training_phases'])
    check(train == {p for p, role in config['phase_roles'].items() if role == 'train'},
          'Historical training roles drifted')
    deployed = read_json(plan['deployment_binding'])
    check(set(deployed['checkpoints']) == set(MODELS), 'Checkpoint inventory changed')
    coupled, whole = Path(plan['coupled_root']), Path(plan['wholefield_root'])
    geometry, geometry_hashes = {}, {}
    for phase in sorted(train):
        path = coupled / 'cartesian_v2/geometry' / phase / 'GEOMETRY_COMPLETE.json'
        data = read_json(path)
        check(data['phase'] == phase and data['pass'], 'Invalid geometry receipt')
        rows = data['pairs']
        geometry[phase] = {r['pair_id']: r for r in rows}
        check(len(rows) == len(geometry[phase]), 'Duplicate geometry IDs')
        geometry_hashes[phase] = sha256(path)
    report, bindings = {}, {}
    for key in MODELS:
        record = deployed['checkpoints'][key]
        path = Path(record['path']).parent / 'BINDING.json'
        binding = read_json(path)
        ids = binding.get('ids', binding.get('train_ids'))
        check(set(p for p, _ in ids) == train, 'Checkpoint training phases differ')
        check(len(ids) == len({tuple(x) for x in ids}), 'Duplicate training IDs')
        check(binding['normalizer'] == deployed['normalizer'], 'Normalizer mismatch')
        if 'data' in binding:
            for phase, expected in binding['data'].items():
                check(phase in train, 'Unexpected phase in whole-field data binding')
                complete = whole / 'data' / phase / 'COMPLETE.json'
                check(sha256(complete) == expected, 'Whole-field metadata hash mismatch')
                payload = read_json(complete)
                check(payload['normalizer'] == deployed['normalizer'], 'Data normalizer differs')
                for row in payload['pairs']:
                    check(row == geometry[phase][row['pair_id']], 'Geometry metadata differs')
        rows = [geometry[p][pair] for p, pair in ids]
        shape = binding.get('shape', [64, 48, 48])
        fine = [box_min_radius(r, shape, 6.766) for r in rows]
        wide = [box_min_radius(r, [48, 48, 48], 27.064) for r in rows]
        report[key] = dict(
            checkpoint=record, binding_path=str(path), binding_sha256=sha256(path),
            phases=sorted(train), domains=len(rows), fine_shape=shape,
            centre_z_min=min(r['redshift'] for r in rows),
            centre_z_max=max(r['redshift'] for r in rows),
            min_fine_radius_mpc_h=min(fine), min_wide_radius_mpc_h=min(wide),
            wide_boxes_containing_observer=sum(r == 0 for r in wide),
            fine_boxes_intersecting_z_lt_0p1=sum(r < 292.5842397676456 for r in fine),
            wide_boxes_intersecting_z_lt_0p1=sum(r < 292.5842397676456 for r in wide))
        bindings[key] = binding
    return dict(checkpoints=report, geometry_sha256=geometry_hashes,
                normalizer=deployed['normalizer'], checkpoint_payloads_verified=False,
                exposure_scope='continuous box bounds; not exact voxel exposure or validation'), bindings


def require_compute():
    if not os.environ.get('SLURM_JOB_ID') or not socket.gethostname().startswith('nid'):
        raise RuntimeError('Payload/FITS work requires an authorized compute-node Slurm step')


def verify_payload(record):
    path = Path(record['path'])
    before = path.stat()
    if 'bytes' in record:
        check(before.st_size == record['bytes'], 'Payload size mismatch')
    check(sha256(path) == record['sha256'], 'Payload hash mismatch: ' + str(path))
    after = path.stat()
    check((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
          'Payload changed during hash')
    return path


def audit_checkpoints(report, bindings):
    import torch
    torch.set_num_threads(1)
    for key, item in report['checkpoints'].items():
        path = verify_payload(item['checkpoint'])
        # Trusted local project artifact, verified against the deployment hash.
        state = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        check(json.loads(json.dumps(state['binding'])) == bindings[key],
              'Embedded checkpoint binding differs: ' + key)
        item['saved_step'] = int(state['step'])
        del state
    report['checkpoint_payloads_verified'] = True


def stage_points(plan, output):
    import fitsio
    import numpy as np
    from workflows.sbi import e2e_coupled_coordinates as coord
    from workflows.abacus_tweb import p3br_build_random_response as response

    check(sha256(coord.CONFIG) == sha256(REPO / plan['distance_config']),
          'Coordinate authority differs from plan')
    coord.table()  # Verify the pinned distance-table bytes.
    observations = output / 'observations'
    evaluation = output / 'evaluation_only'
    observations.mkdir(); evaluation.mkdir()
    report = {}
    for phase in plan['frozen_training_phases'] + plan['development_phases']:
        check(phase not in SEALED, 'Final-test phase forbidden')
        root = Path(plan['coupled_root']) / 'observations' / phase
        receipt_path = root / 'OBSERVED_COMPLETE.json'
        receipt = read_json(receipt_path)
        check(receipt['pass'] and receipt['phase'] == phase
              and receipt['target_columns_present'] is False, 'Invalid observation receipt')
        source = verify_payload(receipt['outputs'][0])
        g = fitsio.read(source, columns=['TARGETID', 'RA', 'DEC', 'Z'])
        check(len(np.unique(g['TARGETID'])) == len(g), 'Observed TARGETID duplicates')
        check(all(np.isfinite(g[k]).all() for k in ('RA', 'DEC', 'Z')), 'Nonfinite sky data')
        cap = response.galactic_cap(g['RA'], g['DEC'])
        edges = np.asarray(plan['catalogue_audit_bin_edges'])
        counts = {}
        for name, number in (('NGC', 1), ('SGC', 0)):
            z = g['Z'][cap == number]
            hist = np.histogram(z, edges)[0]
            under, over = int((z < edges[0]).sum()), int((z > edges[-1]).sum())
            check(int(hist.sum()) + under + over == len(z), 'Histogram accounting mismatch')
            counts[name] = dict(bins=hist.tolist(), underflow=under, overflow=over, rows=len(z))
        item = dict(source=receipt['outputs'][0], receipt_sha256=sha256(receipt_path),
                    rows=len(g), min_z=float(g['Z'].min()), max_z=float(g['Z'].max()),
                    counts=counts)
        if phase in plan['development_phases']:
            keep = (g['Z'] >= .01) & (g['Z'] < .15)
            low = g[keep]; ids = low['TARGETID']
            xyz = coord.sky_mpc_h(low['RA'], low['DEC'], low['Z'])
            obs_path = observations / f'{phase}_POINTS.npz'
            with obs_path.open('xb') as stream:
                np.savez_compressed(stream, TARGETID=ids, RA=low['RA'], DEC=low['DEC'],
                                    Z=low['Z'], CAP=cap[keep], xyz_observed=xyz)
            parent_receipt_path = root / 'PARENT_COMPLETE.json'
            parent_receipt = read_json(parent_receipt_path)
            check(parent_receipt['pass'] and parent_receipt['phase'] == phase,
                  'Invalid parent receipt')
            parent_path = verify_payload(parent_receipt['outputs'][0])
            parent = fitsio.read(parent_path, columns=['TARGETID', 'RA', 'DEC', 'Z',
                                                     'Z_COSMO', 'R_MAG_APP'])
            check(np.array_equal(parent['TARGETID'], np.arange(1, len(parent) + 1)),
                  'Parent row-address contract failed')
            check(np.all((ids >= 1) & (ids <= len(parent))), 'Observed ID outside parent')
            p = parent[ids - 1]
            check(np.array_equal(p['TARGETID'], ids), 'Parent identity mismatch')
            for name, tol in (('RA', 1e-10), ('DEC', 1e-10), ('Z', 2e-7)):
                check(np.allclose(p[name], low[name], atol=tol, rtol=0), 'Sky join failed: '+name)
            real = coord.sky_mpc_h(p['RA'], p['DEC'], p['Z_COSMO'])
            eval_path = evaluation / f'{phase}_POSITIONS.npz'
            with eval_path.open('xb') as stream:
                np.savez_compressed(stream, TARGETID=ids, Z_COSMO=p['Z_COSMO'],
                                    R_MAG_APP=p['R_MAG_APP'], xyz_realspace=real,
                                    native_xyz=(real - 1000.) % 2000.)
            item['staged'] = dict(rows=len(low), observations=str(obs_path),
                observations_sha256=sha256(obs_path), evaluation_only=str(eval_path),
                evaluation_sha256=sha256(eval_path), parent_receipt_sha256=sha256(parent_receipt_path),
                contains_tidal_targets=False, halo_indices_used=False)
            del parent, p
        report[phase] = item
        print(json.dumps(dict(event='catalogue', phase=phase, counts=counts)), flush=True)
        del g
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=DEFAULT_PLAN)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--metadata-only', action='store_true')
    args = parser.parse_args()
    plan = read_json(args.plan)
    validate_plan(plan)
    if not args.metadata_only:
        require_compute()
    args.output.mkdir(parents=True, exist_ok=False)
    report, bindings = metadata_audit(plan)
    report.update(schema='cfm-lowz-input-audit-v1', source_sha256=sha256(__file__),
                  plan_sha256=sha256(args.plan), deployment_binding_sha256=sha256(plan['deployment_binding']),
                  host=socket.gethostname(), job=os.environ.get('SLURM_JOB_ID'),
                  model_inputs_ready=False, lowz_performance_measured=False,
                  bin_edges=plan['catalogue_audit_bin_edges'], passed=False)
    save_json(args.output / 'METADATA.json', report)
    if not args.metadata_only:
        audit_checkpoints(report, bindings)
        norm = Path(plan['coupled_root']) / 'cartesian_v2/normalization/NORMALIZATION_COMPLETE.json'
        check(sha256(norm) == report['normalizer'], 'Frozen normalizer hash mismatch')
        save_json(args.output / 'FROZEN_NORMALIZER.json', read_json(norm))
        report['catalogues'] = stage_points(plan, args.output)
    report['passed'] = True
    report['point_inputs_ready'] = not args.metadata_only
    report['remaining'] = plan['remaining_model_input_gates']
    save_json(args.output / 'AUDIT.json', report)
    print('POINT_CANARY_COMPLETE' if not args.metadata_only else 'METADATA_ONLY_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
