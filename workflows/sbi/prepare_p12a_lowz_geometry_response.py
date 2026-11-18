"""Low-z P12-A parent support/boundary arrays without a fictitious selection fit.

P12-A's two survey features depend on geometry, not n(z). Use the existing exact
voxel-centre sampler and a separate geometry receipt. This does not qualify
selection, folds, normalization, posterior inputs or model training readiness.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import time

import numpy as np

from workflows.sbi import prepare_p12a_lowz_inputs as raw
from workflows.sbi.p12a_parent_response import sample_centres
from workflows.sbi.prepare_cfm_lowz_inputs import require_compute, save_json
from workflows.abacus_tweb import p3br_build_random_response as randoms
from workflows.abacus_tweb.p6_refit_fullcap_selection import radius_to_redshift_grid


def phase_guard(phase, policy):
    if phase not in policy['training'] + policy['development_and_calibration']:
        raise PermissionError('No sealed or unknown phase access: ' + phase)


def prepare(phase, plan, policy, geometry, deadline):
    phase_guard(phase, policy)
    root = Path(plan['output_root']) / phase
    complete = raw.read(root / 'RAW_PREPARATION_COMPLETE.json')
    if complete['phase'] != phase or complete['pass_raw_preparation'] is not True:
        raise ValueError('Unqualified raw phase')
    manifest = root / 'p3_fields/field_manifest.json'
    if raw.sha256(manifest) != complete['fields_sha256']:
        raise ValueError('Raw field binding changed')
    fields = raw.read(manifest)
    raw.require_matching_domains(fields, plan, geometry)
    points_path = Path(fields['points'])
    points_hash = raw.sha256(points_path)
    p1_path = root / 'p1_canonical/manifest.json'
    if raw.sha256(p1_path) != fields['p1_manifest_sha256']:
        raise ValueError('P1 identity changed')
    if Path(raw.read(p1_path)['points']).resolve() != points_path.resolve():
        raise ValueError('P1/P3 point paths differ')
    angular_path = Path(fields['angular_support']['source'])
    if raw.sha256(angular_path) != fields['angular_support']['sha256']:
        raise ValueError('Angular support changed')
    angular = randoms.load_map(angular_path)
    points = np.load(points_path, mmap_mode='r')
    index_path = Path(fields['canonical_index'])
    if raw.sha256(index_path) != fields['canonical_index_sha256']:
        raise ValueError('Context membership changed')
    with np.load(index_path) as index:
        context = np.asarray(index['context'], dtype=bool)
    if context.shape != (len(points),):
        raise ValueError('Context/point row mismatch')
    cap_id = np.asarray(points[:, 3], dtype=np.uint8)
    if not np.isin(cap_id, [0, 1]).all():
        raise ValueError('Unknown cap')
    output = root / 'p12a_lowz_geometry_response'
    output.mkdir(exist_ok=False)
    save_json(output / 'GEOMETRY.json', geometry)
    paths = {k: output / (k + '_by_parent.npy') for k in ('support', 'distance')}
    support = np.lib.format.open_memmap(paths['support'], mode='w+', dtype=bool, shape=(len(points),))
    distance = np.lib.format.open_memmap(paths['distance'], mode='w+', dtype=np.float32, shape=(len(points),))
    support[:] = False
    distance[:] = np.nan
    for cap, name in randoms.CAPS:
        mask = angular['support'].astype(bool) & (angular['domain'] // 2 == cap)
        angle = randoms.angular_boundary_distance(mask)
        # Radius alone cannot recover the sign of an observed negative redshift.
        # Use the label-independent P1 context mask to exclude those rows and
        # the sentinel before geometry sampling, not a truth-validity mask.
        ids = np.flatnonzero((cap_id == cap) & context)
        for start in range(0, len(ids), 250000):
            if time.monotonic() > deadline:
                raise TimeoutError('Approved preparation budget exhausted')
            rows = ids[start:start+250000]
            distance[rows], support[rows] = sample_centres(
                points[rows], fields['components'][name], mask, angle, geometry)
    support.flush()
    distance.flush()
    if not support.any() or not np.isfinite(distance[support]).all() or (distance[support] < 0).any():
        raise ValueError('Invalid geometric survey response')
    # Diagnostic bins use observed-coordinate radii only, never target columns.
    edges = raw.domain_from(plan)['diagnostic_edges']
    counts, supported, inside = (np.zeros(len(edges)-1, dtype=np.int64) for _ in range(3))
    for start in range(0, len(points), 250000):
        xyz = np.asarray(points[start:start+250000, :3], dtype=float)
        z = np.interp(np.linalg.norm(xyz, axis=1), geometry['cosmology']['radius_grid_mpc'],
                      geometry['cosmology']['redshift_grid'])
        selected = context[start:start+len(xyz)]
        valid = np.isfinite(distance[start:start+len(xyz)]) & selected
        mask = np.asarray(support[start:start+len(xyz)])
        counts += np.histogram(z[selected], edges)[0]
        inside += np.histogram(z[valid], edges)[0]
        supported += np.histogram(z[mask], edges)[0]
    if not np.array_equal(counts, inside):
        raise ValueError('Science-range parent outside registered raw grid')
    if raw.sha256(points_path) != points_hash:
        raise ValueError('Points changed during response generation')
    row = dict(schema='p12a-lowz-geometry-response-v1', phase=phase,
        redshift_contract=raw.domain_from(plan), rows=len(points), support_rows=int(support.sum()),
        labels_read=False, geometry_features_ready=True, selection_fitted=False, model_inputs_ready=False,
        raw_receipt_sha256=raw.sha256(root / 'RAW_PREPARATION_COMPLETE.json'),
        p3_manifest_sha256=raw.sha256(manifest), points_sha256=points_hash,
        canonical_index_sha256=fields['canonical_index_sha256'],
        context_rows=int(context.sum()), excluded_context_rows=int((~context).sum()),
        angular_map_sha256=fields['angular_support']['sha256'],
        geometry_sha256=raw.sha256(output / 'GEOMETRY.json'),
        arrays={k: str(v) for k, v in paths.items()}, array_sha256={k: raw.sha256(v) for k, v in paths.items()},
        diagnostics=dict(edges=edges, observed_rows=counts.tolist(), inside_grid=inside.tolist(),
                         random_supported_rows=supported.tolist()))
    save_json(output / 'GEOMETRY_RESPONSE_COMPLETE.json', row)
    return row


def run(seconds=1200, workers=2):
    require_compute()
    if not 0 < seconds <= 1800 or workers not in (1, 2):
        raise ValueError('Outside bounded preparation resources')
    plan, _, policy = raw.load_plan()
    root = Path(plan['output_root'])
    radii, redshifts = radius_to_redshift_grid(0., .60)
    geometry = dict(redshift_contract=raw.domain_from(plan),
        cosmology=dict(radius_grid_mpc=radii.tolist(), redshift_grid=redshifts.tolist()),
        purpose='geometry-only; no radial selection or normalization fitted')
    sources = [Path(__file__), raw.REPO / 'workflows/sbi/p12a_parent_response.py',
               raw.REPO / 'workflows/abacus_tweb/p3br_build_random_response.py']
    binding = {str(p): raw.sha256(p) for p in sources}
    phases = policy['training'] + policy['development_and_calibration']
    deadline = time.monotonic() + seconds
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(prepare, p, plan, policy, geometry, deadline) for p in phases]
        rows = [f.result() for f in futures]
    if any(raw.sha256(p) != binding[str(p)] for p in sources):
        raise ValueError('Source changed during geometry response preparation')
    save_json(root / 'GEOMETRY_RESPONSES_COMPLETE.json', dict(phases=phases, records=rows,
        code_bindings=binding, job=os.environ['SLURM_JOB_ID'], model_inputs_ready=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seconds', type=int, default=1200)
    p.add_argument('--workers', type=int, choices=(1, 2), default=2)
    a = p.parse_args()
    run(a.seconds, a.workers)
