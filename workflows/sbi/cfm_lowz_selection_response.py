"""New low-z selection/response preparation, never a historical-weight retrofit.

Statistics use label-free observations and random support in native Mpc/h.
The finite-volume estimator is deliberately explicit and provisional: a pooled
count closure is an algebraic identity, not an independent validation gate.
Payload stages require a separately authorized compute allocation. No defaults
in the old observation, geometry, selection or model loaders are modified.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path

import numpy as np

from shared.phase_policy import REPO, bound_policy
from workflows.sbi.prepare_cfm_lowz_inputs import (
    check, read_json, require_compute, save_json, sha256, verify_payload,
)

DEFAULT_PLAN = REPO / 'configs/cfm_lowz_selection_response_20261029.json'
CAPS = ('SGC', 'NGC')


def code_bindings():
    paths = (Path(__file__), REPO / 'workflows/sbi/e2e_coupled_coordinates.py',
             REPO / 'workflows/sbi/prepare_cfm_lowz_inputs.py',
             REPO / 'workflows/abacus_tweb/p3a_build_canonical_fields.py',
             REPO / 'workflows/abacus_tweb/p3br_build_random_response.py',
             REPO / 'shared/phase_policy.py')
    return {str(p.relative_to(REPO)): sha256(p) for p in paths}


def load_plan(path=DEFAULT_PLAN):
    plan = read_json(path)
    check(plan['schema'] == 'cfm-lowz-selection-response-v1', 'Wrong low-z schema')
    policy = bound_policy(plan)
    for key in ('source_campaign', 'coordinates'):
        check(sha256(REPO / plan[key]['path']) == plan[key]['sha256'], key + ' drift')
    source = read_json(REPO / plan['source_campaign']['path'])
    check(bound_policy(source) == policy, 'Source phase policy differs')
    check(plan['science_redshift'] == [0.01, 0.55]
          and plan['context_redshift'] == [0., .60]
          and plan['excluded_sentinel_redshift'] == [.585, .595], 'Unregistered radial scope')
    check(plan['selection_bin_width'] == .005 and plan['minimum_pooled_count'] == 20,
          'Unregistered selection estimator')
    check(plan['selection_units'] == '(Mpc/h)^-3'
          and plan['smoothing_radius_mpc_h'] == 7.
          and plan['fresh_training_required'] is True
          and plan['model_inputs_ready'] is False, 'Physical/readiness policy differs')
    root = Path(plan['output_root']).resolve()
    check(root.is_relative_to(Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v3'))
          and root != Path(source['root']).resolve()
          and not root.is_relative_to(Path(source['root']).resolve()), 'Unsafe output root')
    return plan, policy, source


def phase_guard(phase, policy, *, fit=False):
    allowed = policy['training'] if fit else policy['training'] + policy['development_and_calibration']
    check(phase in allowed, 'Phase outside authorized input scope: ' + phase)


def bin_edges(plan):
    lo, hi = plan['context_redshift']
    return np.round(np.linspace(lo, hi, round((hi - lo) / plan['selection_bin_width']) + 1), 12)


def radial_mask(z, plan):
    z = np.asarray(z)
    lo, hi = plan['context_redshift']
    a, b = plan['excluded_sentinel_redshift']
    return np.isfinite(z) & (z > 0) & (z >= lo) & (z < hi) & ~((z >= a) & (z < b))


def selected_phases(policy, omitted=None):
    if omitted is not None:
        phase_guard(omitted, policy, fit=True)
    return [p for p in policy['training'] if p != omitted]


def fit_statistics(records, plan, policy, omitted=None):
    """Require the exact full/OOF training set; do not accept development records."""
    phases = selected_phases(policy, omitted)
    check(set(records) == set(phases), 'Fit requires exactly the registered training phases')
    edges = bin_edges(plan)
    shape = (2, len(edges) - 1)
    counts, volume = np.zeros(shape), np.zeros(shape)
    for phase in phases:
        row = records[phase]
        check(row['phase'] == phase and row['label_free'] is True
              and row['plan_sha256'] == records[phases[0]]['plan_sha256'], 'Statistics identity differs')
        check(np.array_equal(row['bin_edges'], edges), 'Statistics radial bins differ')
        n, v = np.asarray(row['counts'], dtype=float), np.asarray(row['volume'], dtype=float)
        check(n.shape == v.shape == shape and np.isfinite(n).all() and np.isfinite(v).all()
              and np.all(n >= 0) and np.all(v >= 0), 'Invalid counts/effective volumes')
        counts += n
        volume += v
    centers = (edges[:-1] + edges[1:]) / 2
    valid = (volume > 0) & (counts >= plan['minimum_pooled_count']) & radial_mask(centers, plan)
    density = np.divide(counts, volume, out=np.zeros_like(counts), where=valid)
    science = (edges[:-1] >= plan['science_redshift'][0]) & (edges[1:] <= plan['science_redshift'][1])
    check(valid[:, science].all(), 'Unpopulated/unsupported science bins; no extrapolation allowed')
    return dict(schema='cfm-lowz-binned-selection-v1', fit_phases=phases,
                out_of_fold_phase=omitted, bin_edges=edges.tolist(),
                density_unit='(Mpc/h)^-3', context_redshift=plan['context_redshift'],
                excluded_sentinel_redshift=plan['excluded_sentinel_redshift'],
                caps={name: dict(density=density[i].tolist(), valid=valid[i].tolist(),
                    pooled_counts=counts[i].tolist(), effective_volume=volume[i].tolist())
                    for i, name in enumerate(CAPS)},
                estimator=plan['selection_estimator'], model_inputs_ready=False,
                closure_is_independent_validation=False)


def density_at(z, selection, cap):
    """No interpolation clamp, unit conversion, or extrapolation into missing bins."""
    check(selection['schema'] == 'cfm-lowz-binned-selection-v1'
          and selection['density_unit'] == '(Mpc/h)^-3', 'Wrong selection type/units')
    z = np.asarray(z, dtype=float)
    edges = np.asarray(selection['bin_edges'])
    check(edges.ndim == 1 and len(edges) >= 2 and np.all(np.diff(edges) > 0), 'Invalid selection bins')
    values = np.asarray(selection['caps'][cap]['density'], dtype=float)
    known = np.asarray(selection['caps'][cap]['valid'], dtype=bool)
    check(values.shape == known.shape == (len(edges) - 1,)
          and np.isfinite(values).all() and np.all(values >= 0), 'Invalid selection density')
    index = np.searchsorted(edges, z, side='right') - 1
    safe = np.clip(index, 0, len(edges) - 2)
    valid = radial_mask(z, selection) & (index >= 0) & (index < len(edges) - 1) & known[safe]
    return np.where(valid, values[safe], 0.), valid


def response_chunk(spec, slices, angular, cap, plan, selection=None, boundary=None):
    """Origin-safe response and matched effective volume; no galaxy truth access.

    Unlike the old Mpc selection, density is fitted in the same Mpc/h coordinate
    frame as this grid, so applying its inherited volume Jacobian would be wrong.
    """
    import healpy as hp
    from scipy.ndimage import gaussian_filter
    from workflows.sbi import e2e_coupled_coordinates as coord
    from workflows.abacus_tweb import p3a_build_canonical_fields as grid

    sigma = 4.0596 / spec.cell_mpc  # fixed physical apodization, also on refinement
    halo = int(4 * sigma + .5)
    gx, gy, gz = grid.coordinate_block(spec, slices, halo=halo)
    radius = np.sqrt(gx * gx + gy * gy + gz * gz)
    shape = radius.shape
    # Give HEALPix a valid temporary vector at the origin; exclude it afterwards.
    safe = np.where(radius > 0, radius, 1.)
    ux = np.broadcast_to(gx, shape) / safe
    uy = np.broadcast_to(gy, shape) / safe
    uz = np.broadcast_to(gz, shape) / safe
    pix = hp.vec2pix(256, np.where(radius > 0, ux, 1.), uy, uz, nest=False)
    z = coord.redshift(radius)
    support_map = angular['support'].astype(bool) & (angular['domain'] // 2 == CAPS.index(cap))
    binary = radial_mask(z, plan) & (radius > 0) & support_map[pix]
    apod = gaussian_filter(binary.astype(np.float32), sigma=sigma, mode='constant', cval=0., truncate=4.)
    trim = tuple(slice(halo, halo + sl.stop - sl.start) for sl in slices)
    z, binary, pix = z[trim], binary[trim], pix[trim]
    apod = np.clip(apod[trim], 0., 1.)
    response = np.asarray(angular['angular_response'][pix], dtype=np.float64) * binary
    volume = response * apod * spec.cell_mpc**3
    result = dict(support_random=binary.astype(np.uint8), angular_response=response.astype(np.float32),
                  exposure_apodized_random=apod, effective_volume=volume,
                  los_x=ux[trim].astype(np.float32), los_y=uy[trim].astype(np.float32),
                  los_z=uz[trim].astype(np.float32), los_valid=(radius[trim] > 0).astype(np.uint8))
    if selection is not None:
        nbar, valid = density_at(z, selection, cap)
        result.update(ntilde_h3_mpc3=nbar.astype(np.float32),
                      selection_valid=(valid & binary).astype(np.uint8),
                      expected_counts_random=(nbar * volume).astype(np.float32))
    if boundary is not None:
        distances = [np.abs(radius[trim] - coord.radius_mpc_h(edge))
                     for edge in plan['context_redshift'] + plan['excluded_sentinel_redshift']]
        result['distance_to_support_boundary'] = (np.minimum(
            boundary[pix] * radius[trim], np.minimum.reduce(distances)) * binary).astype(np.float32)
    check(all(np.isfinite(v).all() for v in result.values()), 'Nonfinite response')
    return result, z


def support_grid(angular, cap, cell):
    """A random-support-only box including the observer, not a galaxy bbox."""
    import healpy as hp
    from workflows.sbi import e2e_coupled_coordinates as coord
    from workflows.abacus_tweb.p3a_build_canonical_fields import GridSpec

    pix = np.flatnonzero(angular['support'].astype(bool) & (angular['domain'] // 2 == CAPS.index(cap)))
    check(len(pix) > 0, 'Empty cap support')
    direction = np.asarray(hp.pix2vec(256, pix))
    outer = float(coord.radius_mpc_h(.60))
    padding = coord.config()['padding_mpc_h'] + outer * hp.max_pixrad(256)
    lo = np.minimum(direction.min(axis=1) * outer, 0.) - padding
    hi = np.maximum(direction.max(axis=1) * outer, 0.) + padding
    origin = np.floor(lo / cell) * cell
    shape = np.ceil((hi - origin) / cell).astype(int)
    return GridSpec(tuple(float(v) for v in origin), tuple(int(v) for v in shape), cell, float(padding))


def source_payload(root, phase, relative, source_hash, target_free=False):
    """Guard receipt identity before hashing or opening any scientific payload."""
    import re
    receipt_path = root / 'observations' / phase / relative
    row = read_json(receipt_path)
    check(row.get('pass') is True and row['phase'] == phase
          and row['config_sha256'] == source_hash, 'Unqualified source receipt')
    if target_free:
        check(row.get('target_columns_present') is False, 'Target-bearing observations')
    check(len(row['outputs']) == 1, 'Ambiguous source payload')
    record = row['outputs'][0]
    for name in (record['path'], str(Path(record['path']).resolve())):
        phases = set(re.findall(r'ph\d{3}', name))
        check(phases == {phase}, 'Mismatched/ambiguous payload phase')
    path = verify_payload(record)
    return path, dict(receipt=str(receipt_path), receipt_sha256=sha256(receipt_path), payload=record)


def statistics(plan_path, phase, refinement=1, selection_path=None):
    require_compute()
    plan, policy, source = load_plan(plan_path)
    plan_hash, implementation = sha256(plan_path), code_bindings()
    phase_guard(phase, policy)  # before all phase payload IO and output creation
    check(refinement in (1, 2), 'Only native or factor-two refinement registered')
    import fitsio
    from workflows.sbi import e2e_coupled_coordinates as coord
    from workflows.abacus_tweb import p3a_build_canonical_fields as grid
    from workflows.abacus_tweb.p3br_build_random_response import galactic_cap
    from workflows.abacus_tweb.p3br_build_random_response import angular_boundary_distance

    coord.table()
    root = Path(source['root'])
    observed, observed_source = source_payload(root, phase, 'OBSERVED_COMPLETE.json',
                                               plan['source_campaign']['sha256'], True)
    angular_path, angular_source = source_payload(root, phase, 'angular/ANGULAR_COMPLETE.json',
                                                 plan['source_campaign']['sha256'])
    selection = None
    selection_hash = None
    if selection_path is not None:
        selection_hash = sha256(selection_path)
        selection = read_json(selection_path)
        expected = selected_phases(policy, selection.get('out_of_fold_phase'))
        check(selection['fit_phases'] == expected
              and selection['plan_sha256'] == plan_hash
              and selection['code_bindings'] == implementation, 'Selection binding differs')
        check(refinement == 1, 'Response writer uses native cells; refine statistics separately')
        name = 'full' if selection['out_of_fold_phase'] is None else 'oof_' + selection['out_of_fold_phase']
        section = 'responses_' + name
    else:
        section = f'statistics_r{refinement}'
    directory = Path(plan['output_root']) / section / phase
    directory.mkdir(parents=True, exist_ok=False)
    with np.load(angular_path, allow_pickle=False) as data:
        angular = {k: data[k] for k in ('support', 'domain', 'angular_response')}
    check(all(v.shape == (12 * 256**2,) for v in angular.values())
          and np.isfinite(angular['angular_response']).all()
          and np.all(angular['angular_response'] >= 0), 'Invalid angular maps')
    rows = fitsio.read(observed, columns=['RA', 'DEC', 'Z'])
    check(all(np.isfinite(rows[k]).all() for k in ('RA', 'DEC', 'Z')), 'Nonfinite observations')
    rows = rows[radial_mask(rows['Z'], plan)]
    cap_id = galactic_cap(rows['RA'], rows['DEC'])
    xyz = coord.sky_mpc_h(rows['RA'], rows['DEC'], rows['Z'])
    edges = bin_edges(plan)
    counts, volume = np.zeros((2, len(edges)-1)), np.zeros((2, len(edges)-1))
    diagnostics, outputs = {}, []
    for cap, name in enumerate(CAPS):
        spec = support_grid(angular, name, coord.config()['raw_cell_mpc_h'] / refinement)
        values, audit = grid.cic_deposit(xyz[cap_id == cap], spec)
        check(audit['lost_weight'] == 0 and np.isclose(values.sum(dtype=np.float64),
              int((cap_id == cap).sum()), rtol=2e-6), 'CIC conservation failure')
        excluded = 0.
        expected_counts = np.zeros(len(edges) - 1)
        boundary = None
        with ExitStack() as stack:
            fields = {}
            chunks = tuple(min(n, 64) for n in spec.shape)
            if selection is not None:
                import h5py
                path = directory / (name + '_response.h5')
                out = stack.enter_context(h5py.File(path, 'x'))
                out.attrs.update(schema='cfm-lowz-response-preparation-v1', phase=phase, cap=name,
                    origin_mpc_h=spec.origin, cell_mpc_h=spec.cell_mpc, distance_unit='Mpc/h',
                    plan_sha256=plan_hash, selection_sha256=selection_hash,
                    model_inputs_ready=False)
                boundary = angular_boundary_distance(angular['support'].astype(bool)
                            & (angular['domain'] // 2 == cap))
            for i, slices in enumerate(grid.iter_chunks(spec.shape, chunks)):
                response, z = response_chunk(spec, slices, angular, name, plan, selection, boundary)
                good = response['effective_volume'] > 0
                counts[cap] += np.histogram(z[good], edges, weights=values[slices][good].astype(float))[0]
                volume[cap] += np.histogram(z[good], edges, weights=response['effective_volume'][good])[0]
                excluded += float(values[slices][~good].sum(dtype=np.float64))
                if selection is not None:
                    expected_counts += np.histogram(z[good], edges,
                                       weights=response['expected_counts_random'][good].astype(float))[0]
                    response['counts'] = values[slices]
                    # Invalid selection is a separate mask, never evidence of a void.
                    response['log_count_ratio_random'] = grid.log_count_ratio(
                        values[slices], response['expected_counts_random'],
                        response['exposure_apodized_random'] * response['selection_valid'], 1e-3, 1e-4)
                    if not fields:
                        fields = {k: out.create_dataset(k, shape=spec.shape, dtype=v.dtype,
                                  chunks=chunks, compression='lzf', shuffle=True) for k, v in response.items()}
                    for key, value in response.items():
                        fields[key][slices] = value
                if i % 100 == 0:
                    print(json.dumps(dict(phase=phase, cap=name, chunk=i)), flush=True)
        if selection is not None:
            outputs.append(dict(path=str(path), sha256=sha256(path), bytes=path.stat().st_size))
        diagnostics[name] = dict(grid=dict(origin_mpc_h=list(spec.origin), shape=list(spec.shape),
             cell_mpc_h=spec.cell_mpc), cic=audit, cic_weight_without_response=excluded,
             catalogue_counts=np.histogram(rows['Z'][cap_id == cap], edges)[0].tolist(),
             expected_counts=None if selection is None else expected_counts.tolist())
        del values
    check(sha256(plan_path) == plan_hash and code_bindings() == implementation,
          'Plan or implementation changed during preparation')
    if selection_path is not None:
        check(sha256(selection_path) == selection_hash, 'Selection changed during preparation')
    result = dict(schema='cfm-lowz-selection-statistics-v1', phase=phase, label_free=True,
                  plan_sha256=plan_hash, code_bindings=implementation, refinement=refinement,
                  bin_edges=edges.tolist(), counts=counts.tolist(), volume=volume.tolist(),
                  sources=dict(observed=observed_source, angular=angular_source),
                  diagnostics=diagnostics, model_inputs_ready=False)
    if selection is not None:
        result.update(schema='cfm-lowz-response-preparation-v1', outputs=outputs,
                      selection=dict(path=str(selection_path), sha256=selection_hash))
    save_json(directory / ('STATISTICS.json' if selection is None else 'RESPONSE_PREPARED.json'), result)
    return result


def fit(plan_path, omitted=None):
    plan, policy, _ = load_plan(plan_path)
    phases = selected_phases(policy, omitted)
    records, sources = {}, {}
    for phase in phases:
        path = Path(plan['output_root']) / 'statistics_r1' / phase / 'STATISTICS.json'
        row = read_json(path)
        check(row['schema'] == 'cfm-lowz-selection-statistics-v1'
              and row['plan_sha256'] == sha256(plan_path)
              and row['code_bindings'] == code_bindings() and row['refinement'] == 1,
              'Statistics contract/code drift')
        records[phase] = row
        sources[phase] = dict(path=str(path), sha256=sha256(path))
    selection = fit_statistics(records, plan, policy, omitted)
    selection.update(plan_sha256=sha256(plan_path), code_bindings=code_bindings(), sources=sources)
    directory = Path(plan['output_root']) / ('full' if omitted is None else 'oof_' + omitted)
    directory.mkdir(parents=True, exist_ok=False)
    save_json(directory / 'SELECTION.json', selection)
    return selection


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, default=DEFAULT_PLAN)
    sub = p.add_subparsers(dest='stage', required=True)
    s = sub.add_parser('statistics')
    s.add_argument('--phase', required=True)
    s.add_argument('--refinement', type=int, choices=(1, 2), default=1)
    f = sub.add_parser('fit')
    f.add_argument('--omitted-phase')
    r = sub.add_parser('response')
    r.add_argument('--phase', required=True)
    r.add_argument('--selection', type=Path, required=True)
    a = p.parse_args()
    if a.stage == 'statistics':
        statistics(a.plan, a.phase, a.refinement)
    elif a.stage == 'fit':
        fit(a.plan, a.omitted_phase)
    else:
        statistics(a.plan, a.phase, selection_path=a.selection)


if __name__ == '__main__':
    main()
