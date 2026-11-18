"""Independent phase/cap closure reporting and bounded low-z volume refinement.

No fit is tuned to development residuals. The volume screen is predeclared at
1% in each delta-z=.005 science bin below .15; it is numerical, not a posterior
calibration or full-range resolution qualification.
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np

from workflows.sbi import cfm_lowz_selection_response as lowz
from workflows.sbi.prepare_cfm_lowz_inputs import require_compute, save_json, sha256


def clipped_chunks(spec, radius):
    origin = np.array(spec.origin)
    start = np.maximum(0, np.floor((-radius-origin)/spec.cell_mpc).astype(int))
    stop = np.minimum(spec.shape, np.ceil((radius-origin)/spec.cell_mpc).astype(int))
    for index in itertools.product(*(range(a, b, 64) for a, b in zip(start, stop))):
        yield tuple(slice(i, min(i+64, int(end))) for i, end in zip(index, stop))


def volume_audit(phase):
    require_compute()
    from workflows.sbi import e2e_coupled_coordinates as coord
    from workflows.abacus_tweb.p3a_build_canonical_fields import GridSpec
    plan, policy, source = lowz.load_plan()
    lowz.phase_guard(phase, policy)
    root = Path(plan['output_root'])
    reference_path = root / 'statistics_r1' / phase / 'STATISTICS.json'
    reference = lowz.read_json(reference_path)
    if reference['code_bindings'] != lowz.code_bindings() or reference['plan_sha256'] != sha256(lowz.DEFAULT_PLAN):
        raise ValueError('Native statistics binding differs')
    angular_path, binding = lowz.source_payload(Path(source['root']), phase, 'angular/ANGULAR_COMPLETE.json',
                                              plan['source_campaign']['sha256'])
    with np.load(angular_path) as data:
        angular = {k: data[k] for k in ('support', 'domain', 'angular_response')}
    edges = lowz.bin_edges(plan)
    edges = edges[(edges >= .01) & (edges <= .15)]
    outputs = {}
    for cap, name in enumerate(lowz.CAPS):
        native = lowz.support_grid(angular, name, coord.config()['raw_cell_mpc_h'])
        volumes = []
        for refinement in (1, 2):
            spec = GridSpec(native.origin, tuple(n*refinement for n in native.shape),
                            native.cell_mpc/refinement, native.padding_mpc)
            total = np.zeros(len(edges)-1)
            for slices in clipped_chunks(spec, float(coord.radius_mpc_h(.15))):
                response, z = lowz.response_chunk(spec, slices, angular, name, plan)
                total += np.histogram(z, edges, weights=response['effective_volume'])[0]
            volumes.append(total)
        full_edges = np.array(reference['bin_edges'])
        mask = (full_edges[:-1] >= .01) & (full_edges[1:] <= .15)
        np.testing.assert_allclose(volumes[0], np.array(reference['volume'])[cap, mask], rtol=1e-9, atol=1e-6)
        if not np.all(volumes[1] > 0):
            raise ValueError('Unsupported refined low-z bin')
        relative = volumes[0]/volumes[1]-1
        outputs[name] = dict(native_volume=volumes[0].tolist(), half_cell_volume=volumes[1].tolist(),
            native_over_refined_minus_one=relative.tolist(), max_abs_fractional_change=float(abs(relative).max()),
            numerical_1pct_screen_pass=bool(np.all(abs(relative) <= .01)))
    output = root / ('LOWZ_VOLUME_AUDIT_' + phase + '.json')
    save_json(output, dict(phase=phase, edges=edges.tolist(), caps=outputs, angular_source=binding,
        reference_sha256=sha256(reference_path), audit_source_sha256=sha256(__file__),
        code_bindings=lowz.code_bindings(), fixed_physical_apodization=True,
        screen_tolerance=.01, full_range_convergence_established=False, model_inputs_ready=False))


def grouped_closure(counts, expected, edges, diagnostics):
    result = []
    for lo, hi in zip(diagnostics[:-1], diagnostics[1:]):
        mask = (edges[:-1] >= lo) & (edges[1:] <= hi)
        n, mu = float(counts[mask].sum()), float(expected[mask].sum())
        result.append(dict(z=[lo, hi], observed=n, expected=mu,
                           expected_over_observed_minus_one=None if n == 0 else mu/n-1))
    return result


def closure_audit():
    require_compute()
    plan, policy, _ = lowz.load_plan()
    root = Path(plan['output_root'])
    full = lowz.read_json(root / 'full/SELECTION.json')
    phases = policy['training'] + policy['development_and_calibration']
    records, inputs = {}, {}
    for phase in phases:
        path = root / 'statistics_r1' / phase / 'STATISTICS.json'
        row = lowz.read_json(path)
        inputs[str(path)] = sha256(path)
        if row['phase'] != phase or row['code_bindings'] != lowz.code_bindings():
            raise ValueError('Statistics binding differs')
        selection_path = root / ('oof_' + phase if phase in policy['training'] else 'full') / 'SELECTION.json'
        selection = lowz.read_json(selection_path)
        expected_phases = [p for p in policy['training'] if p != phase]
        if selection['fit_phases'] != expected_phases:
            raise ValueError('Independent selection phase leakage')
        inputs[str(selection_path)] = sha256(selection_path)
        bycap = {}
        for i, cap in enumerate(lowz.CAPS):
            counts, volume = np.array(row['counts'])[i], np.array(row['volume'])[i]
            expected = volume * np.array(selection['caps'][cap]['density'])
            full_expected = volume * np.array(full['caps'][cap]['density'])
            response_path = root / 'responses_full' / phase / 'RESPONSE_PREPARED.json'
            response = lowz.read_json(response_path)
            inputs[str(response_path)] = sha256(response_path)
            np.testing.assert_allclose(response['counts'], row['counts'], rtol=0, atol=0)
            np.testing.assert_allclose(response['volume'], row['volume'], rtol=0, atol=0)
            np.testing.assert_allclose(response['diagnostics'][cap]['expected_counts'], full_expected,
                                       rtol=2e-6, atol=1e-5)
            bycap[cap] = grouped_closure(counts, expected, np.array(row['bin_edges']), plan['diagnostic_bin_edges'])
        records[phase] = dict(role='train_oof' if phase in policy['training'] else 'development',
                              selection_sha256=sha256(selection_path), caps=bycap)
    save_json(root / 'INDEPENDENT_CLOSURE_DIAGNOSTICS.json', dict(records=records, inputs=inputs,
        audit_source_sha256=sha256(__file__), no_development_tuning=True,
        pooled_closure_is_validation=False, response_arithmetic_pass=True, model_inputs_ready=False))


def payload_audit():
    """Rehash payloads and independently replay scalar response identities."""
    require_compute()
    import h5py
    import healpy as hp
    from workflows.sbi import e2e_coupled_coordinates as coord
    plan, policy, _ = lowz.load_plan()
    root = Path(plan['output_root'])
    selection_path = root / 'full/SELECTION.json'
    selection = lowz.read_json(selection_path)
    if selection['fit_phases'] != policy['training']:
        raise ValueError('Full selection phase mismatch')
    result = {}
    for phase in policy['training'] + policy['development_and_calibration']:
        receipt_path = root / 'responses_full' / phase / 'RESPONSE_PREPARED.json'
        receipt = lowz.read_json(receipt_path)
        if (receipt['phase'] != phase or receipt['selection']['sha256'] != sha256(selection_path)
                or receipt['code_bindings'] != lowz.code_bindings()):
            raise ValueError('Response receipt binding changed')
        angular_path = Path(receipt['sources']['angular']['payload']['path'])
        if sha256(angular_path) != receipt['sources']['angular']['payload']['sha256']:
            raise ValueError('Angular payload hash mismatch')
        with np.load(angular_path) as data:
            angular = {k: data[k] for k in ('support', 'domain', 'angular_response')}
        bycap = {}
        for cap, record in zip(lowz.CAPS, receipt['outputs']):
            path = Path(record['path'])
            if path.stat().st_size != record['bytes'] or sha256(path) != record['sha256']:
                raise ValueError('Response payload hash mismatch: ' + str(path))
            support_map = angular['support'].astype(bool) & (angular['domain']//2 == lowz.CAPS.index(cap))
            pix = np.flatnonzero(support_map)
            chosen = pix[np.linspace(0, len(pix)-1, 8, dtype=int)]
            directions = np.asarray(hp.pix2vec(256, chosen)).T
            count, supported = 0, 0
            with h5py.File(path) as handle:
                if handle.attrs['cap'] != cap or handle.attrs['phase'] != phase:
                    raise ValueError('HDF5 phase/cap mismatch')
                origin = np.asarray(handle.attrs['origin_mpc_h'])
                cell = float(handle.attrs['cell_mpc_h'])
                for redshift in (.02, .10, .40):
                    xyz = directions * float(coord.radius_mpc_h(redshift))
                    ids = np.floor((xyz-origin)/cell).astype(int)
                    for index in ids:
                        key = tuple(index)
                        centre = origin+(index+.5)*cell
                        radius = float(np.linalg.norm(centre))
                        z = float(coord.redshift(radius))
                        pixel = hp.vec2pix(256, *centre)
                        binary = bool(support_map[pixel] and lowz.radial_mask(z, plan))
                        edges = np.asarray(selection['bin_edges'])
                        radial_bin = int(np.searchsorted(edges, z, side='right')-1)
                        valid = binary and bool(selection['caps'][cap]['valid'][radial_bin])
                        nbar = selection['caps'][cap]['density'][radial_bin]
                        apod = float(handle['exposure_apodized_random'][key])
                        angular_response = float(angular['angular_response'][pixel])*binary
                        expected = nbar*angular_response*apod*cell**3
                        if bool(handle['support_random'][key]) != binary or bool(handle['selection_valid'][key]) != valid:
                            raise ValueError('Independent support/selection replay failed')
                        np.testing.assert_allclose(handle['expected_counts_random'][key], expected, rtol=2e-6, atol=1e-7)
                        np.testing.assert_allclose([handle['los_'+axis][key] for axis in 'xyz'], centre/radius,
                                                   rtol=2e-6, atol=1e-7)
                        if not np.isfinite(handle['counts'][key]) or handle['counts'][key] < 0:
                            raise ValueError('Invalid raw counts')
                        count += 1
                        supported += int(binary)
            bycap[cap] = dict(samples=count, supported_samples=supported, payload_sha256=record['sha256'])
        result[phase] = bycap
    save_json(root / 'PAYLOAD_AUDIT.json', dict(phases=result, numerical_identity_pass=True,
        all_payload_hashes_verified=True, audit_source_sha256=sha256(__file__), model_inputs_ready=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('volume', 'closure', 'payload'))
    parser.add_argument('--phase', choices=('ph001', 'ph006'))
    a = parser.parse_args()
    if a.stage == 'volume':
        if a.phase is None:
            parser.error('volume requires --phase')
        volume_audit(a.phase)
    elif a.stage == 'closure':
        closure_audit()
    else:
        payload_audit()
