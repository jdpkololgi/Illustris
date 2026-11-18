"""Match CIC count refinement to the separate low-z volume diagnostic.

This reads only observed positions and random geometry in two permitted phases.
Per-phase count/volume ratios diagnose the operator, not a fitted selection;
development ratios must never be substituted into training products.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from workflows.sbi import cfm_lowz_selection_response as lowz
from workflows.sbi import diagnose_cfm_lowz_volume as volume
from workflows.sbi.prepare_cfm_lowz_inputs import require_compute, save_json, sha256


def cropped_counts(native, xyz, radius, factor):
    """Crop the SAME lattice with a conservative CIC margin around a sphere.

    Keep every point whose CIC stencil intersects the crop, including points
    just outside it. Deposited weight outside the crop is reported, not treated
    as a conservation failure: every retained voxel must equal the full grid.
    Cap boxes need only cover their own footprint, not the entire sphere.
    """
    from workflows.abacus_tweb.p3a_build_canonical_fields import GridSpec, cic_deposit
    if factor not in (1, 2, 4) or not np.isfinite(radius) or radius <= 0:
        raise ValueError('Unsupported count refinement')
    cell = native.cell_mpc/factor
    origin = np.asarray(native.origin)
    extent = radius+3*native.cell_mpc
    start = np.maximum(0, np.floor((-extent-origin)/cell).astype(int))
    stop = np.minimum(np.array(native.shape)*factor, np.ceil((extent-origin)/cell).astype(int))
    local_origin = origin+start*cell
    shape = stop-start
    if (shape <= 0).any():
        raise ValueError('Source grid does not intersect the diagnostic region')
    spec = GridSpec(tuple(local_origin), tuple(shape), cell, native.padding_mpc)
    xyz = np.asarray(xyz, dtype=float)
    keep = ((xyz >= local_origin-.5*cell) & (xyz < local_origin+(shape+.5)*cell)).all(axis=1)
    counts, audit = cic_deposit(xyz[keep], spec)
    audit.update(input_rows_total=len(xyz), rows_with_no_stencil_overlap=int((~keep).sum()),
                 lost_weight_scope='outside diagnostic crop, not survey deposition loss',
                 crop_preserves_full_lattice_cells=True)
    return spec, counts, audit


def run(phase):
    require_compute()
    if phase not in ('ph001', 'ph006'):
        raise ValueError('Only the two registered numerical canaries are permitted')
    import fitsio
    from workflows.sbi import e2e_coupled_coordinates as coord
    from workflows.abacus_tweb.p3br_build_random_response import galactic_cap
    from workflows.abacus_tweb.p3a_build_canonical_fields import iter_chunks
    plan, policy, source = lowz.load_plan()
    lowz.phase_guard(phase, policy)
    root = Path(plan['output_root'])
    output = root/('MATCHED_COUNT_REFINEMENT_' + phase + '.json')
    if output.exists():
        raise FileExistsError(output)
    diagnostic_path = root/('VOLUME_REFINEMENT_DIAGNOSTIC_' + phase + '.json')
    reference = lowz.read_json(diagnostic_path)
    bindings = lowz.code_bindings()
    if reference['phase'] != phase or reference['code_bindings'] != bindings:
        raise ValueError('Volume diagnostic binding differs')
    edges = np.asarray(reference['edges'])
    if not np.array_equal(edges, lowz.bin_edges(plan)[2:11]):
        raise ValueError('Unexpected diagnostic interval')
    statistics = lowz.read_json(root/'statistics_r1'/phase/'STATISTICS.json')
    observed, observed_binding = lowz.source_payload(Path(source['root']), phase,
        'OBSERVED_COMPLETE.json', plan['source_campaign']['sha256'], True)
    angular_path, angular_binding = lowz.source_payload(Path(source['root']), phase,
        'angular/ANGULAR_COMPLETE.json', plan['source_campaign']['sha256'])
    if angular_binding != reference['angular_source']:
        raise ValueError('Angular source differs from volume diagnostic')
    with np.load(angular_path, allow_pickle=False) as data:
        angular = {k: data[k] for k in ('support', 'domain', 'angular_response')}
    rows = fitsio.read(observed, columns=['RA', 'DEC', 'Z'])
    if not all(np.isfinite(rows[k]).all() for k in ('RA', 'DEC', 'Z')):
        raise ValueError('Nonfinite observed coordinate')
    rows = rows[lowz.radial_mask(rows['Z'], plan)]
    caps = galactic_cap(rows['RA'], rows['DEC'])
    xyz = coord.sky_mpc_h(rows['RA'], rows['DEC'], rows['Z'])
    implementation = sha256(__file__)
    results = {}
    for cap_id, cap in enumerate(lowz.CAPS):
        native = lowz.support_grid(angular, cap, coord.config()['raw_cell_mpc_h'])
        levels = {}
        for factor in (1, 2, 4):
            spec, counts, audit = cropped_counts(native, xyz[caps == cap_id],
                                                 float(coord.radius_mpc_h(edges[-1])), factor)
            totals, volumes = np.zeros(len(edges)-1), np.zeros(len(edges)-1)
            for slices in iter_chunks(spec.shape, tuple(min(n, 64) for n in spec.shape)):
                response, z = lowz.response_chunk(spec, slices, angular, cap, plan)
                valid = response['effective_volume'] > 0
                totals += np.histogram(z[valid], edges, weights=counts[slices][valid].astype(float))[0]
                volumes += np.histogram(z, edges, weights=response['effective_volume'])[0]
            del counts
            expected_volume = reference['caps'][cap]['levels'][str(factor)]['effective']
            np.testing.assert_allclose(volumes, expected_volume, rtol=1e-9, atol=1e-6)
            if factor == 1:
                np.testing.assert_allclose(totals, np.asarray(statistics['counts'])[cap_id, 2:10],
                                           rtol=1e-6, atol=1e-5)
            if not np.all(volumes > 0) or not np.all(totals > 0):
                raise ValueError('Missing diagnostic counts/volume')
            levels[str(factor)] = dict(counts=totals.tolist(), effective_volume=volumes.tolist(),
                                      phase_count_over_volume=(totals/volumes).tolist(), deposition=audit)
            print(phase, cap, 'matched refinement', factor, 'complete', flush=True)
        comparisons = {}
        for coarse, fine in ((1, 2), (2, 4), (1, 4)):
            a, b = levels[str(coarse)], levels[str(fine)]
            ga = volume.group_volume(a['counts'], edges)/volume.group_volume(a['effective_volume'], edges)
            gb = volume.group_volume(b['counts'], edges)/volume.group_volume(b['effective_volume'], edges)
            comparisons[f'{coarse}_over_{fine}_minus_one'] = dict(
                counts=volume.relative_change(a['counts'], b['counts']),
                phase_count_over_volume=volume.relative_change(a['phase_count_over_volume'], b['phase_count_over_volume']),
                grouped_phase_count_over_volume=volume.relative_change(ga, gb))
        results[cap] = dict(levels=levels, comparisons=comparisons)
    if lowz.code_bindings() != bindings or sha256(__file__) != implementation:
        raise ValueError('Implementation changed during diagnostic')
    save_json(output, dict(phase=phase, job=os.environ['SLURM_JOB_ID'], caps=results, edges=edges.tolist(),
        volume_diagnostic_sha256=sha256(diagnostic_path), diagnostic_source_sha256=implementation,
        code_bindings=bindings, observed_source=observed_binding, angular_source=angular_binding,
        selection_fit=False, development_used_for_fitting=False, model_inputs_ready=False))
    print('SAVED', output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('ph001', 'ph006'), required=True)
    run(parser.parse_args().phase)
