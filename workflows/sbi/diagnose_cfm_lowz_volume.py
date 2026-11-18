"""Post-screen, geometry-only refinement diagnostic; never promotes readiness.

The .01-.05 interval was chosen AFTER the original .01-.15 screen localized
its >1% discrepancies below .035. Keep that original evidence unchanged.
Compare nested native/half/quarter cells with a fixed physical mask smoothing.
An analytic unsmoothed-volume reference separates integration of the angular
footprint from smoothing sensitivity. Counts are NOT re-deposited here, so a
refined denominator must not be substituted into the native selection fit.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from workflows.sbi import cfm_lowz_selection_response as lowz
from workflows.sbi.audit_cfm_lowz_survey import clipped_chunks
from workflows.sbi.prepare_cfm_lowz_inputs import require_compute, save_json, sha256


def analytic_volume(angular, cap, radii):
    """Exact piecewise-constant HEALPix footprint integral, WITHOUT apodization."""
    import healpy as hp
    radii = np.asarray(radii, dtype=float)
    if not (radii.ndim == 1 and len(radii) >= 2 and np.isfinite(radii).all()
            and radii[0] > 0 and np.all(np.diff(radii) > 0)):
        raise ValueError('Invalid radial shell edges')
    support = angular['support'].astype(bool) & (angular['domain']//2 == lowz.CAPS.index(cap))
    response = np.asarray(angular['angular_response'], dtype=float)
    if response.shape != support.shape or not np.isfinite(response).all() or (response < 0).any():
        raise ValueError('Invalid angular response')
    solid_angle = hp.nside2pixarea(hp.npix2nside(len(support))) * response[support].sum()
    return solid_angle * np.diff(radii**3) / 3.


def relative_change(coarse, fine):
    coarse, fine = np.asarray(coarse, dtype=float), np.asarray(fine, dtype=float)
    if (coarse.shape != fine.shape or not np.isfinite(coarse).all()
            or not np.isfinite(fine).all() or (coarse <= 0).any() or (fine <= 0).any()):
        raise ValueError('Invalid positive volume comparison')
    return (coarse / fine - 1.).tolist()


def group_volume(values, edges, grouped=(.01, .03, .05)):
    edges, values = np.asarray(edges), np.asarray(values)
    if values.shape != (len(edges)-1,) or not all(np.any(np.isclose(edges, x, atol=1e-12, rtol=0)) for x in grouped):
        raise ValueError('Grouped boundaries must coincide with fine bin edges')
    return np.array([values[(edges[:-1] >= lo-1e-12) & (edges[1:] <= hi+1e-12)].sum()
                     for lo, hi in zip(grouped[:-1], grouped[1:])])


def integrate(spec, angular, cap, plan, edges):
    from workflows.sbi import e2e_coupled_coordinates as coord
    effective, geometric = np.zeros(len(edges)-1), np.zeros(len(edges)-1)
    for slices in clipped_chunks(spec, float(coord.radius_mpc_h(edges[-1]))):
        response, z = lowz.response_chunk(spec, slices, angular, cap, plan)
        effective += np.histogram(z, edges, weights=response['effective_volume'])[0]
        geometric += np.histogram(z, edges, weights=response['angular_response'].astype(float)*spec.cell_mpc**3)[0]
    return dict(effective=effective.tolist(), geometric=geometric.tolist())


def run(phase):
    require_compute()
    if phase not in ('ph001', 'ph006'):
        raise ValueError('Only the two registered numerical canaries are permitted')
    from workflows.sbi import e2e_coupled_coordinates as coord
    from workflows.abacus_tweb.p3a_build_canonical_fields import GridSpec
    plan, policy, source = lowz.load_plan()
    lowz.phase_guard(phase, policy)
    root = Path(plan['output_root'])
    output = root / ('VOLUME_REFINEMENT_DIAGNOSTIC_' + phase + '.json')
    if output.exists():
        raise FileExistsError(output)
    original_path = root / ('LOWZ_VOLUME_AUDIT_' + phase + '.json')
    original = lowz.read_json(original_path)
    statistics_path = root / 'statistics_r1' / phase / 'STATISTICS.json'
    statistics = lowz.read_json(statistics_path)
    bindings = lowz.code_bindings()
    if (statistics['code_bindings'] != bindings or original['code_bindings'] != bindings
            or statistics['plan_sha256'] != sha256(lowz.DEFAULT_PLAN)
            or original['reference_sha256'] != sha256(statistics_path)):
        raise ValueError('Original statistics/audit binding changed')
    angular_path, angular_binding = lowz.source_payload(Path(source['root']), phase,
        'angular/ANGULAR_COMPLETE.json', plan['source_campaign']['sha256'])
    with np.load(angular_path, allow_pickle=False) as data:
        angular = {k: data[k] for k in ('support', 'domain', 'angular_response')}
    edges = lowz.bin_edges(plan)
    edges = edges[(edges >= .01) & (edges <= .05)]
    implementation = sha256(__file__)
    results = {}
    for cap in lowz.CAPS:
        native = lowz.support_grid(angular, cap, coord.config()['raw_cell_mpc_h'])
        analytic = analytic_volume(angular, cap, coord.radius_mpc_h(edges))
        levels = {}
        for factor in (1, 2, 4):
            spec = GridSpec(native.origin, tuple(n*factor for n in native.shape),
                            native.cell_mpc/factor, native.padding_mpc)
            levels[str(factor)] = integrate(spec, angular, cap, plan, edges)
            levels[str(factor)]['cell_mpc_h'] = spec.cell_mpc
            if factor in (1, 2):
                old = original['caps'][cap]['native_volume' if factor == 1 else 'half_cell_volume']
                np.testing.assert_allclose(levels[str(factor)]['effective'], old[:len(edges)-1], rtol=1e-9, atol=1e-6)
            print(phase, cap, 'refinement', factor, 'complete', flush=True)
        comparisons = {}
        for coarse, fine in ((1, 2), (2, 4), (1, 4)):
            key = str(coarse) + '_over_' + str(fine) + '_minus_one'
            comparisons[key] = {kind: relative_change(levels[str(coarse)][kind], levels[str(fine)][kind])
                                for kind in ('effective', 'geometric')}
            comparisons[key]['grouped_effective'] = relative_change(
                group_volume(levels[str(coarse)]['effective'], edges),
                group_volume(levels[str(fine)]['effective'], edges))
        results[cap] = dict(levels=levels, comparisons=comparisons,
            analytic_unapodized_volume=analytic.tolist(),
            geometric_over_analytic_minus_one={str(f): relative_change(levels[str(f)]['geometric'], analytic)
                                               for f in (1, 2, 4)})
    if lowz.code_bindings() != bindings or sha256(__file__) != implementation:
        raise ValueError('Implementation changed during diagnostic')
    save_json(output, dict(phase=phase, job=os.environ['SLURM_JOB_ID'], edges=edges.tolist(),
        grouped_edges=[.01, .03, .05], caps=results, original_audit_sha256=sha256(original_path),
        statistics_sha256=sha256(statistics_path), angular_source=angular_binding,
        code_bindings=bindings, diagnostic_source_sha256=implementation,
        scope='post-screen geometry-only diagnostic; not full-range qualification',
        counts_redeposited=False, original_screen_replaced=False, model_inputs_ready=False))
    print('SAVED', output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('ph001', 'ph006'), required=True)
    run(parser.parse_args().phase)
