"""Unit-density forward response using the SAME CIC kernel as observed counts.

This is geometry-only numerical evidence, not a replacement selection fit.
The Gaussian-apodized mask remains a separate geometry feature. Here the
expected native-cell count is the integral of angular intensity times the CIC
assignment basis, approximated with nested subcell quadrature.
"""
from __future__ import annotations

import argparse
import itertools
import os
from pathlib import Path

import numpy as np

from workflows.sbi import cfm_lowz_selection_response as lowz
from workflows.sbi import e2e_coupled_coordinates as coord
from workflows.sbi.prepare_cfm_lowz_inputs import require_compute, save_json, sha256


def weighted_cic(xyz, weights, spec, out):
    xyz, weights = np.asarray(xyz, float), np.asarray(weights, float)
    if (xyz.ndim != 2 or xyz.shape[1] != 3 or weights.shape != (len(xyz),)
            or not np.isfinite(xyz).all() or not np.isfinite(weights).all() or (weights < 0).any()
            or out.shape != tuple(spec.shape) or out.dtype != np.float64):
        raise ValueError('Invalid weighted CIC inputs')
    position = (xyz-np.asarray(spec.origin))/spec.cell_mpc-.5
    base = np.floor(position).astype(int)
    fraction = position-base
    retained = 0.
    for corner in itertools.product((0, 1), repeat=3):
        index = base+corner
        weight = weights*np.prod(np.where(np.array(corner), fraction, 1-fraction), axis=1)
        valid = ((index >= 0) & (index < np.asarray(spec.shape))).all(axis=1)
        np.add.at(out, tuple(index[valid].T), weight[valid])
        retained += float(weight[valid].sum())
    return retained


def project_unit_response(native, angular, cap, plan, factor, radius):
    import healpy as hp
    from workflows.abacus_tweb.p3a_build_canonical_fields import GridSpec, iter_chunks, coordinate_block
    if factor not in (2, 4):
        raise ValueError('Only registered half/quarter-cell quadrature')
    # The sphere of audited cell centres is separated from the integration-box
    # boundary by three native cells, exceeding the CIC kernel support.
    origin = np.asarray(native.origin)
    extent = radius+3*native.cell_mpc
    start = np.maximum(0, np.floor((-extent-origin)/native.cell_mpc).astype(int))
    stop = np.minimum(native.shape, np.ceil((extent-origin)/native.cell_mpc).astype(int))
    local = GridSpec(tuple(origin+start*native.cell_mpc), tuple(stop-start), native.cell_mpc, native.padding_mpc)
    fine = GridSpec(local.origin, tuple(n*factor for n in local.shape), local.cell_mpc/factor, local.padding_mpc)
    out = np.zeros(local.shape, dtype=np.float64)
    support = angular['support'].astype(bool) & (angular['domain']//2 == lowz.CAPS.index(cap))
    input_weight, retained_weight = 0., 0.
    for slices in iter_chunks(fine.shape, tuple(min(n, 64) for n in fine.shape)):
        axes = coordinate_block(fine, slices)
        radius_grid = np.sqrt(sum(a*a for a in axes))
        shape = radius_grid.shape
        points = np.stack([np.broadcast_to(a, shape).ravel() for a in axes], axis=-1)
        safe = points.copy()
        safe[radius_grid.ravel() == 0, 0] = 1.
        pixels = hp.vec2pix(256, *safe.T)
        valid = (radius_grid.ravel() > 0) & lowz.radial_mask(coord.redshift(radius_grid.ravel()), plan) & support[pixels]
        weights = angular['angular_response'][pixels[valid]].astype(float)*fine.cell_mpc**3
        retained_weight += weighted_cic(points[valid], weights, local, out)
        input_weight += float(weights.sum())
    return local, out, dict(input_integrated_weight=input_weight, deposited_weight=retained_weight,
                           outside_crop_weight=input_weight-retained_weight)


def run(phase):
    require_compute()
    if phase != 'ph001':
        raise ValueError('Training-only operator canary: ph001')
    from workflows.abacus_tweb.p3a_build_canonical_fields import iter_chunks
    plan, policy, source = lowz.load_plan()
    lowz.phase_guard(phase, policy)
    root = Path(plan['output_root'])
    output = root/'CIC_UNIT_RESPONSE_DIAGNOSTIC_ph001.json'
    if output.exists():
        raise FileExistsError(output)
    angular_path, binding = lowz.source_payload(Path(source['root']), phase,
        'angular/ANGULAR_COMPLETE.json', plan['source_campaign']['sha256'])
    with np.load(angular_path, allow_pickle=False) as data:
        angular = {k: data[k] for k in ('support', 'domain', 'angular_response')}
    edges = lowz.bin_edges(plan)[2:11]
    results = {}
    for cap in lowz.CAPS:
        native = lowz.support_grid(angular, cap, coord.config()['raw_cell_mpc_h'])
        levels = {}
        previous = None
        for factor in (2, 4):
            spec, expected, audit = project_unit_response(native, angular, cap, plan, factor,
                                                          float(coord.radius_mpc_h(edges[-1])))
            totals, apodized = np.zeros(len(edges)-1), np.zeros(len(edges)-1)
            l1, reference_sum = 0., 0.
            for slices in iter_chunks(spec.shape, tuple(min(n, 64) for n in spec.shape)):
                response, z = lowz.response_chunk(spec, slices, angular, cap, plan)
                valid = response['support_random'].astype(bool)
                totals += np.histogram(z[valid], edges, weights=expected[slices][valid])[0]
                apodized += np.histogram(z, edges, weights=response['effective_volume'])[0]
                if previous is not None:
                    selected = valid & (z >= edges[0]) & (z < edges[-1])
                    l1 += float(np.abs(expected[slices][selected]-previous[slices][selected]).sum())
                    reference_sum += float(expected[slices][selected].sum())
            levels[str(factor)] = dict(retained_unit_density_expected_counts=totals.tolist(),
                apodized_volume=apodized.tolist(), quadrature_audit=audit)
            if previous is not None:
                levels[str(factor)]['previous_vs_current_cellwise_l1_relative'] = l1/reference_sum
            previous = expected
            print(phase, cap, 'CIC response quadrature', factor, 'complete', flush=True)
        coarse, fine = (np.asarray(levels[str(f)]['retained_unit_density_expected_counts']) for f in (2, 4))
        results[cap] = dict(levels=levels, half_over_quarter_minus_one=(coarse/fine-1).tolist(),
            apodized_over_matched_cic_minus_one=(np.asarray(levels['4']['apodized_volume'])/fine-1).tolist())
    save_json(output, dict(phase=phase, job=os.environ['SLURM_JOB_ID'], edges=edges.tolist(), caps=results,
        angular_source=binding, source_sha256=sha256(__file__), code_bindings=lowz.code_bindings(),
        intensity='unit radial density times angular_response on registered support',
        galaxy_counts_read=False, selection_fit=False, model_inputs_ready=False))
    print('SAVED', output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('ph001',), required=True)
    run(parser.parse_args().phase)
