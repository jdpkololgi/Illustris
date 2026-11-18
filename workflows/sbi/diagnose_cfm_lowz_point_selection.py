"""Grid-free point/solid-angle reference for the provisional CIC estimator.

Diagnostic only: per-phase curves include cosmic structure and are not selection
fits. No development-phase curve may become a training conditioning product.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from workflows.sbi import cfm_lowz_selection_response as lowz
from workflows.sbi import e2e_coupled_coordinates as coord
from workflows.sbi.diagnose_cfm_lowz_volume import analytic_volume, relative_change
from workflows.sbi.prepare_cfm_lowz_inputs import require_compute, save_json, sha256


def point_histogram(redshift, pixels, support, edges):
    redshift, pixels = np.asarray(redshift), np.asarray(pixels)
    if redshift.shape != pixels.shape or pixels.dtype.kind not in 'iu' or np.any(pixels < 0) or np.any(pixels >= len(support)):
        raise ValueError('Invalid point-pixel correspondence')
    inside = np.asarray(support, bool)[pixels]
    return np.histogram(redshift[inside], edges)[0], np.histogram(redshift[~inside], edges)[0]


def run(phase):
    require_compute()
    if phase not in ('ph001', 'ph006'):
        raise ValueError('Two-canary diagnostic only')
    import fitsio
    import healpy as hp
    from workflows.abacus_tweb.p3br_build_random_response import galactic_cap
    plan, policy, source = lowz.load_plan()
    lowz.phase_guard(phase, policy)
    root = Path(plan['output_root'])
    output = root/('POINT_SELECTION_DIAGNOSTIC_' + phase + '.json')
    if output.exists():
        raise FileExistsError(output)
    observed, observed_binding = lowz.source_payload(Path(source['root']), phase,
        'OBSERVED_COMPLETE.json', plan['source_campaign']['sha256'], True)
    angular_path, angular_binding = lowz.source_payload(Path(source['root']), phase,
        'angular/ANGULAR_COMPLETE.json', plan['source_campaign']['sha256'])
    with np.load(angular_path, allow_pickle=False) as data:
        angular = {k: data[k] for k in ('support', 'domain', 'angular_response')}
    rows = fitsio.read(observed, columns=['RA', 'DEC', 'Z'])
    if not all(np.isfinite(rows[k]).all() for k in ('RA', 'DEC', 'Z')):
        raise ValueError('Nonfinite observed coordinate')
    rows = rows[lowz.radial_mask(rows['Z'], plan)]
    caps = galactic_cap(rows['RA'], rows['DEC'])
    pixels = hp.ang2pix(256, rows['RA'], rows['DEC'], lonlat=True)
    edges = lowz.bin_edges(plan)[2:31]
    statistics_path = root/'statistics_r1'/phase/'STATISTICS.json'
    statistics = lowz.read_json(statistics_path)
    if statistics['code_bindings'] != lowz.code_bindings():
        raise ValueError('Native statistics binding changed')
    result = {}
    for i, cap in enumerate(lowz.CAPS):
        support = angular['support'].astype(bool) & (angular['domain']//2 == i)
        selected = caps == i
        counts, outside = point_histogram(rows['Z'][selected], pixels[selected], support, edges)
        volume = analytic_volume(angular, cap, coord.radius_mpc_h(edges))
        native_counts = np.asarray(statistics['counts'])[i, 2:30]
        native_volume = np.asarray(statistics['volume'])[i, 2:30]
        if not np.all(counts > 0):
            raise ValueError('Missing point counts in diagnostic bin')
        direct = counts/volume
        result[cap] = dict(point_counts=counts.tolist(), point_counts_outside_angular_support=outside.tolist(),
            analytic_unapodized_volume=volume.tolist(), point_count_over_volume=direct.tolist(),
            native_cic_count_over_point_count_minus_one=relative_change(native_counts, counts),
            native_effective_over_analytic_volume_minus_one=relative_change(native_volume, volume),
            native_ratio_over_point_ratio_minus_one=relative_change(native_counts/native_volume, direct))
    save_json(output, dict(phase=phase, job=os.environ['SLURM_JOB_ID'], edges=edges.tolist(), caps=result,
        observed_source=observed_binding, angular_source=angular_binding,
        native_statistics_sha256=sha256(statistics_path), source_sha256=sha256(__file__),
        code_bindings=lowz.code_bindings(), selection_fit=False, model_inputs_ready=False,
        caveat='Per-phase ratios contain cosmic structure; diagnostic operator comparison only'))
    print('SAVED', output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('ph001', 'ph006'), required=True)
    run(parser.parse_args().phase)
