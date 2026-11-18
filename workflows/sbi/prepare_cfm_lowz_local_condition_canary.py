"""Target-free local condition canary with point-fit, CIC-matched expectations.

Not a full training condition release: wide context, global normalization and
the final domain/loss-mask contract remain separate. Never read matter files.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from workflows.sbi import cfm_lowz_selection_response as lowz
from workflows.sbi import e2e_coupled_coordinates as coord
from workflows.sbi import e2e_coupled_operators as op
from workflows.sbi import e2e_coupled_condition_products as condition
from workflows.sbi.diagnose_cfm_lowz_cic_response import weighted_cic
from workflows.sbi.prepare_cfm_lowz_target_canary import select_canaries
from workflows.sbi.prepare_cfm_lowz_inputs import require_compute, save_json, sha256


def density_at(z, selection, cap):
    if (selection['schema'] != 'lowz-point-selection-candidate-v1'
            or selection['density_unit'] != '(Mpc/h)^-3'):
        raise ValueError('Explicit native-coordinate point selection required')
    z = np.asarray(z)
    edges = np.asarray(selection['bin_edges'])
    index = np.searchsorted(edges, z, side='right')-1
    safe = np.clip(index, 0, len(edges)-2)
    values = np.asarray(selection['density'])[lowz.CAPS.index(cap)]
    known = np.asarray(selection['valid'])[lowz.CAPS.index(cap)]
    valid = (index >= 0) & (index < len(edges)-1) & np.isfinite(z) & known[safe]
    return np.where(valid, values[safe], 0.), valid


def project_crop(origin, shape, cell, angular, cap, plan, selection, factor):
    """Integrate intensity against native CIC basis, with one-cell source halo."""
    import healpy as hp
    from workflows.abacus_tweb.p3a_build_canonical_fields import GridSpec, coordinate_block, iter_chunks
    if factor not in (2, 4):
        raise ValueError('Unregistered quadrature')
    padded = GridSpec(tuple(np.asarray(origin)-cell), tuple(np.asarray(shape)+2), cell, 0.)
    fine = GridSpec(padded.origin, tuple(n*factor for n in padded.shape), cell/factor, 0.)
    expected = np.zeros(padded.shape, dtype=np.float64)
    support = angular['support'].astype(bool) & (angular['domain']//2 == lowz.CAPS.index(cap))
    for slices in iter_chunks(fine.shape, tuple(min(n, 64) for n in fine.shape)):
        axes = coordinate_block(fine, slices)
        radius = np.sqrt(sum(a*a for a in axes))
        xyz = np.stack([np.broadcast_to(a, radius.shape).ravel() for a in axes], axis=-1)
        safe = xyz.copy(); safe[radius.ravel() == 0, 0] = 1.
        pixels = hp.vec2pix(256, *safe.T)
        z = coord.redshift(radius.ravel())
        density, known = density_at(z, selection, cap)
        valid = known & (radius.ravel() > 0) & lowz.radial_mask(z, plan) & support[pixels]
        weights = density[valid]*angular['angular_response'][pixels[valid]]*fine.cell_mpc**3
        weighted_cic(xyz[valid], weights, padded, expected)
    return expected[1:-1, 1:-1, 1:-1].copy()


def run(phase):
    require_compute()
    if phase not in ('ph001', 'ph006'):
        raise ValueError('Two-canary scope only')
    import h5py
    from workflows.abacus_tweb.p3a_build_canonical_fields import GridSpec, coordinate_block
    plan, policy, source = lowz.load_plan()
    lowz.phase_guard(phase, policy)
    root = Path(plan['output_root'])
    directory = root/'local_condition_canaries'/phase
    if directory.exists():
        raise FileExistsError(directory)
    geometry_path = root/('GEOMETRY_CANDIDATES_' + phase + '.json')
    geometric = lowz.read_json(geometry_path)
    rows = select_canaries(geometric['selected_nonoverlapping_candidates'], phase)
    selection_path = root/'point_selection_candidates_v1/full.json'
    selection = lowz.read_json(selection_path)
    if selection['fit_phases'] != policy['training'] or selection['omitted_phase'] is not None:
        raise ValueError('Full selection must use exactly the 20 training phases')
    angular_path, angular_binding = lowz.source_payload(Path(source['root']), phase,
        'angular/ANGULAR_COMPLETE.json', plan['source_campaign']['sha256'])
    with np.load(angular_path, allow_pickle=False) as data:
        angular = {k: data[k] for k in ('support', 'domain', 'angular_response')}
    response_path = root/'responses_full'/phase/'RESPONSE_PREPARED.json'
    if sha256(response_path) != geometric['response_receipt_sha256']:
        raise ValueError('Geometry/response provenance mismatch')
    receipt = lowz.read_json(response_path)
    directory.mkdir(parents=True, exist_ok=False)
    implementation = sha256(__file__)
    outputs = []
    for row in rows:
        cap = row['cap']; raw_record = receipt['outputs'][lowz.CAPS.index(cap)]
        if sha256(raw_record['path']) != raw_record['sha256']:
            raise ValueError('Raw response payload changed')
        start = np.asarray(row['center'])+op.layout()['joint_raw_start_from_first_center']
        shape = tuple(2*n for n in op.layout()['joint_density_shape'])
        cell = row['grid']['cell_mpc_h']
        origin = np.asarray(row['grid']['origin_mpc_h'])+start*cell
        levels = {}
        for factor in (2, 4):
            levels[factor] = project_crop(origin, shape, cell, angular, cap, plan, selection, factor)
            print(phase, cap, 'local expected counts quadrature', factor, 'complete', flush=True)
        spec = GridSpec(tuple(origin), shape, cell, 0.)
        axes = coordinate_block(spec, tuple(slice(0, n) for n in shape))
        z = coord.redshift(np.sqrt(sum(x*x for x in axes)))
        nbar, known = density_at(z, selection, cap)
        with h5py.File(raw_record['path'], 'r') as source_file:
            raw = {key: condition.padded_extract(source_file[key], start, shape)
                   for key in condition.LOCAL_CHANNELS
                   if key not in ('observer_redshift', 'ntilde_h3_mpc3', 'expected_counts_random', 'log_count_ratio_random')}
        valid = known & (raw['support_random'] > 0)
        expected = levels[4]
        raw.update(observer_redshift=z, ntilde_h3_mpc3=nbar,
                   expected_counts_random=expected,
                   log_count_ratio_random=np.where(valid, np.log((raw['counts']+.001)/(expected+.001)), 0.))
        joint = np.stack([op.mean_pool(condition.transformed(key, raw[key]), 2).astype('f4')
                          for key in condition.LOCAL_CHANNELS])
        support = op.mean_pool(raw['support_random'], 2).astype('f4')
        if joint.shape != (12, 64, 48, 48) or not np.isfinite(joint).all():
            raise ValueError('Local condition shape/finiteness failure')
        l1 = float(np.abs(levels[2][valid]-expected[valid]).sum()/expected[valid].sum())
        output = directory/(row['pair_id']+'.h5')
        with h5py.File(output, 'x') as handle:
            handle.attrs.update(schema='cfm-lowz-local-condition-canary-v1', phase=phase,
                                pair_id=row['pair_id'], contains_matter=False, model_inputs_ready=False,
                                expected_count_operator='point selection times angular response integrated through native CIC')
            for key, value in dict(joint=joint, support=support,
                                   selection_valid_fraction=op.mean_pool(valid.astype(float), 2).astype('f4'),
                                   native_expected_counts=expected.astype('f4')).items():
                handle.create_dataset(key, data=value, compression='lzf', shuffle=True)
        outputs.append(dict(pair_id=row['pair_id'], path=str(output), sha256=sha256(output),
                            half_quarter_expected_l1_relative=l1, joint_shape=list(joint.shape)))
    if sha256(__file__) != implementation:
        raise ValueError('Local condition implementation changed')
    save_json(directory/'LOCAL_CONDITION_CANARY_COMPLETE.json', dict(phase=phase, job=os.environ['SLURM_JOB_ID'],
        outputs=outputs, local_channels=list(condition.LOCAL_CHANNELS),
        selection_sha256=sha256(selection_path), geometry_sha256=sha256(geometry_path),
        angular_source=angular_binding, source_sha256=implementation,
        contains_matter=False, wide_context_ready=False, normalizer_fitted=False,
        training_loader_ready=False, model_inputs_ready=False))
    print('LOCAL_CONDITION_CANARY_PASS', phase, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('ph001', 'ph006'), required=True)
    run(parser.parse_args().phase)
