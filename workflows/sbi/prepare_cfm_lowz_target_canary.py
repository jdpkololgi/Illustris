"""Two low-z geometry-selected R7 target charts, not a complete training panel.

Reuse native matter and unchanged interpolation/physical charts. Choose one
lowest-centre-redshift candidate per cap using geometry only. Record actual
support in each owned-core redshift bin; midpoint bins are not coverage claims.
"""
from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path

import numpy as np

from workflows.sbi import cfm_lowz_selection_response as lowz
from workflows.sbi import e2e_coupled_contract as contract
from workflows.sbi import e2e_coupled_coordinates as coord
from workflows.sbi import e2e_coupled_target_products as target
from workflows.sbi import e2e_coupled_operators as op
from workflows.sbi import e2e_field_error_budget as native_ops
from workflows.sbi.prepare_cfm_lowz_inputs import require_compute, save_json, sha256


def select_canaries(rows, phase):
    chosen = []
    for cap in lowz.CAPS:
        eligible = [r for r in rows if r['phase'] == phase and r['cap'] == cap]
        if not eligible:
            raise ValueError('Both caps require a low-z geometry candidate')
        row = dict(min(eligible, key=lambda r: (r['redshift'], tuple(r['center']))))
        row['pair_id'] = phase + '_' + cap + '_lowz_target_canary'
        chosen.append(row)
    return chosen


def coverage(rows, receipt):
    import h5py
    from workflows.sbi.e2e_coupled_condition_products import padded_extract
    edges = np.array([.01, .03, .05, .075, .1, .15, .25, .35, .45, .55])
    records = {}
    for cap, source in zip(lowz.CAPS, receipt['outputs']):
        with h5py.File(source['path'], 'r') as handle:
            dataset = handle['support_random']
            for row in (r for r in rows if r['cap'] == cap):
                owned = []
                for offset in ([0, 0, 0], [32, 0, 0]):
                    start = np.asarray(row['center'])+offset-16
                    support = padded_extract(dataset, start, [32, 32, 32]) > 0
                    axes = [row['grid']['origin_mpc_h'][i]+(start[i]+np.arange(32)+.5)*row['grid']['cell_mpc_h']
                            for i in range(3)]
                    mesh = np.meshgrid(*axes, indexing='ij', sparse=True)
                    z = coord.redshift(np.sqrt(sum(v*v for v in mesh)))
                    owned.append(dict(supported_raw_voxels_by_z=np.histogram(z[support], edges)[0].tolist(),
                                      raw_voxels_by_z=np.histogram(z, edges)[0].tolist(),
                                      support_fraction=float(support.mean())))
                np.testing.assert_array_equal([r['support_fraction'] for r in owned], row['science_core_support_fractions'])
                records[row['pair_id']] = owned
    return dict(edges=edges.tolist(), by_pair=records,
                raw_support_coverage_is_not_a_final_fine_target_loss_mask=True)


def run(phase, workers=32):
    require_compute()
    if phase not in ('ph001', 'ph006') or workers not in (16, 32, 64):
        raise ValueError('Unregistered target-canary scope/resources')
    import h5py
    from scipy import fft
    plan, policy, source = lowz.load_plan()
    lowz.phase_guard(phase, policy)
    if not contract.SHARED_POLICY or sha256(contract.CONFIG) != plan['source_campaign']['sha256']:
        raise ValueError('Launch with the explicit shared 20/3/2 source data configuration')
    root = Path(plan['output_root'])
    geometry_path = root/('GEOMETRY_CANDIDATES_' + phase + '.json')
    geometric = lowz.read_json(geometry_path)
    if geometric['phase'] != phase or not geometric['original_support_and_nonoverlap_rules_preserved']:
        raise ValueError('Unqualified geometry candidates')
    if any(sha256(path) != value for path, value in geometric['code_bindings'].items()):
        raise ValueError('Geometry code binding changed')
    rows = select_canaries(geometric['selected_nonoverlapping_candidates'], phase)
    response_path = root/'responses_full'/phase/'RESPONSE_PREPARED.json'
    if sha256(response_path) != geometric['response_receipt_sha256']:
        raise ValueError('Response receipt changed')
    covered = coverage(rows, lowz.read_json(response_path))
    # Verify native counts independently before any FFT or target output.
    native_path = Path(source['root'])/'matter'/phase/'DENSITY_COMPLETE.json'
    native = contract.verify_receipt(native_path)
    if (native['phase'] != phase or native['ngrid'] != 2048 or native['box_mpc_h'] != 2000.
            or native['subsample_fraction'] != .10 or native['processed_files'] != 136):
        raise ValueError('Unqualified native particle density')
    directory = root/'target_canaries'/phase
    directory.mkdir(parents=True, exist_ok=False)
    code = {str(Path(module.__file__).resolve()): sha256(module.__file__)
            for module in (lowz, coord, target, op, native_ops, contract)}
    code[str(Path(__file__).resolve())] = sha256(__file__)
    binding = dict(geometry_receipt_sha256=sha256(geometry_path),
                   native_receipt_sha256=sha256(native_path), code_bindings=code,
                   layout_sha256=sha256(op.LAYOUT), canary_pairs=rows)
    save_json(directory/'BINDING.json', binding)
    print(phase, 'native hash verified; starting R7 FFT', flush=True)
    spectrum, mean = op.smoothed_spectrum(native['outputs'][0]['path'], workers)
    field = fft.irfftn(spectrum.copy(), s=(2048,)*3, workers=workers, overwrite_x=True)
    density = target.make_density_stage(field, rows, phase, binding, directory, mean)
    del field
    gc.collect()
    tensors = []
    for i, component in enumerate(op.COMPONENTS):
        field = native_ops.inverse_component(spectrum, 2000., component, workers=workers)
        tensors.append(target.make_tensor_stage(field, rows, phase, binding, directory, i))
        del field
        gc.collect()
        print(phase, 'tensor', component, 'complete', flush=True)
    del spectrum
    gc.collect()
    components = [np.load(r['outputs'][0]['path'], mmap_mode='r') for r in tensors]
    qa = []
    with h5py.File(density['outputs'][0]['path'], 'r') as handle:
        for ordinal, row in enumerate(rows):
            rho = handle[row['pair_id']]['rho_joint'][:]
            wide = handle[row['pair_id']]['coarse_rho_extended'][:]
            tensor = np.stack([a[ordinal] for a in components], axis=-1)
            qa.append(dict(pair_id=row['pair_id'], **target.density_qa(rho, wide), **target.fullbox_qa(rho, tensor)))
    if any(sha256(path) != expected for path, expected in code.items()):
        raise ValueError('Implementation changed during target preparation')
    save_json(directory/'TARGET_CANARY_COMPLETE.json', dict(phase=phase, job=os.environ['SLURM_JOB_ID'],
        binding=binding, density=density, tensors=tensors, qa=qa, raw_support_coverage=covered,
        target_canary_pass=True, gaussian_radius_mpc_h=7., smoothing_count=1,
        galaxy_query_estimand_qualified=False, final_training_panel=False, model_inputs_ready=False))
    print('TARGET_CANARY_PASS', phase, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('ph001', 'ph006'), required=True)
    parser.add_argument('--workers', type=int, choices=(16, 32, 64), default=32)
    args = parser.parse_args()
    run(args.phase, args.workers)
