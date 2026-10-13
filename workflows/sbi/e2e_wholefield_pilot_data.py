"""Finite multiwindow fine domains; reuse the existing R7 target and observations.

Targets and observations are separate. No sealed phases, normalization fit,
new particle deposition, or changes to the frozen model/data products.
"""
import argparse
import gc
import json
from pathlib import Path
import time

import h5py
import numpy as np
from scipy import fft

from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_coordinates as coord
from workflows.sbi import e2e_coupled_conditions as conditions
from workflows.sbi import e2e_coupled_condition_products as products
from workflows.sbi import e2e_coupled_observations as obs
from workflows.sbi import e2e_coupled_operators as op
from workflows.sbi import e2e_coupled_views as views
from workflows.abacus_tweb import p3a_build_canonical_fields as grid_ops
from workflows.abacus_tweb import p3br_build_random_response as response_ops

ROOT = Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/wholefield_pilot_20261013_v1')
SHAPE = (96, 64, 64)
EVAL = ('ph012', 'ph013', 'ph014', 'ph015', 'ph016', 'ph017')
COARSE_CROP = (slice(12, 36), slice(16, 32), slice(16, 32))
OLD_CROP = (slice(16, 80), slice(8, 56), slice(8, 56))


def guard(phase):
    if phase not in (*c.TRAIN, *EVAL):
        raise PermissionError('Only original training and exposed evaluation phases')


def select_rows(phase, rows):
    guard(phase)
    chosen = []
    for cap in ('NGC', 'SGC'):
        values = sorted((r for r in rows if r['cap'] == cap), key=lambda r: r['pair_id'])
        if phase in c.TRAIN:
            if len(values) != 64:
                raise ValueError('Expected original 64 training pairs per cap')
            # 48 domains/phase x 96x64x64 equals 128 old 64x48x48 volumes.
            chosen.extend(values[i] for i in np.linspace(0, 63, 24, dtype=int))
        else:
            if len(values) != 8:
                raise ValueError('Expected all eight evaluation pairs per cap')
            chosen.extend(values)
    return chosen


def build_fine_observations(grid, midpoint, cap, counts, response_data):
    support, response, boundary, selection = response_data
    shape = np.array(SHAPE)*2
    start = np.asarray(midpoint)-np.array(SHAPE)
    spec = grid_ops.GridSpec(tuple(grid['origin_mpc_h']), tuple(grid['shape']), 3.383, grid['padding_mpc_h'])
    raw_counts = products.padded_extract(counts, start, shape)
    result = np.empty((12, *SHAPE), dtype='f4')
    for i in range(0, int(shape[0]), 16):
        sl = tuple(slice(int(a), int(a+n)) for a, n in zip(start+[i, 0, 0], [16, shape[1], shape[2]]))
        values, z = obs.response_chunk(spec, sl, support, response, boundary, selection, cap)
        values['counts'] = raw_counts[i:i+16]
        values['log_count_ratio_random'] = grid_ops.log_count_ratio(values['counts'], values['expected_counts_random'],
                                                                  values['exposure_apodized_random'], 1e-3, 1e-4)
        axes = [np.arange(s.start, s.stop) for s in sl]
        valid = ((axes[0][:, None, None] >= 0) & (axes[0][:, None, None] < spec.shape[0]) &
                 (axes[1][None, :, None] >= 0) & (axes[1][None, :, None] < spec.shape[1]) &
                 (axes[2][None, None, :] >= 0) & (axes[2][None, None, :] < spec.shape[2]))
        for name in values:
            values[name] = np.where(valid, values[name], 0)
        values['observer_redshift'] = z
        for j, name in enumerate(products.LOCAL_CHANNELS):
            result[j, i//2:(i+16)//2] = op.mean_pool(products.transformed(name, values[name]), 2)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite observations')
    return result


def observations(phase, rows):
    guard(phase)
    directory = ROOT/'data'/phase
    directory.mkdir(parents=True, exist_ok=True)
    ar = c.verify_receipt(c.ROOT/'observations'/phase/'angular/ANGULAR_COMPLETE.json')
    with np.load(ar['outputs'][0]['path']) as f:
        angular = {k: f[k] for k in f.files}
    selection_path = Path(c.config()['observation']['fixed_selection_manifest'])
    selection = json.loads(selection_path.read_text())
    for cap in ('NGC', 'SGC'):
        mask = angular['support'].astype(bool) & (angular['domain']//2 == (cap == 'NGC'))
        response = (mask, angular['angular_response'], response_ops.angular_boundary_distance(mask), selection)
        raw_path = coord.ROOT/'observations'/phase/(cap+'_COMPLETE.json')
        raw = coord.verify_receipt(raw_path, payload=False)
        with h5py.File(raw['outputs'][0]['path'], 'r') as file:
            for row in [r for r in rows if r['cap'] == cap]:
                pair = row['pair_id']; dest = directory/pair
                dest.mkdir(exist_ok=True)
                if (dest/'OBSERVATIONS.json').exists():
                    continue
                midpoint = np.asarray(row['center'])+[16, 0, 0]
                local = build_fine_observations(row['grid'], midpoint, cap, file['counts'], response)
                old = conditions.crop_context(conditions.load_pair(phase, pair), phase)
                error = float(np.max(abs(local[(slice(None), *OLD_CROP)]-old['joint'])))
                if error > 2e-5:
                    raise ValueError('Expanded observation does not reproduce old central view: '+str(error))
                path = dest/'OBSERVATIONS.npz'
                with path.open('xb') as stream:
                    np.savez_compressed(stream, joint=local, wide=old['wide'])
                c.atomic_json(dest/'OBSERVATIONS.json', dict(phase=phase, pair_id=pair, contains_matter=False,
                     output=c.file_record(path, content_hash=True), raw_receipt=c.file_record(raw_path, content_hash=True),
                     old_condition=c.file_record(coord.ROOT/'conditions'/phase/f'{pair}.json', content_hash=True),
                     central_channel_maxabs=error, passed=True))


def targets(phase, rows, workers):
    guard(phase)
    needed = [r for r in rows if not (ROOT/'data'/phase/r['pair_id']/'TARGETS.json').exists()]
    if not needed:
        return
    native_path = c.ROOT/'matter'/phase/'DENSITY_COMPLETE.json'
    native = c.verify_receipt(native_path, payload=False)
    if native['phase'] != phase or native['ngrid'] != 2048 or native['box_mpc_h'] != 2000.:
        raise ValueError('Wrong native target')
    payload = native['outputs'][0]
    if Path(payload['path']).stat().st_size != payload['bytes']:
        raise ValueError('Native target size changed')
    print(json.dumps(dict(stage='target_fft', phase=phase, domains=len(needed))), flush=True)
    spectrum, _ = op.smoothed_spectrum(payload['path'], workers)
    field = fft.irfftn(spectrum, s=(2048,)*3, workers=workers, overwrite_x=True)
    del spectrum; gc.collect()
    for ordinal, row in enumerate(needed):
        midpoint = np.asarray(row['center'])+[16, 0, 0]
        starts = [v-n+2*np.arange(n) for v, n in zip(midpoint, SHAPE)]
        rho = 1+op.sample_averaged_local(field, starts, 2, row['grid'])
        coarse, residual = op.encode(rho)
        original_path = coord.ROOT/'targets'/phase/f"{row['pair_id']}.json"
        original = coord.verify_receipt(original_path, payload=False)
        with h5py.File(original['outputs'][0]['path'], 'r') as saved:
            old_rho = saved['rho_joint'][:]
            wide = saved['coarse_rho_extended'][4:52, 4:52, 4:52]
        error = float(np.max(abs(rho[OLD_CROP]-old_rho)/old_rho))
        mass = float(np.max(abs(coarse-wide[COARSE_CROP])/coarse))
        if min(rho.min(), wide.min()) <= 0 or not np.isfinite(rho).all() or max(error, mass) > 2e-6:
            raise ValueError('Expanded target parity/positivity/mass failed')
        dest = ROOT/'data'/phase/row['pair_id']; path = dest/'TARGETS.npz'
        with path.open('xb') as stream:
            np.savez_compressed(stream, rho=rho.astype('f4'), coarse_wide=wide.astype('f4'), residual=residual.astype('f4'))
        c.atomic_json(dest/'TARGETS.json', dict(phase=phase, pair_id=row['pair_id'], passed=True,
             output=c.file_record(path, content_hash=True), native_receipt=c.file_record(native_path, content_hash=True),
             old_target=c.file_record(original_path, content_hash=True), central_density_relative=error,
             coarse_alignment_relative=mass, native_payload_hash_verified_upstream=True))
        if ordinal % 8 == 0:
            print(json.dumps(dict(stage='targets', phase=phase, domains=ordinal+1)), flush=True)
    del field; gc.collect()


def run(phases, workers):
    c.require_compute()
    for phase in phases:
        guard(phase)
    ROOT.mkdir(parents=True, exist_ok=True)
    normalizer = c.sha256(coord.ROOT/'normalization/NORMALIZATION_COMPLETE.json')
    views.load_chart(normalizer)
    source = {str(Path(module.__file__).resolve()): c.sha256(module.__file__) for module in (op, conditions, products, obs, views)}
    source[str(Path(__file__).resolve())] = c.sha256(__file__)
    start = time.monotonic()
    for phase in phases:
        marker = ROOT/'data'/phase/'COMPLETE.json'
        if marker.exists():
            old = json.loads(marker.read_text())
            if old['sources'] != source:
                raise ValueError('Data source changed')
            continue
        geometry_path = coord.ROOT/'geometry'/phase/'GEOMETRY_COMPLETE.json'
        geo = coord.verify_receipt(geometry_path, payload=False)
        rows = select_rows(phase, geo['pairs'])
        observations(phase, rows)
        targets(phase, rows, workers)
        c.atomic_json(marker, dict(phase=phase, pairs=rows, shape=list(SHAPE), normalizer=normalizer,
             sources=source, geometry=c.file_record(geometry_path, content_hash=True), passed=True,
             seconds=time.monotonic()-start, training=phase in c.TRAIN))
        print(json.dumps(dict(stage='phase_complete', phase=phase, domains=len(rows), seconds=time.monotonic()-start)), flush=True)
    print('DATA_SEGMENT_COMPLETE', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phases', nargs='+', default=list(c.TRAIN)+list(EVAL))
    p.add_argument('--workers', type=int, default=32)
    a = p.parse_args(); run(a.phases, a.workers)
