"""Whole-footprint truth-only boundary experiment; sealed phases fail closed.

No galaxy-density inference or retraining. Each phase commits separately.
Full-box truth uses the existing R7, native2048, block-averaged reference.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import time

import h5py
import numpy as np
from scipy import fft, ndimage

from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_coordinates as coord
from workflows.sbi import e2e_coupled_geometry as geo
from workflows.sbi import e2e_coupled_operators as grids
from workflows.sbi import e2e_coupled_target_products as targets
from workflows.sbi import e2e_footprint_tidal_operator as isolated
from workflows.sbi.e2e_coupled_product_audit import read_target

TRAIN = ('ph007', 'ph008')
EVAL = ('ph016', 'ph017')
WEIGHTS = np.array([1., 2., 2., 1., 2., 1.])


def guard(phase):
    if phase not in TRAIN + EVAL:
        raise PermissionError('Only registered training007/008 and exposed016/017 allowed')


def pool(a, factor):
    shape = a.shape[:3]
    if any(n % factor for n in shape):
        raise ValueError('Non-divisible grid')
    return a.reshape(*(v for n in shape for v in (n//factor, factor)), *a.shape[3:]).mean((1, 3, 5))


def crop(a, start, shape):
    start = np.array(start, int); shape = np.array(shape, int)
    lo, hi = np.maximum(start, 0), np.minimum(start+shape, a.shape[:3])
    out = np.zeros((*shape, *a.shape[3:]), a.dtype)
    if np.all(hi > lo):
        out[tuple(slice(x, y) for x, y in zip(lo-start, hi-start))] = a[
            tuple(slice(x, y) for x, y in zip(lo, hi))]
    return out


def log(**values):
    print(json.dumps(values), flush=True)


def response(phase, cap, factor):
    guard(phase)
    path = coord.ROOT/'observations'/phase/(cap+'_COMPLETE.json')
    record = coord.verify_receipt(path, payload=False)
    if record['phase'] != phase:
        raise ValueError('Response phase mismatch')
    item = record['outputs'][0]
    if c.sha256(item['path']) != item['sha256']:
        raise ValueError('Response hash mismatch')
    with h5py.File(item['path']) as f:
        mask = f['support_random'][:]
    if not np.isin(mask, (0, 1)).all():
        raise ValueError('Expected original binary support, not response weights')
    occupied = [np.flatnonzero(mask.any(axis=tuple(j for j in range(3) if j != i))) for i in range(3)]
    low = np.array([v[0]//2*2 for v in occupied])
    high = np.array([(v[-1]//2+1)*2 for v in occupied])
    mask = pool(crop(mask, low, high-low).astype('f4'), factor)
    grid = record['grid']; coord.validate_grid(grid)
    origin = np.array(grid['origin_mpc_h'])+(low+factor/2)*3.383
    return dict(cap=cap, grid=grid, low=low, high=high, support=mask,
                origin=origin, shape=mask.shape, receipt=c.file_record(path, content_hash=True))


def core_extract(field, rows, cap, low, factor):
    values = []
    for row in rows:
        if row['cap'] != cap:
            continue
        for side in range(2):
            raw = np.array(row['center']) + [32*side, 0, 0] - 16
            if np.any((raw-low) % factor):
                raise ValueError('Core/source lattice mismatch')
            values.append(pool(crop(field, (raw-low)//factor, (32//factor,)*3), 2//factor))
    return np.stack(values)


def prepare(phase, out, workers=32, factor=2):
    c.require_compute(); guard(phase)
    if factor not in (1, 2):
        raise ValueError('Only raw/fine refinement registered')
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    gp = coord.ROOT/'geometry'/phase/'GEOMETRY_COMPLETE.json'
    geometry = coord.verify_receipt(gp, payload=False); geo.verify_rows(geometry['pairs'], phase)
    rows = geometry['pairs']
    native_path = c.ROOT/'matter'/phase/'DENSITY_COMPLETE.json'
    native = c.verify_receipt(native_path)
    if native['phase'] != phase or native['ngrid'] != 2048 or native['box_mpc_h'] != 2000.:
        raise ValueError('Native truth identity mismatch')
    caps = [response(phase, name, factor) for name in ('NGC', 'SGC')]
    for src in caps:
        core_lows = np.array([np.array(r['center'])+[32*s,0,0]-16
            for r in rows if r['cap']==src['cap'] for s in (0,1)])
        low = np.minimum(src['low'], core_lows.min(0))
        high = np.maximum(src['high'], (core_lows+32).max(0))
        src['support'] = crop(src['support'], (low-src['low'])//factor, (high-low)//factor)
        src.update(low=low, high=high, shape=src['support'].shape,
            origin=np.array(src['grid']['origin_mpc_h'])+(low+factor/2)*3.383)
    binding = dict(phase=phase, factor=factor, job=os.environ['SLURM_JOB_ID'],
        source_sha256=c.sha256(__file__), operator_sha256=c.sha256(isolated.__file__),
        grid_operator_sha256=c.sha256(grids.__file__), geometry=c.file_record(gp, content_hash=True),
        native=c.file_record(native_path, content_hash=True), masks=[s['receipt'] for s in caps],
        full_footprints=True, mask_then_smooth=False, exterior_model_fitted=False,
        sealed_phase_access=False, threshold=.2)
    c.atomic_json(out/'BINDING.json', binding)
    log(phase=phase, stage='native_R7_FFT', seconds=time.monotonic()-started)
    spectrum, mean = grids.smoothed_spectrum(native['outputs'][0]['path'], workers)
    field = fft.irfftn(spectrum, s=(2048,)*3, workers=workers, overwrite_x=True)
    del spectrum; gc.collect()
    for src in caps:
        before = time.monotonic()
        starts = [np.arange(a, b, factor) for a, b in zip(src['low'], src['high'])]
        delta = grids.sample_averaged_local(field, starts, factor, src['grid'])
        if not np.isfinite(delta).all() or np.min(delta) <= -1:
            raise ValueError('Invalid physical density')
        src['delta'] = delta*src['support']
        src['core_density'] = core_extract(delta, rows, src['cap'], src['low'], factor)
        src['core_support'] = core_extract(src['support'], rows, src['cap'], src['low'], factor)
        src['core_distance'] = core_extract(ndimage.distance_transform_edt(
            np.pad(src['support']==1, 1))[1:-1,1:-1,1:-1]*3.383*factor,
            rows, src['cap'], src['low'], factor)
        with h5py.File(out/(src['cap']+'_SOURCE.h5'), 'x') as f:
            f.attrs.update(origin_mpc_h=src['origin'], raw_low=src['low'], cell_mpc_h=3.383*factor)
            f.create_dataset('delta', data=delta, compression='lzf')
            f.create_dataset('support', data=src['support'], compression='lzf')
        del delta
        log(phase=phase, cap=src['cap'], stage='sample_full_cap', shape=src['shape'], seconds=time.monotonic()-before)
    del field; gc.collect()
    for target in caps:
        selected = [r for r in rows if r['cap'] == target['cap']]
        # The target lattice includes every registered core, even outside the source bounding box.
        core_lows = np.array([np.array(r['center'])+[32*s,0,0]-16 for r in selected for s in (0,1)])
        low = core_lows.min(0); high = (core_lows+32).max(0)
        shape = tuple((high-low)//factor)
        origin = np.array(target['grid']['origin_mpc_h'])+(low+factor/2)*3.383
        totals = np.zeros((len(selected)*2,16,16,16,6), np.float64)
        self_isolated = np.zeros_like(totals)
        for src in caps:
            before = time.monotonic()
            kernels = isolated.kernel(src['shape'], shape, (origin-src['origin'])/(3.383*factor))
            pad = kernels.shape[1:]; spectrum = fft.rfftn(src['delta'], s=pad, workers=workers)
            log(phase=phase, source=src['cap'], target=target['cap'], stage='kernel', padded_shape=pad,
                seconds=time.monotonic()-before)
            for component in range(6):
                k = fft.rfftn(kernels[component], workers=workers)
                value = fft.irfftn(spectrum*k, s=pad, workers=workers)[tuple(slice(0,n) for n in shape)]
                core = core_extract(value, rows, target['cap'], low, factor)
                totals[...,component] += core
                if src['cap'] == target['cap']:
                    self_isolated[...,component] = core
                del k, value
            del spectrum, kernels; gc.collect()
            log(phase=phase, source=src['cap'], target=target['cap'], stage='isolated_complete', seconds=time.monotonic()-before)
        # Strict periodic self-cap comparator. The other observed cap remains isolated.
        # Its rectangular period and missing k=0 are explicit, not simulation-box periodicity.
        before = time.monotonic()
        periodic = grids.tensor_from_delta(target['delta'], 3.383*factor, workers, dc=False)
        periodic = core_extract(periodic, rows, target['cap'], target['low'], factor)
        periodic += totals-self_isolated
        dc_mean = float(target['delta'].mean())
        periodic_dc = periodic.copy(); periodic_dc[..., [0,3,5]] += dc_mean/3
        truth = []; sources = []; identities = []
        for row in selected:
            rp = coord.ROOT/'targets'/phase/(row['pair_id']+'.json')
            rec = coord.verify_receipt(rp)
            if rec['phase'] != phase or rec['pair_id'] != row['pair_id']:
                raise ValueError('Target identity mismatch')
            t = read_target(rec['outputs'][0]['path'], phase, row['pair_id'])
            truth.extend(t['fullbox_tensor_cores'])
            sources.append(c.file_record(rp, content_hash=True))
            density = targets.owned_core_arrays(t['rho_joint'])-1
            i = len(truth)-2
            if not np.allclose(target['core_density'][i:i+2], density, atol=2e-6, rtol=0):
                raise ValueError('Full-cap truth density does not replay registered cores')
            identities.extend([dict(pair_id=row['pair_id'], phase=phase, cap=row['cap'],
                shell=row['shell'], stratum=row['support_stratum'], side=s,
                center_mpc_h=(np.array(row['grid']['origin_mpc_h'])+
                    (np.array(row['center'])+[32*s,0,0])*3.383).tolist()) for s in (0,1)])
        truth = np.stack(truth)
        expected = core_extract(target['delta'], rows, target['cap'], target['low'], factor)
        trace_error = float(np.max(abs(totals[...,[0,3,5]].sum(-1)-expected)))
        if trace_error > 2e-6 or not np.isfinite(totals).all():
            raise ValueError('Isolated full-footprint trace failed: '+str(trace_error))
        path = out/(target['cap']+'_CORES.npz')
        np.savez_compressed(path, truth=truth, isolated=totals, periodic=periodic,
            periodic_dc=periodic_dc, density=target['core_density'], support=target['core_support'],
            distance=target['core_distance'])
        c.atomic_json(out/(target['cap']+'_COMPLETE.json'), dict(passed=True, phase=phase,
            identities=identities, output=c.file_record(path, content_hash=True), targets=sources,
            trace_max_abs=trace_error, periodic_box_shape=list(target['shape']),
            periodic_source_mean=dc_mean, cell_mpc_h=3.383*factor))
        log(phase=phase, cap=target['cap'], stage='cores_complete', seconds=time.monotonic()-before)
    c.atomic_json(out/'COMPLETE.json', dict(passed=True, phase=phase, factor=factor,
        seconds=time.monotonic()-started, binding_sha256=c.sha256(out/'BINDING.json'),
        caps=[c.file_record(out/(cap+'_COMPLETE.json'),content_hash=True) for cap in ('NGC','SGC')]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--workers', type=int, default=32); p.add_argument('--factor', type=int, default=2)
    a=p.parse_args(); prepare(a.phase,a.out,a.workers,a.factor)


if __name__ == '__main__':
    main()
