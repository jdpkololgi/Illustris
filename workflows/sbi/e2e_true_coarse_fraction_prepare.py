"""Build only ph012/ph013 true coarse canvases with the original R7 operator."""
import argparse
import gc
import os
import sys
import time

import e2e_true_coarse_fraction_saved as io
sys.path.insert(0, str(io.BASE/'source_78156ac3417555ef269f'))
import h5py
import numpy as np
from scipy import fft
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_operators as op
from workflows.sbi import e2e_coupled_views as views
from workflows.sbi.e2e_global_canvas import Geometry


def run(phase):
    io.compute_guard()
    if phase not in io.PHASES:
        raise PermissionError('ph012/ph013 only')
    started = time.monotonic()
    dest = io.OUT/'true_coarse'/phase; dest.mkdir(parents=True, exist_ok=False)
    manifest = io.record(io.OUT/'MANIFEST.json')
    freeze = io.read(io.BASE/'SELECTION_FREEZE.json'); io.record(io.BASE/'SELECTION_FREEZE.json')
    chart = views.load_chart(freeze['normalizer'])
    native_path = c.ROOT/'matter'/phase/'DENSITY_COMPLETE.json'
    native = io.read(native_path); io.record(native_path)
    if native['phase'] != phase or native['ngrid'] != 2048 or native['box_mpc_h'] != 2000.:
        raise ValueError('Original physical native density required')
    io.verify(native['outputs'][0])
    prepared = io.read(io.FACTOR/'PREPARED.json'); io.record(io.FACTOR/'PREPARED.json')
    sources = [io.record(p) for p in (__file__, io.__file__, op.__file__, views.__file__, sys.modules[Geometry.__module__].__file__)]
    print(io.json.dumps(dict(stage='true_coarse_fft', phase=phase)), flush=True)
    spectrum, mean = op.smoothed_spectrum(native['outputs'][0]['path'], 32)
    field = fft.irfftn(spectrum, s=(2048,)*3, workers=32, overwrite_x=True)
    del spectrum; gc.collect()
    products = []
    for cap in ('NGC', 'SGC'):
        plan_path = io.BASE/'fullcap_inputs'/phase/cap/'PLAN.json'; plan = io.read(plan_path); io.record(plan_path)
        cg = Geometry.make(plan['midpoints'], 'coarse')
        starts = [8*(int(a)+np.arange(int(n))) for a, n in zip(cg.origin, cg.shape)]
        physical = 1+op.sample_averaged_local(field, starts, 8, plan['grid'])
        if not np.isfinite(physical).all() or physical.min() <= 0:
            raise ValueError('Invalid true coarse density')
        z = ((np.log(physical)-chart['coarse_logrho']['mean'][0])/chart['coarse_logrho']['std'][0]).astype('f4')
        checks = []
        for row in (r for r in prepared['cases'] if r['phase'] == phase and r['cap'] == cap):
            io.verify(row['file'])
            with h5py.File(row['input']) as f:
                expected = f['true_wide'][:]
            sl = cg.crop(row['window_index'])[2:]
            actual = physical[sl]
            np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-6)
            checks.append(dict(pair_id=row['pair_id'], maxabs=float(abs(actual-expected).max())))
        if len(checks) != 8:
            raise ValueError('Incomplete true-coarse overlap validation')
        path = dest/(cap+'_TRUE_COARSE.h5')
        with h5py.File(path, 'x') as f:
            f.create_dataset('rho', data=physical, compression='lzf')
            f.create_dataset('logrho_standardized', data=z, compression='lzf')
            f.attrs.update(phase=phase, cap=cap, coarse_origin_raw_cells=np.asarray(cg.origin)*8,
                normalizer_sha256=freeze['normalizer'], smoothing_mpc_h=7., smoothing_count=1,
                cell_mpc_h=27.064, posterior_qualified=False)
        products.append(dict(cap=cap, output=io.record(path), checks=checks, shape=list(map(int, cg.shape))))
        print(io.json.dumps(dict(stage='true_coarse_saved', phase=phase, cap=cap, seconds=time.monotonic()-started)), flush=True)
    del field; gc.collect()
    for rec in sources:
        if io.sha(rec['path']) != rec['sha256']:
            raise ValueError('Source changed during preparation')
    io.write(dest/'COMPLETE.json', dict(passed=True, phase=phase, products=products, native_count_mean=mean,
        inputs=list(io.CACHE.values()), sources=sources, manifest=manifest,
        scope='one parent seed, two reused phases, diagnostic only', job=os.environ['SLURM_JOB_ID'],
        seconds=time.monotonic()-started, confirmation_access=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--phase', choices=io.PHASES, required=True)
    args = p.parse_args(); run(args.phase)
