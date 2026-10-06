"""Development-only conservative reconstruction; no truth subcells as inputs."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2e_operator_split as s
import numpy as np
import h5py

OUT = s.b.ROOT/'verification/density_only_reconstruction_20261005_v1'


def reconstruct(rho, method):
    """Two subcells/axis; central or MC-limited slopes, exact parent mass.

    Cell centers are at +/-1/4 of a parent cell. A common per-parent slope
    multiplier guarantees positivity without changing the parent average.
    One outer coarse cell is needed only to estimate interior slopes.
    """
    rho = np.asarray(rho, dtype=np.float64)
    if method not in ('constant', 'linear', 'monotone'):
        raise ValueError(method)
    if rho.ndim != 3 or not np.isfinite(rho).all() or np.min(rho) <= 0:
        raise ValueError('Positive finite density required')
    slopes = []
    for axis in range(3):
        left = rho-np.roll(rho, 1, axis)
        right = np.roll(rho, -1, axis)-rho
        value = .5*(left+right)
        if method == 'monotone':
            value = np.where(left*right > 0, np.sign(value)*np.minimum(
                abs(value), np.minimum(2*abs(left), 2*abs(right))), 0.)
        if method == 'constant':
            value *= 0
        edge = [slice(None)]*3
        for index in (0, -1):
            edge[axis] = index
            value[tuple(edge)] = 0
        slopes.append(value)
    excursion = .25*sum(abs(v) for v in slopes)
    factor = np.minimum(1., .999*rho/np.maximum(excursion, 1e-300))
    fine = np.empty(tuple(2*n for n in rho.shape))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                fine[i::2, j::2, k::2] = rho+factor*sum(
                    (index-.5)*.5*v for index, v in zip((i,j,k), slopes))
    np.testing.assert_allclose(s.b.op.mean_pool(fine, 2), rho, atol=1e-14, rtol=1e-14)
    assert fine.min() > 0
    return fine


def run(all_cases=False):
    global OUT
    if all_cases:
        OUT = OUT.with_name(OUT.name+'_all64')
    s.b.c.require_compute()
    OUT.mkdir(parents=True, exist_ok=False)
    records = []
    for phase in ('ph012', 'ph013'):
        s.b.auth.frozen(phase, s.b.FREEZE)
        rows = s.b.auth.geometry(phase, s.b.FREEZE)
        old = json.loads((s.OUT/phase/'SPLIT.json').read_text())
        for row in (r for r in rows if all_cases or r['pair_id'] in s.SELECTED[phase]):
            pid, cap = row['pair_id'], row['cap']
            ref = next(r for r in old['cases'] if r['pair_id'] == pid)
            with np.load(s.OUT/phase/(cap+'_CORE_TENSORS.npz')) as f:
                index = list(f['pair_id']).index(pid)
                baseline = f['pad32'][index]
            results = []
            for pad in ((32,) if all_cases else (16, 32)):
                # Extra collar cell estimates slopes, then is removed before solve.
                with h5py.File(s.b.OUT/phase/(cap+'_TRUE_SOURCE.h5')) as f:
                    core = s.b.cuts(row, np.asarray(f.attrs['source_origin_mpc_h']))
                    cut = tuple(slice(v.start-pad-1, v.stop+pad+1) for v in core)
                    assert all(v.start >= 0 and v.stop <= n for v,n in zip(cut, f['rho'].shape))
                    rho = f['rho'][cut]
                center = rho[1:-1,1:-1,1:-1]
                target_shape = (32,16,16)
                coarse = s.patch_tensor(center, target_shape, pad).reshape(2,16,16,16,6)
                for method in ('constant', 'linear', 'monotone'):
                    fine = reconstruct(rho, method)[2:-2,2:-2,2:-2]
                    tensor = s.patch_tensor(fine, (64,32,32), 2*pad)
                    averaged = np.stack([s.b.op.mean_pool(tensor[...,k],2) for k in range(6)], -1)
                    correction = averaged.reshape(2,16,16,16,6)-coarse
                    trace = float(abs(correction[...,[0,3,5]].sum(-1)).max())
                    assert trace < 1e-9
                    eigen = s.b.tidal.eigensystem(baseline+correction).mean((1,2,3))
                    rec = dict(method=method, padding=pad, eigenvalues=eigen.tolist(), trace_error=trace)
                    results.append(rec)
                    print(json.dumps(dict(phase=phase,pair_id=pid,**rec)), flush=True)
                    del tensor, averaged, fine; gc.collect()
            record = dict(phase=phase,pair_id=pid,fullbox=ref['fullbox'],baseline=ref['pad32'],
                          oracle=ref.get('refinement',{}).get('32',{}).get('pad32_plus_local'),results=results)
            s.b.c.atomic_json(OUT/(pid+'.json'), record)
            records.append(record)
    s.b.c.atomic_json(OUT/'COMPLETE.json', dict(cases=records, passed=True,
        inputs='Only saved coarse true-density cells enter reconstruction; full-box/subcell truth is scoring-only.',
        source_sha256=s.b.c.sha256(__file__),freeze_sha256=s.b.c.sha256(s.b.FREEZE),
        job=os.environ['SLURM_JOB_ID'],posterior_qualified=False))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--all-cases',action='store_true');a=p.parse_args()
    run(a.all_cases)
