"""Frozen saved-draw conditioning/decoder cross; development phases only.

Posterior draw distance from one truth is NOT a measured model-error field.
Crossed shapes and masses are diagnostics, never posterior candidates.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2e_true_density_closure as b
import e2e_sampler_factor_diagnostic as saved
import numpy as np
import h5py
from scipy import fft
from scipy.stats import rankdata

OUT = b.ROOT / 'verification/conditioning_ablation_20261005_v1'
CROP = (slice(16, 32), slice(18, 30), slice(18, 30))
EDGES = (0., .04, .08, .16, .32, np.inf)


def shape_mass(rho):
    """Separate dimensionless shape from physical mass without clipping."""
    mass, latent = b.op.encode(rho)
    shape = rho / b.op.lift(mass)
    np.testing.assert_allclose(b.op.mean_pool(shape), 1., atol=2e-14, rtol=0)
    np.testing.assert_allclose(b.op.decode(mass, latent), rho, atol=1e-13, rtol=1e-13)
    return shape, mass, latent


def crossed(shape, mass):
    rho = shape * b.op.lift(mass)
    error = float(np.max(abs(b.op.mean_pool(rho) / mass - 1)))
    assert np.isfinite(rho).all() and rho.min() > 0 and error < 2e-14
    return rho, error


def masks(shape):
    axes = np.meshgrid(*[2*np.pi*fft.fftfreq(n, d=6.766) for n in shape], indexing='ij', sparse=True)
    k = np.sqrt(sum(a*a for a in axes))
    return [(k > lo) & (k <= hi) for lo, hi in zip(EDGES[:-1], EDGES[1:])]


def fine_metrics(shape, mass, cores, band_masks):
    rho, error = crossed(shape, mass)
    residual = rho - b.op.lift(mass)
    tensor = b.op.tensor_from_delta(residual, 6.766, workers=8)
    rms = saved.shear_rms(np.stack([tensor[sl] for sl in cores]))
    power = abs(fft.fftn(residual, norm='ortho', workers=8))**2
    return dict(shear_rms=rms, variance=float(np.mean(residual**2)),
                power=[float(power[m].mean()) for m in band_masks], mass_error=error)


def pearson(x, y):
    x, y = np.asarray(x).ravel(), np.asarray(y).ravel()
    x, y = x-x.mean(), y-y.mean()
    norm = np.sqrt(np.sum(x*x)*np.sum(y*y))
    return None if norm < 1e-25 else float(np.sum(x*y)/norm)


def relations(cases, arm, xname, yname):
    x = np.array([r['native'][arm][xname] for r in cases])
    y = np.log(np.array([r['native'][arm][yname] for r in cases]))
    xcenter, ycenter = x-x.mean(1, keepdims=True), y-y.mean(1, keepdims=True)
    ranksx = np.array([rankdata(v) for v in x]); ranksy = np.array([rankdata(v) for v in y])
    return dict(arm=arm, distance=xname, response='log_'+yname,
                within_pair_pearson=pearson(xcenter, ycenter),
                within_pair_rank_correlation=pearson(ranksx-ranksx.mean(1, keepdims=True),
                                                    ranksy-ranksy.mean(1, keepdims=True)),
                per_pair_pearson=[pearson(a, z) for a, z in zip(x, y)],
                between_pair_pearson=pearson(x.mean(1), y.mean(1)))


def summarize(cases):
    result = []
    for phase in ('ph012', 'ph013', 'pooled'):
        rows = [r for r in cases if phase == 'pooled' or r['phase'] == phase]
        controls = []
        for name in rows[0]['cross']:
            entry = dict(arm=name)
            for metric in ('shear_rms', 'variance', 'power'):
                pred = np.mean([v[metric] for r in rows for v in r['cross'][name]], axis=0)
                truth = np.mean([r['truth'][metric] for r in rows], axis=0)
                entry[metric] = dict(prediction=np.asarray(pred).tolist(), truth=np.asarray(truth).tolist(),
                                    ratio=np.asarray(pred/truth).tolist())
            controls.append(entry)
        correlations = []
        if phase != 'pooled':
            for arm in rows[0]['native']:
                for x in ('wide_log_distance', 'joint_log_distance'):
                    for y in ('fine_shear_rms', 'physical_variance', 'shape_variance', 'latent_variance'):
                        correlations.append(relations(rows, arm, x, y))
        result.append(dict(phase=phase, pairs=len(rows), controls=controls, correlations=correlations))
    return result


def run(limit=0):
    b.c.require_compute()
    dest = OUT if not limit else OUT.with_name(OUT.name+f'_pilot{limit}')
    dest.mkdir(parents=True, exist_ok=False)
    cases, inputs = [], []
    band_masks = masks((64, 48, 48))
    for phase in ('ph012', 'ph013'):
        b.auth.frozen(phase, b.FREEZE)
        path = saved.OUT / 'results' / f'COMPLETE_{phase}.json'
        complete = json.loads(path.read_text()); assert complete['passed']
        inputs.append(b.c.file_record(path, content_hash=True))
        chosen = complete['cases'][:limit] if limit else complete['cases']
        for row in chosen:
            before = time.monotonic()
            assert b.c.sha256(row['input']['path']) == row['input']['sha256']
            assert b.c.sha256(row['fields']['path']) == row['fields']['sha256']
            with h5py.File(row['input']['path']) as f:
                trho, twide, start = f['true_rho'][:], f['true_wide'][:], f['core_start'][:]
            cores = [saved.slices(start+[16*i, 0, 0], (16, 16, 16)) for i in (0, 1)]
            tq, tm, tu = shape_mass(trho)
            truth = fine_metrics(tq, tm, cores, band_masks)
            assert abs(truth['shear_rms']/row['truth_scale']['fine_shear_rms']-1) < 3e-6
            native = {name: {k: [] for k in ('wide_log_distance', 'joint_log_distance',
                      'joint_log_bias', 'mass_ratio', 'fine_shear_rms', 'physical_variance',
                      'shape_variance', 'latent_variance')} for name in
                      ('canvas_raw', 'canvas_coarse_local_fine', 'local_both')}
            cross = {a+'_'+m: [] for a in ('truthshape', 'oracleshape', 'sampledshape')
                     for m in ('truemass', 'sampledmass')}
            parity = 0.
            with h5py.File(row['fields']['path']) as f:
                for draw in range(32):
                    shapes = {}
                    for name in (*native, 'oracle_coarse_local_fine'):
                        rho = f[name+'/rho'][draw].astype(float)
                        wide = f[name+'/wide'][draw].astype(float)
                        q, mass, u = shape_mass(rho)
                        np.testing.assert_allclose(mass, wide[CROP], atol=2e-6, rtol=2e-6)
                        shapes[name] = q, mass
                        if name in native:
                            values = dict(wide_log_distance=float(np.sqrt(np.mean(np.log(wide/twide)**2))),
                                joint_log_distance=float(np.sqrt(np.mean(np.log(mass/tm)**2))),
                                joint_log_bias=float(np.mean(np.log(mass/tm))), mass_ratio=float(mass.mean()/tm.mean()),
                                fine_shear_rms=row['arms'][name]['scale'][draw]['fine_shear_rms'],
                                physical_variance=float(np.mean((rho-b.op.lift(mass))**2)),
                                shape_variance=float(np.mean((q-1)**2)), latent_variance=float(np.mean(u*u)))
                            for k, v in values.items(): native[name][k].append(v)
                    sm = shapes['canvas_coarse_local_fine'][1]
                    for label, q in (('truthshape', tq), ('oracleshape', shapes['oracle_coarse_local_fine'][0]),
                                     ('sampledshape', shapes['canvas_coarse_local_fine'][0])):
                        for mlabel, mass in (('truemass', tm), ('sampledmass', sm)):
                            metric = fine_metrics(q, mass, cores, band_masks)
                            cross[label+'_'+mlabel].append(metric)
                    replay = cross['sampledshape_sampledmass'][-1]['shear_rms']
                    expected = row['arms']['canvas_coarse_local_fine']['scale'][draw]['fine_shear_rms']
                    parity = max(parity, abs(replay/expected-1)); assert parity < 3e-6
            rec = dict(phase=phase, pair_id=row['pair_id'], truth=truth, native=native, cross=cross,
                       fine_rms_relative_replay_error=parity, seconds=time.monotonic()-before)
            b.c.atomic_json(dest/(row['pair_id']+'.json'), rec); cases.append(rec)
            print(json.dumps(dict(phase=phase, pair_id=row['pair_id'], seconds=rec['seconds'], parity=parity)), flush=True)
    result = dict(cases=cases, summary=summarize(cases), passed=True, posterior_qualified=False,
                  sources=inputs, source_sha256=b.c.sha256(__file__), job=os.environ['SLURM_JOB_ID'],
                  independent_phases=2, caveat='Dependent windows and shared draws; correlations are not causal or model-error estimates. Crosses use privileged truth masses/shapes.')
    b.c.atomic_json(dest/'COMPLETE.json', result)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--limit', type=int, default=0); a = p.parse_args()
    run(a.limit)
