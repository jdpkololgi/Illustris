"""Saved-data-only A/C and sampled-coarse power addendum. No field generation.

Never equates the saved local oracle with the full-cap Stage C posterior.
All file hashing and array analysis must run in an interactive compute step.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time

import h5py
import numpy as np
import e2e_true_coarse_fraction_metrics as metrics

REPO = Path(__file__).resolve().parents[2]
PARENT = Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v3')
OLD = PARENT/'cfm_conditioning_augmentation_20261006_v3'
BASE = PARENT/'coherent_exterior_20261002_v1'
FACTOR = BASE/'verification/sampler_factor_20261005_v1'
OUT = PARENT/'true_coarse_fraction_diagnostic_20261008_v1'
CONFIG = REPO/'configs/e2e_true_coarse_fraction_diagnostic_20261008_v1.json'
PHASES = ('ph012', 'ph013')
SETTINGS = [('parent', 0.), ('control', 0.)]+[('augmented', s) for s in (0., .025, .05, .1, .2)]
FINE_SHA = '88b2c7d1c5174e64dc98bb6f555bf555c876ae02496f237a71f7ee8c31e5cced'
CROP = (slice(16, 32), slice(18, 30), slice(18, 30))
CACHE = {}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(16 << 20), b''):
            digest.update(part)
    return digest.hexdigest()


def record(path):
    path = Path(path)
    stat = path.stat(); key = str(path)
    if key not in CACHE:
        CACHE[key] = dict(path=key, bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=sha(path))
    elif CACHE[key]['bytes'] != stat.st_size or CACHE[key]['mtime_ns'] != stat.st_mtime_ns:
        raise ValueError('Input changed during analysis: '+key)
    return CACHE[key]


def verify(rec):
    got = record(rec['path'])
    if got['sha256'] != rec['sha256']:
        raise ValueError('Input hash mismatch: '+rec['path'])
    return got


def write(path, value):
    path = Path(path)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def compute_guard():
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith('login'):
        raise RuntimeError('Interactive compute step required')
    if os.environ.get('PYTHONPATH') or os.environ.get('PYTHONHOME') or os.environ.get('PYTHONNOUSERSITE') != '1':
        raise RuntimeError('Required isolated cosmic_env hygiene missing')


def lift(x):
    for axis in (-3, -2, -1):
        x = np.repeat(x, 4, axis=axis)
    return x


def slices(start, shape):
    return tuple(slice(int(a), int(a+n)) for a, n in zip(start, shape))


def key(arm, strength):
    return f'{arm}_s{strength:g}'


def load_parent():
    """Reuse existing Stage C core products; no density or tide regeneration."""
    cases = []; fractions = []; truths = []; densities = []; tdensities = []
    for phase in PHASES:
        for cap in ('NGC', 'SGC'):
            eigen = []; rho = []
            for first in range(0, 32, 4):
                path = OLD/'cores'/phase/cap/'parent/s0'/f'd{first:03d}/COMPLETE.json'
                rec = read(path); record(path); verify(rec['binding']); verify(rec['output'])
                if not rec['passed'] or rec['phase'] != phase or rec['cap'] != cap or rec['first'] != first or rec['draws'] != 4:
                    raise ValueError('Parent core address mismatch')
                with np.load(rec['output']['path']) as f:
                    eigen.append(f['eigen']); rho.append(f['density'])
                    current = [f[k] for k in ('truth_eigen', 'truth_density', 'galaxy_counts', 'pair_id')]
                    if first == 0:
                        fixed = current; meta = rec['cases']
                    else:
                        if meta != rec['cases']:
                            raise ValueError('Parent core metadata drift')
                        for a, b in zip(fixed, current):
                            np.testing.assert_array_equal(a, b)
            eigen = np.concatenate(eigen, axis=2)
            density = np.concatenate(rho, axis=1)
            target, trho, weight, names = fixed
            if eigen.shape != (2, 8, 32, 2, 4096, 3) or np.any(weight.sum(-1) <= 0):
                raise ValueError('Stage C geometry/weights mismatch')
            if list(names) != [r['pair_id'] for r in meta]:
                raise ValueError('Case order mismatch')
            lab = (eigen > .2).sum(-1); tlab = (target > .2).sum(-1)
            frac = np.stack([((lab == k)*weight[None, :, None]).sum(-1)/weight.sum(-1)[None, :, None] for k in range(4)], -1)
            truth = np.stack([((tlab == k)*weight).sum(-1)/weight.sum(-1) for k in range(4)], -1)
            fractions.append(frac); truths.append(truth); cases.extend(meta)
            densities.append(density.mean(-1)); tdensities.append(trho.mean(-1))
    if len(cases) != 32 or len({r['pair_id'] for r in cases}) != 32:
        raise ValueError('Complete registered 32-pair panel required')
    return np.concatenate(fractions, axis=1), np.concatenate(truths), cases, np.concatenate(densities), np.concatenate(tdensities)


def summarize_power(rows, counts):
    result = []
    for phase in PHASES:
        panel = [r for r in rows if r['phase'] == phase]
        if len(panel) != 16:
            raise ValueError('Sixteen paired windows per phase required')
        for name in panel[0]['spectra']:
            for i in range(len(panel[0]['spectra'][name])):
                values = [r['spectra'][name][i] for r in panel]
                v = values[0]
                per_draw = np.mean([q['per_draw_power'] for q in values], axis=0)
                truth = float(np.mean([q['truth_power'] for q in values]))
                if truth <= 0:
                    raise ValueError('Nonpositive reference power')
                entry = dict(phase=phase, setting_factor=name, k_min=v['k_min'], k_max=v['k_max'],
                    interval=v['interval'], modes_per_window=v['modes'], windows=len(panel),
                    truth_power=truth, sample_power=float(per_draw.mean()),
                    sample_truth_power_ratio=float(per_draw.mean()/truth),
                    mc_ratio=metrics.interval(counts@per_draw/32/truth))
                for field in ('mean_power', 'posterior_variance_power', 'mean_error_power', 'mean_error_minus_draw_noise'):
                    entry[field] = float(np.mean([q[field] for q in values]))
                result.append(entry)
    return result


def run():
    compute_guard(); started = time.monotonic()
    config = read(CONFIG)
    if config['phases'] != list(PHASES) or config['draws'] != 32 or config['training'] or config['model_selection']:
        raise ValueError('Diagnostic contract changed')
    OUT.mkdir(exist_ok=False)
    sources = [record(p) for p in (CONFIG, __file__, metrics.__file__)]
    old_decision = record(OLD/'decision/COMPLETE.json')
    freeze = read(BASE/'SELECTION_FREEZE.json'); record(BASE/'SELECTION_FREEZE.json')
    if freeze['checkpoints']['fine'] != FINE_SHA:
        raise ValueError('Wrong frozen fine checkpoint')
    record(PARENT/'cfm_pilot_20260924_v1/fine_17/CHECKPOINT_026624.pt')
    if CACHE[str(PARENT/'cfm_pilot_20260924_v1/fine_17/CHECKPOINT_026624.pt')]['sha256'] != FINE_SHA:
        raise ValueError('Parent checkpoint payload changed')
    prepared = read(FACTOR/'PREPARED.json'); record(FACTOR/'PREPARED.json')
    if not prepared['passed'] or prepared['freeze_sha256'] != record(BASE/'SELECTION_FREEZE.json')['sha256']:
        raise ValueError('Saved-window freeze mismatch')
    rows = prepared['cases']
    if len(rows) != 32 or {r['phase'] for r in rows} != set(PHASES):
        raise ValueError('Development-only 32-case panel required')
    for phase in PHASES:
        p = FACTOR/'results'/f'COMPLETE_{phase}.json'; completion = read(p); record(p)
        if not completion['passed'] or completion['draws'] != 32 or completion['nfe'] != 128 or completion['checkpoints']['fine'] != FINE_SHA:
            raise ValueError('Saved parent oracle contract mismatch')
    manifest = dict(config=config, config_record=record(CONFIG), source_records=sources,
        git_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        original_decision=old_decision, parent_checkpoint_sha256=FINE_SHA,
        job=os.environ['SLURM_JOB_ID'], node=socket.gethostname(), scope=config['scope'],
        B_status='matched_fullcap_true_coarse_authorized_not_yet_executed',
        interpretation_rule_frozen_before_scoring=True)
    write(OUT/'MANIFEST.json', manifest)
    _, counts = metrics.resamples()
    frac, truth, cases, parent_density, true_density = load_parent()
    a_reports = []
    parent_score_path = OLD/'scores/parent_s0/COMPLETE.json'
    parent_score = read(parent_score_path); record(parent_score_path)
    for phase in PHASES:
        for group in ('interior', 'boundary'):
            ix = [j for j, r in enumerate(cases) if r['phase'] == phase and r['support_stratum'] == group]
            if len(ix) != 8:
                raise ValueError('Stage C stratum changed')
            for mi, method in enumerate(('original', 'corrected')):
                report = metrics.fractions_report(frac[mi, ix], truth[ix], counts)
                prior = next(q for q in parent_score['moments'] if q['phase'] == phase and q['group'] == 'support_stratum='+group and q['operator'] == method and q['weighting'] == 'galaxy')
                np.testing.assert_allclose(report['joint']['ratio'], prior['joint']['trace_adjusted_mse_over_variance'], atol=1e-12, rtol=0)
                np.testing.assert_allclose(report['joint']['bias'], prior['joint']['bias'], atol=1e-12, rtol=0)
                if group == 'interior':
                    expected = parent_score['mc_sensitivity'][phase+'/'+method+'/interior_ratio']
                    np.testing.assert_allclose([report['joint']['mc']['ratio']['lower'], report['joint']['mc']['ratio']['upper']], [expected['lower'], expected['upper']], atol=1e-10, rtol=0)
                a_reports.append(dict(phase=phase, group=group, operator=method, **report))
    print(json.dumps(dict(stage='A_reconciled', reports=len(a_reports))), flush=True)
    coarse_x = []; coarse_y = []; power_rows = []; inventory = []
    for row in rows:
        phase, cap, pid = (row[k] for k in ('phase', 'cap', 'pair_id'))
        if phase not in PHASES or cap not in ('NGC', 'SGC'):
            raise PermissionError('Development panel only')
        verify(row['file']); verify(row['truth'])
        baseline_path = FACTOR/'results'/(pid+'.json'); baseline = read(baseline_path); record(baseline_path); verify(baseline['fields'])
        with h5py.File(row['input']) as f:
            true_rho = f['true_rho'][:].astype(float); true_wide = f['true_wide'][:].astype(float)
            canvas_z = f['canvas_z'][:]
            valid = np.isfinite(f['canvas_rho'][0])
        with h5py.File(baseline['fields']['path']) as f:
            if f['oracle_coarse_local_fine/rho'].shape != (32, 64, 48, 48):
                raise ValueError('Saved oracle draw bank incomplete')
            wide = f['canvas_raw/wide'][:].astype(float)
            parent_rho = f['canvas_raw/rho'][:].astype(float)
            inventory.append(dict(pair_id=pid, input=baseline['fields'], saved_parent_true_coarse=True,
                sampler='isolated_production_window', operator='historical_local_plus_wide',
                eligible_for_fullcap_A_B=False))
        core = np.asarray(row['core_start'])
        if np.any(core % 4):
            raise ValueError('Regional core is not coarse-block aligned')
        xc = []; yc = []
        for side in (0, 1):
            start = np.array([16, 18, 18])+core//4+[4*side, 0, 0]
            crop = slices(start, (4, 4, 4))
            xc.append(wide[(slice(None), *crop)].mean((1, 2, 3)))
            yc.append(float(true_wide[crop].mean()))
        cx, cy = np.stack(xc, -1), np.array(yc)
        ix = next(j for j, c in enumerate(cases) if c['pair_id'] == pid)
        # Independent saved full-cap density means establish exact regional support.
        np.testing.assert_allclose(cx, parent_density[ix], rtol=0, atol=3e-6)
        np.testing.assert_allclose(cy, true_density[ix], rtol=0, atol=3e-6)
        coarse_x.append(cx); coarse_y.append(cy)
        basefine = lift(wide[(slice(None), *CROP)])
        truthfine = true_rho-lift(true_wide[CROP])
        power = {}
        for legacy in (False, True):
            suffix = '/legacy' if legacy else '/primary'
            power['coarse'+suffix] = metrics.spectra(wide-1, true_wide-1, 27.064, (0, .01, .02, .04, .08, .16, np.inf), legacy)
        for arm, strength in SETTINGS:
            if arm == 'parent':
                rho = parent_rho
            else:
                chunks = []
                for first in range(0, 32, 4):
                    path = OLD/'fields'/phase/cap/arm/f's{strength:g}'/f'd{first:03d}/COMPLETE.json'
                    receipt = read(path); record(path); verify(receipt['field'])
                    if not receipt['passed'] or receipt['phase'] != phase or receipt['cap'] != cap or receipt['first'] != first or receipt['arm'] != arm or receipt['strength'] != strength or receipt['nfe'] != 128:
                        raise ValueError('Augmentation field address mismatch')
                    with h5py.File(receipt['field']['path']) as f:
                        raw = (np.asarray(row['fine_origin_mpc_h'])-np.asarray(f.attrs['origin_mpc_h']))/6.766
                        start = np.rint(raw).astype(int)
                        np.testing.assert_allclose(raw, start, atol=1e-8, rtol=0)
                        cs = (np.asarray(row['midpoint'])-192-np.asarray(f.attrs['coarse_origin_raw_cells']))//8
                        if np.any(start < 0) or np.any(cs < 0):
                            raise ValueError('Negative field crop')
                        np.testing.assert_array_equal(f['coarse_logrho_standardized'][(slice(None), *slices(cs, (48, 48, 48)))], canvas_z[first:first+4])
                        chunk = f['rho'][(slice(None), *slices(start, (64, 48, 48)))].astype(float)
                        np.testing.assert_array_equal(np.isfinite(chunk), np.broadcast_to(valid, chunk.shape))
                        chunks.append(chunk)
                rho = np.where(valid[None], np.concatenate(chunks), basefine)
            residual = rho-basefine
            for legacy in (False, True):
                suffix = '/legacy' if legacy else '/primary'
                power[key(arm, strength)+suffix] = metrics.spectra(residual, truthfine, 6.766, (0, .04, .08, .16, .32, np.inf), legacy)
        power_rows.append(dict(pair_id=pid, phase=phase, cap=cap, uncovered_fraction=float(1-valid.mean()), spectra=power))
        print(json.dumps(dict(stage='saved_power_and_coarse', pair_id=pid, completed=len(power_rows), seconds=time.monotonic()-started)), flush=True)
    coarse_x, coarse_y = np.asarray(coarse_x), np.asarray(coarse_y)
    coarse_reports = []
    for phase in PHASES:
        for group in ('interior', 'boundary'):
            ix = [j for j, r in enumerate(rows) if r['phase'] == phase and ('_'+group+'_') in r['pair_id']]
            if len(ix) != 8:
                raise ValueError('Coarse stratum mismatch')
            coarse_reports.append(dict(phase=phase, group=group, weighting='equal-volume regional density; equal paired cases',
                joint=metrics.moments(coarse_x[ix], coarse_y[ix], counts)))
    addendum = dict(passed=True, evaluation_complete=True, diagnostic_only=True,
        scope=config['scope'], source_decision=old_decision, config=record(CONFIG), sources=sources,
        summary=summarize_power(power_rows, counts), cases=power_rows,
        interpretation='Descriptive missing sampled-coarse power readout; original selection decision is not overwritten or reopened',
        caveats=['Overlapping rectangular windows, not independent Fourier modes or phase replications',
            'Same historical coarse-only fill for uncovered generated cells; true physical window reference retained',
            'Primary half-open bands follow the new request; legacy bands retained for exact comparison'])
    write(OUT/'SAMPLED_COARSE_POWER_ADDENDUM.json', addendum)
    with (OUT/'SAVED_ARRAYS.npz').open('xb') as stream:
        np.savez_compressed(stream, parent_fractions=frac, truth_fractions=truth,
            parent_pair_id=np.array([r['pair_id'] for r in cases]), coarse_draws=coarse_x,
            coarse_truth=coarse_y, coarse_pair_id=np.array([r['pair_id'] for r in rows]))
    for rec in sources:
        if sha(rec['path']) != rec['sha256']:
            raise ValueError('Diagnostic source changed during execution')
    result = dict(passed=True, saved_field_analysis_complete=True, A=a_reports, C=coarse_reports,
        B_status='not_scored_saved_oracle_is_not_matched_to_Stage_C', oracle_inventory=inventory,
        power_addendum=record(OUT/'SAMPLED_COARSE_POWER_ADDENDUM.json'),
        arrays=record(OUT/'SAVED_ARRAYS.npz'), manifest=record(OUT/'MANIFEST.json'),
        inputs=list(CACHE.values()), seconds=time.monotonic()-started,
        job=os.environ['SLURM_JOB_ID'], node=socket.gethostname(), scope=config['scope'],
        posterior_qualified=False, selection=False, confirmation_access=False)
    write(OUT/'SAVED_FIELDS_COMPLETE.json', result)
    print(json.dumps(dict(stage='saved_fields_complete', seconds=result['seconds'], receipt=str(OUT/'SAVED_FIELDS_COMPLETE.json'))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.parse_args(); run()
