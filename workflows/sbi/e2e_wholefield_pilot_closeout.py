"""Deterministic descriptive closeout of saved pilot evaluations.

Reads small score records only. No fits, field generation, scheduler calls,
checkpoint selection, uncertainty recalibration, or sealed-phase access.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from workflows.sbi.e2e_wholefield_pilot_readout import aggregate, check_metrics
from workflows.sbi.e2e_wholefield_pilot_stability import (
    ROOT, MODELS, EARLY, FINAL, tasks, result_path, verify_result,
)

PHASES = tuple(f'ph{i:03d}' for i in range(12, 18))
SPLITS = {'development': PHASES[:2], 'exposed_evaluation': PHASES[2:]}
COMPONENTS = ['core0_density', 'core0_shear_spread', 'core0_shear_curvature',
              'core1_density', 'core1_shear_spread', 'core1_shear_curvature']


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def development_point(record, epoch):
    if not record['available']:
        return dict(epoch=epoch, available=False, fixed_noise_cfm_loss=None,
                    reason=record.get('reason', 'Unavailable in saved record'))
    if {x[0] for x in record['ids']} != set(PHASES[:2]):
        raise ValueError('Changed development loss support')
    return dict(epoch=epoch, available=True, fixed_noise_cfm_loss=record['fixed_noise_cfm_loss'])


def central_rank_coverage(ranks):
    """Ranks 2..30 cover 29/33 under continuous exchangeable truth+32 draws.

    This avoids interpreting interpolated sample quantiles as exact population
    90-percent intervals. It is a descriptive diagnostic, not a hypothesis test.
    """
    ranks = np.asarray(ranks)
    if np.any((ranks < 0) | (ranks > 32)) or np.any(ranks != np.rint(ranks)):
        raise ValueError('Expected integer ranks zero through thirty-two')
    return ((ranks >= 2) & (ranks <= 30)).mean(0)


def paired(left, right, keys=('phase', 'pair', 'seed')):
    def indexed(rows):
        values = {tuple(r[k] for k in keys): r for r in rows}
        if len(values) != len(rows):
            raise ValueError('Duplicate matched-case address')
        return values
    a, b = indexed(left), indexed(right)
    if not a or a.keys() != b.keys():
        raise ValueError('Incomplete or mismatched comparison')
    output = []
    for key in sorted(a):
        np.testing.assert_allclose(a[key]['result']['fullbox']['truth'],
                                   b[key]['result']['fullbox']['truth'], rtol=0, atol=0)
        output.append((a[key], b[key]))
    return output


def describe(rows):
    out = aggregate(rows)
    if out is None:
        return None
    full = [r['result']['fullbox'] for r in rows]
    errors = np.array([np.array(r['mean'])-r['truth'] for r in full])
    variance = np.array([r['sd'] for r in full])**2
    closure = np.array([r['result']['closure_truth_shift'] for r in rows])
    mean_variance = variance.mean(0)
    out.update(
        phase_count=len({r['phase'] for r in rows}),
        unique_region_pairs=len({(r['phase'], r['pair']) for r in rows}),
        redshift_range=[min(r['redshift'] for r in rows), max(r['redshift'] for r in rows)],
        component_mean_bias=errors.mean(0).tolist(),
        component_fraction_negative_error=(errors < 0).mean(0).tolist(),
        component_truth_mean=np.array([r['truth'] for r in full]).mean(0).tolist(),
        component_rms_posterior_sd=np.sqrt(mean_variance).tolist(),
        component_central_rank_coverage_29_of_33=central_rank_coverage(
            np.rint(32*np.array([r['rank'] for r in full]))).tolist(),
        central_rank_reference_probability=29/33,
        component_mse_over_mean_variance=((errors**2).mean(0)/mean_variance).tolist(),
        closure_truth_shift_rms=np.sqrt((closure**2).mean(0)).tolist(),
        closure_adjusted_component_rmse=np.sqrt(((errors-closure)**2).mean(0)).tolist(),
        density_voxel_rms_case_rmse=float(np.sqrt(np.mean([
            r['result']['density_voxels']['rmse_mean']**2 for r in rows]))),
        density_voxel_mean_sd=float(np.mean([
            r['result']['density_voxels']['mean_posterior_sd'] for r in rows])))
    if all(r['result']['reference_fullbox'] is not None for r in rows):
        refs = [r['result']['reference_fullbox'] for r in rows]
        out['reference_energy_score'] = float(np.mean([r['energy_score'] for r in refs]))
        out['reference_component_coverage90'] = np.array([r['coverage90'] for r in refs], float).mean(0).tolist()
        out['reference_component_crps'] = np.array([r['crps'] for r in refs]).mean(0).tolist()
        out['reference_component_central_rank_coverage_29_of_33'] = central_rank_coverage(
            np.rint(32*np.array([r['rank'] for r in refs]))).tolist()
        differences = np.array([r['energy_score']-b['energy_score'] for r, b in zip(full, refs)])
        out['reference_energy_difference_median'] = float(np.median(differences))
        out['reference_energy_fraction_lower'] = float(np.mean(differences < 0))
    return out


def compare(left, right):
    pairs = paired(left, right)
    out = dict(cases=len(pairs), direction='left minus right; negative score difference favors left')
    for name, extract in (
        ('density_crps', lambda r: r['density_voxels']['mean_crps']),
        ('energy', lambda r: r['fullbox']['energy_score']),
        ('density_coverage90', lambda r: r['density_voxels']['coverage90'])):
        differences = np.array([extract(a['result'])-extract(b['result']) for a, b in pairs])
        out[name] = dict(mean_difference=float(differences.mean()), median_difference=float(np.median(differences)),
                         fraction_negative=float((differences < 0).mean()))
    return out


def read_primary(root):
    rows, files = [], []
    source = sha(Path(__file__).with_name('e2e_wholefield_pilot_evaluate.py'))
    for phase in PHASES:
        geometry = read(root/'data'/phase/'COMPLETE.json')
        if len(geometry['pairs']) != 16:
            raise ValueError('Changed primary geometry')
        for row in geometry['pairs']:
            for architecture in MODELS:
                for seed in (17, 29):
                    for coarse in ('sampled', 'oracle'):
                        task = dict(phase=phase, pair=row['pair_id'], seed=seed, coarse=coarse, nfe=128)
                        path = result_path(root, architecture, task, FINAL)
                        verify_result(path, task, source, FINAL)
                        binding = read(path.parent/'BINDING.json')
                        if Path(binding['checkpoint']['path']).parent.name != f'{architecture}_{MODELS[architecture]}_{seed}':
                            raise ValueError('Evaluation checkpoint/model mismatch')
                        result = read(path)
                        check_metrics(result['summaries'], result['fullbox'])
                        check_metrics(result['summaries'], result['closure_mean_diagnostic'])
                        rows.append(dict(**task, architecture=architecture, cap=row['cap'],
                                         shell=row['shell'], stratum=row['support_stratum'],
                                         redshift=row['redshift'], result=result))
                        files.append(dict(path=str(path), sha256=sha(path)))
    if len(rows) != 768:
        raise ValueError('Incomplete final-checkpoint panel')
    for phase in PHASES:
        for coarse in ('sampled', 'oracle'):
            selected = [r for r in rows if r['phase'] == phase and r['coarse'] == coarse]
            matches = paired([r for r in selected if r['architecture'] == 'unet'],
                             [r for r in selected if r['architecture'] == 'transformer'])
            if coarse == 'sampled':
                for a, b in matches:
                    if a['result']['reference_fullbox'] != b['result']['reference_fullbox']:
                        raise ValueError('Architecture reference scores are not matched')
    return rows, files


def learning_curves(root):
    output = []
    for architecture, prediction in MODELS.items():
        for seed in (17, 29):
            directory = root/'fits'/f'{architecture}_{prediction}_{seed}'
            bound = read(directory/'BINDING.json')
            if set(bound['data']) & set(PHASES):
                raise ValueError('Evaluation phase present in training binding')
            development = []
            for epoch in (4, 8, 16, 24, 32):
                path = directory/f'DEVELOPMENT_{epoch*312:06d}.json'
                record = read(path)
                development.append(dict(development_point(record, epoch),
                                        source=dict(path=str(path), sha256=sha(path))))
            train = [json.loads(line) for line in (directory/'TRAIN.jsonl').read_text().splitlines()]
            if len({r['step'] for r in train}) != len(train):
                raise ValueError('Duplicate logged training updates')
            windows = []
            for lo, hi in ((0, 4), (4, 8), (8, 16), (16, 24), (24, 32)):
                selected = [r for r in train if lo < r['epoch'] <= hi]
                windows.append(dict(epoch_start_exclusive=lo, epoch_end_inclusive=hi,
                                    logged_updates=len(selected),
                                    loss_mean=float(np.mean([r['loss'] for r in selected])),
                                    mse_mean=float(np.mean([r['mse'] for r in selected])),
                                    median_preclip_gradient_norm=float(np.median([r['gradient_norm'] for r in selected])),
                                    fraction_logged_updates_clipped=float(np.mean([r['gradient_norm'] > .5 for r in selected]))))
            output.append(dict(architecture=architecture, seed=seed, development=development,
                               training_windows=windows, caveat='Training windows average logged updates, not every update; CFM loss is not a posterior score.'))
    return output


def checkpoint_stability(root, final_rows):
    output = []
    source = sha(Path(__file__).with_name('e2e_wholefield_pilot_evaluate.py'))
    for architecture in MODELS:
        early, final, files = [], [], []
        for task in tasks(root):
            path = result_path(root, architecture, task, EARLY)
            if not path.exists():
                continue
            verify_result(path, task, source, EARLY)
            result = read(path)
            check_metrics(result['summaries'], result['fullbox'])
            check_metrics(result['summaries'], result['closure_mean_diagnostic'])
            candidates = [r for r in final_rows if r['architecture'] == architecture
                          and all(r[k] == task[k] for k in ('phase', 'pair', 'seed', 'coarse'))]
            if len(candidates) != 1:
                raise ValueError('Missing or duplicate final checkpoint counterpart')
            final_row = candidates[0]
            if result['coarse_receipts'] != final_row['result']['coarse_receipts']:
                raise ValueError('Checkpoint comparison does not reuse identical coarse draws')
            early.append(dict(final_row, result=result))
            final.append(final_row)
            files.append(dict(path=str(path), sha256=sha(path)))
        record = dict(architecture=architecture, completed=len(early), expected=64,
                      complete=len(early) == 64, no_checkpoint_selection=True,
                      groups=[], stratified_differences=[], files=files)
        if len(early) == 64:
            for coarse in ('sampled', 'oracle'):
                a = [r for r in early if r['coarse'] == coarse]
                b = [r for r in final if r['coarse'] == coarse]
                matches = paired(a, b)
                changes = []
                for left, right in matches:
                    x = np.asarray(left['result']['summaries'])
                    y = np.asarray(right['result']['summaries'])
                    changes.append(np.sqrt(((x-y)**2).mean(0))/np.maximum(y.std(0, ddof=1), 1e-12))
                record['groups'].append(dict(coarse=coarse, epoch24=describe(a), epoch32=describe(b),
                    epoch24_minus_epoch32=compare(a, b),
                    paired_draw_rms_over_epoch32_sd_median=np.median(changes, axis=0).tolist(),
                    paired_draw_rms_over_epoch32_sd_max=np.max(changes, axis=0).tolist()))
                for key in ('phase', 'seed', 'shell', 'stratum'):
                    for value in sorted({r[key] for r in a}):
                        left = [r for r in a if r[key] == value]
                        right = [r for r in b if r[key] == value]
                        record['stratified_differences'].append(dict(coarse=coarse, kind=key, value=value,
                            epoch24_minus_epoch32=compare(left, right)))
        output.append(record)
    return output


def reconcile_groups(groups):
    """Independent scalar weighted sums cross-check phase/split aggregation."""
    checks = 0
    for architecture in MODELS:
        for coarse in ('sampled', 'oracle'):
            for split, phases in SPLITS.items():
                total = next(g['scores'] for g in groups if g['kind'] == 'split'
                             and g['value'] == split and g['architecture'] == architecture and g['coarse'] == coarse)
                pieces = [g['scores'] for g in groups if g['kind'] == 'phase'
                          and g['value'] in phases and g['architecture'] == architecture and g['coarse'] == coarse]
                if sum(s['cases'] for s in pieces) != total['cases']:
                    raise ValueError('Phase/split case totals disagree')
                checks += 1
                for key in ('density_voxel_crps', 'density_voxel_coverage90', 'energy_score',
                            'summary_component_crps', 'summary_component_coverage90', 'component_mean_bias'):
                    values = total[key] if isinstance(total[key], list) else [total[key]]
                    for j, value in enumerate(values):
                        expected = math.fsum((s[key][j] if isinstance(s[key], list) else s[key])*s['cases']
                                             for s in pieces)/total['cases']
                        if not math.isclose(expected, value, rel_tol=1e-13, abs_tol=1e-14):
                            raise ValueError('Phase/split metric mismatch: '+key)
                        checks += 1
    return checks


def closeout(root):
    rows, files = read_primary(root)
    groups, comparisons = [], []
    definitions = [('phase', phase, [r for r in rows if r['phase'] == phase]) for phase in PHASES]
    for split, phases in SPLITS.items():
        selected = [r for r in rows if r['phase'] in phases]
        definitions.append(('split', split, selected))
        for key in ('cap', 'shell', 'stratum', 'seed'):
            for value in sorted({r[key] for r in selected}):
                definitions.append((f'{split}_{key}', value, [r for r in selected if r[key] == value]))
    for kind, value, selected in definitions:
        for architecture in MODELS:
            for coarse in ('sampled', 'oracle'):
                chosen = [r for r in selected if r['architecture'] == architecture and r['coarse'] == coarse]
                groups.append(dict(kind=kind, value=value, architecture=architecture, coarse=coarse,
                                   scores=describe(chosen)))
        for coarse in ('sampled', 'oracle'):
            chosen = [r for r in selected if r['coarse'] == coarse]
            comparisons.append(dict(kind=kind, value=value, coarse=coarse,
                unet_minus_transformer=compare([r for r in chosen if r['architecture'] == 'unet'],
                                              [r for r in chosen if r['architecture'] == 'transformer'])))
    return dict(schema='wholefield-scientific-closeout-v1', generated_utc=datetime.now(timezone.utc).isoformat(),
                source=sha(__file__), primary_cases=len(rows), files=files, components=COMPONENTS,
                groups=groups, independent_aggregation_checks=reconcile_groups(groups),
                architecture_comparisons=comparisons, learning_curves=learning_curves(root),
                stability=checkpoint_stability(root, rows), posterior_qualified=False,
                limitations=[
                    'Descriptive equal-case means; four exposed evaluation phases are not fresh blind confirmation.',
                    'Regions, seeds, components and voxels are dependent, not independent simulation trials.',
                    'Thirty-two posterior draws; interpolated nominal 90-percent coverage is not an exact 90-percent benchmark.',
                    'Reference scores are paired only for sampled coarse; oracle is unavailable on real DESI.',
                    'Shared frozen exterior is not jointly reconditioned on the new fine field.',
                    'Energy score uses six components in their registered physical units, not a whitened joint-calibration test.',
                    'Backbone and output parameterization differ together; no isolated architecture causal claim.',
                    'Stability subset is development-only, geometry-selected and does not select a new checkpoint.',
                    'Same-noise draw displacement measures coupling sensitivity, not by itself a distance between posterior distributions.',
                    'Ratios of MSE to mean variance are descriptive; finite-ensemble mean noise and conditional bias contribute.'
                ])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = closeout(args.root)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(primary_cases=result['primary_cases'],
        stability=[{k: s[k] for k in ('architecture', 'completed', 'expected', 'complete')} for s in result['stability']])) )
