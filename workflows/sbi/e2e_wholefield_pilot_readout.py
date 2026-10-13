"""Read saved pilot scores with independent arithmetic and explicit denominators.

No fitting, scheduler calls, target access, calibration adjustment or pass gate.
Incomplete panels are reported as incomplete, never silently pooled as complete.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_metrics(draws, saved):
    """Independent sorted-ensemble CRPS and loop-based energy verification."""
    x = np.asarray(draws, dtype=float)
    y = np.asarray(saved['truth'], dtype=float)
    if x.shape != (32, 6) or y.shape != (6,) or not np.isfinite(x).all():
        raise ValueError('Expected thirty-two finite six-component draws')
    n = len(x)
    weights = (2*np.arange(n)+1-n)[:, None]
    crps = np.mean(abs(x-y), axis=0)-np.sum(weights*np.sort(x, axis=0), axis=0)/n**2
    pair_distances = sum(np.linalg.norm(a-b) for i, a in enumerate(x) for b in x[:i])
    energy = sum(np.linalg.norm(a-y) for a in x)/n-pair_distances/n**2
    interval = np.percentile(x, [5, 95], axis=0)
    expected = dict(mean=np.mean(x, axis=0), sd=np.std(x, axis=0, ddof=1),
                    crps=crps, energy_score=energy, lower90=interval[0], upper90=interval[1],
                    coverage90=(interval[0] <= y)&(y <= interval[1]),
                    rank=np.sum(x < y, axis=0)/n)
    for key, value in expected.items():
        np.testing.assert_allclose(saved[key], value, atol=1e-10, rtol=1e-10, err_msg=key)


def aggregate(rows):
    """Descriptive equal-case means; seeds/cores/voxels are not independent trials."""
    if not rows:
        return None
    full = [r['result']['fullbox'] for r in rows]
    crps = np.asarray([r['crps'] for r in full])
    cover = np.asarray([r['coverage90'] for r in full], dtype=float)
    error = np.asarray([r['mean'] for r in full])-np.asarray([r['truth'] for r in full])
    sd = np.asarray([r['sd'] for r in full])
    ranks = np.rint(32*np.asarray([r['rank'] for r in full])).astype(int)
    result = dict(cases=len(rows), summary_component_coverage90=cover.mean(0).tolist(),
                  summary_component_crps=crps.mean(0).tolist(),
                  summary_component_rmse=np.sqrt((error**2).mean(0)).tolist(),
                  summary_component_mean_sd=sd.mean(0).tolist(),
                  summary_component_rank_counts=[np.bincount(ranks[:, j], minlength=33).tolist() for j in range(6)],
                  energy_score=float(np.mean([r['energy_score'] for r in full])),
                  density_voxel_coverage90=float(np.mean([r['result']['density_voxels']['coverage90'] for r in rows])),
                  density_voxel_crps=float(np.mean([r['result']['density_voxels']['mean_crps'] for r in rows])))
    baseline = [r['result']['reference_fullbox'] for r in rows]
    if all(b is not None for b in baseline):
        result['paired_reference_crps_difference'] = (crps-np.asarray([b['crps'] for b in baseline])).mean(0).tolist()
        result['paired_reference_energy_difference'] = float(np.mean([
            a['energy_score']-b['energy_score'] for a, b in zip(full, baseline)]))
    return result


def readout(root):
    root = Path(root)
    rows, files = [], []
    for path in sorted((root/'evaluation').glob('*_*/step9984/ph01[2-7]/*/*/fullcap/nfe*/COMPLETE.json')):
        binding_path = path.parent/'BINDING.json'
        result = json.loads(path.read_text())
        binding = json.loads(binding_path.read_text())
        if not result['passed'] or result['binding'] != digest(binding_path):
            raise ValueError('Invalid completed case: '+str(path))
        model = path.relative_to(root/'evaluation').parts[0]
        architecture, prediction, seed = model.split('_')
        if binding['draws'] != 32 or binding['operator'] != 'fullcap':
            raise ValueError('Mixed panel contract')
        check_metrics(result['summaries'], result['fullbox'])
        check_metrics(result['summaries'], result['closure_mean_diagnostic'])
        rows.append(dict(architecture=architecture, prediction=prediction, seed=int(seed),
                         phase=binding['phase'], pair=binding['pair'], coarse=binding['coarse'],
                         nfe=binding['nfe'], result=result))
        files.append(dict(path=str(path), sha256=digest(path)))
    panels = []
    for architecture in ('unet', 'transformer'):
        for coarse in ('sampled', 'oracle'):
            selected = [r for r in rows if r['architecture'] == architecture and r['coarse'] == coarse and r['nfe'] == 128]
            per_phase = []
            for phase in (f'ph{i:03d}' for i in range(12, 18)):
                phase_rows = [r for r in selected if r['phase'] == phase]
                addresses = {(r['pair'], r['seed']) for r in phase_rows}
                if len(addresses) != len(phase_rows):
                    raise ValueError('Duplicate primary case')
                per_phase.append(dict(phase=phase, complete=len(addresses) == 32,
                                      expected_cases=32, scores=aggregate(phase_rows)))
            splits = []
            for name, phases in (('development', {'ph012', 'ph013'}),
                                 ('exposed_evaluation', {'ph014', 'ph015', 'ph016', 'ph017'})):
                split_rows = [r for r in selected if r['phase'] in phases]
                complete = all(p['complete'] for p in per_phase if p['phase'] in phases)
                splits.append(dict(split=name, complete=complete, expected_cases=32*len(phases),
                                   scores=aggregate(split_rows), weighting='equal observed cases; equal phases only when complete'))
            panels.append(dict(architecture=architecture, coarse=coarse, phases=per_phase, splits=splits))
    refinements = []
    for fine in (r for r in rows if r['nfe'] == 256):
        paired = [r for r in rows if r['nfe'] == 128 and all(r[k] == fine[k] for k in ('architecture', 'seed', 'phase', 'pair', 'coarse'))]
        if len(paired) != 1:
            continue
        a, b = np.asarray(paired[0]['result']['summaries']), np.asarray(fine['result']['summaries'])
        scale = b.std(0, ddof=1)
        refinements.append(dict(**{k: fine[k] for k in ('architecture', 'seed', 'phase', 'pair', 'coarse')},
                                component_paired_rms=np.sqrt(((a-b)**2).mean(0)).tolist(),
                                component_paired_rms_over_sd=(np.sqrt(((a-b)**2).mean(0))/np.maximum(scale, 1e-12)).tolist()))
    return dict(schema='exploratory-wholefield-readout-v1', source=digest(__file__), files=files,
                summary_components=['core0_density', 'core0_shear_spread', 'core0_shear_curvature',
                                    'core1_density', 'core1_shear_spread', 'core1_shear_curvature'],
                completed_addresses=len(rows), expected_addresses=772,
                all_addresses_complete=len(rows) == 772, panels=panels, refinements=refinements,
                posterior_qualified=False,
                limitations=['Descriptive finite-ensemble scores, not a statistical pass rule.',
                             'Nominal 90-percent interpolated intervals use only 32 draws; finite-ensemble coverage differs from exact-posterior coverage.',
                             'Ranks have 33 discrete levels; uniformity is descriptive here, with phase and spatial dependence retained.',
                             'Six exposed phases; two seeds, cores and voxels are not independent simulations.',
                             'Shared frozen exterior is not reconditioned on replacement density.',
                             'Oracle denotes true coarse input, not a full-cap truth exterior.',
                             'Partial panel averages reflect completed cases only.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = readout(args.root)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(completed=report['completed_addresses'], complete=report['all_addresses_complete'])))
