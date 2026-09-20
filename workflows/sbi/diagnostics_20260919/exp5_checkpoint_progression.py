"""Had the completed Abacus VDM fits converged by 20480 updates? (read-only)

The Gaussian reference continuation showed that 4096 updates were materially
under-trained and that 16x more training halved covariance error. The completed
A/B/C/D campaign trained 20480 updates. If its metrics were still descending
there, its ABSOLUTE calibration statements are under-training-confounded, while
its matched arm CONTRASTS are not, since every arm received the same budget.

This reads only published case reports from the frozen run. It opens no payload,
no truth and no checkpoint, and writes nothing into the run root.

Ensemble size changes across the progression (32 draws at 5120/10240, 64 or 128
at 20480), so only ensemble-size-unbiased quantities are compared:
  * density_crps and tidal_energy use fair U-statistic pair corrections;
  * mean_sample_power_ratio is a mean over draws of a per-draw quantity;
  * coverage is compared as |coverage90 - attainable90|, each case using its own
    finite-M attainable value.
density_rmse is EXCLUDED: it scores the M-draw posterior mean, whose error falls
with M, so it is not comparable across the progression.
"""
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

RUN = Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v2/wide_pipeline_v1/vdm_context_20260917_v1')
CHECKPOINTS = (5120, 10240, 20480)


def collect(run):
    """Matched (arm, seed, anchor) triples present at all three checkpoints."""
    index = defaultdict(dict)
    folder = run / 'analysis' / 'cases'
    for name in sorted(os.listdir(folder)):
        if not name.endswith('.json'):
            continue
        record = json.loads((folder / name).read_text())
        task = record['task']
        if task['purpose'] != 'main' or task['steps'] != 250:
            continue
        if task.get('coarse_mode') not in (None, 'sampled'):
            continue
        primary, spectra = record['primary'], record['spectra']
        index[(task['arm'], task['replica'], task['anchor'])][task['checkpoint']] = dict(
            draws=task['count'],
            density_crps=primary['density_crps'],
            tidal_energy=primary['tidal_energy'],
            coverage_gap=abs(primary['density_coverage90'] - primary['attainable90']),
            power_ratio=np.asarray(spectra['mean_sample_power_ratio'], dtype=float))
    return {k: v for k, v in index.items() if set(CHECKPOINTS) <= set(v)}


def power_law_slope(values):
    """Least-squares slope of log(metric) against log(updates)."""
    x = np.log(np.asarray(CHECKPOINTS, dtype=float))
    y = np.log(np.asarray(values, dtype=float))
    return float(np.polyfit(x, y, 1)[0])


def analyse(triples):
    metrics = ('density_crps', 'tidal_energy', 'coverage_gap', 'power_deficit')
    out = {}
    for arm in sorted({k[0] for k in triples}):
        for metric in metrics:
            per_checkpoint, improved_second = {c: [] for c in CHECKPOINTS}, []
            for key, cells in triples.items():
                if key[0] != arm:
                    continue
                series = []
                for c in CHECKPOINTS:
                    cell = cells[c]
                    value = (float(np.abs(1 - cell['power_ratio']).sum())
                             if metric == 'power_deficit' else cell[metric])
                    per_checkpoint[c].append(value)
                    series.append(value)
                improved_second.append(series[2] < series[1])
            means = [float(np.mean(per_checkpoint[c])) for c in CHECKPOINTS]
            first = (means[1] - means[0]) / means[0]
            second = (means[2] - means[1]) / means[1]
            slope = power_law_slope(means)
            out[f'{arm}:{metric}'] = dict(
                means=means, anchors=len(improved_second),
                fractional_change_5120_to_10240=first,
                fractional_change_10240_to_20480=second,
                deceleration=float(second / first) if first else None,
                anchors_improving_on_second_doubling=int(sum(improved_second)),
                log_log_slope=slope,
                projected_if_16x_more=float(means[2] * 16 ** slope))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=RUN)
    parser.add_argument('--out', type=Path,
                        default=Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/diagnostics_20260919'))
    args = parser.parse_args()
    triples = collect(args.run)
    if not triples:
        raise SystemExit('no matched checkpoint triples found')
    stats = analyse(triples)
    draws = sorted({cells[c]['draws'] for cells in triples.values() for c in CHECKPOINTS})
    result = dict(
        schema='e2e-diagnostic-checkpoint-progression-v1', run=str(args.run),
        checkpoints=list(CHECKPOINTS), matched_triples=len(triples),
        ensemble_sizes_present=draws,
        panel='ph004 development anchors only; development-panel progression, not a held-out claim',
        caveat=('Ensemble size rises at 20480, which lowers the variance of the 20480 '
                'estimates but does not bias the compared quantities. density_rmse is '
                'excluded as ensemble-size dependent. A log-log slope from three points '
                'is a descriptive trend, not a fitted convergence law, and the projection '
                'assumes that trend continues, which is exactly what is not established.'),
        statistics=stats)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / 'EXP5_CHECKPOINT_PROGRESSION.json'
    path.write_text(json.dumps(result, indent=1, sort_keys=True))

    print(f'matched triples {len(triples)}   ensemble sizes {draws}\n')
    header = f"{'arm:metric':22s} {'5120':>9s} {'10240':>9s} {'20480':>9s} {'d1%':>8s} {'d2%':>8s} {'slope':>7s} {'impr':>6s}"
    print(header)
    for key, value in stats.items():
        m = value['means']
        print(f"{key:22s} {m[0]:9.4f} {m[1]:9.4f} {m[2]:9.4f} "
              f"{100*value['fractional_change_5120_to_10240']:+8.2f} "
              f"{100*value['fractional_change_10240_to_20480']:+8.2f} "
              f"{value['log_log_slope']:+7.3f} "
              f"{value['anchors_improving_on_second_doubling']:3d}/{value['anchors']}")
    print('\nWROTE', path)


if __name__ == '__main__':
    main()
