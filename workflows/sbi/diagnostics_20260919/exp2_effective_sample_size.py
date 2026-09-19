"""Effective sample size of the frozen VDM case panel (read-only diagnostic).

Question: the registered protocol treats the PHASE as the replicate unit, so the
two development / six confirmation phases bound every interval. That is correct
only if per-anchor scores inside a phase are strongly correlated. This measures
the intraclass correlation (ICC) of the paired arm-difference scores across
anchors within a phase, and converts it to an effective replicate count.

Reads only published case reports from a completed, frozen experiment. It opens
no payload arrays, no truth, no checkpoints, and writes nothing into the run root.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

RUN = Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v2/wide_pipeline_v1/vdm_context_20260917_v1')
SCORES = ('density_crps', 'tidal_energy', 'density_coverage90', 'density_rmse')
PAIRS = (('A', 'B'), ('B', 'C'), ('C', 'D'))


def load_final_cases(run):
    """Final registered ensembles only: checkpoint 20480, main, sampled coarse."""
    rows = {}
    folder = run / 'analysis' / 'cases'
    for name in sorted(os.listdir(folder)):
        if not name.endswith('.json'):
            continue
        record = json.loads((folder / name).read_text())
        task = record['task']
        if task['checkpoint'] != 20480 or task['purpose'] != 'main':
            continue
        # A/B/C carry no coarse_mode; D must be the sampled shared-parent mode.
        if task.get('coarse_mode') not in (None, 'sampled'):
            continue
        if task['count'] not in (64, 128):
            continue
        key = (task['arm'], task['replica'], record['phase'], task['anchor'], task['count'])
        rows[key] = {s: record['primary'][s] for s in SCORES if s in record['primary']}
    return rows


def icc_oneway(groups):
    """One-way random-effects ICC(1) plus its variance components.

    groups: list of equal-length arrays, one per cluster (phase).
    Returns ICC, between-cluster MS, within-cluster MS, cluster count, group size.
    """
    k = len(groups)
    m = len(groups[0])
    if k < 2 or any(len(g) != m for g in groups) or m < 2:
        return None
    values = np.array(groups, dtype=float)
    grand = values.mean()
    cluster_means = values.mean(axis=1)
    msb = m * ((cluster_means - grand) ** 2).sum() / (k - 1)
    msw = ((values - cluster_means[:, None]) ** 2).sum() / (k * (m - 1))
    icc = (msb - msw) / (msb + (m - 1) * msw) if (msb + (m - 1) * msw) > 0 else 0.0
    return dict(icc=float(icc), ms_between=float(msb), ms_within=float(msw),
                clusters=k, per_cluster=m)


def analyse(rows):
    out = {}
    phases = sorted({k[2] for k in rows})
    seeds = sorted({k[1] for k in rows})
    counts = sorted({k[4] for k in rows})
    for lo, hi in PAIRS:
        for score in SCORES:
            groups, labels = [], []
            for seed in seeds:
                for phase in phases:
                    anchors = sorted({(k[3], k[4]) for k in rows
                                      if k[0] == lo and k[1] == seed and k[2] == phase})
                    diffs = []
                    for anchor, count in anchors:
                        a = rows.get((lo, seed, phase, anchor, count))
                        b = rows.get((hi, seed, phase, anchor, count))
                        if a is None or b is None or score not in a or score not in b:
                            continue
                        diffs.append(b[score] - a[score])
                    if len(diffs) >= 2:
                        groups.append(diffs)
                        labels.append(f'seed{seed}_{phase}')
            if len(groups) < 2:
                continue
            size = min(len(g) for g in groups)
            trimmed = [g[:size] for g in groups]
            stats = icc_oneway(trimmed)
            if stats is None:
                continue
            flat = np.array(trimmed, dtype=float).ravel()
            cluster_means = np.array([np.mean(g) for g in trimmed])
            n_total = flat.size
            icc = max(stats['icc'], 0.0)
            n_eff = n_total / (1 + (size - 1) * icc)
            se_naive = flat.std(ddof=1) / np.sqrt(n_total)
            se_cluster = cluster_means.std(ddof=1) / np.sqrt(len(cluster_means))
            out[f'{lo}->{hi}:{score}'] = dict(
                cells=labels, cell_means=[float(x) for x in cluster_means],
                anchors_per_cell=size, n_total=int(n_total),
                mean_difference=float(flat.mean()),
                icc=stats['icc'], n_effective=float(n_eff),
                se_treating_anchors_independent=float(se_naive),
                se_treating_cells_as_clusters=float(se_cluster),
                power_ratio_cluster_over_naive=float(se_cluster / se_naive) if se_naive > 0 else None,
                ms_between=stats['ms_between'], ms_within=stats['ms_within'])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=RUN)
    parser.add_argument('--out', type=Path,
                        default=Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/diagnostics_20260919'))
    args = parser.parse_args()
    rows = load_final_cases(args.run)
    result = dict(
        schema='e2e-diagnostic-effective-sample-size-v1',
        run=str(args.run), cases_used=len(rows),
        caveat=('Anchors inside a phase share large-scale modes; ICC measures how much. '
                'Two development phases give only a noisy between-cluster estimate. '
                'Read-only over published case reports; no payload or truth access.'),
        statistics=analyse(rows))
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / 'EXP2_EFFECTIVE_SAMPLE_SIZE.json'
    path.write_text(json.dumps(result, indent=1, sort_keys=True))
    print('WROTE', path, 'cases', len(rows))
    for key, value in sorted(result['statistics'].items()):
        if 'density_crps' in key or 'tidal_energy' in key:
            print(f"{key:28s} ICC={value['icc']:+.4f}  n_eff={value['n_effective']:6.1f}"
                  f"  of n={value['n_total']}  SE ratio={value['power_ratio_cluster_over_naive']:.2f}")


if __name__ == '__main__':
    main()
