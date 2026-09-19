"""Descriptive, all-cell learning curves; never select the best checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import statistics


METRICS = ('mean_rms', 'covariance_relative', 'variance_ratio', 'octant_coverage')


def aggregate(rows):
    if not rows:
        raise ValueError('empty comparison')
    return dict(cells=len(rows), passed=sum(r['passed'] for r in rows),
                power_passed=sum(all(.9 <= x <= 1.1 for x in r['power_ratio']) for r in rows),
                metrics={k: dict(mean=statistics.mean(r[k] for r in rows),
                                  minimum=min(r[k] for r in rows),
                                  maximum=max(r[k] for r in rows)) for k in METRICS},
                power_minimum=min(min(r['power_ratio']) for r in rows),
                power_maximum=max(max(r['power_ratio']) for r in rows))


def summarize(done, manifest):
    ext, base = manifest['extension'], manifest['base']
    if done['sources'] != manifest['sources']:
        raise ValueError('source mismatch')
    identities = set()
    for r in done['evaluations']:
        if r['draws'] != base['draws']:
            raise ValueError('curve draw count mismatch')
        key = (r['fit'], r['update'], r['case'], r['nfe'])
        if key in identities:
            raise ValueError('duplicate evaluation')
        identities.add(key)
    expected = {(i['name'], u, c, n)
                for i in manifest['items'] for u in ext['checkpoints']
                for c in (range(base['cases']) if i['fixed'] is None else [i['fixed']])
                for n in ext['nfe_by_objective'][i['objective']]}
    if identities != expected:
        raise ValueError('incomplete or unexpected learning curve')
    precise_ids = [(r['fit'], r['case']) for r in done['precision']]
    expected_precise = {(i['name'], c) for i in manifest['items']
                        for c in (range(base['cases']) if i['fixed'] is None else [i['fixed']])}
    if len(precise_ids) != len(set(precise_ids)) or set(precise_ids) != expected_precise:
        raise ValueError('incomplete or duplicate precision evaluation')
    for r in done['precision']:
        if (r['update'] != ext['updates'] or r['draws'] != ext['precision_draws']
                or r['nfe'] != max(ext['nfe_by_objective'][r['objective']])):
            raise ValueError('precision configuration mismatch')
    groups = {}
    for objective in base['objectives']:
        low, high = ext['nfe_by_objective'][objective]
        for regime in ('fixed', 'amortised'):
            select = lambda r: r['objective'] == objective and regime in r['fit']
            matched = lambda r: select(r) and r['case'] in base['fixed_cases']
            curve = {str(u): aggregate([r for r in done['evaluations']
                                      if matched(r) and r['update'] == u and r['nfe'] == high])
                     for u in ext['checkpoints']}
            precision = [r for r in done['precision'] if matched(r)]
            all_precision = [r for r in done['precision'] if select(r)]
            late_rows = [r for r in done['evaluations'] if matched(r)
                         and r['update'] in ext['checkpoints'][-2:] and r['nfe'] == high]
            sensitivity = []
            for r in done['evaluations']:
                if not matched(r) or r['nfe'] != low:
                    continue
                other = next(x for x in done['evaluations']
                             if (x['fit'], x['case'], x['update'], x['nfe']) ==
                             (r['fit'], r['case'], r['update'], high))
                sensitivity.append(dict(fit=r['fit'], case=r['case'], update=r['update'],
                                        differences={k: other[k]-r[k] for k in METRICS}))
            groups[f'{objective}_{regime}'] = dict(
                primary_nfe=high, curve=curve, precision_matched=aggregate(precision),
                precision_all_cases=aggregate(all_precision), nfe_sensitivity=sensitivity,
                reproducible_matched_four_gate=all(r['passed'] for r in late_rows+precision),
                final_all_cases_four_gate=all(r['passed'] for r in all_precision),
                final_matched_power_gate=all(all(.9 <= p <= 1.1 for p in r['power_ratio'])
                                             for r in precision))
    return dict(groups=groups, checkpoints=ext['checkpoints'],
                curve_ensembles=len(identities), precision_ensembles=len(precise_ids),
                note='Matched cases0/1, both seeds; no best-checkpoint selection. '
                     'Coverage is exact posterior mass in sample octant intervals, not population TARP. '
                     'Covariance is a16-probe diagnostic, not full-field certification.')


def report(root, output):
    manifest = json.loads((root/'manifest.json').read_text())
    done = json.loads((root/'COMPLETE.json').read_text())
    summary = summarize(done, manifest)
    summary['analysis_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    summary['run_root'] = str(root.resolve())
    parent = Path(manifest['parent'])/'results'
    if hashlib.sha256((parent/'COMPLETE.json').read_bytes()).hexdigest() != manifest['parent_complete_sha256']:
        raise ValueError('parent completion receipt changed')
    controls = json.loads((parent/'controls.json').read_text())
    summary['curve_covariance_thresholds'] = [max(manifest['base']['covariance_tolerance'], n['covariance_p99'])
                                             for n in controls['nulls']]
    replay = []
    for r in done['evaluations']:
        if r['objective'] != 'cfm' or r['update'] != 4096 or r['nfe'] != 256:
            continue
        old = json.loads((parent/r['fit']/f'evaluation_4096_{r["case"]}_256.json').read_text())
        delta = max(abs(r[k]-old[k]) for k in METRICS)
        if delta > 1e-6:
            raise ValueError('original CFM baseline evaluation not reproduced')
        replay.append(dict(fit=r['fit'], case=r['case'], maximum_metric_difference=delta))
    summary['cfm_baseline_replay'] = replay
    output.mkdir(parents=True, exist_ok=True)
    for name in ('manifest.json', 'COMPLETE.json'):
        shutil.copy2(root/name, output/name)
    for path in root.glob('worker_*.json'):
        shutil.copy2(path, output/path.name)
    for item in manifest['items']:
        shutil.copy2(root/'results'/item['name']/'precision/nulls.json',
                     output/f'precision_nulls_{item["name"]}.json')
    shutil.copy2(parent/'controls.json', output/'original_controls.json')
    for name in ('replay_default.log', 'replay_deterministic.log'):
        shutil.copy2(root/name, output/(name+'.txt'))
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    lines = ['# Unchanged-training Gaussian reference continuation', '',
             '|Model|Updates|Mean RMS|Covariance error|Variance ratio|Octant coverage|Passed cells|',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name, group in summary['groups'].items():
        for update, row in group['curve'].items():
            values = '|'.join(f'{row["metrics"][k]["mean"]:.4f}' for k in METRICS)
            lines.append(f'|{name}|{update}|{values}|{row["passed"]}/{row["cells"]}|')
        row = group['precision_matched']
        values = '|'.join(f'{row["metrics"][k]["mean"]:.4f}' for k in METRICS)
        lines.append(f'|{name} (2048 draws)|65536|{values}|{row["passed"]}/{row["cells"]}|')
    lines += ['', summary['note'], '',
              'Full draws/checkpoints remain in Scratch; these receipts contain every registered cell.']
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    colors = {'vdm_fixed':'tab:orange', 'vdm_amortised':'tab:red',
              'cfm_fixed':'tab:blue', 'cfm_amortised':'tab:green'}
    for name, group in summary['groups'].items():
        for ax, metric in zip(axes, METRICS[:2]):
            points = [group['curve'][str(u)]['metrics'][metric] for u in summary['checkpoints']]
            ax.plot(summary['checkpoints'], [p['mean'] for p in points], 'o-', label=name,
                    color=colors[name])
            ax.fill_between(summary['checkpoints'], [p['minimum'] for p in points],
                            [p['maximum'] for p in points], alpha=.12, color=colors[name])
            ax.set(xscale='log', xlabel='Training updates', ylabel=metric.replace('_', ' '))
            ax.grid(alpha=.2)
    axes[0].axhline(.1, color='black', ls=':', label='mean gate')
    axes[0].legend(fontsize=8)
    thresholds = summary['curve_covariance_thresholds']
    axes[1].axhspan(min(thresholds), max(thresholds), color='grey', alpha=.25,
                   label='512-draw covariance gate range')
    axes[1].legend(fontsize=7)
    fig.suptitle('Matched observations0/1; both seeds; bands show cell range, not confidence intervals')
    fig.savefig(output/'learning_curves.png', dpi=180)
    plt.close(fig)
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(output.iterdir()) if p.is_file() and p.name != 'SHA256.json'}
    (output/'SHA256.json').write_text(json.dumps(hashes, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    report(args.root, args.output)
