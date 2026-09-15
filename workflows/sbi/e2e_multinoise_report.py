"""Verify and summarize the registered joint-noise architecture comparison."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from workflows.sbi import e2e_wide_pipeline as p
from workflows.sbi.e2e_fine_learning_test import aggregate, gates


def grouped(probes):
    result = {}
    for group in ('fit', 'transfer'):
        for ratio in sorted({r['ratio'] for r in probes}):
            rows = [r for r in probes if r['group'] == group and r['ratio'] == ratio]
            result[f'{group}/{ratio}'] = {k: np.median([r['metrics'][k] for r in rows], axis=0).tolist()
                                        for k in ('gain', 'error_power', 'error_over_truth_power', 'noise_amplitude', 'power_ratio')}
            result[f'{group}/{ratio}']['velocity_mse'] = float(np.median([r['velocity_mse'] for r in rows]))
    return result


def validate_probes(rows, panel, ratios):
    expected = {(anchor, ratio, rep, p.seed_for(panel['seed'], anchor, f'evaluation-{rep}', 'fine'), group)
                for group in ('fit', 'transfer') for anchor in panel[group+'_anchors']
                for ratio in ratios for rep in range(panel['evaluation_noise_replicates'])}
    actual = [(r['anchor_id'], r['ratio'], r['rep'], r['seed'], r['group']) for r in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('missing/duplicate/mismatched probe panel')


def verify(data, root):
    cfg = data['registration']['config']; panel = data['registration']['panel']
    for path, expected in data['registration']['source_sha256'].items():
        if p.sha256(p.REPO/path) != expected:
            raise ValueError('source mismatch: '+path)
    for path, expected in data['checkpoints'].items():
        if p.sha256(root/path) != expected:
            raise ValueError('checkpoint mismatch: '+path)
    for r in data['fields']:
        path = root/f'diffusion_{r["label"]}_{r["anchor_id"]}_true_coarse.h5'
        if p.sha256(path) != r['sample_sha256']:
            raise ValueError('draw mismatch: '+str(path))
    if sorted(b['arm'] for b in data['results']) != sorted(cfg['arms']):
        raise ValueError('missing/duplicate architecture')
    validate_probes(data['baseline'], panel, panel['ratios'])
    common = None
    baseline_keys = [(r['anchor_id'], r['ratio'], r['rep'], r['seed']) for r in data['baseline']]
    for branch in data['results']:
        if [c['update'] for c in branch['curve']] != cfg['evaluate_at']:
            raise ValueError('checkpoint evaluation sequence mismatch')
        validate_probes(branch['intermediate'], panel, cfg['intermediate_ratios'])
        if [r['update'] for r in branch['training']] != list(range(1, cfg['updates']+1)):
            raise ValueError('training update sequence mismatch')
        exposure = [(r['anchor_id'], r['noise_seed'], r['ratio'], r['time']) for r in branch['training']]
        if common is not None and exposure != common:
            raise ValueError('unpaired exposure')
        common = exposure
        if set(r['anchor_id'] for r in branch['training']) != set(panel['fit_anchors']):
            raise ValueError('transfer training contamination')
        for point in branch['curve']:
            validate_probes(point['probes'], panel, panel['ratios'])
            if [(r['anchor_id'], r['ratio'], r['rep'], r['seed']) for r in point['probes']] != baseline_keys:
                raise ValueError('unpaired evaluation')
        evalseeds = {r['seed'] for point in branch['curve'] for r in point['probes']}
        if evalseeds & {r['noise_seed'] for r in branch['training']}:
            raise ValueError('evaluation noise contamination')
        if gates(data['baseline'], branch['curve'][-1]['probes'], panel) != branch['gate']:
            raise ValueError('gate mismatch')
    return dict(source_hashes=True, checkpoint_hashes=True, draw_hashes=True, paired_exposure=True,
                paired_evaluation=True, evaluation_noise_disjoint=True, transfer_not_fitted=True, recomputed_gates=True)


def summarize(data):
    counts = dict(updates=sum(len(b['training']) for b in data['results']),
                  probes=len(data['baseline'])+sum(len(c['probes']) for b in data['results'] for c in b['curve'])+
                         sum(len(b['intermediate']) for b in data['results']),
                  draws=len(data['fields']), checkpoints=len(data['checkpoints']))
    if not data['complete'] or counts != dict(updates=18432, probes=1788, draws=42, checkpoints=18):
        raise ValueError('incomplete registered comparison')
    result = {k: data[k] for k in ('registration', 'checkpoints', 'fields', 'elapsed_seconds', 'training_ready', 'calibration_pass', 'smoke')}
    result.update(counts=counts, baseline=aggregate(data['baseline']), branches={}, draw_summary={})
    for label in ('parent384', *data['registration']['config']['arms']):
        for group in ('fit', 'transfer'):
            rows = [r for r in data['fields'] if r['label'] == label and r['group'] == group]
            result['draw_summary'][f'{label}/{group}'] = {k: np.median([r['metrics'][k] for r in rows], axis=0).tolist()
                                                       for k in ('gain', 'power_ratio', 'error_over_truth_power')}
            result['draw_summary'][f'{label}/{group}']['below_minus_one'] = float(np.median([r['density_below_minus_one_fraction'] for r in rows]))
    for b in data['results']:
        grad = np.array([r['gradient_norm_before_clip'] for r in b['training']])
        curves = {str(c['update']): dict(summary=c['summary'], gate=c['gate']) for c in b['curve']}
        loss_by_regime = {}
        for start in (0, 1536, 2560):
            rows = b['training'][start:start+512]
            for name, low, high in (('near', 0, .2), ('middle', .2, 2), ('high', 2, 81)):
                values = [r['primary_loss'] for r in rows if r['ratio'] is not None and low <= r['ratio'] < high]
                loss_by_regime[f'{start+1}-{start+512}/{name}'] = float(np.mean(values))
        alignment = {}
        for group in ('fit', 'transfer'):
            for ratio in (.05, .2):
                rows = [r for r in b['curve'][-1]['probes'] if r['group'] == group and r['ratio'] == ratio]
                alignment[f'{group}/{ratio}'] = np.median([np.array(r['metrics']['noise_correlated_error_power'])/
                                                         np.maximum(r['metrics']['error_power'], 1e-30) for r in rows], axis=0).tolist()
        result['branches'][b['arm']] = dict(parameters=b['parameters'], elapsed_seconds=b['elapsed_seconds'],
             gate=b['gate'], curve=curves, intermediate=grouped(b['intermediate']), noise_aligned_error_fraction=alignment,
             clipped_fraction=float(np.mean(grad > data['registration']['config']['clip'])),
             gradient_norm_quantiles=np.quantile(grad, [0, .5, .9, 1]).tolist(), loss_by_regime=loss_by_regime)
    return result


def plot(summary, out):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for j, ratio in enumerate((.05, .2)):
        for arm, branch in summary['branches'].items():
            steps = sorted(map(int, branch['curve']))
            noise = []; error = []
            for step in steps:
                checks = [r for r in branch['curve'][str(step)]['gate']['transfer']['checks'] if r['ratio'] == ratio]
                noise.append(max(r['noise_amplitude'] for r in checks))
                error.append(max(r['error_vs_parent'] for r in checks))
            axes[0, j].plot(steps, noise, marker='o', label=arm)
            axes[1, j].plot(steps, error, marker='o', label=arm)
        axes[0, j].axhline(.2, color='k', ls=':'); axes[1, j].axhline(.25, color='k', ls=':')
        axes[0, j].set(title=f'Transfer ratio {ratio}', ylabel='Worst-phase high-k noise amplitude', yscale='log')
        axes[1, j].set(xlabel='Updates', ylabel='Worst-phase high-k error / parent', yscale='log')
    axes[0, 0].legend(fontsize=7)
    fig.suptitle('One model per arm across all noise levels; lower-band gains are an additional gate')
    fig.savefig(out/'learning_curves.png', dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('input', type=Path); ap.add_argument('output', type=Path)
    args = ap.parse_args(); data = json.loads(args.input.read_text()); summary = summarize(data)
    summary.update(verification=verify(data, args.input.parent), raw_path=str(args.input.resolve()),
                   raw_sha256=p.sha256(args.input), report_sha256=p.sha256(__file__))
    p.write_json(args.output/'SUMMARY.json', summary); plot(summary, args.output)
    print(json.dumps(dict(counts=summary['counts'], gates={k: v['gate'] for k, v in summary['branches'].items()}), indent=2))


if __name__ == '__main__':
    main()
