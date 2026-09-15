"""Archive matched EDM-informed ablation evidence, without checkpoint promotion."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from workflows.sbi import e2e_wide_pipeline as p


def verify_artifacts(data, root):
    for relative, expected in data['registration']['source_sha256'].items():
        if p.sha256(p.REPO/relative) != expected:
            raise ValueError(f'source drift: {relative}')
    for key, expected in data['checkpoints'].items():
        arm, update = key.split('/')
        if p.sha256(root/arm/f'step_{384+int(update):06d}.pt') != expected:
            raise ValueError(f'checkpoint mismatch: {key}')
    for row in data['fields']:
        path = root/f'{row["method"]}_{row["label"]}_{row["anchor_id"]}_true_coarse.h5'
        if p.sha256(path) != row['sample_sha256']:
            raise ValueError(f'field mismatch: {path}')
    paired = None
    for branch in data['results']:
        seeds = [(r['anchor_id'], r['noise_seed']) for r in branch['training']]
        if paired is None:
            paired = seeds
        elif seeds != paired:
            raise ValueError('unpaired training noise')
        evaluation = {r['seed'] for point in branch['curve'] for r in point['probes']}
        if evaluation & {s for _, s in seeds}:
            raise ValueError('training/evaluation noise overlap')
    return dict(source_hashes=True, checkpoint_hashes=True, field_hashes=True,
                paired_training_noise=True, separate_evaluation_noise=True)


def summarize(data):
    counts = dict(updates=sum(len(r['training']) for r in data['results']),
                  probes=sum(len(c['probes']) for r in data['results'] for c in r['curve']),
                  draws=len(data['fields']), checkpoints=len(data['checkpoints']))
    if not data['complete'] or len(data['results']) != 6 or counts != dict(updates=3072, probes=1440, draws=42, checkpoints=18):
        raise ValueError('incomplete registered experiment')
    result = {k: data[k] for k in ('registration', 'checkpoints', 'elapsed_seconds', 'training_ready', 'calibration_pass')}
    result.update(counts=counts, branches={}, fields={})
    result['field_sha256'] = {f'{r["label"]}/{r["anchor_id"]}': r['sample_sha256'] for r in data['fields']}
    for branch in data['results']:
        arm = branch['arm']; phase_curves = {}
        for point in branch['curve']:
            for group in ('fit', 'transfer'):
                for phase in ('ph000', 'ph002', 'ph003'):
                    for ratio in (.05, .2):
                        rows = [r for r in point['probes'] if r['group'] == group and r['phase'] == phase and r['ratio'] == ratio]
                        if len(rows) != 2:
                            raise ValueError('missing paired phase probes')
                        phase_curves[f'{point["update"]}/{group}/{phase}/{ratio}'] = {
                            k: np.median([r['metrics'][k] for r in rows], axis=0).tolist()
                            for k in ('noise_amplitude', 'error_power', 'gain')}
        gradients = np.array([r['gradient_norm_before_clip'] for r in branch['training']])
        sigma = np.array([r['sigma'] for r in branch['training']])
        result['branches'][arm['name']] = dict(
            arm=arm, parameter_count=branch['parameter_count'], initial_max_abs=branch['initial_max_abs'],
            curve={str(c['update']): c['summary'] for c in branch['curve']},
            phase_curves=phase_curves, gate=branch['gate'],
            clipping_fraction=float(np.mean(gradients > arm['clip'])),
            gradient_norm_quantiles=np.quantile(gradients, [0, .5, .9, 1]).tolist(),
            gradient_group_quantiles={g: np.quantile([r['gradient_groups'][g] for r in branch['training']],
                                                    [0, .5, .9, 1]).tolist()
                                      for g in ('base', 'film', 'receptive')},
            sigma_quantiles=np.quantile(sigma, [0, .1, .5, .9, 1]).tolist(),
            sigma_histogram=dict(edges=[0, .05, .2, 1, 5, 20, 'inf'],
                                 counts=np.histogram(sigma, [0, .05, .2, 1, 5, 20, np.inf])[0].tolist()),
            loss_medians_by_128_updates=[float(np.median([r['loss'] for r in branch['training'][i:i+128]]))
                                       for i in range(0, 512, 128)])
    for label in ['parent'] + list(result['branches']):
        for group in ('fit', 'transfer'):
            rows = [r for r in data['fields'] if r['label'] == label and r['group'] == group]
            if len(rows) != 3:
                raise ValueError('missing diagnostic fields')
            result['fields'][f'{label}/{group}'] = dict(
                power_ratio=np.median([r['metrics']['power_ratio'] for r in rows], axis=0).tolist(),
                density_below_minus_one_fraction=float(np.median([r['density_below_minus_one_fraction'] for r in rows])))
    return result


def plot(data, out):
    colors = ['#444444', '#888888', '#b47717', '#3179ae', '#8954a3', '#26906a']
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for i, group in enumerate(('fit', 'transfer')):
        for j, ratio in enumerate((.05, .2)):
            ax = axes[i, j]
            for (name, branch), color in zip(data['branches'].items(), colors):
                steps = sorted(map(int, branch['curve']))
                noise = [branch['curve'][str(s)][f'{group}/{ratio}']['noise_amplitude'][-1] for s in steps]
                ax.plot(steps, noise, marker='o', color=color, label=name)
            ax.axhline(.2, color='0.2', linestyle=':', label='registered amplitude threshold')
            ax.set(title=f'{group}: noise/signal = {ratio}', xlabel='Additional diagnostic updates',
                   ylabel='High-k residual noise / injected noise', ylim=(-.1, 1.1))
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle('Matched diffusion denoiser ablations: three phases, training data only\n'
                 'Parent weights and initial predictions identical; gate also requires error reduction and signal preservation.')
    fig.savefig(out/'learning_curves.png', dpi=160); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path); parser.add_argument('output', type=Path)
    args = parser.parse_args()
    data = json.loads(args.input.read_text()); summary = summarize(data)
    summary['verification'] = verify_artifacts(data, args.input.parent)
    summary.update(raw_path=str(args.input.resolve()), raw_sha256=p.sha256(args.input), report_sha256=p.sha256(__file__))
    p.write_json(args.output/'SUMMARY.json', summary)
    plot(summary, args.output)
    print(json.dumps(dict(counts=summary['counts'], fields=summary['fields'],
                          gates={k: v['gate'] for k, v in summary['branches'].items()}), indent=2))


if __name__ == '__main__':
    main()
