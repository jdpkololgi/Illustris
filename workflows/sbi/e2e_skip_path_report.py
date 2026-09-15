"""Paired skip-only comparison, with original six-arm experiment unchanged."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from workflows.sbi import e2e_wide_pipeline as p
from workflows.sbi.e2e_multinoise_report import verify, grouped


def summarize(data, old):
    cfg = data['registration']['config']
    counts = dict(updates=sum(len(b['training']) for b in data['results']),
                  new_probes=sum(len(c['probes']) for b in data['results'] for c in b['curve'])+
                             sum(len(b['intermediate']) for b in data['results']),
                  reused_parent_probes=len(data['baseline']), draws=len(data['fields']),
                  checkpoints=len(data['checkpoints']), response_records=sum(len(b['response']) for b in data['results']))
    if not data['complete'] or counts != dict(updates=9216, new_probes=864, reused_parent_probes=60, draws=18, checkpoints=9, response_records=36):
        raise ValueError('incomplete approved skip comparison')
    if data['baseline'] != old['baseline']:
        raise ValueError('changed parent baseline')
    summary = {k: data[k] for k in ('registration', 'checkpoints', 'fields', 'elapsed_seconds', 'training_ready', 'calibration_pass')}
    summary.update(counts=counts, branches={}, comparison={}, draw_summary={})
    for b in data['results']:
        arm = b['arm']; previous = next(v for v in old['results'] if v['arm'] == arm)
        keys = ('update', 'anchor_id', 'noise_seed', 'ratio', 'time')
        if [tuple(r[k] for k in keys) for r in b['training']] != [tuple(r[k] for k in keys) for r in previous['training']]:
            raise ValueError('skip contrast exposure drift')
        if b['parameters'] != previous['parameters']:
            raise ValueError('skip contrast parameter count drift')
        response = {}
        for group in ('fit', 'transfer'):
            for ratio in (.05, .2):
                rows = [r for r in b['response'] if r['group'] == group and r['ratio'] == ratio]
                response[f'{group}/{ratio}'] = {k: np.median([r['metrics'][k] for r in rows], axis=0).tolist() for k in rows[0]['metrics']}
        grad = np.array([r['gradient_norm_before_clip'] for r in b['training']])
        summary['branches'][arm] = dict(parameters=b['parameters'], elapsed_seconds=b['elapsed_seconds'], gate=b['gate'],
             curve={str(c['update']): dict(summary=c['summary'], gate=c['gate']) for c in b['curve']},
             intermediate=grouped(b['intermediate']), response=response, clipped_fraction=float(np.mean(grad > cfg['clip'])),
             loss_means_by_512=[float(np.mean([r['primary_loss'] for r in b['training'][i:i+512]])) for i in range(0, 3072, 512)])
        summary['comparison'][arm] = dict(original_gate=previous['gate'], skip_gate=b['gate'],
             original_curve={str(c['update']): dict(summary=c['summary'], gate=c['gate']) for c in previous['curve']})
        for group in ('fit', 'transfer'):
            for chart, source in (('original', old), ('skip', data)):
                rows = [r for r in source['fields'] if r['label'] == arm and r['group'] == group]
                summary['draw_summary'][f'{chart}/{arm}/{group}'] = {
                    k: np.median([r['metrics'][k] for r in rows], axis=0).tolist() for k in ('gain', 'power_ratio', 'error_over_truth_power')}
    return summary


def plot(summary, out):
    arms = summary['registration']['config']['arms']
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for j, arm in enumerate(arms):
        for label, curve, color in [('original bounded chart', summary['comparison'][arm]['original_curve'], '0.5'),
                                     ('near-identity skip', summary['branches'][arm]['curve'], '#167d96')]:
            steps = sorted(map(int, curve))
            for i, metric in enumerate(('noise_amplitude', 'error_vs_parent')):
                values = [max(r[metric] for r in curve[str(s)]['gate']['transfer']['checks'] if r['ratio'] == .05) for s in steps]
                axes[i, j].plot(steps, values, marker='o', label=label, color=color)
                axes[i, j].axhline(.2 if i == 0 else .25, color='k', ls=':')
                axes[i, j].set(yscale='log', xlabel='Updates', ylabel=metric, title=arm if i == 0 else None)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Paired skip-only contrast: worst-phase transfer at noise ratio .05\nSame network initialization, raw scaling, exposure, optimizer and gates')
    fig.savefig(out/'skip_comparison.png', dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('input', type=Path); ap.add_argument('output', type=Path)
    args = ap.parse_args(); data = json.loads(args.input.read_text()); old_path = Path(data['registration']['original_path'])
    if p.sha256(old_path) != data['registration']['original_sha256']:
        raise ValueError('original result hash mismatch')
    old = json.loads(old_path.read_text()); summary = summarize(data, old)
    summary.update(verification=verify(data, args.input.parent), original_verification=verify(old, old_path.parent),
                   paired_skip_exposure=True, raw_path=str(args.input.resolve()), raw_sha256=p.sha256(args.input),
                   report_sha256=p.sha256(__file__))
    p.write_json(args.output/'SKIP_SUMMARY.json', summary); plot(summary, args.output)
    print(json.dumps(dict(counts=summary['counts'], gates={k: v['gate'] for k, v in summary['branches'].items()}), indent=2))


if __name__ == '__main__':
    main()
