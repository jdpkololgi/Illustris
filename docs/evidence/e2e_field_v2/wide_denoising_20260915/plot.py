"""Reproduce localization figure and supplementary scalar summaries from receipts."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent
s = json.loads((root/'SUMMARY.json').read_text())
d = json.loads(Path(s['audit_path']).read_text())
extra = {}
for method in ('cfm', 'diffusion'):
    for stage in ('coarse', 'fine'):
        for ratio in (.05, .2, 1., 5., 20.):
            rows = [r for r in d['probes'] if r['step'] == 384 and r['method'] == method
                    and r['stage'] == stage and r['ratio'] == ratio]
            extra[f'{method}/{stage}/{ratio}'] = {
                'median_velocity_mse': float(np.median([r['velocity_mse'] for r in rows])),
                'median_clean_mse': float(np.median([r['clean_mse'] for r in rows])),
                'median_high_k_noise_correlated_error_fraction': float(np.median([
                    r['metrics']['noise_correlated_error_power'][-1]/r['metrics']['error_power'][-1] for r in rows])),
                'median_high_k_error_over_total_truth_power': float(np.median([
                    r['metrics']['error_power'][-1]/sum(r['metrics']['truth_power']) for r in rows]))}
with (root/'SUPPLEMENTARY.json').open('x') as f:
    json.dump(extra, f, indent=2, sort_keys=True, allow_nan=False)
    f.write('\n')

fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
for col, method in enumerate(('cfm', 'diffusion')):
    ax = axes[0, col]
    control = s['true_coarse'][f'all/{method}']
    for name, key, color in [('Ordinary draw', 'actual_draw0_power_ratio', '#b54e43'),
                             ('True-coarse control', 'oracle_power_ratio', '#3575a5')]:
        ax.plot(range(4), control[key], 'o-', label=name, color=color)
    ax.axhline(1, color='0.4', linewidth=1, linestyle=':')
    ax.set(yscale='log', ylim=(.2, 100), xticks=range(4),
           xticklabels=['0–.08', '.08–.16', '.16–.32', '>.32'],
           xlabel='Local spatial frequency k [h/Mpc]', ylabel='Generated / true power',
           title=f'{method.upper()}: 384 updates')
    ax.legend(fontsize=9)
    ax = axes[1, col]
    ratios = [.05, .2, 1., 5., 20.]
    for step, color in [(192, '0.55'), (384, '#713c89')]:
        values = [s['probe_summary'][f'{step}/{method}/fine/{r}']['noise_leakage_over_input_sigma'][-1]
                  for r in ratios]
        ax.plot(ratios, values, 'o-', color=color, label=f'{step} updates')
    ax.axhline(1, color='0.4', linewidth=1, linestyle=':')
    ax.axhline(0, color='0.4', linewidth=1, linestyle=':')
    ax.set(xscale='log', xlabel='Injected noise / signal amplitude',
           ylabel='Fine high-k residual noise / input noise', ylim=(-.3, 1.1))
    ax.legend(fontsize=9)
fig.suptitle('Training-panel diagnosis: 24 anchors, three simulation phases\n'
             'True coarse repairs the largest scale; fine-scale noise remains', fontsize=12)
fig.savefig(root/'denoising_localization.png', dpi=160)
plt.close(fig)
print(json.dumps(extra, indent=2))
