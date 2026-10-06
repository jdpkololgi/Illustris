#!/usr/bin/env python3
"""Small-array resolution comparison and evidence packaging after compute."""
import argparse
import json
from pathlib import Path
import numpy as np
from abacus_eigenvalue_statistics import marginal, pyplot, PAIRS, COLORS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run', type=Path)
    ap.add_argument('output', type=Path)
    args = ap.parse_args()
    complete = json.loads((args.run/'COMPLETE.json').read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    datasets = {}
    summary = {'run': str(args.run), 'job_id': complete['job_id'], 'measurements': {}}
    for n in (256, 512):
        for label in ('abacus', 'random_phase'):
            with np.load(args.run/f'{label}_n{n}.npz') as data:
                d = {k: data[k] for k in data.files}
            datasets[label, n] = d
            x = (d['xedges'][1:]+d['xedges'][:-1])/2
            diff = np.abs(d['pdf']-marginal(x))*np.diff(d['xedges'])
            summary['measurements'][f'{label}_n{n}'] = {
                'sigma_delta': float(d['sigma']),
                'mean_eigenvalues_over_sigma': (d['means']/d['sigma']).tolist(),
                'pdf_integrated_absolute_difference_over_minus4_to6': diff.sum(axis=1).tolist(),
                'fraction_outside_pdf_range': d['tail_fraction'].tolist(),
                'trace_max_absolute_error': float(d['trace_error']),
                'spectral_trace_max_relative_error': float(d['spectral_closure']),
            }
        a, b = datasets['abacus', n], datasets['random_phase', n]
        mask = np.isfinite(a['density_power']) & (a['density_power'] > np.nanmax(a['density_power'])*1e-8)
        mismatch = float(np.max(np.abs(a['density_power'][mask]/b['density_power'][mask]-1)))
        if mismatch > 1e-4:
            raise AssertionError(f'Random phases changed density power: {mismatch}')
        summary[f'random_phase_density_power_relative_error_n{n}'] = mismatch
    lo, hi = datasets['abacus', 256], datasets['abacus', 512]
    # Rebinning was not identical: interpolate positive spectra in log k/log P.
    # Compare only populated bins with >=100 full-mode equivalents.
    mask = (lo['modes'] >= 100) & np.isfinite(lo['k']) & (lo['k'] < .2)
    k = lo['k'][mask]
    ratios = []
    for i in range(3):
        good = np.isfinite(hi['k']) & (hi['power'][i] > 0)
        interpolated = np.exp(np.interp(np.log(k), np.log(hi['k'][good]), np.log(hi['power'][i, good])))
        ratios.append(lo['power'][i, mask]/interpolated)
    summary['resolution_comparison'] = {
        'description': '256 / log-interpolated 512 autos; finite bins >=100 modes; extra cell windows differ',
        'k': k.tolist(), 'ratios': np.array(ratios).tolist(),
    }
    plt = pyplot()
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    for i in range(3):
        axs[0].plot(k, ratios[i], color=COLORS[i], label=rf'$P_{{{i+1}{i+1}}}$')
        x = (hi['xedges'][1:]+hi['xedges'][:-1])/2
        axs[1].plot(x, lo['pdf'][i]-hi['pdf'][i], color=COLORS[i], label=rf'$\lambda_{i+1}$')
    axs[0].axhline(1, color='#333333', ls='--')
    axs[0].set(xlabel=r'$k\ [h\,\mathrm{Mpc}^{-1}]$', ylabel=r'$P_{256}/P_{512}$', title='Resolution sensitivity of autos')
    axs[1].axhline(0, color='#333333', ls='--')
    axs[1].set(xlim=(-2, 3), xlabel=r'$\lambda/\sigma_\delta$', ylabel='Normalized PDF difference (256 − 512)', title='Resolution sensitivity of PDFs')
    for ax in axs:
        ax.legend()
    for ext in ['png', 'pdf']:
        fig.savefig(args.output/f'resolution_comparison.{ext}', dpi=180)
    plt.close(fig)
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.4), layout='constrained')
    for a, label in enumerate(('abacus', 'random_phase')):
        d = datasets[label, 512]
        use = (d['modes'] >= 100) & (d['k'] <= .2) & np.isfinite(d['k'])
        style = '-' if a == 0 else ':'
        for p, (i, j) in enumerate(PAIRS):
            axs[0 if i == j else 1].plot(d['k'][use], d['power'][p, use],
                color=COLORS[p % 3], ls=style, label=rf'$P_{{{i+1}{j+1}}}$' if a == 0 else None)
        for p, pair in enumerate(('12', '13', '23')):
            axs[2].plot(d['k'][use], d['r'][p, use], color=COLORS[p], ls=style,
                label=rf'$r_{{{pair}}}$' if a == 0 else None)
        if a == 0:
            axs[0].plot(d['k'][use], d['density_power'][use]/9, '--', color='#222222', label=r'$P_{\delta_R}/9$')
    axs[0].set_yscale('log')
    axs[1].set_yscale('symlog', linthresh=100)
    axs[1].axhline(0, color='#999999', lw=.6)
    axs[2].set(ylim=(-.5, 1.02), ylabel=r'$r_{ij}$')
    for ax in axs[:2]:
        ax.set_ylabel(r'$P(k)\ [(\mathrm{Mpc}/h)^3]$')
    for ax in axs:
        ax.set_xscale('log')
        ax.set_xticks([.02, .05, .1, .2], labels=['0.02', '0.05', '0.10', '0.20'])
        ax.set_xlabel(r'$k\ [h\,\mathrm{Mpc}^{-1}]$')
        ax.legend(fontsize=9)
    fig.suptitle('Abacus z=0.2, R=7 Mpc/h, 512³ — solid: measured; dotted: random-phase control')
    for ext in ('png', 'pdf'):
        fig.savefig(args.output/f'spectra_focus_n512.{ext}', dpi=180)
    plt.close(fig)
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k != 'resolution_comparison'}, indent=2))

if __name__ == '__main__':
    main()
