#!/usr/bin/env python3
"""Volume-weighted periodic Abacus tidal PDFs and signed auto/cross spectra.

Default input is the documented ph000 z0.2 count-density cache. No holdout
phases are read. Block means retain an additional cell window; no assignment
window or particle shot-noise correction is applied. Gaussian control uses
random phases with exactly the same discrete density Fourier amplitudes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy import fft
from scipy.special import erfc

SOURCE = Path('/pscratch/sd/d/dkololgi/abacus/p8_deterministic_v1/true_field_context_v1/density_blockmean_ngrid1024.npy')
PAIRS = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
COLORS = ['#2474a6', '#cf8626', '#9256a3']


def marginal(x):
    def largest(t):
        return np.sqrt(5)/(12*np.pi)*(20*t*np.exp(-4.5*t*t)
            +np.sqrt(2*np.pi)*(20*t*t-1)*np.exp(-2.5*t*t)*erfc(-np.sqrt(2)*t)
            +3*np.sqrt(3*np.pi)*np.exp(-3.75*t*t)*erfc(-np.sqrt(3)*t/2))
    return np.array([largest(-x), np.sqrt(15)/(2*np.sqrt(np.pi))*np.exp(-3.75*x*x), largest(x)])


def pyplot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    return plt


def analytic_plot(out):
    plt = pyplot()
    x = np.linspace(-2.5, 2.5, 1601)
    fig, ax = plt.subplots(figsize=(8, 4.8), layout='constrained')
    for i, y in enumerate(marginal(x)):
        ax.plot(x, y, color=COLORS[i], label=rf'$\lambda_{i+1}$')
    ax.set(xlabel=r'$\lambda/\sigma_\delta$', ylabel=r'$\sigma_\delta\,p(\lambda)$',
           title='Gaussian tidal-field prediction — not an Abacus measurement')
    ax.legend()
    for ext in ['png', 'pdf']:
        fig.savefig(out / f'analytic_pdfs.{ext}', dpi=180)
    plt.close(fig)


def density(source, n):
    a = np.load(source, mmap_mode='r')
    if a.ndim != 3 or len(set(a.shape)) != 1 or a.shape[0] % n:
        raise ValueError('Input must be cubic with dimension divisible by requested mesh')
    f = a.shape[0] // n
    b = np.empty((n, n, n), dtype=np.float32)
    for i in range(n):
        b[i] = a[i*f:(i+1)*f].reshape(f, n, f, n, f).mean(axis=(0, 2, 4), dtype=np.float64)
    if not np.isfinite(b).all() or b.min() < 0 or b.mean(dtype=np.float64) <= 0:
        raise ValueError('Expected finite nonnegative count-density input')
    mean = b.mean(dtype=np.float64)
    b /= mean
    b -= 1
    return b, float(mean)


def solve(delta, box, radius, workers, randomize=False):
    n = delta.shape[0]
    axes = [2*np.pi*fft.fftfreq(n, d=box/n), 2*np.pi*fft.fftfreq(n, d=box/n),
            2*np.pi*fft.rfftfreq(n, d=box/n)]
    # Nyquist planes removed to keep mixed derivatives Hermitian and trace exact.
    q = [axes[0][:, None, None], axes[1][None, :, None], axes[2][None, None, :]]
    k2 = (q[0]**2+q[1]**2+q[2]**2).astype(np.float32)
    d = fft.rfftn(delta, workers=workers)
    d[n//2, :, :] = 0
    d[:, n//2, :] = 0
    d[:, :, -1] = 0
    d[0, 0, 0] = 0
    if randomize:
        rng = np.random.default_rng(61006)
        w = fft.rfftn(rng.standard_normal(delta.shape, dtype=np.float32), workers=workers)
        d *= w / np.maximum(np.abs(w), np.finfo(np.float32).tiny) * np.conj(d) / np.maximum(np.abs(d), np.finfo(np.float32).tiny)
        del w
    d *= np.exp(-0.5*radius**2*k2)
    smooth = fft.irfftn(d, s=delta.shape, workers=workers)
    inv = np.zeros_like(k2)
    np.divide(1., k2, out=inv, where=k2 > 0)
    comps = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
    tensors = np.empty((6, n, n, n), dtype=np.float32)
    for c, (i, j) in enumerate(comps):
        tensors[c] = fft.irfftn(d*(q[i]*q[j]*inv), s=delta.shape, workers=workers)
    eig = np.empty((3, n, n, n), dtype=np.float32)
    for left in range(n):
        matrix = np.empty((n, n, 3, 3), dtype=np.float32)
        for c, (i, j) in enumerate(comps):
            matrix[..., i, j] = matrix[..., j, i] = tensors[c, left]
        eig[:, left] = np.linalg.eigvalsh(matrix).transpose(2, 0, 1)
    trace_error = float(np.max(np.abs(eig.sum(axis=0)-smooth)))
    if trace_error > 2e-5*max(1., float(np.max(np.abs(smooth)))):
        raise AssertionError(f'Trace mismatch: {trace_error}')
    return eig, smooth, k2, trace_error


def summarize(eig, smooth, k2, box, workers):
    n = smooth.shape[0]
    sigma = float(np.std(smooth, dtype=np.float64))
    edges = np.linspace(-4, 6, 201)
    hist, tails = [], []
    means = [float(e.mean(dtype=np.float64)) for e in eig]
    for e in eig:
        counts = np.histogram(e, bins=edges*sigma)[0]
        hist.append(counts/e.size/np.diff(edges))
        tails.append(float(1-counts.sum()/e.size))
    transforms = [fft.rfftn(e-means[i], workers=workers) for i, e in enumerate(eig)]
    density_fft = fft.rfftn(smooth, workers=workers)
    k = np.sqrt(k2)
    kedges = np.geomspace(2*np.pi/box, np.pi*n/box, 45)
    # Correct multiplicities for a real FFT (full shells, not half-shell means).
    weight = np.ones(k.shape[-1])*2
    weight[0] = weight[-1] = 1
    index = np.searchsorted(kedges, k, side='right')-1
    valid = (index >= 0) & (index < len(kedges)-1)
    idx = index[valid]
    wt = np.broadcast_to(weight, k.shape)[valid]
    counts = np.bincount(idx, weights=wt, minlength=len(kedges)-1)
    def avg(v):
        total = np.bincount(idx, weights=v[valid]*wt, minlength=len(counts))
        return np.divide(total, counts, out=np.full_like(total, np.nan), where=counts > 0)
    norm = box**3/n**6
    powers = np.array([avg((transforms[i]*transforms[j].conj()).real)*norm for i, j in PAIRS])
    pd = avg(np.abs(density_fft)**2)*norm
    total = powers[:3].sum(axis=0)+2*powers[3:].sum(axis=0)
    good = np.isfinite(pd) & (pd > np.nanmax(pd)*1e-8)
    closure = float(np.max(np.abs(total[good]/pd[good]-1)))
    if closure > 1e-4:
        raise AssertionError(f'Spectral trace closure failed: {closure}')
    corr = np.array([powers[p]/np.sqrt(powers[i]*powers[j]) for p, (i, j) in enumerate(PAIRS) if i != j])
    if np.nanmax(np.abs(corr)) > 1.0001:
        raise AssertionError('Cross-correlation outside [-1,1]')
    return dict(xedges=edges, pdf=np.array(hist), tail_fraction=np.array(tails), sigma=sigma,
                means=np.array(means), k=avg(k), modes=counts, power=powers, density_power=pd,
                r=corr, spectral_closure=closure)


def plot_results(results, out, n, radius):
    plt = pyplot()
    sim, control = results
    x = (sim['xedges'][1:]+sim['xedges'][:-1])/2
    fig, axs = plt.subplots(1, 3, figsize=(13, 4), layout='constrained')
    for i, ax in enumerate(axs):
        ax.plot(x, sim['pdf'][i], color=COLORS[i], label='Abacus')
        ax.plot(x, control['pdf'][i], color='#777777', ls=':', label='Random-phase control')
        ax.plot(x, marginal(x)[i], color='#222222', ls='--', label='Doroshkevich')
        ax.set(xlim=(-2.5, 4), ylim=(0, None), xlabel=r'$\lambda/\sigma_\delta$',
               title=rf'$\lambda_{i+1}$')
    axs[0].set_ylabel(r'$\sigma_\delta p(\lambda)$')
    axs[0].legend(fontsize=8)
    fig.suptitle(f'Abacus c000 ph000, z=0.2; R={radius:g} Mpc/h; {n}³ full-box mesh')
    for ext in ['png', 'pdf']:
        fig.savefig(out/f'pdfs_n{n}.{ext}', dpi=180)
    plt.close(fig)
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.5), layout='constrained')
    for a, result in enumerate(results):
        style = '-' if a == 0 else ':'
        for p, (i, j) in enumerate(PAIRS):
            ax = axs[0] if i == j else axs[1]
            ax.plot(result['k'], result['power'][p], ls=style, color=COLORS[p % 3],
                    label=rf'$P_{{{i+1}{j+1}}}$' if a == 0 else None)
        for p, label in enumerate(['12', '13', '23']):
            axs[2].plot(result['k'], result['r'][p], ls=style, color=COLORS[p], label=label if a == 0 else None)
    axs[0].plot(sim['k'], sim['density_power']/9, '--', color='#222222', label=r'$P_{\delta_R}/9$')
    axs[0].set_yscale('log')
    axs[1].set_yscale('symlog', linthresh=1)
    axs[2].set(ylim=(-1.05, 1.05), ylabel=r'$r_{ij}(k)$')
    for ax in axs:
        ax.set_xscale('log')
        ax.set_xlabel(r'$k\ [h\,\mathrm{Mpc}^{-1}]$')
        ax.legend(fontsize=8)
    for ax in axs[:2]:
        ax.set_ylabel(r'$P(k)\ [(\mathrm{Mpc}/h)^3]$')
    fig.suptitle(f'{n}³, R={radius:g} Mpc/h — solid: Abacus; dotted: random-phase control')
    for ext in ['png', 'pdf']:
        fig.savefig(out/f'spectra_n{n}.{ext}', dpi=180)
    plt.close(fig)


def self_test():
    x = np.linspace(-8, 8, 16001)
    p = marginal(x)
    integrator = getattr(np, 'trapezoid', np.trapz)
    assert np.allclose(integrator(p, x, axis=1), 1, atol=1e-7)
    assert np.allclose(integrator(p*x, x, axis=1), [-3/np.sqrt(10*np.pi), 0, 3/np.sqrt(10*np.pi)], atol=1e-7)
    rng = np.random.default_rng(7)
    delta = rng.normal(size=(16,)*3).astype(np.float32)
    for randomized in [False, True]:
        e, d, k2, err = solve(delta, 100, 7, 1, randomized)
        result = summarize(e, d, k2, 100, 1)
        assert err < 1e-6 and result['spectral_closure'] < 1e-4
    print('PASS: PDF normalization/means; synthetic trace/spectral closure and cross bounds', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, default=SOURCE)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--ngrid', type=int, nargs='+', default=[256, 512])
    ap.add_argument('--box', type=float, default=2000)
    ap.add_argument('--radius', type=float, default=7)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--analytic-only', action='store_true')
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.self_test:
        self_test()
    analytic_plot(args.output)
    if args.analytic_only:
        return
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Simulation analysis requires an approved Slurm compute allocation')
    provenance = json.loads(args.source.with_name('downsample_manifest.json').read_text())
    manifest = dict(source=str(args.source), cache_provenance=provenance, args=vars(args).copy(),
                    source_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    job_id=os.environ['SLURM_JOB_ID'], results={},
                    caveats=['One phase; no cosmic-variance errors', '10% particle density; no shot-noise subtraction',
                             'Extra block-average cell window; no deconvolution',
                             'Nyquist planes removed; high-k modes require resolution comparison',
                             'Random-phase control has fixed amplitudes, not an exact Gaussian ensemble'])
    for n in args.ngrid:
        print(f'Start n={n}', flush=True)
        delta, mean = density(args.source, n)
        results = []
        for randomize in [False, True]:
            e, d, k2, error = solve(delta, args.box, args.radius, args.workers, randomize)
            result = summarize(e, d, k2, args.box, args.workers)
            result['trace_error'] = error
            tag = 'random_phase' if randomize else 'abacus'
            np.savez(args.output/f'{tag}_n{n}.npz', **result)
            manifest['results'][f'{tag}_n{n}'] = {k: float(result[k]) for k in ['sigma', 'trace_error', 'spectral_closure']}
            manifest['results'][f'{tag}_n{n}']['input_mean'] = mean
            results.append(result)
            del e, d, k2
        plot_results(results, args.output, n, args.radius)
        del delta
    (args.output/'COMPLETE.json').write_text(json.dumps(manifest, indent=2, default=str)+'\n')
    print('COMPLETE', flush=True)

if __name__ == '__main__':
    main()
