"""Independent numerical checks before any fresh-phase exterior scoring."""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_spatial_exterior as spatial
from workflows.sbi import e2e_footprint_tidal_operator as tidal


def run(inputs, samples, prior_path, out):
    c.require_compute()
    from workflows.sbi import e2e_loa_field_sampling as sampling
    from workflows.sbi import e2e_global_sampler as sampler
    from workflows.sbi import e2e_cfm_pilot_evaluate as old
    from workflows.sbi import e2e_coupled_views as views
    from workflows.sbi.e2e_exterior_panel import CoreTidal, load
    out.mkdir(parents=True, exist_ok=False)
    sampling.setup(True)
    models, hashes, normalizer = old.load_models(17, 13312, 26624)
    chart = views.load_chart(normalizer)
    # Sparse finite-volume sources permit a fully independent direct cuboid sum.
    rho = np.ones((64, 48, 48))
    points = [(16, 16, 16), (23, 25, 26), (45, 20, 22)]
    amplitudes = [.8, -.4, 1.3]
    for point, amplitude in zip(points, amplitudes):
        rho[point] += amplitude
    actual = CoreTidal(rho.shape, 0)(rho).reshape(32, 16, 16, 6)
    maximum = 0.
    for pos in ((0, 0, 0), (6, 8, 10), (18, 13, 6), (29, 4, 6)):
        target = np.array(pos)+16
        expected = sum(amplitude*tidal.cube_tensor(*(target-point)) for point, amplitude in zip(points, amplitudes))
        maximum = max(maximum, float(np.max(abs(actual[pos]-expected))))
    if maximum > 1e-12:
        raise ValueError('CUDA tensor disagrees with independent cuboid sum')
    fit = json.loads(prior_path.read_text())
    numerical, extent = [], []
    selections = [('ph012', 'NGC', 's0', 'interior'), ('ph012', 'SGC', 's3', 'boundary'),
                  ('ph013', 'NGC', 's3', 'boundary'), ('ph013', 'SGC', 's0', 'interior')]
    for phase, cap, shell, kind in selections:
        identity = f'{phase}_{cap}_{shell}_{kind}_00'
        panel = json.loads((inputs/phase/'PREPARED.json').read_text())
        case = next(row for row in panel['cases'] if row['pair_id'] == identity)
        arrays = [load(item) for item in case['conditions']]
        record = json.loads((samples/phase/identity/'COMPLETE.json').read_text())
        full = load(next(v for v in record['outputs'] if Path(v['path']).stem == 'shared_density'))
        selected = {name: load(next(v for v in record['outputs'] if Path(v['path']).stem == name))
                    for name in ('unconditional', 'conditional', 'learned')}
        reference = sampler.generate(models, arrays, chart, identity, count=4, nfe=128)
        refined = sampler.generate(models, arrays, chart, identity, count=4, nfe=256)
        a = np.concatenate((reference['rho'][0], reference['rho'][1][:, 32:]), axis=1)
        b = np.concatenate((refined['rho'][0], refined['rho'][1][:, 32:]), axis=1)
        spread = float(np.sqrt(np.mean(full['rho'].var(0, ddof=1))))
        numerical.append(dict(case=identity, nfe256_density_rmse_over_sd=float(np.sqrt(np.mean((a-b)**2))/spread),
                              saved_density_maxabs=float(np.max(abs(a.astype('f4')-full['rho'][:4])))))
        if numerical[-1]['saved_density_maxabs'] != 0:
            raise ValueError('Saved shared-state draws do not replay')
        support = full['support']
        # Freeze one large-domain density realization and vary only the
        # gravitational source extent, avoiding confounding by new random draws.
        pad = 64; shape = tuple(n+2*pad for n in support.shape)
        prior = spatial.CudaSpatialPrior(shape, fit, workers=4)
        operators = {p: CoreTidal(tuple(n+2*p for n in support.shape), p) for p in (16, 32, 64)}
        for name in ('unconditional', 'conditional', 'learned'):
            comparisons = {p: [] for p in (16, 32)}
            prior_domain = []
            tick = time.monotonic()
            for draw in range(4):
                seed = int(c.digest(['spatial-exterior-v1', identity, draw])[:15], 16)
                field, diagnostic = spatial.complete_density(a[draw], support, prior, np.random.default_rng(seed), name, pad)
                eigen = tidal.eigensystem(operators[64](field))
                for p in (16, 32):
                    edge = pad-p
                    cut = tuple(slice(edge, n-edge) for n in field.shape)
                    comparisons[p].append(tidal.eigensystem(operators[p](field[cut]))-eigen)
                prior_domain.append(eigen)
            posterior_sd = float(np.sqrt(np.mean(selected[name]['tidal'].var(0, ddof=1))))
            row = dict(case=identity, method=name,
                       source_extent_rms_over_sd={str(p): float(np.sqrt(np.mean(np.asarray(v)**2))/posterior_sd)
                                                 for p, v in comparisons.items()},
                       larger_domain_mean_shift_over_sd=float(np.sqrt(np.mean((np.asarray(prior_domain).mean(0)-selected[name]['tidal'].mean(0))**2))/posterior_sd),
                       larger_domain_draws=4, seconds=time.monotonic()-tick,
                       domain_shift_is_unpaired_monte_carlo=True)
            extent.append(row); print(json.dumps(row), flush=True)
        del operators, prior
    passed = maximum < 1e-12 and max(v['nfe256_density_rmse_over_sd'] for v in numerical) < .1
    c.atomic_json(out/'CHECKS.json', dict(operator_direct_maxabs=maximum, sampler=numerical, extent=extent,
                  numerical_passed=passed, source_extent16_passed=all(r['source_extent_rms_over_sd']['16'] < .1 for r in extent),
                  source_extent32_passed=all(r['source_extent_rms_over_sd']['32'] < .1 for r in extent),
                  prior_domain_distribution_convergence_established=False,
                  native_density_resolution_convergence_established=False,
                  source_sha256=c.sha256(__file__), job=os.environ['SLURM_JOB_ID']))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--samples', type=Path, required=True); p.add_argument('--prior', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True); a = p.parse_args()
    run(a.inputs, a.samples, a.prior, a.out)
