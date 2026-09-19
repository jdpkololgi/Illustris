"""Truth-only effect size for the proposed coupled-field primary gate (read-only).

The proposed gate (docs/e2e_coupled_resource_proposal_20260918.md, decision item 3)
requires >=10% lower matched cross-core variogram score for J-VDM over I-VDM on a
14-vector of block-mean-subtracted DCT-II probes. That gate is only meetable if the
TRUE conditional cross-core dependence of those probes is large enough to register.

This measures that dependence directly from the prepared coupled TRUTH rectangles,
then converts the measured 14x14 covariance into the maximum variogram gain any
correct posterior could show over a correct-marginals/independent-cores posterior.

Reads target products read-only. Writes nothing into the coupled data root. These
are training-role phases; no sealed phase and no confirmation prediction is touched.
"""
import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np

TARGETS = Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/coupled_20260918_v1/cartesian_v2/targets')
CORES = (((16, 32), (16, 32), (16, 32)), ((32, 48), (16, 32), (16, 32)))
MODES = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1))
CORE_N, BLOCK = 16, 4


def probe_weights():
    """Seven separable DCT-II modes, 4-cubed block means removed, unit L2."""
    index = np.arange(CORE_N)
    axis = lambda k: np.cos(np.pi * k * (index + 0.5) / CORE_N)
    weights = []
    for mode in MODES:
        w = np.einsum('i,j,k->ijk', axis(mode[0]), axis(mode[1]), axis(mode[2]))
        blocked = w.reshape(CORE_N // BLOCK, BLOCK, CORE_N // BLOCK, BLOCK, CORE_N // BLOCK, BLOCK)
        w = (blocked - blocked.mean(axis=(1, 3, 5), keepdims=True)).reshape(CORE_N, CORE_N, CORE_N)
        weights.append(w / np.linalg.norm(w))
    return np.array(weights)


def block_means(core):
    shaped = core.reshape(CORE_N // BLOCK, BLOCK, CORE_N // BLOCK, BLOCK, CORE_N // BLOCK, BLOCK)
    return shaped.mean(axis=(1, 3, 5)).ravel()


def domain_files(targets, per_phase, roles=('train',)):
    picked = []
    for phase in sorted(os.listdir(targets)):
        folder = targets / phase
        if not folder.is_dir():
            continue
        names = sorted(n for n in os.listdir(folder)
                       if n.endswith('.h5') and not n.startswith('density_generation'))
        kept = 0
        for name in names:
            if kept >= per_phase:
                break
            picked.append(folder / name)
            kept += 1
    return picked


def collect(files, roles):
    weights = probe_weights()
    probes, coarse, used = [], [], []
    for path in files:
        with h5py.File(path, 'r') as handle:
            if handle.attrs.get('role') not in roles:
                continue
            rho = np.asarray(handle['rho_joint'], dtype=np.float64)
        delta = rho - 1.0
        row, blocks = [], []
        for (x0, x1), (y0, y1), (z0, z1) in CORES:
            core = delta[x0:x1, y0:y1, z0:z1]
            row.extend(float((w * core).sum()) for w in weights)
            blocks.extend(block_means(core))
        probes.append(row)
        coarse.append(blocks)
        used.append(str(path.name))
    return np.array(probes), np.array(coarse), used


def residualise(values, regressors):
    """Least-squares residuals of `values` after removing `regressors` (with intercept)."""
    design = np.hstack([np.ones((len(regressors), 1)), regressors])
    coef, *_ = np.linalg.lstsq(design, values, rcond=None)
    return values - design @ coef


def variogram_gain(cov, draws=6000, truths=6000, seed=11):
    """Max variogram-score gain of the correct dependent posterior over one with
    correct marginals but independent cores, for the measured 14x14 covariance."""
    rng = np.random.default_rng(seed)
    dim = cov.shape[0] // 2
    scale = np.sqrt(np.diag(cov))
    corr = cov / np.outer(scale, scale)
    chol = np.linalg.cholesky(corr + 1e-10 * np.eye(len(corr)))
    independent = corr.copy()
    independent[:dim, dim:] = 0.0
    independent[dim:, :dim] = 0.0
    chol_independent = np.linalg.cholesky(independent + 1e-10 * np.eye(len(corr)))
    pairs = [(i, i + dim) for i in range(dim)]
    left = [i for i, _ in pairs]
    right = [j for _, j in pairs]

    def sample(factor, n):
        return rng.standard_normal((n, len(corr))) @ factor.T

    def score(forecast, truth):
        gamma = (np.abs(forecast[:, left] - forecast[:, right]) ** 0.5).mean(0)
        return float((((np.abs(truth[:, left] - truth[:, right]) ** 0.5) - gamma) ** 2).sum(1).mean())

    truth = sample(chol, truths)
    dependent = score(sample(chol, draws), truth)
    marginal = score(sample(chol_independent, draws), truth)
    return dict(variogram_dependent=dependent, variogram_independent=marginal,
                max_achievable_gain_percent=100.0 * (marginal - dependent) / marginal)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--targets', type=Path, default=TARGETS)
    parser.add_argument('--per-phase', type=int, default=64)
    parser.add_argument('--out', type=Path,
                        default=Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/diagnostics_20260919'))
    args = parser.parse_args()

    files = domain_files(args.targets, args.per_phase)
    probes, coarse, used = collect(files, ('train',))
    if len(probes) < 200:
        raise SystemExit(f'too few training domains collected: {len(probes)}')
    dim = probes.shape[1] // 2

    standard = (probes - probes.mean(0)) / probes.std(0, ddof=1)
    raw_corr = np.corrcoef(standard, rowvar=False)
    matched_raw = [float(raw_corr[i, i + dim]) for i in range(dim)]

    residual = residualise(probes, coarse)
    partial_corr = np.corrcoef(residual, rowvar=False)
    matched_partial = [float(partial_corr[i, i + dim]) for i in range(dim)]

    gain_raw = variogram_gain(np.cov(standard, rowvar=False))
    residual_standard = (residual - residual.mean(0)) / residual.std(0, ddof=1)
    gain_partial = variogram_gain(np.cov(residual_standard, rowvar=False))

    result = dict(
        schema='e2e-diagnostic-probe-effect-size-v1',
        targets=str(args.targets), domains=len(probes), regressors=coarse.shape[1],
        cores=[[list(a) for a in core] for core in CORES], modes=[list(m) for m in MODES],
        core_separation_mpc_h=16 * 6.766,
        matched_cross_core_correlation_raw=matched_raw,
        matched_cross_core_correlation_given_coarse=matched_partial,
        mean_abs_matched_raw=float(np.mean(np.abs(matched_raw))),
        mean_abs_matched_given_coarse=float(np.mean(np.abs(matched_partial))),
        variogram_gain_raw=gain_raw,
        variogram_gain_given_coarse=gain_partial,
        registered_gate_percent=10.0,
        caveat=('Training-role truth rectangles only; overlapping context crops are not '
                'independent volumes, so correlations carry more uncertainty than the count '
                'suggests. Partial correlations remove the 4-cubed coarse block means of both '
                'cores, the quantity I and J share exactly. Population Gaussian surrogate for '
                'the variogram mapping; not a claim about any trained posterior.'),
        example_files=used[:3])
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / 'EXP3_PROBE_EFFECT_SIZE.json'
    path.write_text(json.dumps(result, indent=1, sort_keys=True))
    print('WROTE', path)
    print(f"domains={len(probes)}  coarse regressors={coarse.shape[1]}")
    print('matched cross-core corr (raw)        ', [round(x, 4) for x in matched_raw])
    print('matched cross-core corr (given coarse)', [round(x, 4) for x in matched_partial])
    print(f"max achievable variogram gain: raw {gain_raw['max_achievable_gain_percent']:.3f}%"
          f"   given coarse {gain_partial['max_achievable_gain_percent']:.3f}%"
          f"   (registered gate 10%)")


if __name__ == '__main__':
    main()
