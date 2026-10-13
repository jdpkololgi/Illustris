"""Finite-domain exploratory draws with frozen coarse inputs and honest scope.

The common local+wide free-space operator is a diagnostic, NOT the full-cap
corrected operator. Report its truth-closure error as well as fullbox scores.
Target files are opened only in the explicitly labelled oracle/scoring branches.
"""
import argparse
import json
from pathlib import Path
import time

import h5py
import numpy as np
import torch

from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_coordinates as coord
from workflows.sbi import e2e_coupled_views as views
from workflows.sbi import e2e_footprint_tidal_operator as tidal
from workflows.sbi.e2e_wholefield_pilot_data import ROOT, SHAPE, EVAL, COARSE_CROP
from workflows.sbi.e2e_wholefield_pilot_models import Condition, FineModel, sample
from workflows.sbi.e2e_wholefield_pilot_train import setup, sources
from workflows.sbi.e2e_wholefield_fullcap_extract import summary
from workflows.sbi.e2e_vdm_context_models import decode_density

BANK = ROOT.parent/'wholefield_comparison_20261009_v1/fullcap_v1'
CORE = (slice(32, 64), slice(24, 40), slice(24, 40))


def read(path):
    return json.loads(Path(path).read_text())


def verify(record):
    if c.sha256(record['path']) != record['sha256']:
        raise ValueError('Source changed: '+record['path'])


def observation(phase, pair, chart):
    if phase not in EVAL:
        raise PermissionError('Only exposed evaluation phases')
    directory = ROOT/'data'/phase/pair
    with np.load(directory/'OBSERVATIONS.npz') as f:
        return tuple(views.scale_channels(f[k], chart[k]) for k in ('joint', 'wide'))


def bank_field(phase, cap, seed, first):
    if phase not in EVAL or cap not in ('NGC', 'SGC') or seed not in (17, 29) or first not in range(0, 32, 4):
        raise PermissionError('Unregistered baseline address')
    key = f'{phase}/{cap}/seed{seed}/d{first:03d}'
    manifest = read(BANK/'MANIFEST.json'); reuse = read(BANK/'REUSE_VERIFIED.json')
    task = next(t for t in manifest['tasks'] if t['key'] == key)
    record = task['density_bound_receipt'] if key in reuse['tasks'] else read(BANK/'tasks'/key/'COMPLETE.json')['density']
    verify(record); payload = read(record['path'])
    if not payload['passed'] or (payload['phase'], payload['cap'], payload['first']) != (phase, cap, first):
        raise ValueError('Wrong baseline density address')
    # Full payload hashes were verified by the completed bank. Do not rehash
    # multi-GB fields for every local pair; retain the bound receipt and size.
    if Path(payload['field']['path']).stat().st_size != payload['field']['bytes']:
        raise ValueError('Baseline payload size changed')
    return payload['field'], record


def integer_crop(start, shape, available):
    rounded = np.rint(start).astype(int)
    if np.max(abs(start-rounded)) > 1e-7 or any(a < 0 or a+n > m for a, n, m in zip(rounded, shape, available)):
        raise ValueError('Requested field outside aligned saved canvas')
    return tuple(slice(int(a), int(a+n)) for a, n in zip(rounded, shape))


def frozen_coarse(row, seed, draw):
    record, receipt = bank_field(row['phase'], row['cap'], seed, draw//4*4)
    midpoint = np.asarray(row['center'])+[16, 0, 0]
    with h5py.File(record['path']) as f:
        z = f['coarse_logrho_standardized']
        cut = integer_crop((midpoint-192-f.attrs['coarse_origin_raw_cells'])/8, (48,)*3, z.shape[1:])
        coarse = z[(draw % 4, *cut)]
    return coarse, receipt


class LocalWideTidal:
    """Common finite free-space closure with unit density beyond the wide cube."""
    def __init__(self):
        self.entries = []
        for shape, offset in (((192,)*3, (80, 88, 88)), (SHAPE, (32, 24, 24))):
            kernel = tidal.kernel(shape, (32, 16, 16), offset=offset)
            self.entries.append((kernel.shape[1:], [torch.fft.rfftn(torch.as_tensor(k, device='cuda')) for k in kernel]))

    @torch.no_grad()
    def tensor(self, rho, wide):
        background = np.asarray(wide, float)
        for axis in range(3):
            background = background.repeat(4, axis=axis)
        residual = np.asarray(rho, float)-background[48:144, 64:128, 64:128]
        result = np.zeros((32, 16, 16, 6))
        for value, (shape, kernels) in zip((background-1, residual), self.entries):
            spectrum = torch.fft.rfftn(torch.as_tensor(value, device='cuda'), s=shape)
            for k, kernel in enumerate(kernels):
                result[..., k] += torch.fft.irfftn(spectrum*kernel, s=shape)[:32, :16, :16].cpu().numpy()
        np.testing.assert_allclose(result[..., [0, 3, 5]].sum(-1), rho[CORE]-1, atol=2e-8, rtol=0)
        return result.reshape(2, 4096, 6)


def metrics(draws, truth):
    draws = np.asarray(draws, float); truth = np.asarray(truth, float)
    lo, hi = np.quantile(draws, [.05, .95], axis=0)
    sd = draws.std(0, ddof=1)
    crps = np.abs(draws-truth).mean(0)-.5*np.abs(draws[:, None]-draws[None, :]).mean((0, 1))
    energy = np.linalg.norm(draws-truth, axis=-1).mean()-.5*np.linalg.norm(draws[:, None]-draws[None, :], axis=-1).mean()
    return dict(mean=draws.mean(0).tolist(), sd=sd.tolist(), truth=truth.tolist(),
        coverage90=((lo <= truth)&(truth <= hi)).tolist(), lower90=lo.tolist(), upper90=hi.tolist(),
        crps=crps.tolist(), energy_score=float(energy), rank=(draws < truth).mean(0).tolist(),
        mean_error_sd=((draws.mean(0)-truth)/np.maximum(sd, 1e-12)).tolist())


def run(architecture, prediction, seed, step, phase, pair, coarse, draws, nfe, operator='fullcap'):
    if phase not in EVAL or coarse not in ('sampled', 'oracle') or not 2 <= draws <= 32:
        raise PermissionError('Invalid exploratory evaluation address')
    setup(seed)
    fit = ROOT/'fits'/f'{architecture}_{prediction}_{seed}'/f'CHECKPOINT_{step:06d}.pt'
    state = torch.load(fit, map_location='cuda', weights_only=False)
    binding = state['binding']
    if binding['sources'] != sources() or state['step'] != step:
        raise ValueError('Fit source/step mismatch')
    chart = views.load_chart(binding['normalizer'])
    rows = read(ROOT/'data'/phase/'COMPLETE.json')['pairs']
    row = dict(next(r for r in rows if r['pair_id'] == pair), phase=phase)
    joint, wide_obs = observation(phase, pair, chart)
    model = FineModel(architecture, prediction).cuda().eval(); model.load_state_dict(state['ema'])
    del state
    if operator not in ('fullcap', 'local'):
        raise ValueError('Unknown operator')
    dest = ROOT/'evaluation'/f'{architecture}_{prediction}_{seed}'/f'step{step}'/phase/pair/coarse/operator/f'nfe{nfe}'
    dest.mkdir(parents=True, exist_ok=True)
    bound = dict(checkpoint=c.file_record(fit, content_hash=True), phase=phase, pair=pair,
                 coarse=coarse, nfe=nfe, draws=draws, inference='independent finite multiwindow fields',
                 operator=operator, exterior=('shared frozen full-cap reference draw; not reconditioned on replacement'
                      if operator == 'fullcap' else 'local-plus-wide unit density exterior; not corrected full-cap'),
                 source=c.sha256(__file__), truth_is_input=coarse == 'oracle')
    if (dest/'BINDING.json').exists():
        if read(dest/'BINDING.json') != bound:
            raise ValueError('Evaluation binding mismatch')
    else:
        c.atomic_json(dest/'BINDING.json', bound)
    if (dest/'COMPLETE.json').exists():
        return
    cs = chart['coarse_logrho']; receipts = []; start = time.monotonic()
    for draw in range(draws):
        output = dest/f'draw{draw:03d}.npz'
        if output.exists():
            continue
        if coarse == 'sampled':
            z, receipt = frozen_coarse(row, seed, draw); receipts.append(receipt)
        else:
            with np.load(ROOT/'data'/phase/pair/'TARGETS.npz') as f:
                z = ((np.log(f['coarse_wide'])-cs['mean'][0])/cs['std'][0]).astype('f4')
        tensor = lambda x: torch.as_tensor(x, device='cuda', dtype=torch.float32)[None]
        cond = Condition(tensor(joint), tensor(wide_obs), tensor(z[None]),
                         'sampled' if coarse == 'sampled' else 'oracle_diagnostic')
        rng = torch.Generator(device='cuda').manual_seed(int(c.digest(['wholefield-pilot-draw', phase, pair, seed, draw])[:15], 16))
        noise = torch.randn((1, 1, *SHAPE), device='cuda', generator=rng)
        latent = sample(model, cond, noise, nfe)
        physical_wide = np.exp(z.astype(float)*cs['std'][0]+cs['mean'][0])
        mass = torch.as_tensor(physical_wide[COARSE_CROP], device='cuda')[None, None]
        rho = decode_density(mass, latent.double()*chart['fine_residual']['std'][0])[0, 0].cpu().numpy()
        if not np.isfinite(rho).all() or rho.min() <= 0:
            raise FloatingPointError('Nonpositive/nonfinite draw')
        temporary = output.with_suffix('.tmp')
        with temporary.open('wb') as f:
            np.savez_compressed(f, rho=rho.astype('f4'), wide=physical_wide.astype('f4'))
        temporary.replace(output)
        print(json.dumps(dict(phase=phase, pair=pair, coarse=coarse, draw=draw, seconds=time.monotonic()-start)), flush=True)
    del model; torch.cuda.empty_cache()
    # Scoring begins here. No truth files were opened by sampled inference above.
    with np.load(ROOT/'data'/phase/pair/'TARGETS.npz') as f:
        truthrho, truthwide = f['rho'].astype(float), f['coarse_wide'].astype(float)
    original = read(coord.ROOT/'targets'/phase/f'{pair}.json')
    with h5py.File(original['outputs'][0]['path']) as f:
        fullbox = tidal.eigensystem(f['fullbox_tensor_cores'][:]).reshape(2, 4096, 3)
    truth_density = truthrho[CORE].reshape(2, 4096)
    true_summary = summary(truth_density, fullbox)
    if operator == 'fullcap':
        from workflows.sbi.e2e_wholefield_pilot_exterior import ReplacementTidal
        physical_operator = ReplacementTidal()
    else:
        physical_operator = LocalWideTidal()
    predictions = []; baseline = []; closure_truth = []; density_draws = []
    for draw in range(draws):
        with np.load(dest/f'draw{draw:03d}.npz') as f:
            rho, wide = f['rho'].astype(float), f['wide'].astype(float)
        density_draws.append(rho[CORE].reshape(2, 4096))
        if operator == 'fullcap':
            key = f'{phase}/{row["cap"]}/seed{seed}/d{draw//4*4:03d}'
            directory = ROOT/'exterior'/key
            receipt = read(directory/'COMPLETE.json')
            if not receipt['passed'] or receipt['key'] != key:
                raise ValueError('Wrong exterior cache')
            path = directory/f'{pair}_d{draw:03d}.npz'
            verify(next(r for r in receipt['files'] if r['path'] == str(path)))
            with np.load(path) as f:
                base = {k: f[k] for k in f.files}
            t = physical_operator.tensor(rho, base)
            truth_t = physical_operator.tensor(truthrho, base)
            reference_t = physical_operator.tensor(base['rho'].astype(float), base)
            reference = summary(base['rho'][CORE].astype(float).reshape(2, 4096), tidal.eigensystem(reference_t))
            with np.load(BANK/'extraction_v1/cores'/key/'CORES.npz') as f:
                index = list(f['pair_id']).index(pair)
                np.testing.assert_allclose(reference, f['draws'][index, draw % 4], atol=2e-8, rtol=0)
            baseline.append(reference)
        else:
            t = physical_operator.tensor(rho, wide)
            truth_t = physical_operator.tensor(truthrho, truthwide)
        predictions.append(summary(rho[CORE].reshape(2, 4096), tidal.eigensystem(t)))
        closure_truth.append(summary(truth_density, tidal.eigensystem(truth_t)))
    predictions = np.asarray(predictions)
    local_summary = np.mean(closure_truth, axis=0)
    density_draws = np.asarray(density_draws)
    lower, upper = np.quantile(density_draws, [.05, .95], axis=0)
    # Efficient empirical CRPS avoids allocating draws x draws x all voxels.
    ordered = np.sort(density_draws, axis=0)
    coefficients = (2*np.arange(1, draws+1)-draws-1)[:, None, None]
    voxel_crps = np.abs(density_draws-truth_density).mean(0)-(coefficients*ordered).sum(0)/draws**2
    voxel = dict(coverage90=float(((lower <= truth_density)&(truth_density <= upper)).mean()),
                 mean_crps=float(voxel_crps.mean()), rmse_mean=float(np.sqrt(((density_draws.mean(0)-truth_density)**2).mean())),
                 mean_posterior_sd=float(density_draws.std(0, ddof=1).mean()),
                 scope='8192 correlated core voxels; not independent coverage trials')
    c.atomic_json(dest/'COMPLETE.json', dict(binding=c.sha256(dest/'BINDING.json'), passed=True,
        fullbox=metrics(predictions, true_summary), closure_mean_diagnostic=metrics(predictions, local_summary),
        reference_fullbox=metrics(baseline, true_summary) if baseline and coarse == 'sampled' else None,
        closure_truth_shift=(local_summary-true_summary).tolist(), summaries=predictions.tolist(),
        truth_under_closure=np.asarray(closure_truth).tolist(), operator=operator,
        density_voxels=voxel,
        coarse_receipts=receipts, seconds=time.monotonic()-start, posterior_qualified=False,
        warning=('Shared frozen exterior; modular finite-domain replacement, not a jointly reconditioned full-cap posterior.'
                 if operator == 'fullcap' else 'Local finite closure only; not corrected full-cap baseline.')))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--architecture', required=True, choices=('unet', 'transformer'))
    p.add_argument('--prediction', required=True, choices=('v', 'x'))
    p.add_argument('--seed', type=int, required=True, choices=(17, 29))
    p.add_argument('--step', type=int, required=True)
    p.add_argument('--phase', required=True, choices=EVAL)
    p.add_argument('--pair', required=True)
    p.add_argument('--coarse', choices=('sampled', 'oracle'), required=True)
    p.add_argument('--draws', type=int, default=32)
    p.add_argument('--nfe', type=int, default=128)
    p.add_argument('--operator', choices=('fullcap', 'local'), default='fullcap')
    run(**vars(p.parse_args()))
