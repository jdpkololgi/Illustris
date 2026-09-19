"""Is the frozen VDM fine-power deficit sampler discretisation or learned-score error?

The completed experiment reports D fine sample/truth power ratios falling to 0.628
in the highest band. Two very different causes are consistent with that: (a) the
ancestral sampler under-injects noise at finite step count, which is a SAMPLING
error fixable without retraining, or (b) the learned score is wrong, which is not.

This re-samples ONE frozen development anchor from a frozen checkpoint at several
step counts, and additionally with a Langevin predictor-corrector, and compares
band power. It reads the published run read-only, writes only to its own output
directory, and never touches truth: band powers are compared BETWEEN samplers, and
the published per-case truth power supplies the absolute reference separately.

Development phase ph004 only. No confirmation phase, no sealed phase, no refit.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from workflows.sbi.e2e_vdm_context_analysis import Bands, EDGES
from workflows.sbi.e2e_vdm_context_data import read_json, spec
from workflows.sbi.e2e_vdm_context_dataset import Products
from workflows.sbi.e2e_vdm_context_models import ContextVDM, coupled_sample, project
from workflows.sbi.e2e_vdm_context_sample import expand_condition, load_model
from workflows.sbi.e2e_vdm_context_tasks import task_seed

RUN = Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v2/wide_pipeline_v1/vdm_context_20260917_v1')
DRAWS = 8


def corrected_sample(model, condition, steps, seeds, corrector, snr, noise_grid=1000):
    """Frozen ancestral predictor plus `corrector` Langevin steps per level.

    The predictor is algebraically identical to the frozen `coupled_sample`. The
    corrector is the standard VP Langevin update using the model's own epsilon
    prediction as the score, and is the ONLY addition.
    """
    if noise_grid % steps:
        raise ValueError('steps must divide the coupled noise grid')
    local = condition.local
    generators = [torch.Generator(device=local.device).manual_seed(int(s)) for s in seeds]
    projector = project if isinstance(model, ContextVDM) and model.arm == 'D' else (lambda x: x)

    def noise():
        return projector(torch.cat([torch.randn((1, 1, *local.shape[2:]), device=local.device,
                                                dtype=local.dtype, generator=g) for g in generators]))

    before = model.training
    model.eval()
    z = noise()
    factor = noise_grid // steps
    b = lambda v: v[:, None, None, None, None]
    try:
        for i in range(steps):
            gt = model.schedule(z.new_full((len(z),), 1 - i / steps))
            gs = model.schedule(z.new_full((len(z),), 1 - (i + 1) / steps))
            at, st = model.schedule.coefficients(gt)
            ass, ss = model.schedule.coefficients(gs)
            c = -torch.expm1(gs - gt)
            mean = b(ass / at) * (z - b(c * st) * model(z, gt, condition))
            eps = noise()
            for _ in range(factor - 1):
                eps = eps + noise()
            z = projector(mean + b(ss * c.sqrt()) * eps / math.sqrt(factor))
            for _ in range(corrector):
                score = -model(z, gs, condition) / b(ss)
                w = noise()
                gnorm = score.flatten(1).norm(dim=1).clamp_min(1e-12)
                wnorm = w.flatten(1).norm(dim=1)
                step = 2 * (snr * wnorm / gnorm) ** 2
                z = projector(z + b(step) * score + b((2 * step).sqrt()) * w)
            if not torch.isfinite(z).all():
                raise FloatingPointError('nonfinite corrected path')
        return z
    finally:
        model.train(before)


def band_power(delta, bands):
    """Mean over draws of the per-band pseudo-power, matching the frozen estimator."""
    values = []
    for field in delta:
        f = bands.fft(field)
        values.append(bands.cross(f, f))
    return np.asarray(values).mean(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=RUN)
    parser.add_argument('--arm', default='B')
    parser.add_argument('--replica', type=int, default=0)
    parser.add_argument('--steps', type=int, nargs='+', default=[125, 250, 500, 1000])
    parser.add_argument('--corrector', type=int, nargs='+', default=[1, 2])
    parser.add_argument('--snr', type=float, nargs='+', default=[0.05, 0.10, 0.20])
    parser.add_argument('--out', type=Path,
                        default=Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/diagnostics_20260919'))
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = spec()
    checkpoint = config['updates']
    ledger = read_json(args.run / 'DRAW_LEDGER.json')
    anchor = sorted(ledger['panels']['refinement'])[0]
    task = next(t for t in ledger['tasks']
                if t['arm'] == args.arm and t['replica'] == args.replica
                and t['checkpoint'] == checkpoint and t['purpose'] == 'main'
                and t['anchor'] == anchor and t['steps'] == 250)
    seeds = [task_seed(task, i, 'fine') for i in range(DRAWS)]

    observations = Products(args.run, ('ph004',), targets=False)
    model, checkpoint_sha = load_model(args.run, args.arm, args.replica, checkpoint, 'fine', device)
    condition = expand_condition(observations.condition(anchor, device=device), DRAWS)
    chart = observations.chart
    bands = Bands(48, 6.766, EDGES)

    def decode(z):
        rho = torch.exp(z.double() * chart['fine']['std'] + chart['fine']['mean'])
        if not torch.isfinite(rho).all() or (rho <= 0).any():
            raise FloatingPointError('nonfinite/nonpositive density')
        return (rho[:, 0] - 1).cpu().numpy()

    results = {}
    with torch.no_grad():
        for steps in args.steps:
            z = coupled_sample(model, condition, steps, seeds)
            results[f'ancestral_{steps}'] = band_power(decode(z), bands).tolist()
            print('ancestral', steps, [round(x, 6) for x in results[f'ancestral_{steps}']], flush=True)
        for corrector in args.corrector:
            for snr in args.snr:
                key = f'corrector{corrector}_snr{snr}_250'
                z = corrected_sample(model, condition, 250, seeds, corrector, snr)
                results[key] = band_power(decode(z), bands).tolist()
                print(key, [round(x, 6) for x in results[key]], flush=True)

    baseline = np.asarray(results['ancestral_250'])
    relative = {k: (np.asarray(v) / np.maximum(baseline, 1e-30)).tolist() for k, v in results.items()}
    record = dict(
        schema='e2e-diagnostic-sampler-scaling-v1',
        run=str(args.run), arm=args.arm, replica=args.replica, checkpoint=checkpoint,
        anchor=anchor, draws=DRAWS, seeds=seeds, device=str(device),
        checkpoint_sha256=checkpoint_sha, band_edges=[float(e) for e in EDGES[:-1]] + ['inf'],
        mean_sample_band_power=results,
        power_relative_to_ancestral_250=relative,
        caveat=('Eight draws at one development anchor: a mechanism screen, not a calibrated '
                'result. Band powers are compared BETWEEN samplers on identical seeds and '
                'conditions; no truth array was read. The corrector is an addition to the '
                'frozen predictor, not a change to any published draw or checkpoint.'))
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f'EXP1_SAMPLER_SCALING_{args.arm}{args.replica}.json'
    path.write_text(json.dumps(record, indent=1, sort_keys=True))
    print('WROTE', path)


if __name__ == '__main__':
    main()
