"""Does the Langevin corrector improve CALIBRATION, or only inflate variance?

exp1 showed that adding corrector steps removes about half of arm B's band-power
deficit. Power is a second moment: more noise always raises it. The scientific
question is whether the corrected ensemble is a BETTER POSTERIOR, which requires
a proper score and coverage against truth, on more than one arm.

This extends exp1 to arms B/C/D over several development anchors and scores
density CRPS and central coverage on the owned science core, using the frozen
`calibration` implementation. Arm D reuses its EXISTING cached shared coarse
parents strictly read-only: a missing cache entry is skipped, never generated,
so the frozen run root is not extended.

Development phase ph004 only. Truth is read through the registered development
reader. No confirmation or sealed phase, no refit, no frozen artifact changed.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from workflows.sbi import e2e_wide_pipeline as existing
from workflows.sbi.e2e_vdm_context_analysis import CORE, Bands, EDGES, checkpoint_hash
from workflows.sbi.e2e_vdm_context_data import read_json, spec
from workflows.sbi.e2e_vdm_context_dataset import Products, coarse_local_crop
from workflows.sbi.e2e_vdm_context_metrics import calibration
from workflows.sbi.e2e_vdm_context_models import coupled_sample, decode_density
from workflows.sbi.e2e_vdm_context_sample import (expand_condition, load_model,
                                                  read_array_receipt, with_coarse)
from workflows.sbi.e2e_vdm_context_tasks import coarse_cache_key, task_seed

from exp1_sampler_scaling import corrected_sample  # noqa: E402  (same directory)

RUN = Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v2/wide_pipeline_v1/vdm_context_20260917_v1')
MICROBATCH = 8


def cached_parents(root, task, ids, coarse_sha, device):
    """Read already-generated shared coarse draws. Never samples, never writes."""
    arrays = {}
    for first in sorted({(i // MICROBATCH) * MICROBATCH for i in ids}):
        draw_ids = list(range(first, first + MICROBATCH))
        key = coarse_cache_key('D', task['replica'], task['checkpoint'], task['domain'],
                               first, task['steps'], task['purpose'])
        folder = root / 'parents' / str(Path(key).parent)
        receipt = folder / f'{first:06d}.json'
        if not receipt.exists():
            return None
        binding = dict(manifest_sha256=existing.sha256(root / 'MANIFEST.json'),
                       checkpoint_sha256=coarse_sha, domain=task['domain'],
                       replica=task['replica'], checkpoint=task['checkpoint'],
                       steps=task['steps'], purpose=task['purpose'], ids=draw_ids)
        rho, _ = read_array_receipt(folder, receipt, binding, 'rho')
        arrays.update(zip(draw_ids, rho))
    return torch.as_tensor(np.stack([arrays[i] for i in ids]), device=device)[:, None]


def score(delta, truth, bands, scale):
    core_draws = np.asarray([d[CORE] for d in delta])
    core_truth = np.asarray(truth)[CORE]
    stats = calibration(core_draws, core_truth)
    power = np.asarray([bands.cross(bands.fft(d), bands.fft(d)) for d in delta]).mean(0)
    truth_power = bands.cross(bands.fft(np.asarray(truth)), bands.fft(np.asarray(truth)))
    return dict(density_crps=float(stats['crps'].mean()),
                standardized_crps=float(stats['crps'].mean() / scale),
                coverage90=float(stats['covered_0.9'].mean()),
                attainable90=float(stats['attainable']['0.9']),
                rmse_mean=float(np.sqrt(np.mean(stats['bias'] ** 2))),
                band_power_ratio=(power / np.maximum(truth_power, 1e-30)).tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=RUN)
    parser.add_argument('--arms', nargs='+', default=['B', 'C', 'D'])
    parser.add_argument('--replica', type=int, default=0)
    parser.add_argument('--anchors', type=int, default=4)
    parser.add_argument('--draws', type=int, default=32)
    parser.add_argument('--corrector', type=int, default=2)
    parser.add_argument('--snr', type=float, default=0.3)
    parser.add_argument('--out', type=Path,
                        default=Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/diagnostics_20260919'))
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = spec()
    checkpoint = config['updates']
    ledger = read_json(args.run / 'DRAW_LEDGER.json')
    bands = Bands(48, 6.766, EDGES)

    truth_reader = Products(args.run, ('ph004',), targets=True)
    observations = Products(args.run, ('ph004',), targets=False)
    chart = observations.chart
    scale = float(np.asarray(chart['fine']['std']))

    records = {}
    with torch.no_grad():
        for arm in args.arms:
            model, sha = load_model(args.run, arm, args.replica, checkpoint, 'fine', device)
            tasks = [t for t in ledger['tasks']
                     if t['arm'] == arm and t['replica'] == args.replica
                     and t['checkpoint'] == checkpoint and t['purpose'] == 'main'
                     and t['steps'] == 250 and t['anchor'].startswith('ph004')
                     and t.get('coarse_mode') in (None, 'sampled')]
            tasks = sorted(tasks, key=lambda t: t['anchor'])[:args.anchors]
            for task in tasks:
                anchor = task['anchor']
                ids = list(range(args.draws))
                seeds = [task_seed(task, i, 'fine') for i in ids]
                base = observations.condition(anchor, device=device)
                condition = expand_condition(base, args.draws)
                rho_coarse = None
                if arm == 'D':
                    coarse_sha = checkpoint_hash(args.run, task, 'coarse')
                    rho_coarse = cached_parents(args.run, task, ids, coarse_sha, device)
                    if rho_coarse is None:
                        print('SKIP (no cached parent)', arm, anchor, flush=True)
                        continue
                    offset = observations.rows[anchor].get('core_offset_raw', [0, 0, 0])
                    condition = with_coarse(base, rho_coarse, chart, offset, task['coarse_mode'])

                def decode(z):
                    if arm == 'D':
                        sl = coarse_local_crop(observations.rows[anchor].get('core_offset_raw', [0, 0, 0]), (0, 0, 0))
                        rho = decode_density(rho_coarse[(slice(None), slice(None), *sl)],
                                             z.double() * chart['residual']['std'])
                    else:
                        rho = torch.exp(z.double() * chart['fine']['std'] + chart['fine']['mean'])
                    return (rho[:, 0] - 1).cpu().numpy()

                truth = truth_reader.raw_targets(anchor)['rho'] - 1
                out = {}
                out['ancestral_250'] = score(decode(coupled_sample(model, condition, 250, seeds)),
                                             truth, bands, scale)
                corrected = corrected_sample(model, condition, 250, seeds, args.corrector, args.snr)
                out[f'corrector{args.corrector}_snr{args.snr}_250'] = score(decode(corrected), truth, bands, scale)
                records[f'{arm}_{anchor}'] = out
                a, c = out['ancestral_250'], out[f'corrector{args.corrector}_snr{args.snr}_250']
                print(f"{arm} {anchor}: CRPS {a['density_crps']:.5f} -> {c['density_crps']:.5f}"
                      f"  C90 {a['coverage90']:.4f} -> {c['coverage90']:.4f}"
                      f"  (attainable {a['attainable90']:.4f})", flush=True)

    record = dict(schema='e2e-diagnostic-sampler-calibration-v1', run=str(args.run),
                  replica=args.replica, draws=args.draws, checkpoint=checkpoint,
                  corrector=args.corrector, snr=args.snr, cases=records,
                  caveat=('Development phase ph004 only, one seed, a few anchors, 32 draws. '
                          'Coverage is over correlated core voxels, not independent posterior '
                          'trials. Arm D reuses existing cached coarse parents read-only. '
                          'A screen for whether the corrector helps calibration, not a '
                          'calibrated-posterior claim or a revision of any published result.'))
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / 'EXP4_SAMPLER_CALIBRATION.json'
    path.write_text(json.dumps(record, indent=1, sort_keys=True))
    print('WROTE', path)


if __name__ == '__main__':
    main()
