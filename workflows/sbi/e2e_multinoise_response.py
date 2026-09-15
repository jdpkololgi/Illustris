"""Post-fit read-only antithetic/clean-input localization; no changed pass gate."""
import argparse
import json
import math
import os
from pathlib import Path
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from workflows.sbi import e2e_wide_pipeline as p
from workflows.sbi.e2e_wide_continue import checked_binding
from workflows.sbi.e2e_wide_denoising_audit import Bands, TRAIN384, clean_estimate
from workflows.sbi.e2e_wide_research_canary import preflight, NORMALIZATION
from workflows.sbi.e2e_multinoise_test import make_model
from workflows.sbi.e2e_multinoise_report import verify
from workflows.sbi.e2e_fixed_noise_response import decomposition


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('input', type=Path); ap.add_argument('output', type=Path)
    args = ap.parse_args(); device = p.runtime(); data = json.loads(args.input.read_text()); root = args.input.parent
    if not data['complete'] or len(data['results']) != 6:
        raise ValueError('completed registered comparison required')
    checked = verify(data, root); c, _, _, _ = preflight(); ds = p.dataset_for(c, NORMALIZATION)
    binding = checked_binding(TRAIN384, p.provenance(c, ds))
    if p.sha256(TRAIN384/'diffusion_fine/step_000384.pt') != data['registration']['parent_sha256']:
        raise ValueError('parent drift')
    cfg = data['registration']['config']; panel = data['registration']['panel']; items = {}
    for group in ('fit', 'transfer'):
        for anchor in panel[group+'_anchors']:
            item = ds[next(i for i, r in enumerate(ds.rows) if r['anchor_id'] == anchor)]
            items[anchor] = dict(target=p.tensor(item['fine_target'], device), condition=p.tensor(item['fine_condition'], device),
                                 wide=p.tensor(item['coarse_condition'], device), group=group)
    bands = Bands(96, 3.383, [0, .08, .16, .32, np.inf]); rows = []; started = time.monotonic(); slices = {}
    scale = ds.normalization['targets']['fine']['std']
    for arm in cfg['arms']:
        model = make_model(arm, c, cfg, device)
        expected = {**binding, 'multinoise': dict(registration=data['registration'], arm=arm)}
        state = p.load_checkpoint(root/arm/'update_003072.pt', expected, 'fine', 'diffusion')
        model.load_state_dict(state['model']); model.eval()
        for ratio in (.05, .2):
            a = 1/math.sqrt(1+ratio**2); b = ratio*a; t = 2*math.atan(ratio)/math.pi
            for anchor, item in items.items():
                if time.monotonic()-started > 600:
                    raise TimeoutError('read-only response cap reached')
                y = item['target']; seed = p.seed_for(panel['seed'], anchor, 'evaluation-0', 'fine')
                noise = torch.randn(y.shape, device=device, generator=torch.Generator(device=device).manual_seed(seed))
                reconstructed = []
                for x in (a*y+b*noise, a*y-b*noise, a*y):
                    pred = model(x, y.new_tensor([t]), item['condition'], wide_condition=item['wide'])
                    reconstructed.append(clean_estimate('diffusion', x, pred, t)[0, 0].cpu().numpy())
                truth = y[0, 0].cpu().numpy(); plus, minus, zero = reconstructed
                metrics = decomposition(bands, (plus-truth)*scale, (minus-truth)*scale, (zero-truth)*scale)
                metrics['clean_input_gain'] = bands.compare(zero, truth)['gain']
                rows.append(dict(arm=arm, ratio=ratio, anchor_id=anchor, group=item['group'], seed=seed, metrics=metrics))
                if ratio == .05 and anchor == panel['transfer_anchors'][0]:
                    slices[arm] = dict(clean=plus[48].tolist(), error=(plus-truth)[48].tolist())
                    slices['reference'] = dict(truth=truth[48].tolist(), noisy=(y+ratio*noise)[0, 0, 48].cpu().tolist())
        del model, state
    summary = {}
    for arm in cfg['arms']:
        for ratio in (.05, .2):
            for group in ('fit', 'transfer'):
                selected = [r for r in rows if r['arm'] == arm and r['ratio'] == ratio and r['group'] == group]
                summary[f'{arm}/{ratio}/{group}'] = {k: np.median([r['metrics'][k] for r in selected], axis=0).tolist() for k in selected[0]['metrics']}
    preflight(); verify(data, root)
    p.write_json(args.output/'RESPONSE.json', dict(complete=True, claim='descriptive localization, not a new gate',
                 main_sha256=p.sha256(args.input), source_sha256=p.sha256(__file__),
                 decomposition_source_sha256=p.sha256(p.REPO/'workflows/sbi/e2e_fixed_noise_response.py'),
                 job_id=os.environ['SLURM_JOB_ID'], verification=checked, rows=rows, summary=summary,
                 forward_calls=216, elapsed_seconds=time.monotonic()-started, training_updates=0, heldout_payloads_read=False))
    p.write_json(args.output/'SLICES.json', dict(anchor=panel['transfer_anchors'][0], ratio=.05, normalized_units=True, slices=slices))
    fig, axes = plt.subplots(2, 4, figsize=(12, 6), constrained_layout=True)
    arrays = [np.array(slices['reference'][k]) for k in ('truth', 'noisy')]+[np.array(slices[arm]['clean']) for arm in cfg['arms']]
    limit = float(np.quantile(np.abs(arrays[0]), .995))
    for ax, arr, title in zip(axes.ravel(), arrays, ['Truth', 'VE-equivalent noisy input']+cfg['arms']):
        im = ax.imshow(arr, cmap='RdBu_r', vmin=-limit, vmax=limit); ax.set_title(title, fontsize=9); ax.set_axis_off()
    fig.colorbar(im, ax=axes.ravel().tolist(), label='Normalized fine residual')
    fig.suptitle('Same transfer slice, noise ratio .05; visual appearance does not replace spectral gates')
    fig.savefig(args.output/'transfer_slice.png', dpi=150); plt.close(fig)
    print('MULTINOISE RESPONSE COMPLETE', flush=True)


if __name__ == '__main__':
    main()
