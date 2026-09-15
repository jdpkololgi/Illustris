"""Adaptive, separately approved paired skip-only follow-on; original run frozen."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import time
import numpy as np
import torch
from workflows.sbi import e2e_wide_pipeline as p
from workflows.sbi.e2e_wide_continue import checked_binding
from workflows.sbi.e2e_wide_denoising_audit import Bands, TRAIN384, clean_estimate
from workflows.sbi.e2e_wide_research_canary import preflight, NORMALIZATION
from workflows.sbi.e2e_fine_learning_test import evaluate, aggregate, gates, draws
from workflows.sbi.e2e_multinoise_models import BoundedResidual, coefficients, loss_for
from workflows.sbi.e2e_multinoise_test import make_model, exposure, PANEL
from workflows.sbi.e2e_multinoise_report import verify
from workflows.sbi.e2e_fixed_noise_response import decomposition

CONFIG = p.REPO/'configs/e2e_skip_path_20260915.json'
ORIGINAL = Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v2/wide_pipeline_v1/multinoise_20260915_58368502/MULTINOISE_COMPLETE.json')


class NearIdentityResidual(BoundedResidual):
    """Only change the fixed skip; identical network, raw scaling and v-MSE.

    D=a*(1+b²)*x+(b/d)*r. At clean input x=a*y and zero head, D=(1-b^4)*y,
    versus (1-b²)*y for the original bounded chart. This matches x/a through
    order b² near clean without division by a/b or a large high-noise skip.
    The extra fixed v term -a*b*x has coefficient bounded by 1/2.
    It changes the initial predictor and raw target, not the success criteria.
    """
    def forward(self, state, time, condition, wide_condition=None, context_present=None):
        a, b, _ = coefficients(time, self.tau)
        return -a*b*state+super().forward(state, time, condition, wide_condition, context_present)


def make_skip(arm, c, cfg, device):
    original = make_model(arm, c, cfg, device)
    return NearIdentityResidual(original.net, original.tau).to(device)


@torch.no_grad()
def response(model, items, panel, bands):
    model.eval(); rows = []
    for ratio in (.05, .2):
        a = 1/math.sqrt(1+ratio**2); b = ratio*a; t = 2*math.atan(ratio)/math.pi
        for anchor, item in items.items():
            y = item['target']; seed = p.seed_for(panel['seed'], anchor, 'evaluation-0', 'fine')
            noise = torch.randn(y.shape, device=y.device, generator=torch.Generator(device=y.device).manual_seed(seed))
            estimates = []
            for x in (a*y+b*noise, a*y-b*noise, a*y):
                v = model(x, y.new_tensor([t]), item['condition'], wide_condition=item['wide'])
                estimates.append(clean_estimate('diffusion', x, v, t)[0, 0].cpu().numpy())
            errors = [(z-item['target_np'])*item['scale'] for z in estimates]
            metrics = decomposition(bands, *errors)
            rows.append(dict(anchor_id=anchor, group=item['group'], ratio=ratio, seed=seed, metrics=metrics))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args(); device = p.runtime(); cfg = json.loads(CONFIG.read_text()); old = json.loads(ORIGINAL.read_text())
    if cfg['arms'] != ['unet_film', 'transformer', 'wavelet'] or cfg['updates'] != 3072 or cfg['maximum_work_seconds'] != 900:
        raise ValueError('unsupported bounded follow-on')
    verify(old, ORIGINAL.parent)
    # The previously used schedule and optimizer must be bit-for-bit unchanged.
    for key in ('updates', 'evaluate_at', 'noise_bins', 'endpoint_every', 'intermediate_ratios', 'train_seed',
                'model_seed', 'learning_rate', 'weight_decay', 'clip', 'tau', 'condition_dropout'):
        if cfg[key] != old['registration']['config'][key]:
            raise ValueError('not a skip-only contrast: '+key)
    c, _, _, _ = preflight(); ds = p.dataset_for(c, NORMALIZATION); panel = json.loads(PANEL.read_text())
    binding = checked_binding(TRAIN384, p.provenance(c, ds)); baseline = old['baseline']
    if p.digest(binding) != old['registration']['parent_binding_sha256']:
        raise ValueError('parent binding mismatch')
    out = p.output_path(c, args.output); out.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), CONFIG, PANEL, p.REPO/'workflows/sbi/e2e_multinoise_models.py',
               p.REPO/'workflows/sbi/e2e_multinoise_test.py', p.REPO/'workflows/sbi/e2e_fine_learning_test.py',
               p.REPO/'workflows/sbi/e2e_fixed_noise_response.py', p.REPO/'workflows/sbi/e2e_multinoise_report.py']
    registration = dict(config=cfg, panel=panel, original_path=str(ORIGINAL), original_sha256=p.sha256(ORIGINAL),
        source_sha256={str(x.relative_to(p.REPO)): p.sha256(x) for x in sources},
        parent_sha256=old['registration']['parent_sha256'], parent_binding_sha256=p.digest(binding),
        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=p.REPO, text=True).strip(),
        job_id=os.environ['SLURM_JOB_ID'], node=socket.gethostname(), adaptive_follow_on=True,
        heldout_payloads_read=False, training_ready=False, full_e2e_training=False)
    p.write_json(out/'SKIP_STARTED.json', registration); started = time.monotonic()
    def cap():
        if time.monotonic()-started > cfg['maximum_work_seconds']:
            raise TimeoutError('approved 15-minute work cap reached; no automatic continuation')
    items = {}
    for group in ('fit', 'transfer'):
        for anchor in panel[group+'_anchors']:
            item = ds[next(i for i, r in enumerate(ds.rows) if r['anchor_id'] == anchor)]
            up = p.coarse_to_fine(ds.inverse_target(item['coarse_target'][0], 'coarse'))
            items[anchor] = dict(target=p.tensor(item['fine_target'], device), condition=p.tensor(item['fine_condition'], device),
                wide=p.tensor(item['coarse_condition'], device), target_np=item['fine_target'][0], group=group,
                scale=ds.normalization['targets']['fine']['std'], up_coarse=up,
                truth=up+ds.inverse_target(item['fine_target'][0], 'fine'))
    bands = Bands(96, 3.383, [0, .08, .16, .32, np.inf]); results = []; fields = []; checkpoints = {}
    for arm in cfg['arms']:
        cap(); branch_started = time.monotonic(); branch = out/arm; branch.mkdir()
        model = make_skip(arm, c, cfg, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
        experiment_binding = {**binding, 'skip_path': dict(registration=registration, arm=arm)}
        item = items[panel['fit_anchors'][0]]
        loss, _, _ = loss_for(model, item['target'], torch.zeros_like(item['target']), .03, item['condition'], item['wide'])
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.zero_grad(set_to_none=True)  # no smoke optimizer update
        curve = []; training = []
        for update in range(cfg['updates']+1):
            cap()
            if update in cfg['evaluate_at']:
                probes = evaluate(model, items, 'diffusion', panel, device, bands)
                curve.append(dict(update=update, probes=probes, summary=aggregate(probes), gate=gates(baseline, probes, panel)))
                p.write_json(branch/f'probe_{update:04d}.json', curve[-1])
                print('SKIP PROBE', arm, update, json.dumps(curve[-1]['gate']), flush=True)
            if update == cfg['updates']:
                break
            anchor = panel['fit_anchors'][update % 3]; item = items[anchor]
            seed = p.seed_for(cfg['train_seed'], anchor, update, 'noise'); generator = torch.Generator(device=device).manual_seed(seed)
            noise = torch.randn(item['target'].shape, device=device, generator=generator)
            t, ratio, _ = exposure(update, cfg)
            model.train(); optimizer.zero_grad(set_to_none=True)
            loss, primary, _ = loss_for(model, item['target'], noise, t, item['condition'], item['wide'])
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite skip loss')
            loss.backward(); norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['clip'], error_if_nonfinite=True)
            optimizer.step()
            training.append(dict(update=update+1, anchor_id=anchor, noise_seed=seed, ratio=ratio, time=t,
                                 loss=float(loss.detach()), primary_loss=float(primary), gradient_norm_before_clip=float(norm)))
            if (update+1) % 128 == 0:
                print('SKIP TRAIN', arm, update+1, float(np.mean([r['loss'] for r in training[-128:]])), flush=True)
            if update+1 in cfg['evaluate_at']:
                path = branch/f'update_{update+1:06d}.pt'
                p.save_checkpoint(path, model=model, optimizer=optimizer, generator=generator, binding=experiment_binding,
                                  stage='fine', method='diffusion', step=update+1, history=training)
                checkpoints[str(path.relative_to(out))] = p.sha256(path)
        cap(); middle = copy.deepcopy(panel); middle['ratios'] = cfg['intermediate_ratios']
        intermediate = evaluate(model, items, 'diffusion', middle, device, bands)
        fields.extend(draws(model, items, ds, 'diffusion', arm, c, panel, out, device, bands))
        responses = response(model, items, panel, bands)
        result = dict(arm=arm, curve=curve, training=training, intermediate=intermediate, response=responses,
                      parameters=sum(x.numel() for x in model.parameters()), gate=curve[-1]['gate'],
                      elapsed_seconds=time.monotonic()-branch_started)
        results.append(result); p.write_json(branch/'BRANCH_COMPLETE.json', result)
        print('SKIP BRANCH COMPLETE', arm, result['elapsed_seconds'], flush=True); del model, optimizer
    preflight()
    if any(p.sha256(p.REPO/k) != v for k, v in registration['source_sha256'].items()) or p.sha256(ORIGINAL) != registration['original_sha256']:
        raise ValueError('follow-on provenance drift')
    if p.sha256(TRAIN384/'diffusion_fine/step_000384.pt') != registration['parent_sha256']:
        raise ValueError('parent drift')
    if len(results) != 3 or len(fields) != 18 or len(checkpoints) != 9:
        raise ValueError('incomplete skip comparison')
    p.write_json(out/'SKIP_COMPLETE.json', dict(registration=registration, baseline=baseline, results=results,
                 fields=fields, checkpoints=checkpoints, complete=True, elapsed_seconds=time.monotonic()-started,
                 training_ready=False, calibration_pass=None, reused_baseline_probes=60, response_forward_calls=108))
    print('SKIP COMPARISON COMPLETE', flush=True)


if __name__ == '__main__':
    main()
