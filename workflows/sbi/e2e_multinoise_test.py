"""Registered bounded six-arm joint-noise test; not production E2E training."""
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
from workflows.sbi.e2e_wide_denoising_audit import Bands, TRAIN384
from workflows.sbi.e2e_wide_research_canary import preflight, NORMALIZATION
from workflows.sbi.e2e_fine_learning_test import evaluate, aggregate, gates, draws
from workflows.sbi.e2e_multinoise_models import build, loss_for

CONFIG = p.REPO/'configs/e2e_multinoise_20260915.json'
PANEL = p.REPO/'configs/e2e_fine_learning_20260915.json'


def exposure(update, cfg):
    # Blocks of three give all three phases every bin, avoiding phase/bin alias.
    gen = torch.Generator().manual_seed(cfg['train_seed']+update)
    u, drop = torch.rand(2, generator=gen).tolist()
    if (update+1) % cfg['endpoint_every'] == 0:
        return 1., None, drop < cfg['condition_dropout']
    low, high = cfg['noise_bins'][(update//3) % len(cfg['noise_bins'])]
    ratio = math.exp(math.log(low)+u*math.log(high/low))
    return 2*math.atan(ratio)/math.pi, ratio, drop < cfg['condition_dropout']


def make_model(arm, c, cfg, device):
    torch.manual_seed(cfg['model_seed'])
    return build(arm, p.build_model(c, 'fine', device), cfg['tau']).to(device)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    device = p.runtime(); c, _, _, _ = preflight()
    cfg = json.loads(CONFIG.read_text()); panel = json.loads(PANEL.read_text())
    if cfg['updates'] != 3072 or len(cfg['arms']) != 6 or cfg['heldout_access'] or cfg['full_e2e_training'] or cfg['automatic_extension']:
        raise ValueError('unsupported registered contract')
    ds = p.dataset_for(c, NORMALIZATION)
    binding = checked_binding(TRAIN384, p.provenance(c, ds))
    parent_path = TRAIN384/'diffusion_fine/step_000384.pt'
    parent = p.load_checkpoint(parent_path, binding, 'fine', 'diffusion')
    out = p.output_path(c, args.output); out.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), CONFIG, PANEL, p.REPO/'workflows/sbi/e2e_multinoise_models.py',
               p.REPO/'workflows/sbi/e2e_fine_learning_test.py', p.REPO/'workflows/sbi/e2e_wide_denoising_audit.py']
    registration = dict(config=cfg, panel=panel, source_sha256={str(x.relative_to(p.REPO)): p.sha256(x) for x in sources},
                        parent_sha256=p.sha256(parent_path), parent_binding_sha256=p.digest(binding),
                        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=p.REPO, text=True).strip(),
                        job_id=os.environ['SLURM_JOB_ID'], node=socket.gethostname(), heldout_payloads_read=False,
                        training_ready=False, full_e2e_training=False, oracle_coarse_sampling=True)
    p.write_json(out/'MULTINOISE_STARTED.json', registration)
    started = time.monotonic()
    def check_cap():
        if time.monotonic()-started > cfg['maximum_work_seconds']:
            raise TimeoutError('registered work cap reached; no automatic extension')
    items = {}
    for group in ('fit', 'transfer'):
        for anchor in panel[group+'_anchors']:
            item = ds[next(i for i, r in enumerate(ds.rows) if r['anchor_id'] == anchor)]
            up = p.coarse_to_fine(ds.inverse_target(item['coarse_target'][0], 'coarse'))
            items[anchor] = dict(target=p.tensor(item['fine_target'], device), condition=p.tensor(item['fine_condition'], device),
                                 wide=p.tensor(item['coarse_condition'], device), target_np=item['fine_target'][0],
                                 scale=ds.normalization['targets']['fine']['std'], group=group, up_coarse=up,
                                 truth=up+ds.inverse_target(item['fine_target'][0], 'fine'))
    # Representative full-size forward/backward on every architecture BEFORE any fit.
    # Discard smoke weights and rebuild identical fresh initialization for training.
    smoke = []; item = items[panel['fit_anchors'][0]]
    for arm in cfg['arms']:
        check_cap(); model = make_model(arm, c, cfg, device); torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize(); stamp = time.monotonic()
        for ratio in (.05, 20.):
            noise = torch.randn(item['target'].shape, device=device, generator=torch.Generator(device=device).manual_seed(91533))
            model.zero_grad(set_to_none=True)
            loss, _, _ = loss_for(model, item['target'], noise, 2*math.atan(ratio)/math.pi, item['condition'], item['wide'])
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        torch.cuda.synchronize()
        smoke.append(dict(arm=arm, seconds=time.monotonic()-stamp, peak_bytes=torch.cuda.max_memory_allocated(),
                          parameters=sum(x.numel() for x in model.parameters()), finite=True))
        print('SMOKE', json.dumps(smoke[-1]), flush=True); del model
    p.write_json(out/'SMOKE.json', smoke)
    bands = Bands(96, 3.383, [0, .08, .16, .32, np.inf])
    model = p.build_model(c, 'fine', device); model.load_state_dict(parent['model'])
    baseline = evaluate(model, items, 'diffusion', panel, device, bands)
    fields = draws(model, items, ds, 'diffusion', 'parent384', c, panel, out, device, bands)
    p.write_json(out/'BASELINE.json', dict(probes=baseline, fields=fields)); del model
    results = []; checkpoints = {}
    for arm in cfg['arms']:
        branch_start = time.monotonic(); branch = out/arm; branch.mkdir()
        model = make_model(arm, c, cfg, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
        experiment_binding = {**binding, 'multinoise': dict(registration=registration, arm=arm)}
        curve = []; training = []
        for update in range(cfg['updates']+1):
            check_cap()
            if update in cfg['evaluate_at']:
                probes = evaluate(model, items, 'diffusion', panel, device, bands)
                curve.append(dict(update=update, probes=probes, summary=aggregate(probes), gate=gates(baseline, probes, panel)))
                p.write_json(branch/f'probe_{update:04d}.json', curve[-1])
                print('PROBE', arm, update, json.dumps(curve[-1]['gate']), flush=True)
            if update == cfg['updates']:
                break
            anchor = panel['fit_anchors'][update % 3]; item = items[anchor]
            noise_seed = p.seed_for(cfg['train_seed'], anchor, update, 'noise')
            generator = torch.Generator(device=device).manual_seed(noise_seed)
            noise = torch.randn(item['target'].shape, device=device, generator=generator)
            t, ratio, drop = exposure(update, cfg)
            clean_weight = cfg['clean_weight'] if arm == 'unet_film_clean' and ratio is not None and ratio <= cfg['clean_max_ratio'] else 0.
            dropped = arm == 'unet_film_drop' and drop
            model.train(); optimizer.zero_grad(set_to_none=True)
            loss, primary, regularizer = loss_for(model, item['target'], noise, t, item['condition'], item['wide'], clean_weight, dropped)
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['clip'], error_if_nonfinite=True)
            optimizer.step()
            training.append(dict(update=update+1, anchor_id=anchor, noise_seed=noise_seed, ratio=ratio, time=t,
                                 context_dropped=dropped, clean_weight=clean_weight, loss=float(loss.detach()),
                                 primary_loss=float(primary), clean_loss=float(regularizer), gradient_norm_before_clip=float(norm)))
            if (update+1) % 128 == 0:
                print('TRAIN', arm, update+1, 'recent_loss', float(np.mean([r['loss'] for r in training[-128:]])), flush=True)
            if update+1 in cfg['evaluate_at']:
                path = branch/f'update_{update+1:06d}.pt'
                p.save_checkpoint(path, model=model, optimizer=optimizer, generator=generator, binding=experiment_binding,
                                  stage='fine', method='diffusion', step=update+1, history=training)
                checkpoints[str(path.relative_to(out))] = p.sha256(path)
        check_cap()
        intermediate_panel = copy.deepcopy(panel); intermediate_panel['ratios'] = cfg['intermediate_ratios']
        intermediate = evaluate(model, items, 'diffusion', intermediate_panel, device, bands)
        fields.extend(draws(model, items, ds, 'diffusion', arm, c, panel, out, device, bands))
        result = dict(arm=arm, curve=curve, training=training, intermediate=intermediate, gate=curve[-1]['gate'],
                      parameters=sum(x.numel() for x in model.parameters()), elapsed_seconds=time.monotonic()-branch_start)
        results.append(result); p.write_json(branch/'BRANCH_COMPLETE.json', result)
        print('BRANCH COMPLETE', arm, result['elapsed_seconds'], flush=True); del model, optimizer
    preflight()
    if checked_binding(TRAIN384, p.provenance(c, ds)) != binding or p.sha256(parent_path) != registration['parent_sha256']:
        raise ValueError('parent provenance drift')
    if any(p.sha256(p.REPO/k) != v for k, v in registration['source_sha256'].items()):
        raise ValueError('experiment source drift')
    if len(results) != 6 or len(fields) != 42 or len(checkpoints) != 18:
        raise ValueError('incomplete registered test')
    p.write_json(out/'MULTINOISE_COMPLETE.json', dict(registration=registration, smoke=smoke, baseline=baseline,
                 results=results, fields=fields, checkpoints=checkpoints, elapsed_seconds=time.monotonic()-started,
                 complete=True, training_ready=False, calibration_pass=None))
    print('MULTINOISE COMPLETE', flush=True)


if __name__ == '__main__':
    main()
