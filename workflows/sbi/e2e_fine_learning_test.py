"""Bounded fine-stage noise-exposure ablation from frozen 384-update parents."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import time

import h5py
import numpy as np
import torch

from workflows.sbi import e2e_wide_pipeline as p
from workflows.sbi.e2e_wide_continue import checked_binding, equal_state
from workflows.sbi.e2e_wide_denoising_audit import Bands, bridge, clean_estimate, TRAIN384
from workflows.sbi.e2e_wide_research_canary import preflight, NORMALIZATION

CONFIG = p.REPO/'configs/e2e_fine_learning_20260915.json'


def coefficients(method, t):
    if method == 'cfm':
        return t, 1-t
    if method == 'diffusion':
        return math.cos(t*math.pi/2), math.sin(t*math.pi/2)
    raise ValueError('unknown method')


def training_time(method, arm, step, seed, ratios):
    if arm == 'uniform_time':
        return float(torch.rand((), generator=torch.Generator().manual_seed(seed)))
    if arm not in ('balanced_noise', 'near_clean'):
        raise ValueError('unknown arm')
    levels = ratios if arm == 'balanced_noise' else ratios[:2]
    return bridge(method, levels[(step//3) % len(levels)])[0]


def prediction_loss(model, target, noise, t, condition, wide, method):
    alpha, sigma = coefficients(method, t)
    state = alpha*target + sigma*noise
    exact = target-noise if method == 'cfm' else alpha*noise-sigma*target
    pred = model(state, target.new_tensor([t]), condition, wide_condition=wide)
    return torch.mean((pred-exact)**2)


def aggregate(rows):
    result = {}
    for group in ('fit', 'transfer'):
        for ratio in (.05, .2, 1., 5., 20.):
            chosen = [r for r in rows if r['group'] == group and r['ratio'] == ratio]
            result[f'{group}/{ratio}'] = {
                k: np.median([r['metrics'][k] for r in chosen], axis=0).tolist()
                for k in ('gain', 'error_power', 'error_over_truth_power', 'noise_amplitude', 'power_ratio')}
            result[f'{group}/{ratio}']['velocity_mse'] = float(np.median([r['velocity_mse'] for r in chosen]))
    return result


@torch.no_grad()
def evaluate(model, items, method, cfg, device, bands):
    model.eval(); rows = []
    for anchor, item in items.items():
        target, condition, wide = item['target'], item['condition'], item['wide']
        for rep in range(cfg['evaluation_noise_replicates']):
            seed = p.seed_for(cfg['seed'], anchor, f'evaluation-{rep}', 'fine')
            noise = torch.randn(target.shape, device=device, generator=torch.Generator(device=device).manual_seed(seed))
            for ratio in cfg['ratios']:
                t, alpha, sigma = bridge(method, ratio)
                state = alpha*target + sigma*noise
                exact = target-noise if method == 'cfm' else alpha*noise-sigma*target
                pred = model(state, target.new_tensor([t]), condition, wide_condition=wide)
                clean = clean_estimate(method, state, pred, t)
                error = clean-target
                expected = (sigma if method == 'cfm' else -sigma)*(pred-exact)
                torch.testing.assert_close(error, expected, atol=3e-5, rtol=3e-5)
                scale = item['scale']
                metric = bands.compare(clean[0,0].cpu().numpy()*scale, item['target_np']*scale,
                                       noise[0,0].cpu().numpy()*scale)
                metric['noise_amplitude'] = (np.array(metric['noise_leakage_coefficient'])/sigma).tolist()
                rows.append(dict(anchor_id=anchor, phase=anchor[:5], group=item['group'], ratio=ratio,
                                 rep=rep, seed=seed, velocity_mse=float(torch.mean((pred-exact)**2)), metrics=metric))
    return rows


@torch.no_grad()
def draws(model, items, ds, method, label, c, cfg, out, device, bands):
    records = []
    for anchor, item in items.items():
        seed = p.seed_for(cfg['seed'], anchor, 'diagnostic-draw', 'fine')
        z = p.sample_field(model, item['condition'], method=method,
                           steps=c['sampling']['cfm_steps' if method == 'cfm' else 'diffusion_steps'],
                           generator=torch.Generator(device=device).manual_seed(seed),
                           solver='heun', wide_condition=item['wide'])
        fine = ds.inverse_target(z[0,0].cpu().numpy(), 'fine')
        delta = item['up_coarse'] + fine
        path = out/f'{method}_{label}_{anchor}_true_coarse.h5'
        with h5py.File(path, 'x') as f:
            f.attrs['oracle_diagnostic_only'] = True
            f.create_dataset('fine_residual', data=fine)
            f.create_dataset('delta_local96', data=delta)
        records.append(dict(anchor_id=anchor, group=item['group'], method=method, label=label,
                            sample_sha256=p.sha256(path), seed=seed,
                            metrics=bands.compare(delta, item['truth']),
                            density_below_minus_one_fraction=float(np.mean(delta < -1))))
    return records


def gates(parent, final, cfg):
    result = {}
    g = cfg['gate']
    # Require each phase separately to pass, with medians over the two noise seeds.
    for group in ('fit', 'transfer'):
        checks = []
        for phase in ('ph000', 'ph002', 'ph003'):
            for ratio in g['near_clean_ratios']:
                old = [r for r in parent if r['group'] == group and r['phase'] == phase and r['ratio'] == ratio]
                new = [r for r in final if r['group'] == group and r['phase'] == phase and r['ratio'] == ratio]
                gain = np.median([r['metrics']['gain'][:3] for r in new], axis=0)
                leakage = float(np.median([abs(r['metrics']['noise_amplitude'][-1]) for r in new]))
                error_ratio = float(np.median([n['metrics']['error_power'][-1]/o['metrics']['error_power'][-1]
                                               for o,n in zip(old,new)]))
                passed = bool(leakage <= g['max_abs_noise_amplitude'] and error_ratio <= g['max_error_vs_parent']
                              and np.all(gain >= g['signal_gain_min']) and np.all(gain <= g['signal_gain_max']))
                checks.append(dict(phase=phase, ratio=ratio, noise_amplitude=leakage,
                                   error_vs_parent=error_ratio, gain=gain.tolist(), passed=passed))
        result[group] = dict(passed=all(r['passed'] for r in checks), checks=checks)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    device = p.runtime(); c,_,_,_ = preflight()
    cfg = json.loads(CONFIG.read_text())
    if cfg['additional_updates'] != 256 or cfg['heldout_access'] or cfg['automatic_extension']:
        raise ValueError('unsupported diagnostic contract')
    ds = p.dataset_for(c, NORMALIZATION)
    base = p.provenance(c, ds); parent_binding = checked_binding(TRAIN384, base)
    out = p.output_path(c, args.output); out.mkdir(parents=True, exist_ok=False)
    source = {str(path.relative_to(p.REPO)): p.sha256(path) for path in
              (Path(__file__), CONFIG, p.REPO/'workflows/sbi/e2e_wide_denoising_audit.py')}
    registration = dict(config=cfg, source_sha256=source, parent_binding_sha256=p.digest(parent_binding),
                        job_id=os.environ['SLURM_JOB_ID'], node=socket.gethostname(),
                        git_head=subprocess.check_output(['git','rev-parse','HEAD'], cwd=p.REPO, text=True).strip(),
                        optimizer='inherited AdamW state and unchanged hyperparameters for every arm',
                        heldout_payloads_read=False, coarse_training=False, training_ready=False)
    p.write_json(out/'LEARNING_STARTED.json', registration)
    started = time.monotonic(); items = {}
    for group in ('fit', 'transfer'):
        for anchor in cfg[group+'_anchors']:
            index = next(i for i,r in enumerate(ds.rows) if r['anchor_id'] == anchor)
            item = ds[index]
            fine = ds.inverse_target(item['fine_target'][0], 'fine')
            up = p.coarse_to_fine(ds.inverse_target(item['coarse_target'][0], 'coarse'))
            items[anchor] = dict(target=p.tensor(item['fine_target'], device),
                                 condition=p.tensor(item['fine_condition'], device),
                                 wide=p.tensor(item['coarse_condition'], device),
                                 target_np=item['fine_target'][0], scale=ds.normalization['targets']['fine']['std'],
                                 up_coarse=up, truth=up+fine, group=group)
    bands = Bands(96, 3.383, [0,.08,.16,.32,np.inf])
    results = []; field_records = []; parents = {}; checkpoint_hashes = {}
    for method in cfg['methods']:
        path = TRAIN384/f'{method}_fine/step_000384.pt'
        state = p.load_checkpoint(path, parent_binding, 'fine', method)
        parents[method] = p.sha256(path)
        baseline = None
        for arm in cfg['arms']:
            model = p.build_model(c, 'fine', device)
            model.load_state_dict(state['model'])
            optimizer = torch.optim.AdamW(model.parameters(), lr=c['training']['learning_rate'],
                                           weight_decay=c['training']['weight_decay'])
            optimizer.load_state_dict(copy.deepcopy(state['optimizer']))
            if not equal_state(model.state_dict(), state['model']) or not equal_state(optimizer.state_dict(), state['optimizer']):
                raise ValueError('branch did not start from identical parent model/optimizer')
            binding = {**parent_binding, 'learning_test': dict(registration=registration, method=method, arm=arm,
                                                               parent_checkpoint_sha256=parents[method])}
            branch = out/f'{method}_{arm}'; branch.mkdir()
            history = copy.deepcopy(state['history']); curve = []; train_records = []
            for update in range(cfg['additional_updates']+1):
                if time.monotonic()-started > cfg['maximum_work_seconds']:
                    raise TimeoutError('bounded learning-test work budget reached; no automatic retry')
                if update in cfg['evaluate_at']:
                    probes = evaluate(model, items, method, cfg, device, bands)
                    if update == 0:
                        if baseline is None:
                            baseline = probes
                            field_records.extend(draws(model, items, ds, method, 'parent', c, cfg, out, device, bands))
                        elif probes != baseline:
                            raise ValueError('nonidentical paired baseline')
                    curve.append(dict(update=update, summary=aggregate(probes), probes=probes))
                    p.write_json(branch/f'probe_{update:03d}.json', curve[-1])
                    print(f'LEARNING {method} {arm} update={update} '
                          f'near_clean_noise={curve[-1]["summary"]["fit/0.05"]["noise_amplitude"][-1]:.6f}', flush=True)
                if update == cfg['additional_updates']:
                    field_records.extend(draws(model, items, ds, method, arm, c, cfg, out, device, bands))
                    break
                item = items[cfg['fit_anchors'][update % 3]]
                anchor = cfg['fit_anchors'][update % 3]
                seed = p.seed_for(cfg['seed'], anchor, f'train-{update}', 'fine')
                generator = torch.Generator(device=device).manual_seed(seed)
                noise = torch.randn(item['target'].shape, device=device, generator=generator)
                t = training_time(method, arm, update, p.seed_for(cfg['seed'], anchor, update, 'time'), cfg['ratios'])
                model.train(); optimizer.zero_grad(set_to_none=True)
                loss = prediction_loss(model, item['target'], noise, t, item['condition'], item['wide'], method)
                if not torch.isfinite(loss):
                    raise FloatingPointError('nonfinite loss')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), c['training']['clip_gradient_norm'], error_if_nonfinite=True)
                optimizer.step()
                row = dict(step=385+update, diagnostic_update=update+1, anchor_id=anchor, time=t,
                           noise_seed=seed, loss=float(loss.detach()), gradient_norm_before_clip=float(norm))
                history.append(row); train_records.append(row)
                if update+1 in cfg['evaluate_at']:
                    path = branch/f'step_{385+update:06d}.pt'
                    p.save_checkpoint(path, model=model, optimizer=optimizer, generator=generator, binding=binding,
                                      stage='fine', method=method, step=385+update, history=history)
                    checkpoint_hashes[f'{method}/{arm}/{update+1}'] = p.sha256(path)
            result = dict(method=method, arm=arm, curve=curve, training=train_records,
                          gate=gates(baseline, curve[-1]['probes'], cfg))
            p.write_json(branch/'BRANCH_COMPLETE.json', result); results.append(result)
            del model, optimizer
    preflight()
    if checked_binding(TRAIN384,p.provenance(c,ds)) != parent_binding or any(p.sha256(p.REPO/path)!=h for path,h in source.items()):
        raise ValueError('source or parent binding drift')
    if len(results)!=6 or len(field_records)!=48 or len(checkpoint_hashes)!=18:
        raise ValueError('incomplete bounded experiment')
    result = dict(registration=registration, parents=parents, results=results, fields=field_records,
                  checkpoints=checkpoint_hashes, elapsed_seconds=time.monotonic()-started,
                  complete=True, training_ready=False, calibration_pass=None)
    p.write_json(out/'LEARNING_COMPLETE.json', result)
    print('FINE LEARNING TEST COMPLETE', flush=True)


if __name__ == '__main__':
    main()
