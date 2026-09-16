"""Frozen neural solver panel and independent Adam-first-moment x clipping probes.

This module deliberately does not modify the hash-bound parent implementations.
Stage with e2e_oracle_conflict stage, then run from its frozen source directory.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import time
import numpy as np
import torch
from workflows.sbi import e2e_wide_pipeline as p
from workflows.sbi import e2e_durable as durable
from workflows.sbi import e2e_clean_limit as base
from workflows.sbi import e2e_diversity_norm as data
from workflows.sbi.e2e_oracle_conflict import verify
from workflows.sbi.e2e_oracle_solver import sample_vp_heun
from workflows.sbi.e2e_wide_models import sample_field
from workflows.sbi.e2e_wide_denoising_audit import Bands
from workflows.sbi.e2e_gradient_step_probe import assign_gradient, evaluate
from workflows.sbi.e2e_loss_conflict import components, gradient_vectors, remove_conflicting_component
from workflows.sbi.e2e_wide_continue import equal_state

CONFIG = p.REPO / 'configs/e2e_frozen_controls_v1.json'


def panel(selection, phases):
    """First registered field per phase/group; no outcome-driven selection."""
    result = []
    for group in ('train', 'transfer'):
        for phase in phases:
            rows = [r for r in selection[group] if r['phase'] == phase]
            if not rows:
                raise ValueError('missing registered phase/group')
            result.append(dict(rows[0], panel_group=group))
    return result


def zero_first_moment(optimizer):
    """Ablate only the historical first moment, NOT variance or bias-correction age."""
    count = 0
    for state in optimizer.state.values():
        if 'exp_avg' not in state or 'exp_avg_sq' not in state or 'step' not in state:
            raise ValueError('initialized Adam state required')
        state['exp_avg'].zero_()
        count += 1
    if not count:
        raise ValueError('empty Adam state')


def apply_step(optimizer, parameters, gradient, clip):
    optimizer.zero_grad(set_to_none=True)
    assign_gradient(parameters, gradient)
    limit = float('inf') if clip is None else clip
    norm = torch.nn.utils.clip_grad_norm_(parameters, limit, error_if_nonfinite=True)
    scale = min(1., limit / (float(norm) + 1e-6))
    optimizer.step()
    return dict(gradient_norm=float(norm), clipping_scale=scale)


def relative_rms(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean((a-b)**2) / max(np.mean(b*b), 1e-30)))


def load_data(manifest, device):
    old = Path(manifest['spec']['parent_root'])
    previous = json.loads((old/'MANIFEST.json').read_text())['spec']
    clean = Path(previous['parent_root'])
    if p.sha256(clean/'MANIFEST.json') != previous['parent_manifest_sha256']:
        raise ValueError('data contract drift')
    spec = json.loads((clean/'MANIFEST.json').read_text())['spec']
    raw = Path(spec['parent_root'])
    if p.sha256(raw/'PREPARED.json') != spec['prepared_sha256']:
        raise ValueError('prepared drift')
    return data.load_items(raw, device)


def load_parent(manifest, prepared, seed, device):
    parent = manifest['parents'][f'{seed}/control']
    branch = Path(manifest['spec']['parent_root']) / f'replica_{seed}/control'
    if p.sha256(branch/'COMPLETE.json') != parent['complete_sha256']:
        raise ValueError('parent receipt drift')
    state, pointer = durable.load(branch, parent['binding'])
    if pointer != parent['checkpoint']:
        raise ValueError('checkpoint drift')
    cfg, model, opt, gen = base.make(prepared, seed, device)
    base.restore(state, model, opt, gen)
    return cfg, model, opt, gen, state


def check_restored(model, opt, state):
    if not equal_state(model.state_dict(), state['model']) or not equal_state(opt.state_dict(), state['optimizer']):
        raise ValueError('checkpoint state changed')


@torch.no_grad()
def solver_panel(root, manifest, cfg, prepared, items, device):
    folder = root/'neural_sampler'
    folder.mkdir(exist_ok=True)
    rows = panel(prepared['selection'], cfg['phases'])
    fine = prepared['original_normalization']['targets']['fine']
    coarse = prepared['original_normalization']['targets']['coarse']
    bands = Bands(96, 3.383, [0, .08, .16, .32, np.inf])
    results = []
    for seed in cfg['replicas']:
        _, model, opt, gen, state = load_parent(manifest, prepared, seed, device)
        for row in rows:
            anchor = row['anchor_id']; item = items[anchor]
            # Last conditioning channel is normalized interpolated TRUE coarse.
            coarse_local = item['condition'][0,-1].cpu().numpy()*coarse['std']+coarse['mean']
            truth = item['target_np']*fine['std']+fine['mean']
            for draw in range(cfg['draws']):
                name = f'seed{seed}_{anchor}_draw{draw}'
                receipt = folder/f'{name}.json'
                if receipt.exists():
                    saved = json.loads(receipt.read_text())
                    if saved['manifest_sha256'] != p.sha256(root/'MANIFEST.json') or p.sha256(folder/saved['draw_file']) != saved['draw_sha256']:
                        raise ValueError('resumed draw drift')
                    results.append(saved); continue
                noise_seed = p.seed_for(916330, anchor, draw, 'frozen-neural-sampler')
                arrays = {}; timings = {}
                for method, nfe in cfg['samplers']:
                    generator = torch.Generator(device=device).manual_seed(noise_seed)
                    start = time.monotonic()
                    if method == 'ddim':
                        x = sample_field(model, item['condition'], 'diffusion', nfe, generator, wide_condition=item['wide'])
                    else:
                        x = sample_vp_heun(model, item['condition'], nfe//2, generator, wide_condition=item['wide'])
                    if not torch.isfinite(x).all():
                        raise FloatingPointError('nonfinite sampler output')
                    key = f'{method}_{nfe}'
                    arrays[key] = x[0,0].cpu().numpy()
                    timings[key] = time.monotonic()-start
                reference = arrays['heun_512']
                refinement = relative_rms(arrays['heun_256'], reference)
                metrics = {}
                for key, normalized in arrays.items():
                    residual = normalized*fine['std']+fine['mean']
                    density = residual+coarse_local
                    metrics[key] = dict(relative_rms_to_reference=relative_rms(normalized, reference),
                        residual_spectrum=bands.compare(residual, truth),
                        density_mean=float(density.mean()), density_std=float(density.std()),
                        density_below_minus_one=float((density < -1).mean()),
                        density_quantiles=np.quantile(density,[.001,.01,.5,.99,.999]).tolist(),
                        seconds=timings[key])
                path = folder/f'{name}.npz'
                np.savez_compressed(path, **arrays)
                saved = dict(replica=seed, anchor_id=anchor, phase=row['phase'], group=row['panel_group'],
                    draw=draw, noise_seed=noise_seed, metrics=metrics, reference_refinement_relative_rms=refinement,
                    reference_resolved=refinement<=cfg['reference_relative_rms_tolerance'],
                    draw_file=path.name, draw_sha256=p.sha256(path), manifest_sha256=p.sha256(root/'MANIFEST.json'))
                durable.publish_json(receipt, saved); results.append(saved)
                print('NEURAL DRAW', seed, anchor, draw, 'reference refinement', refinement, flush=True)
        check_restored(model, opt, state)
        del model, opt, gen, state
    return results


def optimizer_panel(root, manifest, cfg, prepared, items, device):
    folder=root/'optimizer'; folder.mkdir(exist_ok=True)
    results=[]
    rows=[r for r in panel(prepared['selection'],cfg['phases']) if r['panel_group']=='train']
    for seed in cfg['replicas']:
        path=folder/f'seed{seed}.json'
        if path.exists():
            saved=json.loads(path.read_text())
            if saved['manifest_sha256']!=p.sha256(root/'MANIFEST.json'):raise ValueError('optimizer receipt drift')
            results.extend(saved['records']);continue
        model_cfg,model,opt,gen,state=load_parent(manifest,prepared,seed,device)
        parameters=list(model.parameters());records=[]
        for row in rows:
            anchor=row['anchor_id'];item=items[anchor]
            eval_seed=p.seed_for(916225,anchor,0,'step-evaluation')
            base.restore(state,model,opt,gen)
            before=evaluate(model,item,eval_seed)
            for q in cfg['gradient_sigmas']:
                base.restore(state,model,opt,gen);model.eval()
                rng=torch.Generator(device=device).manual_seed(p.seed_for(916224,anchor,q,'step-gradient'))
                eps=torch.randn(item['target'].shape,device=device,generator=rng)
                aux=torch.randn(item['target'].shape,device=device,generator=rng)
                gradients=gradient_vectors(components(model,item,eps,aux,q),parameters)
                primary=gradients['denoising'];identity=gradients['identity']
                origin=torch.cat([x.detach().flatten() for x in parameters]).double()
                choices=dict(denoising_only=primary,identity_strong=primary+identity,
                             projected_identity=primary+remove_conflicting_component(primary,identity))
                for momentum in cfg['momentum']:
                    for clipping in cfg['clipping']:
                        control=None
                        for variant in cfg['variants']:
                            base.restore(state,model,opt,gen)
                            if momentum=='zero_first':zero_first_moment(opt)
                            stats=apply_step(opt,parameters,choices[variant],model_cfg['clip'] if clipping else None)
                            delta=torch.cat([x.detach().flatten() for x in parameters]).double()-origin
                            if variant=='denoising_only':control=delta.clone()
                            records.append(dict(replica=seed,anchor_id=anchor,phase=row['phase'],gradient_sigma=q,
                                momentum=momentum,clipping=clipping,variant=variant,**stats,
                                raw_primary_dot_aux=float((primary*(choices[variant]-primary)).sum()),
                                actual_primary_dot_displacement=float((primary*delta).sum()),
                                incremental_primary_dot_vs_control=float((primary*(delta-control)).sum()),
                                displacement_norm=float(delta.norm()),before=before,after=evaluate(model,item,eval_seed)))
                base.restore(state,model,opt,gen)
                del gradients,primary,identity,origin,choices,control
        check_restored(model,opt,state)
        durable.publish_json(path,dict(manifest_sha256=p.sha256(root/'MANIFEST.json'),records=records,unchanged_model_optimizer=True))
        results.extend(records);print('OPTIMIZER FACTORIAL',seed,len(records),flush=True)
        del model,opt,gen,state
    return results


def run(root, mode):
    device=p.runtime();manifest=verify(root);cfg=json.loads(CONFIG.read_text())
    if cfg['schema']!='e2e-frozen-controls-v1' or any(cfg[k] for k in ('full_e2e_training','saved_updated_weights','heldout_access')):
        raise ValueError('outside frozen control scope')
    prepared,items=load_data(manifest,device)
    with durable.single_writer(root):
        records=(solver_panel if mode=='sampler' else optimizer_panel)(root,manifest,cfg,prepared,items,device)
        verify(root)
        durable.publish_json(root/f'{mode.upper()}.json',dict(complete=True,mode=mode,records=records,
            config_sha256=p.sha256(CONFIG),manifest_sha256=p.sha256(root/'MANIFEST.json'),
            job_id=os.environ.get('SLURM_JOB_ID'),node=socket.gethostname(),unchanged_checkpoints=True,
            scope='Fine-only true-coarse conditioned solver panel; or independent one-step fitting-panel optimizer probes. No E2E training, saved updated weights, or posterior calibration claim.'))
    print('COMPLETE',mode,len(records),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--mode',choices=('sampler','optimizer'),required=True)
    args=parser.parse_args();run(args.root,args.mode)
