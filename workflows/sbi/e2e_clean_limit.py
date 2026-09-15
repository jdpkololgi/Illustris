"""Frozen fifteen-field optimization, near-zero exposure, then shift diagnosis.

No E2E sampling, no held-out phase, no automatic budget extension. All arrays
and full-size neural operations require an approved compute allocation.
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import signal
import socket
import time
import numpy as np
import torch
from workflows.sbi import e2e_wide_pipeline as p
from workflows.sbi import e2e_diversity_norm as d
from workflows.sbi import e2e_diversity_norm_report as report
from workflows.sbi import e2e_durable as durable
from workflows.sbi.e2e_wide_denoising_audit import Bands
from workflows.sbi.e2e_multinoise_models import loss_for
from workflows.sbi.e2e_multinoise_test import exposure
from workflows.sbi.e2e_skip_path_test import make_skip
from workflows.sbi.e2e_wide_continue import equal_state

CONFIG = p.REPO / 'configs/e2e_clean_limit_20260915.json'


def read_spec():
    spec = json.loads(CONFIG.read_text())
    if (spec['schema'] != 'e2e-clean-limit-v1' or spec['field_count'] != 15 or
            spec['normalization'] != 'current' or spec['replicas'] != [0, 1] or
            any(spec[k] for k in ('full_e2e_training', 'heldout_access', 'automatic_extension', 'training_ready'))):
        raise ValueError('outside registered scope')
    return spec


def schedule(update, cfg, spec, arm):
    """Only first of six bins changes; preserve field, Gaussian and endpoint draws."""
    t, ratio, drop = exposure(update, cfg)
    near = spec['near_zero']
    if arm != 'near_zero' or ratio is None or (update // 3) % 6 != near['replace_bin']:
        return t, ratio, drop
    gen = torch.Generator().manual_seed(cfg['train_seed'] + update)
    u = float(torch.rand(2, generator=gen)[0])
    if u < near['zero_probability']:
        ratio = 0.
    else:
        u = (u - near['zero_probability']) / (1 - near['zero_probability'])
        q = near['positive_uniform_fraction']
        if u < q:
            ratio = near['maximum'] * u / q
        else:
            ratio = near['log_min'] * math.exp((u-q)/(1-q) * math.log(near['maximum']/near['log_min']))
    return 2 * math.atan(ratio) / math.pi, ratio, drop


def step(model, optimizer, items, train_ids, cfg, spec, arm, update, generator):
    anchor = d.field_for(update, 15, train_ids)
    item = items[anchor]
    seed = p.seed_for(cfg['train_seed'], train_ids[update % 3], update, 'noise')
    generator.manual_seed(seed)
    noise = torch.randn(item['target'].shape, device=item['target'].device, generator=generator)
    t, ratio, _ = schedule(update, cfg, spec, arm)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    # Preserve v-MSE, including its unidentifiable random target at exactly zero.
    # D(x,0)=x structurally; a zero-noise training step is NOT a denoising success.
    loss, _, _ = loss_for(model, item['target'], noise, t, item['condition'], item['wide'])
    if not torch.isfinite(loss):
        raise FloatingPointError('nonfinite objective')
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['clip'], error_if_nonfinite=True)
    optimizer.step()
    return dict(update=update+1, anchor_id=anchor, noise_seed=seed, time=t,
                ratio=ratio, loss=float(loss.detach()), gradient_norm=float(norm))


def make(prepared, replica, device):
    cfg = copy.deepcopy(prepared['config'])
    cfg.update(cfg['replicates'][replica])
    model = d.AffineChart(make_skip('unet_film', p.read_config(), cfg, device),
                          prepared['normalizations']['current']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    generator = torch.Generator(device=device)
    return cfg, model, optimizer, generator


def restore(state, model, optimizer, generator):
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(copy.deepcopy(state['optimizer']))
    p.restore_rng(state['rng'], generator)


def verify_snapshot(root):
    manifest = json.loads((root / 'MANIFEST.json').read_text())
    if Path(manifest['source']).resolve() != p.REPO.resolve():
        raise ValueError('must launch from frozen source snapshot')
    if manifest['spec'] != read_spec():
        raise ValueError('configuration drift')
    for rel, sha in manifest['source_sha256'].items():
        if p.sha256(p.REPO / rel) != sha:
            raise ValueError('snapshot drift: ' + rel)
    return manifest


def smoke(root):
    """Full-size source/parent parity, exact Adam/RNG resume, and endpoint gradients."""
    manifest=verify_snapshot(root);spec=manifest['spec'];device=p.runtime()
    prepared,items=d.load_items(Path(spec['parent_root']),device)
    if p.sha256(Path(spec['parent_root'])/'PREPARED.json')!=spec['prepared_sha256']:
        raise ValueError('prepared drift')
    cfg,model,optimizer,generator=make(prepared,0,device)
    train_ids=[r['anchor_id'] for r in prepared['selection']['train']]
    parents=[old_parent(prepared,r,spec) for r in spec['replicas']]
    if any(sha!=manifest['parent_sha256'][str(r)] for r,(_,sha) in enumerate(parents)):
        raise ValueError('parent drift')
    state=parents[0][0];restore(state,model,optimizer,generator)
    if not equal_state(model.state_dict(),state['model']) or not equal_state(optimizer.state_dict(),state['optimizer']):
        raise ValueError('parent model/Adam load mismatch')
    if not equal_state(p.rng_state(generator),state['rng']):raise ValueError('parent RNG mismatch')
    selected={a:items[a] for a in (train_ids[0],prepared['selection']['transfer'][0]['anchor_id'])}
    before=probe(model,selected,cfg,spec,prepared,state['step'],None)
    old=json.loads((Path(spec['parent_root'])/'replica_0/n15_current/probe_3072.json').read_text())
    reference={(r['anchor_id'],r['kind'],r['ratio'],r.get('rep')):r for r in old['rows']}
    parity=0
    for row in before['rows']:
        key=(row['anchor_id'],row['kind'],row['ratio'],row.get('rep'))
        if key in reference:
            for metric in ('gain','error_power','power_ratio'):
                np.testing.assert_allclose(row['metrics'][metric],reference[key]['metrics'][metric],rtol=1e-4,atol=1e-7)
            parity+=1
    folder=root/'smoke_resume';folder.mkdir(exist_ok=False)
    binding=dict(smoke_manifest=p.sha256(root/'MANIFEST.json'))
    history=copy.deepcopy(state['history']);start=state['step']
    for u in range(start,start+4):history.append(step(model,optimizer,items,train_ids,cfg,spec,'optimization',u,generator))
    durable.save(folder,model=model,optimizer=optimizer,generator=generator,binding=binding,
                 stage='fine',method='diffusion',step=start+4,history=history)
    for u in range(start+4,start+8):step(model,optimizer,items,train_ids,cfg,spec,'optimization',u,generator)
    expected=copy.deepcopy(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),rng=p.rng_state(generator)))
    replay,_=durable.load(folder,binding)
    restore(replay,model,optimizer,generator)
    for u in range(start+4,start+8):step(model,optimizer,items,train_ids,cfg,spec,'optimization',u,generator)
    actual=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),rng=p.rng_state(generator))
    if not equal_state(expected,actual):raise ValueError('GPU uninterrupted/resume not exact')
    endpoints=[];item=items[train_ids[0]]
    for ratio in (0.,1e-5,.001,.05,None):
        optimizer.zero_grad(set_to_none=True)
        noise=torch.randn(item['target'].shape,device=device,generator=generator)
        t=1. if ratio is None else 2*math.atan(ratio)/math.pi
        loss,_,_=loss_for(model,item['target'],noise,t,item['condition'],item['wide'])
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        if not torch.isfinite(loss):raise ValueError('nonfinite endpoint loss')
        endpoints.append(dict(ratio=ratio,loss=float(loss),gradient_norm=float(norm)))
    restore(state,model,optimizer,generator)
    torch.cuda.reset_peak_memory_stats();stamp=time.monotonic()
    for u in range(start,start+64):step(model,optimizer,items,train_ids,cfg,spec,'optimization',u,generator)
    torch.cuda.synchronize();seconds=(time.monotonic()-stamp)/64
    stamp=time.monotonic();probe(model,selected,cfg,spec,prepared,start+64,None)
    probe_per_field=(time.monotonic()-stamp)/len(selected)
    # Conservative wall-time estimates include all panels and 30% headroom.
    estimates=dict(optimization=1.3*((spec['optimization_end']-start)*seconds+len(spec['optimization_probes'])*27*probe_per_field),
                   coverage=1.3*((spec['coverage_end']-spec['optimization_end'])*seconds+len(spec['coverage_probes'])*27*probe_per_field))
    if estimates['optimization']>10000 or estimates['coverage']>6500:
        raise RuntimeError('measured runtime exceeds frozen batch limits; review before submission')
    durable.publish_json(root/'SMOKE.json',dict(passed=True,manifest_sha256=p.sha256(root/'MANIFEST.json'),
        exact_gpu_resume=True,parent_probe_parity=parity,endpoints=endpoints,seconds_per_update=seconds,
        seconds_per_probe_field=probe_per_field,estimated_seconds=estimates,
        peak_gpu_bytes=torch.cuda.max_memory_allocated(),node=socket.gethostname(),job_id=os.environ['SLURM_JOB_ID']))
    print('SMOKE PASSED',seconds,probe_per_field,estimates,flush=True)


def old_parent(prepared, replica, spec):
    root = Path(spec['parent_root'])
    registration = json.loads((root / f'replica_{replica}/STARTED.json').read_text())
    binding = {**prepared['base_binding'], 'diversity_norm':
               dict(registration=registration, n=15, normalization='current')}
    path = root / f"replica_{replica}/n15_current/update_{spec['parent_update']:06d}.pt"
    return p.load_checkpoint(path, binding, 'fine', 'diffusion'), p.sha256(path)


def branch_contract(root, manifest, prepared, replica, arm):
    spec = manifest['spec']
    if arm == 'optimization':
        parent, parent_sha = old_parent(prepared, replica, spec)
        if parent_sha != manifest['parent_sha256'][str(replica)]:
            raise ValueError('parent checkpoint drift')
    else:
        parent_root = root / f'replica_{replica}/optimization'
        receipt = json.loads((parent_root / 'COMPLETE.json').read_text())
        if not receipt['complete'] or receipt['update'] != spec['optimization_end']:
            raise ValueError('optimization is not complete')
        parent, pointer = durable.load(parent_root, receipt['binding'])
        if pointer != receipt['checkpoint'] or parent['step'] != spec['optimization_end']:
            raise ValueError('optimization completion/checkpoint mismatch')
        parent_sha = pointer['sha256']
    binding = {**prepared['base_binding'], 'clean_limit': dict(
        manifest_sha256=p.sha256(root/'MANIFEST.json'), replica=replica, arm=arm, parent_sha256=parent_sha)}
    return parent, binding


def probe(model, items, cfg, spec, prepared, update, checkpoint):
    cfg = dict(cfg, clean_ratios=spec['clean_ratios'], noisy_ratios=spec['noisy_ratios'])
    scale = prepared['original_normalization']['targets']['fine']['std']
    bands = Bands(96, 3.383, [0, .08, .16, .32, np.inf])
    rows = d.evaluate(model, items, cfg, scale, bands)
    expected = len(items) * (len(cfg['clean_ratios']) + cfg['evaluation_replicates'] * len(cfg['noisy_ratios']))
    if len(rows) != expected or not report.finite_tree(rows):
        raise ValueError('invalid evaluation panel')
    return dict(update=update, rows=rows, checkpoint=checkpoint, complete=True)


def train(root, replica, arm, smoke_interrupt_at=None):
    manifest = verify_snapshot(root)
    spec = manifest['spec']
    device = p.runtime()
    if p.sha256(Path(spec['parent_root'])/'PREPARED.json') != spec['prepared_sha256']:
        raise ValueError('prepared receipt drift')
    prepared, items = d.load_items(Path(spec['parent_root']), device)
    cfg, model, optimizer, generator = make(prepared, replica, device)
    train_ids = [r['anchor_id'] for r in prepared['selection']['train']]
    parent, binding = branch_contract(root, manifest, prepared, replica, arm)
    if smoke_interrupt_at is not None and (arm!='optimization' or replica!=0 or
            not spec['parent_update']<smoke_interrupt_at<=spec['parent_update']+32):
        raise ValueError('signal smoke must be within the first 32 seed-zero continuation updates')
    end = spec['optimization_end'] if arm == 'optimization' else spec['coverage_end']
    probes = spec['optimization_probes'] if arm == 'optimization' else spec['coverage_probes']
    folder = root / f'replica_{replica}' / arm
    with durable.single_writer(folder):
        if (folder/'COMPLETE.json').exists():
            complete = json.loads((folder/'COMPLETE.json').read_text())
            state, pointer = durable.load(folder, binding)
            if complete['binding'] != binding or complete['checkpoint'] != pointer or state['step'] != end:
                raise ValueError('invalid completed branch')
            print('ALREADY COMPLETE', folder, flush=True)
            return
        state = parent
        if (folder/'LATEST.json').exists():
            state, pointer = durable.load(folder, binding)
        restore(state, model, optimizer, generator)
        history = copy.deepcopy(state['history'])
        update = state['step']
        if smoke_interrupt_at is not None and smoke_interrupt_at<=update:
            raise ValueError('signal-smoke point already reached')
        if not parent['step'] <= update <= end or len(history) != update:
            raise ValueError('invalid resume range/history')
        if not (folder/'LATEST.json').exists():
            pointer = durable.save(folder, model=model, optimizer=optimizer, generator=generator,
                binding=binding, stage='fine', method='diffusion', step=update, history=history)
        stop = []
        def requested(signum, frame):
            stop.append(signum)
        signal.signal(signal.SIGUSR1, requested)
        signal.signal(signal.SIGTERM, requested)
        started = time.monotonic()
        print('START', json.dumps(dict(replica=replica, arm=arm, update=update, end=end,
            node=socket.gethostname(), job=os.environ.get('SLURM_JOB_ID'), binding=p.digest(binding))), flush=True)
        while True:
            if stop:
                if pointer['step'] != update:
                    pointer = durable.save(folder, model=model, optimizer=optimizer, generator=generator,
                        binding=binding, stage='fine', method='diffusion', step=update, history=history)
                print('STOPPED SAFELY', update, stop, flush=True)
                raise SystemExit(75)
            if update in probes:
                path = folder / f'probe_{update:06d}.json'
                if path.exists():
                    saved = json.loads(path.read_text())
                    if saved['checkpoint'] != pointer or not saved['complete']:
                        raise ValueError('probe/checkpoint mismatch')
                else:
                    durable.publish_json(path, probe(model, items, cfg, spec, prepared, update, pointer))
                print('PROBE', replica, arm, update, flush=True)
            if update == end:
                break
            history.append(step(model, optimizer, items, train_ids, cfg, spec, arm, update, generator))
            update += 1
            if update==smoke_interrupt_at:
                os.kill(os.getpid(),signal.SIGUSR1)
            if update % spec['checkpoint_every'] == 0 or update in probes or stop:
                pointer = durable.save(folder, model=model, optimizer=optimizer, generator=generator,
                    binding=binding, stage='fine', method='diffusion', step=update, history=history)
                print('CHECKPOINT', replica, arm, update,
                      float(np.mean([r['loss'] for r in history[-512:]])), flush=True)
        verify_snapshot(root)
        durable.publish_json(folder/'COMPLETE.json', dict(complete=True, update=update, binding=binding,
            checkpoint=pointer, probes={str(u):p.sha256(folder/f'probe_{u:06d}.json') for u in probes},
            seconds=time.monotonic()-started, heldout_payloads_read=False, training_ready=False))
        print('COMPLETE', replica, arm, update, flush=True)


def resume_check(root):
    """Validate the actual interrupted runner against uninterrupted parent replay."""
    manifest=verify_snapshot(root);spec=manifest['spec'];device=p.runtime()
    prepared,items=d.load_items(Path(spec['parent_root']),device)
    parent,binding=branch_contract(root,manifest,prepared,0,'optimization')
    state,pointer=durable.load(root/'replica_0/optimization',binding)
    if state['step']!=spec['parent_update']+16:
        raise ValueError('expected two eight-update signal-interrupted steps')
    cfg,model,optimizer,generator=make(prepared,0,device)
    restore(parent,model,optimizer,generator)
    ids=[r['anchor_id'] for r in prepared['selection']['train']]
    history=copy.deepcopy(parent['history'])
    for u in range(parent['step'],state['step']):
        history.append(step(model,optimizer,items,ids,cfg,spec,'optimization',u,generator))
    if not all((equal_state(model.state_dict(),state['model']),
                equal_state(optimizer.state_dict(),state['optimizer']),
                equal_state(p.rng_state(generator),state['rng']),history==state['history'])):
        raise ValueError('signal-interrupted process replay mismatch')
    rows=json.loads((root/'replica_0/optimization/probe_003072.json').read_text())['rows']
    frozen=json.loads((Path(spec['parent_root'])/'FROZEN.json').read_text())
    baseline={(r['anchor_id'],r['ratio'],r.get('rep')):r for r in frozen['parent384'] if r['kind']=='noisy'}
    groups=report.groups(prepared,15)
    report_smoke=dict(groups={k:grouped(rows,groups[k],prepared,baseline) for k in ('exposed','transfer')},
        limit=limit_shape(rows,groups['transfer']),associations=report.associations(rows,prepared,15),
        predictors=shift_predictors(rows,prepared),paired=report.paired_effect(rows,rows,groups['transfer']))
    if not report.finite_tree(report_smoke):raise ValueError('full-panel reporting smoke failed')
    durable.publish_json(root/'REPORT_SMOKE.json',report_smoke)
    durable.publish_json(root/'RESTART_TEST.json',dict(passed=True,exact=True,update=state['step'],
        manifest_sha256=p.sha256(root/'MANIFEST.json'),checkpoint=pointer,
        expected_signal_exit=75,report_smoke_sha256=p.sha256(root/'REPORT_SMOKE.json'),
        job_id=os.environ['SLURM_JOB_ID']))
    print('SIGNAL RESTART EXACT',state['step'],flush=True)


def grouped(rows, anchors, prepared, parent):
    # Legacy physical gates are defined only at .05/.2. Other ratios are diagnostics.
    rr = [r for r in rows if r['kind']=='clean' or r['ratio'] in (.05,.2)]
    metadata = {r['anchor_id']:r for r in prepared['metadata']}
    out = report.with_metadata(rr, anchors, parent, metadata)
    for ratio in sorted({r['ratio'] for r in rows if r['kind']=='noisy'}):
        noisy = [r for r in rows if r['kind']=='noisy' and r['ratio']==ratio and r['anchor_id'] in anchors]
        out.setdefault(str(ratio), {}).update(
            diagnostic_noise_left=report.med([abs(r['metrics']['noise_amplitude'][-1]) for r in noisy]),
            diagnostic_noisy_highk_error=report.med([r['metrics']['error_power'][-1] for r in noisy]))
    return out


def shift_predictors(rows, prepared):
    """Descriptive leave-one-phase-out ridge predictions, not a causal test.

    Train-only feature scaling in each fold. No hyperparameter search. Each fold
    has eight transfer cutouts for fitting and four for evaluation; only three
    phases exist and some target footprints overlap. Report all fold residuals.
    """
    metadata = {r['anchor_id']:r for r in prepared['metadata']}
    transfer = {r['anchor_id'] for r in prepared['selection']['transfer']}
    rr = [r for r in rows if r['anchor_id'] in transfer and r['kind']=='clean' and r['ratio']==.05]
    phases = np.array([r['phase'] for r in rr])
    y = np.array([r['rms']/metadata[r['anchor_id']]['stats']['std'] for r in rr])
    sets = dict(target_moments=['mean','std'],
                target_distribution=['mean','std','skew','excess_kurtosis','q01','q99'],
                observation=['redshift','observed_fraction','mean_galaxy_count','support_mean','angular_response_mean'])
    sets['combined'] = sets['target_moments'] + sets['observation']
    result = {}
    for name, features in {'intercept':[], **sets}.items():
        x = np.array([[metadata[r['anchor_id']]['stats'][k] for k in features] for r in rr])
        predictions = np.empty(len(y))
        for phase in sorted(set(phases)):
            train = phases != phase
            if features:
                mean = x[train].mean(0)
                scale = np.maximum(x[train].std(0), 1e-8)
                xt = (x[train]-mean)/scale
                weights = np.linalg.solve(xt.T@xt + len(xt)*np.eye(len(features)), xt.T@(y[train]-y[train].mean()))
                predictions[~train] = y[train].mean() + ((x[~train]-mean)/scale)@weights
            else:
                predictions[~train] = y[train].mean()
        result[name] = dict(features=features, mse=float(np.mean((predictions-y)**2)),
            per_field=[dict(anchor_id=r['anchor_id'],phase=r['phase'],truth=float(a),prediction=float(b))
                       for r,a,b in zip(rr,y,predictions)])
    return result


def limit_shape(rows, anchors):
    fields=[]
    for anchor in anchors:
        clean=sorted([r for r in rows if r['anchor_id']==anchor and r['kind']=='clean' and r['ratio']<=.05],
                     key=lambda r:r['ratio'])
        small=[r for r in clean if 0<r['ratio']<=.001 and r['rms']>0]
        slope=float(np.polyfit(np.log([r['ratio'] for r in small]),np.log([r['rms'] for r in small]),1)[0]) if len(small)>2 else None
        fields.append(dict(anchor_id=anchor,exact_zero_max_abs=clean[0]['max_abs'],
            rms_log_slope_below_001=slope,
            monotonic_rms_from_zero=all(b['rms']>=a['rms'] for a,b in zip(clean,clean[1:])),
            curve=[{k:r[k] for k in ('ratio','rms','bias','rms_over_injected')} for r in clean]))
    return dict(fields=fields,interpretation='Positive-noise bias and its rate, not learned exact-zero identity; float32 limits smallest probes.')


def analyze(root):
    p.require_compute()
    manifest = verify_snapshot(root)
    spec = manifest['spec']
    old = Path(spec['parent_root'])
    prepared = json.loads((old/'PREPARED.json').read_text())
    if p.sha256(old/'PREPARED.json') != spec['prepared_sha256']:
        raise ValueError('prepared drift')
    frozen = json.loads((old/'FROZEN.json').read_text())
    if p.sha256(old/'FROZEN.json') != manifest['frozen_sha256']:
        raise ValueError('reference diagnostic drift')
    parent = {(r['anchor_id'],r['ratio'],r.get('rep')):r for r in frozen['parent384'] if r['kind']=='noisy'}
    groups = report.groups(prepared,15)
    results = []; pairs=[]; inputs={}
    with durable.single_writer(root/'analysis'):
        for replica in spec['replicas']:
            final_rows = {}
            for arm in ('optimization','control','near_zero'):
                folder = root/f'replica_{replica}'/arm
                complete = json.loads((folder/'COMPLETE.json').read_text())
                checkpoint, pointer = durable.load(folder, complete['binding'])
                if not complete['complete'] or pointer != complete['checkpoint']:
                    raise ValueError('branch incomplete')
                cfg=copy.deepcopy(prepared['config']);cfg.update(cfg['replicates'][replica])
                train_ids=[r['anchor_id'] for r in prepared['selection']['train']]
                if len(checkpoint['history'])!=checkpoint['step']:
                    raise ValueError('history length mismatch')
                for u,row in enumerate(checkpoint['history']):
                    mode='near_zero' if arm=='near_zero' and u>=spec['optimization_end'] else 'control'
                    t,ratio,_=schedule(u,cfg,spec,mode)
                    if (row['update']!=u+1 or row['anchor_id']!=d.field_for(u,15,train_ids) or
                            row['noise_seed']!=p.seed_for(cfg['train_seed'],train_ids[u%3],u,'noise') or
                            row['time']!=t or row['ratio']!=ratio or not np.isfinite(row['loss'])):
                        raise ValueError('field/noise/objective schedule mismatch')
                curve=[]; raw_curve=[]
                for u, sha in sorted(complete['probes'].items(), key=lambda v:int(v[0])):
                    path=folder/f'probe_{int(u):06d}.json'
                    if p.sha256(path) != sha:
                        raise ValueError('probe drift')
                    inputs[str(path.relative_to(root))]=sha
                    rows=json.loads(path.read_text())['rows'];raw_curve.append(rows)
                    curve.append(dict(update=int(u),groups={k:grouped(rows,groups[k],prepared,parent) for k in ('exposed','transfer')}))
                late={k:report.paired_effect(raw_curve[-2],raw_curve[-1],groups[k]) for k in ('exposed','transfer')}
                plateau = all(abs(late[k]['median']['clean_rms_ratio']-1)<=spec['plateau_fraction'] and
                              abs(late[k]['median']['noisy_highk_error_ratio']-1)<=spec['plateau_fraction']
                              for k in late)
                result=dict(replica=replica,arm=arm,curve=curve,late_paired_change=late,
                    descriptive_last_interval_plateau=plateau,
                    clean_limit={k:limit_shape(raw_curve[-1],groups[k]) for k in ('exposed','transfer')},
                    associations=report.associations(raw_curve[-1],prepared,15),
                    leave_phase_out=shift_predictors(raw_curve[-1],prepared))
                results.append(result);final_rows[arm]=raw_curve[-1]
            pairs.append(dict(replica=replica,near_zero_vs_control={
                k:{str(r):report.paired_effect(final_rows['control'],final_rows['near_zero'],groups[k],r)
                   for r in (.001,.005,.01,.025,.05,.2)} for k in ('exposed','transfer')}))
        summary=dict(complete=True,results=results,paired=pairs,inputs=inputs,
            manifest_sha256=p.sha256(root/'MANIFEST.json'),heldout_payloads_read=False,training_ready=False,
            caveats=['Endpoint identity is structural, not learned.',
                     'A positive-noise Bayes denoiser need not exactly preserve every clean input.',
                     'Only three development phases; NGC-to-SGC cutouts are not independent cosmologies.',
                     'Descriptive last-interval stability is not proof of convergence.',
                     'Observation associations and small-sample ridge comparisons are not causal identification.',
                     'Log-noise embedding floor .002 unchanged; raw time and sine/cosine inputs still vary.'])
        if not report.finite_tree(summary):
            raise ValueError('nonfinite summary')
        dest=root/'analysis/SUMMARY.json'
        if dest.exists():
            if json.loads(dest.read_text()) != summary:
                raise ValueError('analysis output conflict')
        else:
            durable.publish_json(dest,summary)
        print('ANALYSIS COMPLETE',str(dest),p.sha256(dest),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=['train','analyze','smoke','resume-check'])
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--replica',type=int,choices=[0,1],default=0)
    ap.add_argument('--arm',choices=['optimization','control','near_zero'],default='optimization')
    ap.add_argument('--smoke-interrupt-at',type=int)
    args=ap.parse_args()
    if args.command=='train':train(args.root,args.replica,args.arm,args.smoke_interrupt_at)
    elif args.command=='smoke':smoke(args.root)
    elif args.command=='resume-check':resume_check(args.root)
    else:analyze(args.root)


if __name__=='__main__':
    main()
