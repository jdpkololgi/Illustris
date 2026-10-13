"""Restartable, train-only multiwindow fine CFM pilot (two architectures/seeds)."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_coordinates as coord
from workflows.sbi import e2e_coupled_views as views
from workflows.sbi.e2e_coupled_cfm_pilot import save, spectral_weights, weighted_loss
from workflows.sbi.e2e_wholefield_pilot_data import ROOT, SHAPE, guard
from workflows.sbi.e2e_wholefield_pilot_models import Condition, FineModel
from workflows.sbi.e2e_vdm_context_models import project

STEPS = 9984
EPOCH_STEPS = 312
CHECKPOINTS = tuple(EPOCH_STEPS*n for n in (4, 8, 16, 24, 32))


def panel(phases):
    rows = []
    for phase in phases:
        guard(phase)
        record = json.loads((ROOT/'data'/phase/'COMPLETE.json').read_text())
        if not record['passed'] or tuple(record['shape']) != SHAPE:
            raise ValueError('Incomplete data')
        rows.extend((phase, row['pair_id']) for row in record['pairs'])
    return rows


def arrays(address, chart, training=True):
    phase, pair = address
    guard(phase)
    if phase not in (c.TRAIN if training else ('ph012', 'ph013')):
        raise PermissionError('Target access outside training/development role')
    directory = ROOT/'data'/phase/pair
    with np.load(directory/'OBSERVATIONS.npz') as f:
        joint = views.scale_channels(f['joint'], chart['joint'])
        wide = views.scale_channels(f['wide'], chart['wide'])
    with np.load(directory/'TARGETS.npz') as f:
        residual = (f['residual']/chart['fine_residual']['std'][0])[None]
        cs = chart['coarse_logrho']
        coarse = ((np.log(f['coarse_wide'])-cs['mean'][0])/cs['std'][0])[None]
    return tuple(np.asarray(a, dtype='f4') for a in (residual, joint, wide, coarse))


def batch(addresses, chart, training=True, device='cuda'):
    values = [arrays(a, chart, training) for a in addresses]
    x, joint, wide, coarse = [torch.as_tensor(np.stack(v), device=device) for v in zip(*values)]
    return project(x), Condition(joint, wide, coarse, 'training_truth' if training else 'oracle_diagnostic')


def selected(ids, seed, step):
    epoch, position = divmod(step, EPOCH_STEPS)
    order = np.random.default_rng(seed+10000*epoch).permutation(len(ids))
    return [ids[i] for i in order[2*position:2*position+2]]


def sources():
    names = ('e2e_wholefield_pilot_train.py', 'e2e_wholefield_pilot_models.py',
             'e2e_wholefield_pilot_data.py', 'e2e_coupled_benchmark_models.py',
             'e2e_gaussian_transformer.py', 'e2e_vdm_context_models.py',
             'e2e_direct_vdm.py', 'e2e_coupled_cfm_pilot.py', 'e2e_coupled_views.py')
    return {n: c.sha256(c.REPO/'workflows/sbi'/n) for n in names}


def spectrum(ids, chart):
    power = np.zeros(SHAPE)
    addresses = []
    for phase in c.TRAIN:
        options = [a for a in ids if a[0] == phase]
        for index in (0, 24):
            address = options[index]
            directory = ROOT/'data'/address[0]/address[1]
            with np.load(directory/'TARGETS.npz') as f:
                x = f['residual']/chart['fine_residual']['std'][0]
            power += np.abs(np.fft.fftn(x, norm='ortho'))**2/26
            addresses.append(address)
    axes = np.meshgrid(*[np.fft.fftfreq(n) for n in SHAPE], indexing='ij', sparse=True)
    shell = np.rint(np.sqrt(sum(k*k for k in axes))*min(SHAPE)).astype(int)
    radial = np.bincount(shell.ravel(), weights=power.ravel())/np.bincount(shell.ravel())
    return torch.as_tensor(spectral_weights(radial[shell], .3), device='cuda', dtype=torch.float32), dict(
        ids=addresses, power=radial.tolist(), alpha=.3, scope='train-only loss weighting; unchanged bridge')


def bridge(x, seed, step):
    # Addressed RNG makes restarts and architecture comparisons use identical inputs.
    generator = torch.Generator(device=x.device).manual_seed(seed+1000003*(step+1))
    t = (torch.arange(len(x), device=x.device)+torch.rand((), device=x.device, generator=generator))/len(x)
    noise = project(torch.randn(x.shape, device=x.device, generator=generator))
    w = t[:, None, None, None, None]
    return (1-w)*noise+w*x, x-noise, t


def update(model, ema, optimizer, weight, x, condition, seed, step):
    model.train(); optimizer.zero_grad(set_to_none=True)
    z, velocity, t = bridge(x, seed, step)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        prediction = model(z, t, condition)
    error = project(prediction.float()-velocity)
    loss = weighted_loss(error, weight)
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite training loss')
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), .5, error_if_nonfinite=True)
    lr = 1e-5+.5*9e-5*(1+math.cos(math.pi*step/STEPS))
    for group in optimizer.param_groups:
        group['lr'] = lr
    optimizer.step()
    with torch.no_grad():
        for target, value in zip(ema.parameters(), model.parameters()):
            target.lerp_(value, .001)
    return dict(loss=float(loss), mse=float(error.square().mean()), gradient_norm=float(norm), lr=lr)


def setup(seed):
    c.require_compute()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one allocated GPU required')
    torch.set_num_threads(4); torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@torch.no_grad()
def development(ema, chart, weight):
    phases = ('ph012', 'ph013')
    if not all((ROOT/'data'/p/'COMPLETE.json').exists() for p in phases):
        return dict(available=False, reason='development preparation still running')
    ids = panel(phases); chosen = []
    for phase in phases:
        options = [a for a in ids if a[0] == phase]
        chosen.extend(options[i] for i in (0, 1, 8, 9))
    rows = []; ema.eval()
    for index in range(0, len(chosen), 2):
        x, condition = batch(chosen[index:index+2], chart, training=False)
        z, target, t = bridge(x, 91271, index)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            prediction = ema(z, t, condition)
        error = project(prediction.float()-target)
        rows.append(float(weighted_loss(error, weight)))
    return dict(available=True, ids=chosen, fixed_noise_cfm_loss=float(np.mean(rows)),
                scope='held-out velocity loss, not posterior calibration')


def run(architecture, prediction, seed, seconds, stop_step=STEPS):
    if seed not in (17, 29) or not 1 <= stop_step <= STEPS:
        raise ValueError('Unregistered training seed/stop')
    setup(seed)
    normalizer = c.sha256(coord.ROOT/'normalization/NORMALIZATION_COMPLETE.json')
    chart = views.load_chart(normalizer); ids = panel(c.TRAIN)
    if len(ids) != 624:
        raise ValueError('Wrong training exposure')
    bound = dict(schema='wholefield-pilot-train-v1', architecture=architecture, prediction=prediction,
                 seed=seed, steps=STEPS, batch=2, shape=list(SHAPE), alpha=.3, ema=.999,
                 lr=[1e-4, 1e-5], normalizer=normalizer, ids=[list(a) for a in ids], sources=sources(),
                 data={p: c.sha256(ROOT/'data'/p/'COMPLETE.json') for p in c.TRAIN},
                 precision='bf16 forward, float32 loss and optimizer', sealed_access=False)
    root = ROOT/'fits'/f'{architecture}_{prediction}_{seed}'
    root.mkdir(parents=True, exist_ok=True)
    binding = root/'BINDING.json'
    if binding.exists():
        if json.loads(binding.read_text()) != bound:
            raise ValueError('Resume binding mismatch')
    else:
        c.atomic_json(binding, bound)
    model = FineModel(architecture, prediction).cuda()
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    latest = root/'LATEST.pt'; step = 0
    if latest.exists():
        state = torch.load(latest, map_location='cuda', weights_only=False)
        if state['binding'] != bound:
            raise ValueError('Checkpoint binding mismatch')
        model.load_state_dict(state['model']); ema.load_state_dict(state['ema'])
        optimizer.load_state_dict(state['optimizer']); step = state['step']; weight = state['weight']
        torch.set_rng_state(state['cpu_rng'].cpu()); torch.cuda.set_rng_state(state['cuda_rng'].cpu())
    else:
        weight, record = spectrum(ids, chart)
        c.atomic_json(root/'SPECTRAL_PROFILE.json', record)
    start = time.monotonic()
    def checkpoint():
        state = dict(binding=bound, step=step, model=model.state_dict(), ema=ema.state_dict(),
                     optimizer=optimizer.state_dict(), weight=weight,
                     cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state())
        save(latest, state)
        if step in CHECKPOINTS:
            save(root/f'CHECKPOINT_{step:06d}.pt', state)
        c.atomic_json(root/'STATUS.json', dict(step=step, target=STEPS, job=os.environ['SLURM_JOB_ID'],
            seconds=time.monotonic()-start, training_complete=step == STEPS,
            posterior_qualified=False, peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30), replace=True)
    while step < stop_step and time.monotonic()-start < seconds:
        tick = time.monotonic(); addresses = selected(ids, seed, step)
        x, cond = batch(addresses, chart)
        row = update(model, ema, optimizer, weight, x, cond, seed, step)
        torch.cuda.synchronize(); step += 1
        row.update(step=step, epoch=step/EPOCH_STEPS, seconds=time.monotonic()-tick,
                   peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30)
        if step == 1 or step % 16 == 0:
            with (root/'TRAIN.jsonl').open('a') as f:
                f.write(json.dumps(row)+'\n')
            print(json.dumps(row), flush=True)
        if step == 1 or step % 128 == 0 or step in CHECKPOINTS:
            checkpoint()
        if step in CHECKPOINTS:
            c.atomic_json(root/f'DEVELOPMENT_{step:06d}.json', development(ema, chart, weight))
    checkpoint()
    print(json.dumps(dict(event='segment_complete', architecture=architecture, seed=seed, step=step)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--architecture', choices=('unet', 'transformer'), required=True)
    parser.add_argument('--prediction', choices=('v', 'x'), required=True)
    parser.add_argument('--seed', type=int, choices=(17, 29), required=True)
    parser.add_argument('--seconds', type=int, default=6600)
    parser.add_argument('--stop-step', type=int, default=STEPS)
    args = parser.parse_args(); run(**vars(args))
