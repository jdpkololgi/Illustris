"""Bounded paired fine continuations; explicit train-only access and restart state."""
import argparse
import copy
import inspect
import json
import os
from pathlib import Path
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch

from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_coordinates as coord
from workflows.sbi import e2e_coupled_views as views
from workflows.sbi import e2e_coupled_cfm_pilot as pilot
from workflows.sbi.e2e_coupled_benchmark_models import CoupledBackbone
from workflows.sbi.e2e_cfm_conditioning_model import AugmentedFine, augment_condition
from workflows.sbi.e2e_vdm_context_models import project

CONFIG = c.REPO/'configs/e2e_cfm_conditioning_augmentation_v1.json'
PARENT = Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/cfm_pilot_20260924_v1/fine_17/CHECKPOINT_026624.pt')
RUN = Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/cfm_conditioning_augmentation_20261006_v3')


def profile():
    p = json.loads(CONFIG.read_text())
    if (p['schema'] != 'cfm-conditioning-augmentation-v1' or p['seed'] != 17
            or p['fine_step'] != 26624 or p['steps'] != 3328 or p['batch'] != 2
            or p['confirmation_access'] or p['posterior_qualified']
            or p['development_phases'] != ['ph012', 'ph013']):
        raise ValueError('unregistered continuation profile')
    return p


def setup():
    c.require_compute()
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise RuntimeError('registered deterministic cuBLAS workspace required')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('one visible GPU required')
    torch.set_num_threads(4); torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)


def addressed_rng(kind, step, device='cuda'):
    return torch.Generator(device=device).manual_seed(
        int(c.digest(['conditioning-augmentation-v1', 17, kind, step])[:15], 16))


def step_inputs(x, condition, arm, step, p):
    if arm not in ('control', 'augmented') or step < 0:
        raise ValueError('invalid arm/update')
    g = addressed_rng('fine-bridge', step, str(x.device))
    t = (torch.arange(len(x), device=x.device)+torch.rand((), device=x.device, generator=g))/len(x)
    eps = project(torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=g))
    a = addressed_rng('coarse-conditioning', step, str(x.device))
    # Draw the same corruption RNG stream for both arms; it cannot perturb fine RNG.
    clean = torch.rand((len(x),), device=x.device, generator=a) < p['clean_probability']
    s = torch.rand((len(x),), device=x.device, generator=a)*p['max_strength']
    s = torch.where(clean, torch.zeros_like(s), s)
    if arm == 'control':
        s = torch.zeros_like(s)
    eta = torch.randn(condition.coarse_wide.shape, device=x.device,
                      dtype=x.dtype, generator=a)
    cond = augment_condition(condition, s, eta)
    w = t[:, None, None, None, None]
    return (1-w)*eps+w*x, x-eps, t, cond


def parent_state():
    release = coord.ROOT/'data_release/DATA_PRODUCTS_QUALIFIED.json'
    if c.sha256(release) != 'cf26b07c16ec0714fcd6d0c318c77b0ab3799497a00eba4195f48a2a42889ced':
        raise ValueError('prepared training data release changed')
    state = torch.load(PARENT, map_location='cpu', weights_only=False)
    b = state['binding']
    if (state['step'] != 26624 or b['seed'] != 17 or b['stage'] != 'fine'
            or b['alpha'] != .30 or b['confirmation_access']
            or b['model'] != c.sha256(inspect.getfile(CoupledBackbone))):
        raise ValueError('parent checkpoint/architecture identity changed')
    views.load_chart(b['normalizer'])
    return state


def initialize(state):
    torch.manual_seed(17)
    model = AugmentedFine().cuda()
    model.load_parent(state['ema'])
    return model


def optimizer(model, p):
    return torch.optim.AdamW(model.parameters(), lr=p['lr'], weight_decay=p['weight_decay'])


def training_batch(ids, step, normalizer):
    # Continue the existing deterministic data order at epoch 32, not epoch 0.
    absolute = 26624+step; epoch, position = divmod(absolute, 832)
    order = np.random.default_rng(17+epoch*10000).permutation(len(ids))
    selected = [ids[i] for i in order[2*position:2*position+2]]
    if len(selected) != 2 or any(phase not in c.TRAIN for phase, _ in selected):
        raise PermissionError('unregistered training batch')
    batch = pilot.batch_pair([pilot.pair(item, normalizer) for item in selected])
    return pilot.condition(batch, 'fine', 'cuda'), selected


def update(model, ema, opt, weight, x, condition, arm, step, p):
    model.train(); opt.zero_grad(set_to_none=True)
    z, velocity, t, cond = step_inputs(x, condition, arm, step, p)
    error = project(model(z, model.schedule(t), cond)-velocity)
    loss = pilot.weighted_loss(error, weight)
    if not torch.isfinite(loss):
        raise FloatingPointError('nonfinite loss')
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), p['clip_grad'], error_if_nonfinite=True)
    opt.step()
    with torch.no_grad():
        for ep, param in zip(ema.parameters(), model.parameters()):
            ep.lerp_(param, 1-p['ema'])
    return dict(loss=float(loss), mse=float(error.square().mean()), grad_norm=float(norm),
                strengths=cond.strength.detach().cpu().tolist())


def source_records():
    names = ('e2e_cfm_conditioning_train.py', 'e2e_cfm_conditioning_model.py',
             'e2e_cfm_conditioning_sample.py', 'e2e_cfm_conditioning_gates.py',
             'e2e_coupled_cfm_pilot.py', 'e2e_coupled_benchmark_models.py',
             'e2e_coupled_views.py', 'e2e_vdm_context_models.py', 'e2e_direct_vdm.py')
    return {name: c.sha256(c.REPO/'workflows/sbi'/name) for name in names}


def run(arm, seconds, stop_step=None):
    setup(); p = profile()
    # The full run must execute from a committed snapshot after a passed smoke.
    if c.REPO != RUN/'source':
        raise PermissionError('scientific training requires the frozen source snapshot')
    receipt = json.loads((RUN/'SMOKE.json').read_text())
    if (not receipt['passed'] or receipt['sources'] != source_records()
            or receipt['config_sha256'] != c.sha256(CONFIG)
            or receipt['parent_sha256'] != c.sha256(PARENT)):
        raise ValueError('smoke/source/parent binding changed')
    replay_path = RUN/'replay/ph012/SGC/parent/s0/d000/COMPLETE.json'
    replay = json.loads(replay_path.read_text())
    if (not replay['passed'] or not replay['exact_replay'] or replay['nfe'] != 128
            or replay['sources'] != source_records()
            or replay['config_sha256'] != c.sha256(CONFIG)):
        raise ValueError('cached-coarse fine replay prerequisite failed')
    state = parent_state(); ids = pilot.training_ids()
    if json.loads(json.dumps(ids)) != json.loads(json.dumps(state['binding']['train_ids'])):
        raise ValueError('training panel changed')
    root = RUN/arm; root.mkdir(parents=True, exist_ok=True)
    bound = dict(schema='cfm-conditioning-training-v1', arm=arm, profile=p,
                 parent_sha256=receipt['parent_sha256'], sources=source_records(),
                 config_sha256=c.sha256(CONFIG), smoke_sha256=c.sha256(RUN/'SMOKE.json'),
                 replay_sha256=c.sha256(replay_path),
                 normalizer=state['binding']['normalizer'], train_ids=ids,
                 source_root=str(c.REPO), confirmation_access=False)
    binding_path = root/'BINDING.json'
    if not binding_path.exists():
        c.atomic_json(binding_path, bound)
    if json.loads(binding_path.read_text()) != json.loads(json.dumps(bound)):
        raise ValueError('resume binding mismatch')
    model = initialize(state); ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = optimizer(model, p); weight = state['weight'].cuda(); step = 0
    latest = root/'LATEST.pt'
    if latest.exists():
        saved = torch.load(latest, map_location='cuda', weights_only=False)
        if saved['binding'] != bound:
            raise ValueError('checkpoint binding mismatch')
        model.load_state_dict(saved['model']); ema.load_state_dict(saved['ema'])
        opt.load_state_dict(saved['optimizer']); step = saved['step']
        if not torch.equal(saved['weight'], weight):
            raise ValueError('loss weighting changed')
    target = p['steps'] if stop_step is None else stop_step
    if not step <= target <= p['steps']:
        raise ValueError('invalid continuation stop')
    start = time.monotonic()
    def checkpoint():
        saved = dict(binding=bound, step=step, model=model.state_dict(),
                     ema=ema.state_dict(), optimizer=opt.state_dict(), weight=weight)
        pilot.save(latest, saved)
        if step == p['steps']:
            pilot.save(root/'CHECKPOINT_003328.pt', saved)
        c.atomic_json(root/'STATUS.json', dict(step=step, target=p['steps'],
            job=os.environ['SLURM_JOB_ID'], seconds=time.monotonic()-start,
            training_complete=step == p['steps'], posterior_qualified=False), replace=True)
    while step < target and time.monotonic()-start < seconds:
        tick = time.monotonic()
        (x, cond), selected = training_batch(ids, step, bound['normalizer'])
        row = update(model, ema, opt, weight, x, cond, arm, step, p)
        torch.cuda.synchronize(); step += 1
        row.update(step=step, arm=arm, ids=selected, seconds=time.monotonic()-tick,
                   peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30)
        if step == 1 or step % 16 == 0:
            with (root/'TRAIN.jsonl').open('a') as stream:
                stream.write(json.dumps(row)+'\n')
            print(json.dumps(row), flush=True)
        if step == 1 or step % 64 == 0:
            checkpoint()
    checkpoint()
    print(json.dumps(dict(event='segment_complete', arm=arm, step=step)), flush=True)


def smoke():
    setup(); p = profile(); start = time.monotonic()
    RUN.mkdir(parents=True, exist_ok=True)
    if (RUN/'SMOKE.json').exists():
        raise FileExistsError('retain prior smoke receipt')
    state = parent_state(); ids = pilot.training_ids()
    (x, cond), selected = training_batch(ids, 0, state['binding']['normalizer'])
    # Both inputs are paired training truth, so preserve the training-source
    # guard. This backbone has no dropout; no_grad below prevents any update.
    model = initialize(state).train()
    parent = CoupledBackbone('fine', 'joint', 24, 3).cuda().train()
    parent.load_state_dict(state['ema'])
    z, velocity, t, clean = step_inputs(x, cond, 'control', 0, p)
    with torch.no_grad():
        expected = parent(z, parent.schedule(t), cond)
        actual = model(z, model.schedule(t), clean)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    del parent, expected, actual
    torch.cuda.empty_cache()
    weight = state['weight'].cuda()
    ema = copy.deepcopy(model).eval().requires_grad_(False); opt = optimizer(model, p)
    rows = []
    # Two real paired updates and an exact serialized restart replay.
    for step in range(2):
        tick = time.monotonic()
        row = update(model, ema, opt, weight, x, cond, 'augmented', step, p)
        torch.cuda.synchronize(); row['seconds'] = time.monotonic()-tick; rows.append(row)
    checkpoint = RUN/'SMOKE_RESTART.pt'
    pilot.save(checkpoint, dict(model=model.state_dict(), ema=ema.state_dict(),
                               optimizer=opt.state_dict()))
    row = update(model, ema, opt, weight, x, cond, 'augmented', 2, p)
    expected = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    expected_ema = {k: v.detach().cpu().clone() for k, v in ema.state_dict().items()}
    saved = torch.load(checkpoint, map_location='cuda', weights_only=False)
    model.load_state_dict(saved['model']); ema.load_state_dict(saved['ema'])
    opt.load_state_dict(saved['optimizer'])
    repeated = update(model, ema, opt, weight, x, cond, 'augmented', 2, p)
    if row != repeated:
        raise ValueError('restart metric mismatch')
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value.cpu(), expected[key], rtol=0, atol=0)
    for key, value in ema.state_dict().items():
        torch.testing.assert_close(value.cpu(), expected_ema[key], rtol=0, atol=0)
    result = dict(passed=True, config_sha256=c.sha256(CONFIG), sources=source_records(),
                  parent_sha256=c.sha256(PARENT), normalizer=state['binding']['normalizer'],
                  exact_zero_parent_parity=True, exact_restart=True,
                  train_only_ids=selected, updates=rows, seconds=time.monotonic()-start,
                  peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,
                  job=os.environ['SLURM_JOB_ID'], posterior_qualified=False)
    c.atomic_json(RUN/'SMOKE.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('action', choices=['smoke', 'train'])
    ap.add_argument('--arm', choices=['control', 'augmented'])
    ap.add_argument('--seconds', type=int, default=3000); ap.add_argument('--stop-step', type=int)
    args = ap.parse_args()
    if args.action == 'smoke':
        smoke()
    elif args.arm is None:
        ap.error('--arm required')
    else:
        run(args.arm, args.seconds, args.stop_step)
