"""Fine-only shared-canvas replay/generation using immutable coarse endpoints."""
import argparse
import json
import os
from pathlib import Path
import time

import h5py
import numpy as np
import torch

from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_views as views
from workflows.sbi import e2e_cfm_conditioning_train as training
from workflows.sbi.e2e_cfm_conditioning_model import AugmentedCondition, corrupt_endpoint
from workflows.sbi.e2e_global_canvas import Geometry, integrate
from workflows.sbi.e2e_vdm_context_models import project, replicate, decode_density, block_mean

BASE = training.RUN.parent/'coherent_exterior_20261002_v1'


def guard(phase, cap, first, arm, strength):
    if phase not in ('ph012', 'ph013') or cap not in ('NGC', 'SGC'):
        raise PermissionError('development caps only')
    if isinstance(first, bool) or first not in range(0, 32, 4):
        raise ValueError('registered four-draw address required')
    if arm not in ('parent', 'control', 'augmented') or strength not in (0., .025, .05, .1, .2):
        raise ValueError('unregistered arm/strength')
    if arm != 'augmented' and strength != 0:
        raise ValueError('only augmented model uses nonzero strength')


def canvas_noise(geometry, phase, cap, first, kind, device='cuda'):
    rows = []
    for draw in range(first, first+4):
        address = (['global-exterior-cap-v1', phase, cap, 'fine', draw]
                   if kind == 'fine' else ['conditioning-canvas-v1', phase, cap, draw])
        rng = torch.Generator(device=device).manual_seed(int(c.digest(address)[:15], 16))
        rows.append(torch.randn((1, 1, *geometry.shape), device=device, generator=rng))
    value = torch.cat(rows)
    return project(value) if kind == 'fine' else value


def run(phase, cap, first, arm, strength, replay=False):
    guard(phase, cap, first, arm, strength); training.setup(); p = training.profile()
    if c.REPO != training.RUN/'source':
        raise PermissionError('run only the committed source snapshot')
    if replay and (arm != 'parent' or strength != 0):
        raise ValueError('replay uses unchanged parent only')
    out = training.RUN/('replay' if replay else 'fields')/phase/cap/arm/f's{strength:g}'/f'd{first:03d}'
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    inp = BASE/'fullcap_inputs'/phase/cap
    cache = BASE/'fullcap_density'/phase/cap/f'd{first:03d}'
    prepared = json.loads((inp/'PREPARED.json').read_text())
    plan = json.loads((inp/'PLAN.json').read_text())
    original = json.loads((cache/'COMPLETE.json').read_text())
    binding = json.loads((cache/'BINDING.json').read_text())
    if (not original['passed'] or not prepared['targetless']
            or original['binding'] != c.sha256(cache/'BINDING.json')
            or binding['phase'] != phase or binding['cap'] != cap
            or binding['first'] != first or binding['draws'] != 4 or binding['nfe'] != 128
            or binding['benchmark'] or binding.get('model_seed', 17) != 17
            or prepared['phase'] != phase or prepared['cap'] != cap
            or prepared['plan_sha256'] != c.sha256(inp/'PLAN.json')
            or binding['prepared_sha256'] != c.sha256(inp/'PREPARED.json')):
        raise ValueError('cached canvas provenance mismatch')
    for record in (original['field'], prepared['outputs'][0]):
        if c.sha256(record['path']) != record['sha256']:
            raise ValueError('cached input payload changed')
    state = training.parent_state()
    if binding['checkpoints']['fine'] != c.sha256(training.PARENT):
        raise ValueError('cached fine parent differs')
    if binding['normalizer'] != state['binding']['normalizer']:
        raise ValueError('normalizer mismatch')
    model = training.initialize(state).eval()
    checkpoint = training.PARENT
    if arm != 'parent':
        checkpoint = training.RUN/arm/'CHECKPOINT_003328.pt'
        fit = torch.load(checkpoint, map_location='cuda', weights_only=False)
        if (fit['step'] != p['steps'] or fit['binding']['arm'] != arm
                or fit['binding']['profile'] != p
                or fit['binding']['sources'] != training.source_records()
                or fit['binding']['parent_sha256'] != c.sha256(training.PARENT)):
            raise ValueError('fine continuation checkpoint mismatch')
        model.load_state_dict(fit['ema'])
    chart = views.load_chart(binding['normalizer'])
    cg, fg = (Geometry.make(plan['midpoints'], stage) for stage in ('coarse', 'fine'))
    with h5py.File(original['field']['path']) as f:
        if (tuple(f['coarse_logrho_standardized'].shape) != (4, *cg.shape)
                or f.attrs['phase'] != phase or f.attrs['cap'] != cap
                or f.attrs['first_draw'] != first or not f.attrs['shared_state']
                or not np.array_equal(f.attrs['coarse_origin_raw_cells'], np.asarray(cg.origin)*8)):
            raise ValueError('coarse geometry/address mismatch')
        endpoint = torch.as_tensor(f['coarse_logrho_standardized'][:], device='cuda')[:, None]
    s = endpoint.new_full((4,), strength)
    eta = canvas_noise(cg, phase, cap, first, 'conditioning')
    noisy = corrupt_endpoint(endpoint, s, eta)
    tensor = lambda x: torch.as_tensor(x, device='cuda', dtype=torch.float32)[None].expand(4, *x.shape)
    conditions = []
    with h5py.File(prepared['outputs'][0]['path']) as f:
        coverage, support = f['coverage'][:], f['support'][:]
        for i in range(len(plan['midpoints'])):
            zw = noisy[cg.crop(i)]
            conditions.append(AugmentedCondition(
                tensor(views.scale_channels(f['joint'][i], chart['joint'])),
                tensor(views.scale_channels(f['wide'][i], chart['wide'])),
                tensor(np.zeros(3, 'f4')), 'joint', replicate(zw[:, :, 16:32, 18:30, 18:30]),
                zw, 'sampled', s))
    loading = time.monotonic()-started; tick = time.monotonic()
    def report(step, value):
        if step % 8 == 0:
            print(json.dumps(dict(phase=phase, cap=cap, first=first, arm=arm,
                                 strength=strength, step=step, seconds=time.monotonic()-tick)), flush=True)
    u, weights = integrate(model, 'fine', conditions,
        canvas_noise(fg, phase, cap, first, 'fine'), fg, p['nfe'], report)
    torch.cuda.synchronize(); sampling_seconds = time.monotonic()-tick
    if not np.array_equal(weights[0, 0].cpu().numpy(), coverage):
        raise ValueError('fine coverage changed')
    offset = np.asarray(fg.origin)//4-np.asarray(cg.origin)
    origin = np.asarray(plan['grid']['origin_mpc_h'])+np.asarray(fg.origin)*6.766
    maximum = 0.; max_difference = 0.; chunks = 0
    new = None if replay else h5py.File(out/'FIELD.h5', 'x')
    try:
        if new is not None:
            ds = new.create_dataset('rho', shape=(4, *fg.shape), dtype='f4',
                chunks=(1, 16, 32, 32), compression='lzf')
            new.create_dataset('coarse_logrho_standardized', data=endpoint[:, 0].cpu().numpy(), compression='lzf')
            new.create_dataset('support', data=support, compression='lzf')
            new.create_dataset('coverage', data=coverage, compression='lzf')
            new.attrs.update(phase=phase, cap=cap, first_draw=first, shared_state=True,
                origin_mpc_h=origin, cell_mpc_h=6.766, coarse_origin_raw_cells=np.asarray(cg.origin)*8,
                strength=strength, arm=arm, posterior_qualified=False)
        with h5py.File(original['field']['path']) as old:
            if tuple(old['rho'].shape) != (4, *fg.shape) or not np.array_equal(old.attrs['origin_mpc_h'], origin):
                raise ValueError('density geometry changed')
            for lo in range(0, fg.shape[0], 16):
                hi = min(lo+16, fg.shape[0]); startc = offset+[lo//4, 0, 0]
                size = np.array([hi-lo, fg.shape[1], fg.shape[2]])//4
                cut = (slice(None), slice(None), *(slice(int(a), int(a+n)) for a, n in zip(startc, size)))
                # Endpoint, never the noisy conditioning, supplies physical mass.
                mass = torch.exp(endpoint[cut].double()*chart['coarse_logrho']['std'][0]
                                 +chart['coarse_logrho']['mean'][0])
                rho = decode_density(mass, u[:, :, lo:hi].double()*chart['fine_residual']['std'][0])
                if not torch.isfinite(rho).all() or not (rho > 0).all():
                    raise ValueError('nonfinite/nonpositive density')
                maximum = max(maximum, float(((block_mean(rho)-mass)/mass).abs().max()))
                values = rho[:, 0].cpu().numpy().astype('f4'); values[:, coverage[lo:hi] == 0] = np.nan
                if replay:
                    expected = old['rho'][:, lo:hi]
                    if not np.array_equal(expected, values, equal_nan=True):
                        raise ValueError('cached-coarse parent fine replay is not exact')
                else:
                    ds[:, lo:hi] = values
                chunks += 1
    finally:
        if new is not None:
            new.close()
    if maximum > p['mass_relative_tolerance']:
        raise ValueError('block-mass failure')
    result = dict(passed=True, replay=replay, exact_replay=replay, phase=phase, cap=cap,
        first=first, draws=4, nfe=p['nfe'], arm=arm, strength=strength,
        coarse_receipt=dict(path=str(cache/'COMPLETE.json'), sha256=c.sha256(cache/'COMPLETE.json')),
        checkpoint=dict(path=str(checkpoint), sha256=c.sha256(checkpoint)),
        runner_sha256=c.sha256(__file__), config_sha256=c.sha256(training.CONFIG),
        sources=training.source_records(), loading_seconds=loading, sampling_seconds=sampling_seconds,
        max_mass_error=maximum, chunks=chunks, job=os.environ['SLURM_JOB_ID'],
        peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30, posterior_qualified=False)
    if not replay:
        result['field'] = c.file_record(out/'FIELD.h5', content_hash=True)
    c.atomic_json(out/'COMPLETE.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--phase', required=True)
    ap.add_argument('--cap', required=True); ap.add_argument('--first', type=int, default=0)
    ap.add_argument('--arm', choices=['parent', 'control', 'augmented'], default='parent')
    ap.add_argument('--strength', type=float, default=0.); ap.add_argument('--replay', action='store_true')
    a = ap.parse_args(); run(a.phase, a.cap, a.first, a.arm, a.strength, a.replay)
