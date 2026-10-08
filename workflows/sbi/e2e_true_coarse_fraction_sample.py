"""Matched full-cap parent fine draws; only coarse endpoint changes to truth.

The integrator, addressed fine noise, observed channels, normalizer, parent
EMA, precision, support, and physical decoder are imported unchanged from
the immutable conditioning-augmentation model snapshot. No training path.
"""
import argparse
import os
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import e2e_true_coarse_fraction_saved as io
sys.path.insert(0, str(io.OLD/'source'))
import h5py
import numpy as np
import torch
from workflows.sbi import e2e_cfm_conditioning_sample as original
from workflows.sbi import e2e_cfm_conditioning_train as training
from workflows.sbi import e2e_coupled_views as views
from workflows.sbi.e2e_cfm_conditioning_model import AugmentedCondition
from workflows.sbi.e2e_global_canvas import Geometry, integrate
from workflows.sbi.e2e_vdm_context_models import replicate, decode_density, block_mean


def run(phase, cap, first):
    io.compute_guard(); original.guard(phase, cap, first, 'parent', 0.)
    training.setup()
    from workflows.sbi.e2e_loa_field_sampling import setup
    setup(True)
    if training.c.REPO != io.OLD/'source':
        raise PermissionError('Immutable parent dependencies required')
    started = time.monotonic(); out = io.OUT/'fields'/phase/cap/f'd{first:03d}'
    out.mkdir(parents=True, exist_ok=False)
    manifest = io.record(io.OUT/'MANIFEST.json')
    config = io.read(io.CONFIG)
    if not config['interpretation']['matched_fullcap_B_authorized'] or config['training'] or config['nfe'] != 128:
        raise ValueError('Unregistered diagnostic')
    model_sources = training.source_records()
    if model_sources != io.read(io.OLD/'SMOKE.json')['sources']:
        raise ValueError('Parent numerical dependencies changed')
    inp = io.BASE/'fullcap_inputs'/phase/cap
    prepared = io.read(inp/'PREPARED.json'); plan = io.read(inp/'PLAN.json')
    prepared_record = io.record(inp/'PREPARED.json'); plan_record = io.record(inp/'PLAN.json')
    if not prepared['targetless'] or prepared['plan_sha256'] != plan_record['sha256'] or prepared['phase'] != phase or prepared['cap'] != cap:
        raise ValueError('Observed input binding changed')
    io.verify(prepared['outputs'][0])
    cache = io.BASE/'fullcap_density'/phase/cap/f'd{first:03d}'
    old = io.read(cache/'COMPLETE.json'); binding = io.read(cache/'BINDING.json')
    io.record(cache/'COMPLETE.json'); br = io.record(cache/'BINDING.json')
    if not old['passed'] or old['binding'] != br['sha256'] or binding['phase'] != phase or binding['cap'] != cap or binding['first'] != first or binding['draws'] != 4 or binding['nfe'] != 128 or binding['benchmark']:
        raise ValueError('Parent draw addresses changed')
    if binding['prepared_sha256'] != prepared_record['sha256'] or binding.get('model_seed', 17) != 17:
        raise ValueError('Parent observed inputs or seed changed')
    parent = io.record(training.PARENT)
    if parent['sha256'] != io.FINE_SHA or binding['checkpoints']['fine'] != io.FINE_SHA:
        raise ValueError('Wrong parent checkpoint')
    state = training.parent_state(); model = training.initialize(state).eval()
    if state['binding']['normalizer'] != binding['normalizer']:
        raise ValueError('Normalizer changed')
    chart = views.load_chart(binding['normalizer'])
    cg, fg = [Geometry.make(plan['midpoints'], stage) for stage in ('coarse', 'fine')]
    true_path = io.OUT/'true_coarse'/phase/'COMPLETE.json'; true_rec = io.read(true_path); io.record(true_path)
    if not true_rec['passed'] or true_rec['phase'] != phase:
        raise ValueError('Incomplete true-coarse preparation')
    source = next(q['output'] for q in true_rec['products'] if q['cap'] == cap); io.verify(source)
    with h5py.File(source['path']) as f:
        if f.attrs['phase'] != phase or f.attrs['cap'] != cap or f.attrs['normalizer_sha256'] != binding['normalizer'] or f['logrho_standardized'].shape != cg.shape:
            raise ValueError('True coarse canvas address changed')
        np.testing.assert_array_equal(f.attrs['coarse_origin_raw_cells'], np.asarray(cg.origin)*8)
        endpoint = torch.as_tensor(f['logrho_standardized'][:], device='cuda')[None, None].expand(4, 1, *cg.shape).clone()
    strength = endpoint.new_zeros(4)
    tensor = lambda x: torch.as_tensor(x, device='cuda', dtype=torch.float32)[None].expand(4, *x.shape)
    conditions = []
    with h5py.File(prepared['outputs'][0]['path']) as f:
        coverage, support = f['coverage'][:], f['support'][:]
        for i in range(len(plan['midpoints'])):
            zw = endpoint[cg.crop(i)]
            conditions.append(AugmentedCondition(
                tensor(views.scale_channels(f['joint'][i], chart['joint'])),
                tensor(views.scale_channels(f['wide'][i], chart['wide'])),
                tensor(np.zeros(3, 'f4')), 'joint', replicate(zw[:, :, 16:32, 18:30, 18:30]),
                zw, 'sampled', strength))
    loading = time.monotonic()-started; tick = time.monotonic()
    def report(step, value):
        if step % 8 == 0:
            print(io.json.dumps(dict(stage='true_coarse_fine', phase=phase, cap=cap, first=first,
                step=step, seconds=time.monotonic()-tick)), flush=True)
    u, weights = integrate(model, 'fine', conditions,
        original.canvas_noise(fg, phase, cap, first, 'fine'), fg, 128, report)
    torch.cuda.synchronize(); sampling = time.monotonic()-tick
    np.testing.assert_array_equal(weights[0, 0].cpu().numpy(), coverage)
    offset = np.asarray(fg.origin)//4-np.asarray(cg.origin)
    origin = np.asarray(plan['grid']['origin_mpc_h'])+np.asarray(fg.origin)*6.766
    maximum = 0.; path = out/'FIELD.h5'
    with h5py.File(path, 'x') as f:
        ds = f.create_dataset('rho', shape=(4, *fg.shape), dtype='f4', chunks=(1, 16, 32, 32), compression='lzf')
        f.create_dataset('coarse_logrho_standardized', data=endpoint[:, 0].cpu().numpy(), compression='lzf')
        f.create_dataset('support', data=support, compression='lzf'); f.create_dataset('coverage', data=coverage, compression='lzf')
        f.attrs.update(phase=phase, cap=cap, first_draw=first, shared_state=True,
            origin_mpc_h=origin, cell_mpc_h=6.766, coarse_origin_raw_cells=np.asarray(cg.origin)*8,
            arm='parent_true_coarse', posterior_qualified=False)
        for lo in range(0, fg.shape[0], 16):
            hi = min(lo+16, fg.shape[0]); start = offset+[lo//4, 0, 0]
            size = np.array([hi-lo, fg.shape[1], fg.shape[2]])//4
            cut = (slice(None), slice(None), *io.slices(start, size))
            mass = torch.exp(endpoint[cut].double()*chart['coarse_logrho']['std'][0]+chart['coarse_logrho']['mean'][0])
            rho = decode_density(mass, u[:, :, lo:hi].double()*chart['fine_residual']['std'][0])
            if not torch.isfinite(rho).all() or not (rho > 0).all():
                raise ValueError('Invalid decoded density')
            maximum = max(maximum, float(((block_mean(rho)-mass)/mass).abs().max()))
            values = rho[:, 0].cpu().numpy().astype('f4'); values[:, coverage[lo:hi] == 0] = np.nan
            ds[:, lo:hi] = values
    if maximum > training.profile()['mass_relative_tolerance']:
        raise ValueError('Mass conservation failure')
    if model_sources != training.source_records():
        raise ValueError('Parent source changed during generation')
    io.write(out/'COMPLETE.json', dict(passed=True, phase=phase, cap=cap, first=first, draws=4, nfe=128,
        field=io.record(path), manifest=manifest, config=io.record(io.CONFIG),
        source=io.record(__file__), model_sources=model_sources, inputs=list(io.CACHE.values()),
        parent_checkpoint=parent, coarse_conditioning='true physical R7 coarse canvas',
        fine_noise_addresses=[['global-exterior-cap-v1', phase, cap, 'fine', d] for d in range(first, first+4)],
        sampling_seconds=sampling, loading_seconds=loading, max_mass_error=maximum,
        peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,
        scope='one parent seed, two reused phases, diagnostic only', posterior_qualified=False,
        job=os.environ['SLURM_JOB_ID'], seconds=time.monotonic()-started))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--phase', choices=io.PHASES, required=True)
    p.add_argument('--cap', choices=('NGC', 'SGC'), required=True); p.add_argument('--first', type=int, required=True)
    a = p.parse_args(); run(a.phase, a.cap, a.first)
