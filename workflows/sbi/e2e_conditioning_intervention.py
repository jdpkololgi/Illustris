"""Fixed-weight, common-noise coarse-conditioning stress controls.

Oracle interpolations and randomized displacements are not posterior draws.
No training, perturbation fitting, calibration tuning or confirmation access.
"""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2e_conditioning_ablation as a
import e2e_sampler_factor_diagnostic as s
import e2e_operator_split as selected
import numpy as np
import h5py
from scipy import fft

b = a.b
OUT = b.ROOT/'verification/conditioning_intervention_20261005_v1'


def randomize_displacement(error, seed):
    """Preserve full Fourier modulus/DC; random real-field phases."""
    target = fft.rfftn(error)
    noise = fft.rfftn(np.random.default_rng(seed).normal(size=error.shape))
    phase = noise/np.maximum(abs(noise), 1e-100)
    spectrum = abs(target)*phase
    spectrum[0, 0, 0] = target[0, 0, 0]
    result = fft.irfftn(spectrum, s=error.shape)
    np.testing.assert_allclose(abs(fft.rfftn(result)), abs(target), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(np.mean(result**2), np.mean(error**2), rtol=1e-12, atol=1e-14)
    return result


def conditions(truth, sample, seed):
    error = sample-truth
    return dict(halfway=truth+.5*error, reflected=truth-error,
                randomized=truth+randomize_displacement(error, seed))


def run(limit=0, draws=32):
    import torch
    from workflows.sbi.e2e_loa_field_sampling import setup
    from workflows.sbi import e2e_cfm_pilot_evaluate as ev
    from workflows.sbi import e2e_coupled_views as views
    from workflows.sbi.e2e_coupled_benchmark_models import FieldCondition
    from workflows.sbi.e2e_vdm_context_models import replicate, decode_density, block_mean
    from workflows.sbi.e2e_global_canvas import Geometry, integrate
    from e2e_checkpoint_shear import LocalOperator

    b.c.require_compute(); setup(True)
    dest = OUT if limit == 0 and draws == 32 else OUT.with_name(OUT.name+f'_pilot{limit}_{draws}')
    dest.mkdir(parents=True, exist_ok=False)
    models, hashes, normalizer = ev.load_models(17, 13312, 26624)
    freeze = json.loads(b.FREEZE.read_text())
    assert hashes == freeze['checkpoints'] and normalizer == freeze['normalizer']
    chart = views.load_chart(normalizer)
    prepared = json.loads((s.OUT/'PREPARED.json').read_text())
    chosen = [r for r in prepared['cases'] if r['pair_id'] in selected.SELECTED[r['phase']]]
    if limit: chosen = chosen[:limit]
    operator = LocalOperator(); band_masks = a.masks((64, 48, 48)); cases = []
    for row in chosen:
        phase, cap, pid = row['phase'], row['cap'], row['pair_id']
        b.auth.frozen(phase, b.FREEZE); before = time.monotonic()
        assert b.c.sha256(row['input']) == row['file']['sha256']
        previous_path = s.OUT/'results'/(pid+'.json')
        previous = json.loads(previous_path.read_text())
        assert b.c.sha256(previous['fields']['path']) == previous['fields']['sha256']
        with h5py.File(row['input']) as f: arrays = {k: f[k][:] for k in f}
        tq, tm, _ = a.shape_mass(arrays['true_rho'])
        start = arrays['core_start']; cores = [s.slices(start+[16*i, 0, 0], (16, 16, 16)) for i in (0, 1)]
        operator.cores = cores
        truth = a.fine_metrics(tq, tm, cores, band_masks)
        plan = json.loads((b.ROOT/'fullcap_inputs'/phase/cap/'PLAN.json').read_text())
        global_fine = Geometry.make(plan['midpoints'], 'fine')
        single = Geometry.make([row['midpoint']], 'fine')
        ztrue = (np.log(arrays['true_wide'])-chart['coarse_logrho']['mean'][0])/chart['coarse_logrho']['std'][0]
        arms = {k: [] for k in ('halfway', 'reflected', 'randomized')}
        endpoint_errors = {}; max_mass = 0.
        fieldpath = dest/(pid+'_DRAWS.h5')
        with h5py.File(fieldpath, 'x') as output, torch.no_grad():
            for name in arms:
                g = output.create_group(name)
                g.create_dataset('rho', shape=(draws, 64, 48, 48), dtype='f4', compression='lzf')
                g.create_dataset('wide', shape=(draws, 48, 48, 48), dtype='f4', compression='lzf')
            for first in range(0, draws, 4):
                count = min(4, draws-first)
                tensor = lambda x: torch.as_tensor(x, device='cuda', dtype=torch.float32)[None].expand(count, *x.shape)
                base = FieldCondition(tensor(views.scale_channels(arrays['joint'], chart['joint'])),
                    tensor(views.scale_channels(arrays['wide'], chart['wide'])), tensor(np.zeros(3, 'f4')), 'joint')
                noise = s.noise_crop(global_fine, row['window_index'], phase, cap, 'fine', first, count)
                condition_rows = [conditions(ztrue, arrays['canvas_z'][draw].astype(float),
                    int(b.c.digest(['conditioning-stress-v1', phase, cap, pid, draw])[:15], 16))
                    for draw in range(first, first+count)]
                zs = {name: np.stack([r[name] for r in condition_rows])[:, None] for name in arms}
                if first == 0:
                    zs['endpoint_sampled'] = arrays['canvas_z'][first:first+count, None]
                    zs['endpoint_oracle'] = np.broadcast_to(ztrue, (count, 1, *ztrue.shape))
                for name, zvalues in zs.items():
                    z = torch.as_tensor(np.array(zvalues), device='cuda', dtype=torch.float32)
                    cond = replace(base, coarse_joint=replicate(z[(slice(None), slice(None), *a.CROP)]),
                                   coarse_wide=z, coarse_source='oracle_diagnostic')
                    latent, _ = integrate(models['fine'], 'fine', [cond], noise, single, 128)
                    wide = torch.exp(z.double()*chart['coarse_logrho']['std'][0]+chart['coarse_logrho']['mean'][0])
                    rho = decode_density(wide[(slice(None), slice(None), *a.CROP)], latent.double()*chart['fine_residual']['std'][0])
                    max_mass = max(max_mass, float(abs(block_mean(rho)/wide[(slice(None), slice(None), *a.CROP)]-1).max()))
                    rho, wide = rho[:, 0].cpu().numpy(), wide[:, 0].cpu().numpy()
                    if name.startswith('endpoint'):
                        oldname = 'canvas_coarse_local_fine' if name.endswith('sampled') else 'oracle_coarse_local_fine'
                        with h5py.File(previous['fields']['path']) as f: old = f[oldname+'/rho'][:count].astype(float)
                        error = float(np.max(abs(rho/old-1))); assert error < 2e-6
                        endpoint_errors[name] = error
                        continue
                    output[name+'/rho'][first:first+count] = rho
                    output[name+'/wide'][first:first+count] = wide
                    for j, (r, w) in enumerate(zip(rho, wide)):
                        draw = first+j
                        q, mass, u = a.shape_mass(r)
                        sm = b.op.mean_pool(arrays['completed_rho'][draw])
                        crosses = {label: a.fine_metrics(q, m, cores, band_masks) for label, m in
                                   [('native', mass), ('truemass', tm), ('sampledmass', sm)]}
                        eig = b.tidal.eigensystem(operator.tensor(r, w)).mean((1, 2, 3))
                        np.testing.assert_allclose(eig.sum(-1), [r[sl].mean()-1 for sl in cores], atol=1e-9, rtol=0)
                        arms[name].append(dict(draw=draw, cross=crosses, eigenvalues=eig.tolist(),
                            log_distance=float(np.sqrt(np.mean(np.log(w/arrays['true_wide'])**2))),
                            shape_variance=float(np.mean((q-1)**2)), latent_variance=float(np.mean(u*u))))
                print(json.dumps(dict(pair_id=pid, draws_done=first+count, seconds=time.monotonic()-before)), flush=True)
        assert max_mass < 2e-14
        rec = dict(phase=phase, pair_id=pid, draws=draws, arms=arms, truth=truth, matched=previous['matched'],
                   endpoint_errors=endpoint_errors, max_mass_error=max_mass,
                   previous=b.c.file_record(previous_path, content_hash=True), input=row['file'],
                   fields=b.c.file_record(fieldpath, content_hash=True), seconds=time.monotonic()-before)
        b.c.atomic_json(dest/(pid+'.json'), rec); cases.append(rec)
    b.c.atomic_json(dest/'COMPLETE.json', dict(cases=cases, passed=True, posterior_qualified=False,
        source_sha256=b.c.sha256(__file__), metric_source_sha256=b.c.sha256(a.__file__),
        checkpoints=hashes, normalizer=normalizer, nfe=128, draws=draws, job=os.environ['SLURM_JOB_ID'],
        scope='Four preselected development pairs, oracle conditioning interventions, common fine noise. No coverage qualification or training.'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--limit', type=int, default=0)
    p.add_argument('--draws', type=int, choices=[4, 32], default=32); args = p.parse_args()
    run(args.limit, args.draws)
