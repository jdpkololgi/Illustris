"""Targetless, cached full-cap shared-state sampling for exterior qualification.

Uses the existing probability-flow consensus unchanged. Larger-domain
calibration is an empirical question; this module makes no posterior claim.
"""
import argparse
from dataclasses import replace
import gc
import hashlib
import json
import os
from pathlib import Path
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import h5py
import numpy as np
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_coordinates as coord
from workflows.sbi import e2e_loa_field_conditions as adapter
from workflows.sbi import e2e_exterior_campaign as campaign

ATLAS = Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/loa_field_atlas_20260930_v1')
STRIDE = np.array([64, 32, 32])


def input_guard(phase):
    config = json.loads(campaign.CONFIG.read_text())
    if phase in ('ph004', 'ph005', 'ph006'):
        from workflows.sbi.e2e_exterior_legacy import guard
        guard(phase)
        return
    if phase != 'Loa' and phase not in config['prepared_training_unseen_phases']:
        raise PermissionError('Only registered targetless phase inputs or Loa allowed')
    if phase in config.get('preserved_sealed_reserve', []):
        raise PermissionError('Sealed programme reserve')


def inference_guard(phase, freeze):
    input_guard(phase)
    if phase in ('ph018', 'ph019'):
        if freeze is None:
            raise PermissionError('Fresh full-cap inference needs complete selection freeze')
        record = json.loads(freeze.read_text())
        if record.get('schema') != 'e2e-exterior-selection-freeze-v1' or phase not in record['final_phases']:
            raise PermissionError('Unregistered final inference authority')
        for name, sha in record['sources'].items():
            if c.sha256(c.REPO/name) != sha:
                raise ValueError('Frozen code changed')


def mock_plan(phase, cap):
    input_guard(phase)
    import fitsio
    from workflows.sbi.e2e_loa_field_atlas import tile_keys
    from workflows.abacus_tweb import p3br_build_random_response as response_ops
    legacy = phase in ('ph004', 'ph005', 'ph006')
    if legacy:
        from workflows.sbi import e2e_exterior_legacy as lg
        observed_path = lg.ROOT/phase/'OBSERVED.json'
        observed = lg.receipt(observed_path, phase)
    else:
        observed_path = c.ROOT/'observations'/phase/'OBSERVED_COMPLETE.json'
        observed = c.verify_receipt(observed_path)
    catalogue = fitsio.read(observed['outputs'][0]['path'], columns=['RA', 'DEC', 'Z'])
    z = catalogue['Z']
    # Same sky partition as the canonical observation builder.
    cap_index = response_ops.galactic_cap(catalogue['RA'], catalogue['DEC'])
    keep = (z >= .15) & (z < .55) & (cap_index == (cap == 'NGC'))
    xyz = coord.sky_mpc_h(catalogue['RA'][keep], catalogue['DEC'][keep], z[keep])
    raw_path = lg.ROOT/phase/(cap+'_RESPONSE.json') if legacy else coord.ROOT/'observations'/phase/(cap+'_COMPLETE.json')
    raw = lg.receipt(raw_path, phase) if legacy else coord.verify_receipt(raw_path)
    grid = raw['grid']; coord.validate_grid(grid)
    keys = np.unique(tile_keys(xyz, grid), axis=0)
    midpoints = keys*STRIDE+STRIDE//2
    # Registered support-only science cores must be covered even in sparse
    # regions with no observed galaxy in their ownership tile.
    gp = lg.ROOT/phase/'GEOMETRY.json' if legacy else coord.ROOT/'geometry'/phase/'GEOMETRY_COMPLETE.json'
    geometry = lg.receipt(gp, phase, False) if legacy else coord.verify_receipt(gp, payload=False)
    extra = []
    for row in geometry['pairs']:
        if row['cap'] == cap:
            for side in (0, 1):
                key = (np.asarray(row['center'])+[32*side, 0, 0])//STRIDE
                extra.append(key*STRIDE+STRIDE//2)
    midpoints = np.unique(np.vstack((midpoints, extra)), axis=0)
    return dict(phase=phase, cap=cap, grid=grid, midpoints=midpoints.tolist(),
                source_observations=c.file_record(observed_path, content_hash=True),
                geometry=c.file_record(gp, content_hash=True), science_galaxies=int(keep.sum()),
                observed_payload=observed['outputs'][0], support_only_extra_tiles=True)


def prepare(phase, cap, out):
    c.require_compute(); input_guard(phase)
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    if phase == 'Loa':
        plan_path = ATLAS/'PLAN.json'; old = json.loads(plan_path.read_text())
        plan = dict(phase=phase, cap=cap, grid=old['grids'][cap],
                    midpoints=[r['midpoint'] for r in old['tiles'] if r['cap'] == cap],
                    source_plan=c.file_record(plan_path, content_hash=True))
        receipt = json.loads((ATLAS/(cap+'_CACHE.json')).read_text())
        if c.sha256(receipt['outputs'][0]['path']) != receipt['outputs'][0]['sha256']:
            raise ValueError('Loa observation cache changed')
        raw_file = h5py.File(ATLAS/(cap+'_observation_cache.h5'))
        coarse_file = None
        counts, coarse = raw_file['counts_raw'], raw_file['wide']
        response = adapter.load_response(cap)
        sources = [c.file_record(ATLAS/(cap+'_CACHE.json'), content_hash=True)]
    else:
        from workflows.abacus_tweb import p3br_build_random_response as response_ops
        plan = mock_plan(phase, cap)
        if phase in ('ph004', 'ph005', 'ph006'):
            from workflows.sbi import e2e_exterior_legacy as lg
            raw_path, coarse_path, angular_path = [lg.ROOT/phase/name for name in
                (cap+'_RESPONSE.json', cap+'_FACTOR4.json', 'ANGULAR.json')]
            raw, cr, ar = [lg.receipt(p, phase) for p in (raw_path, coarse_path, angular_path)]
            if cr['raw_receipt_sha256'] != c.sha256(raw_path):
                raise ValueError('Legacy coarse/raw input identity changed')
        else:
            raw_path = coord.ROOT/'observations'/phase/(cap+'_COMPLETE.json')
            coarse_path = coord.ROOT/'conditions'/phase/(cap+'_FACTOR4.json')
            raw, cr = coord.verify_receipt(raw_path), coord.verify_receipt(coarse_path)
            if cr['binding']['response_receipt_sha256'] != c.sha256(raw_path):
                raise ValueError('Coarse/raw input identity changed')
            angular_path = c.ROOT/'observations'/phase/'angular/ANGULAR_COMPLETE.json'
            ar = c.verify_receipt(angular_path)
        with np.load(ar['outputs'][0]['path']) as f:
            angular = {k: f[k] for k in f.files}
        mask = angular['support'].astype(bool) & (angular['domain']//2 == (cap == 'NGC'))
        selection_path = Path(c.config()['observation']['fixed_selection_manifest'])
        if raw['selection']['sha256'] != c.sha256(selection_path):
            raise ValueError('Selection changed')
        selection = json.loads(selection_path.read_text())
        response = (mask, angular['angular_response'], response_ops.angular_boundary_distance(mask), selection)
        raw_file, coarse_file = h5py.File(raw['outputs'][0]['path']), h5py.File(cr['outputs'][0]['path'])
        counts, coarse = raw_file['counts'], coarse_file
        sources = [c.file_record(p, content_hash=True) for p in (raw_path, coarse_path, angular_path, selection_path)]
    from workflows.sbi.e2e_global_canvas import Geometry
    fg = Geometry.make(plan['midpoints'], 'fine')
    support = np.zeros(fg.shape, np.float32)
    coverage = np.zeros(fg.shape, np.uint16)
    hashes = []
    path = out/'CONDITIONS.h5'
    try:
        with h5py.File(path, 'x') as f:
            n = len(plan['midpoints'])
            j = f.create_dataset('joint', shape=(n, 12, 64, 48, 48), dtype='f4',
                                 chunks=(1, 12, 8, 48, 48), compression='lzf')
            w = f.create_dataset('wide', shape=(n, 12, 48, 48, 48), dtype='f4',
                                 chunks=(1, 12, 8, 48, 48), compression='lzf')
            for i, midpoint in enumerate(plan['midpoints']):
                arrays = adapter.build(plan['grid'], midpoint, cap, counts, coarse, response)
                j[i], w[i] = arrays['joint'], arrays['wide']
                digest = hashlib.sha256()
                for key in ('joint', 'wide'):
                    digest.update(arrays[key].tobytes())
                hashes.append(digest.hexdigest())
                sl = fg.crop(i)[2:]
                support[sl] = np.maximum(support[sl], arrays['support']); coverage[sl] += 1
                if i % 100 == 0:
                    print(json.dumps(dict(phase=phase, cap=cap, prepared=i+1, total=n, seconds=time.monotonic()-start)), flush=True)
            f.create_dataset('support', data=support, compression='lzf')
            f.create_dataset('coverage', data=coverage, compression='lzf')
            f.attrs.update(phase=phase, cap=cap, targetless=True)
    finally:
        raw_file.close()
        if coarse_file is not None:
            coarse_file.close()
    c.atomic_json(out/'PLAN.json', plan)
    record = dict(phase=phase, cap=cap, plan_sha256=c.sha256(out/'PLAN.json'), windows=len(plan['midpoints']),
                  outputs=[c.file_record(path, content_hash=True)], sources=sources,
                  condition_sha256=hashes, seconds=time.monotonic()-start, targetless=True,
                  source_sha256=c.sha256(__file__), adapter_sha256=c.sha256(adapter.__file__),
                  job=os.environ['SLURM_JOB_ID'])
    c.atomic_json(out/'PREPARED.json', record)
    print(json.dumps({k: v for k, v in record.items() if k not in ('condition_sha256', 'sources')}), flush=True)


def run(root, out, first=0, count=4, nfe=128, benchmark=False, freeze=None, model_seed=17, comparison=None):
    c.require_compute()
    import torch
    from workflows.sbi import e2e_loa_field_sampling as sampling
    from workflows.sbi import e2e_cfm_pilot_evaluate as old
    from workflows.sbi import e2e_coupled_views as views
    from workflows.sbi.e2e_coupled_benchmark_models import FieldCondition
    from workflows.sbi.e2e_vdm_context_models import project, replicate, decode_density, block_mean
    from workflows.sbi.e2e_global_canvas import Geometry, integrate
    prepared = json.loads((root/'PREPARED.json').read_text())
    phase, cap = prepared['phase'], prepared['cap']
    inference_guard(phase, freeze)
    from workflows.sbi.e2e_cfm_mixture_contract import sampler_seed_guard
    registered = None if comparison is None else json.loads(comparison.read_text())
    sampler_seed_guard(model_seed, phase, benchmark, freeze, registered)
    if count != 4 or first < 0 or first % 4 or nfe != 128:
        raise ValueError('Registered four-draw NFE128 batch required')
    if not prepared['targetless'] or prepared['plan_sha256'] != c.sha256(root/'PLAN.json'):
        raise ValueError('Input plan changed')
    if c.sha256(prepared['outputs'][0]['path']) != prepared['outputs'][0]['sha256']:
        raise ValueError('Condition cache changed')
    plan = json.loads((root/'PLAN.json').read_text())
    sampling.setup(True)
    cg, fg = (Geometry.make(plan['midpoints'], stage) for stage in ('coarse', 'fine'))
    models, hashes, normalizer = old.load_models(model_seed, 13312, 26624); chart = views.load_chart(normalizer)
    if registered is not None:
        if (first+count > 32 or hashes != registered['checkpoints'][str(model_seed)]
                or normalizer != registered['normalizer']
                or c.sha256(root/'PREPARED.json') != registered['prepared'][phase+'/'+cap]
                or c.sha256(__file__) != registered['sources']['workflows/sbi/e2e_exterior_fullcap.py']):
            raise ValueError('Mixture generation binding changed')
    if freeze is not None:
        selected = json.loads(freeze.read_text())
        if (hashes != selected['checkpoints'] or normalizer != selected['normalizer']
                or nfe != selected['nfe'] or count != 4 or first+count > selected['draws']):
            raise ValueError('Frozen sampler or checkpoints changed')
    out.mkdir(parents=True, exist_ok=False)
    binding = dict(phase=phase, cap=cap, first=first, draws=count, nfe=nfe, benchmark=benchmark,
                   model_seed=model_seed,
                   comparison_sha256=None if comparison is None else c.sha256(comparison),
                   checkpoints=hashes, normalizer=normalizer, prepared_sha256=c.sha256(root/'PREPARED.json'),
                   runner_sha256=c.sha256(__file__),
                   integrator_sha256=c.sha256(c.REPO/'workflows/sbi/e2e_global_canvas.py'),
                   freeze_sha256=None if freeze is None else c.sha256(freeze), posterior_qualified=False)
    c.atomic_json(out/'BINDING.json', binding)
    base = []; start = time.monotonic()
    tensor = lambda x: torch.as_tensor(x, device='cuda', dtype=torch.float32)[None].expand(count, *x.shape)
    with h5py.File(prepared['outputs'][0]['path']) as f:
        support, expected_coverage = f['support'][:], f['coverage'][:]
        for i in range(len(plan['midpoints'])):
            joint, wide = f['joint'][i], f['wide'][i]
            digest = hashlib.sha256(joint.tobytes()+wide.tobytes()).hexdigest()
            if digest != prepared['condition_sha256'][i]:
                raise ValueError('Condition array changed')
            base.append(FieldCondition(tensor(views.scale_channels(joint, chart['joint'])),
                        tensor(views.scale_channels(wide, chart['wide'])), tensor(np.zeros(3, 'f4')), 'wide'))
    loading = time.monotonic()-start; start = time.monotonic(); torch.cuda.reset_peak_memory_stats()

    def noise(geometry, stage):
        rows = []
        for draw in range(first, first+count):
            address = ['global-cap-v1', cap, stage, draw] if phase == 'Loa' else ['global-exterior-cap-v1', phase, cap, stage, draw]
            rng = torch.Generator(device='cuda').manual_seed(int(c.digest(address)[:15], 16))
            rows.append(torch.randn((1, 1, *geometry.shape), device='cuda', generator=rng))
        value = torch.cat(rows)
        return project(value) if stage == 'fine' else value

    def report(stage):
        def callback(step, state):
            if step % 8 == 0:
                print(json.dumps(dict(phase=phase, cap=cap, first=first, stage=stage, step=step,
                                      seconds=time.monotonic()-start)), flush=True)
        return callback

    effective_nfe = 2 if benchmark else nfe
    z, cw = integrate(models['coarse'], 'coarse', base, noise(cg, 'coarse'), cg, effective_nfe, report('coarse'))
    fine = []
    for i, b in enumerate(base):
        zw = z[cg.crop(i)]
        fine.append(replace(b, region='joint', coarse_joint=replicate(zw[:, :, 16:32, 18:30, 18:30]),
                            coarse_wide=zw, coarse_source='sampled'))
    u, fw = integrate(models['fine'], 'fine', fine, noise(fg, 'fine'), fg, effective_nfe, report('fine'))
    torch.cuda.synchronize()
    elapsed = time.monotonic()-start
    if not np.array_equal(fw[0, 0].cpu().numpy(), expected_coverage):
        raise ValueError('Coverage mismatch')
    common = dict(phase=phase, cap=cap, first=first, draws=count, windows=len(base), loading_seconds=loading,
                  sampling_seconds=elapsed, peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,
                  fine_shape=list(map(int, fg.shape)), coarse_shape=list(map(int, cg.shape)),
                  binding=c.sha256(out/'BINDING.json'), job=os.environ['SLURM_JOB_ID'])
    if benchmark:
        record = dict(common, passed=True, posterior_output=False, nfe=2)
        c.atomic_json(out/'BENCHMARK.json', record); print(json.dumps(record), flush=True); return
    offset = np.asarray(fg.origin)//4-np.asarray(cg.origin)
    origin = np.asarray(plan['grid']['origin_mpc_h'])+np.asarray(fg.origin)*6.766
    mass = 0.; path = out/'FIELD.h5'
    with h5py.File(path, 'x') as f:
        ds = f.create_dataset('rho', shape=(count, *fg.shape), dtype='f4', chunks=(1, 16, 32, 32), compression='lzf')
        f.create_dataset('support', data=support, compression='lzf')
        f.create_dataset('coverage', data=expected_coverage, compression='lzf')
        f.create_dataset('coarse_logrho_standardized', data=z[:, 0].cpu().numpy(), compression='lzf')
        f.attrs.update(origin_mpc_h=origin, cell_mpc_h=6.766, phase=phase, cap=cap, first_draw=first,
                       posterior_qualified=False, shared_state=True, coarse_origin_raw_cells=np.asarray(cg.origin)*8)
        for lo in range(0, fg.shape[0], 16):
            hi = min(lo+16, fg.shape[0]); startc = offset+[lo//4, 0, 0]
            size = np.array([hi-lo, fg.shape[1], fg.shape[2]])//4
            cut = (slice(None), slice(None), *(slice(int(a), int(a+n)) for a, n in zip(startc, size)))
            coarse = torch.exp(z[cut].double()*chart['coarse_logrho']['std'][0]+chart['coarse_logrho']['mean'][0])
            rho = decode_density(coarse, u[:, :, lo:hi].double()*chart['fine_residual']['std'][0])
            if not torch.isfinite(rho).all() or not (rho > 0).all():
                raise ValueError('Invalid full-cap density')
            mass = max(mass, float(((block_mean(rho)-coarse)/coarse).abs().max()))
            values = rho[:, 0].cpu().numpy().astype('f4'); values[:, expected_coverage[lo:hi] == 0] = np.nan
            ds[:, lo:hi] = values
    if mass > 2e-6:
        raise ValueError('Full-cap block conservation failed')
    record = dict(common, passed=True, max_mass_error=mass, field=c.file_record(path, content_hash=True), posterior_qualified=False)
    c.atomic_json(out/'COMPLETE.json', record); print(json.dumps(record), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['prepare', 'run'])
    p.add_argument('--phase'); p.add_argument('--cap', choices=['NGC', 'SGC']); p.add_argument('--root', type=Path)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--first', type=int, default=0)
    p.add_argument('--benchmark', action='store_true'); p.add_argument('--freeze', type=Path)
    a = p.parse_args()
    if a.action == 'prepare': prepare(a.phase, a.cap, a.out)
    else: run(a.root, a.out, a.first, benchmark=a.benchmark, freeze=a.freeze)
