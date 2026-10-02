"""Matched connected-field exterior development panel, with targetless inference."""
import argparse
import json
import os
from pathlib import Path
import time
import h5py
import numpy as np
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_coordinates as coord
from workflows.sbi import e2e_coupled_conditions as conditions
from workflows.sbi import e2e_coupled_target_products as targets
from workflows.sbi import e2e_exterior_campaign as campaign
from workflows.sbi import e2e_spatial_exterior as spatial
from workflows.sbi import e2e_footprint_tidal_operator as tidal

METHODS = ('zero', 'unconditional', 'conditional', 'learned', 'periodic')


def prepare(phase, out):
    campaign.phase_guard(phase, 'development')
    c.require_compute()
    from workflows.sbi import e2e_loa_field_conditions as adapter
    from workflows.sbi.e2e_cfm_field_gallery import compare_channels
    from workflows.abacus_tweb import p3br_build_random_response as response_ops
    gp = coord.ROOT/'geometry'/phase/'GEOMETRY_COMPLETE.json'
    geometry = coord.verify_receipt(gp, payload=False)
    selection_path = Path(c.config()['observation']['fixed_selection_manifest'])
    selection = json.loads(selection_path.read_text())
    ap = c.ROOT/'observations'/phase/'angular/ANGULAR_COMPLETE.json'
    ar = c.verify_receipt(ap)
    with np.load(ar['outputs'][0]['path']) as f:
        angular = {k: f[k] for k in f.files}
    out.mkdir(parents=True, exist_ok=False)
    cases = []
    for cap in ('NGC', 'SGC'):
        raw_path = coord.ROOT/'observations'/phase/(cap+'_COMPLETE.json')
        coarse_path = coord.ROOT/'conditions'/phase/(cap+'_FACTOR4.json')
        raw, coarse = coord.verify_receipt(raw_path), coord.verify_receipt(coarse_path)
        if coarse['binding']['response_receipt_sha256'] != c.sha256(raw_path):
            raise ValueError('Coarse input lineage changed')
        if raw['selection']['sha256'] != c.sha256(selection_path):
            raise ValueError('Selection lineage changed')
        mask = angular['support'].astype(bool) & (angular['domain']//2 == (cap == 'NGC'))
        response = (mask, angular['angular_response'], response_ops.angular_boundary_distance(mask), selection)
        with h5py.File(raw['outputs'][0]['path']) as r, h5py.File(coarse['outputs'][0]['path']) as w:
            for row in (v for v in geometry['pairs'] if v['cap'] == cap):
                pair_id = row['pair_id']; mid = np.asarray(row['center'])+[16, 0, 0]
                arrays = [adapter.build(row['grid'], mid+[64*i, 0, 0], cap, r['counts'], w, response)
                          for i in range(2)]
                original = conditions.crop_context(conditions.load_pair(phase, pair_id), phase)
                if not all(v['passed'] for v in compare_channels(arrays[0], original)):
                    raise ValueError('Registered left observation replay failed')
                for name, shift in [('joint', 32), ('wide', 8)]:
                    if not np.allclose(arrays[0][name][:, shift:], arrays[1][name][:, :-shift], rtol=2e-5, atol=2e-5):
                        raise ValueError('Overlapping observations inconsistent')
                directory = out/pair_id; directory.mkdir()
                files = []
                for i, values in enumerate(arrays):
                    path = directory/f'conditions_{i}.npz'; np.savez_compressed(path, **values)
                    files.append(c.file_record(path, content_hash=True))
                # A separate target object is used ONLY by the later score action.
                truth, source = campaign.target(phase, pair_id, 'development')
                path = directory/'truth.npz'; np.savez_compressed(path, **truth)
                cases.append(dict(phase=phase, pair_id=pair_id, cap=cap, shell=row['shell'],
                                  support_stratum=row['support_stratum'], conditions=files,
                                  truth=c.file_record(path, content_hash=True), truth_source=source))
                print('prepared', pair_id, flush=True)
    c.atomic_json(out/'PREPARED.json', dict(phase=phase, cases=cases,
                  geometry=c.file_record(gp, content_hash=True), job=os.environ['SLURM_JOB_ID']))


class CoreTidal:
    """Free-space convolution to the original contiguous two-core target only."""
    def __init__(self, source_shape, padding):
        import torch
        self.t = torch
        self.target_shape = (32, 16, 16)
        kernels = tidal.kernel(source_shape, self.target_shape,
                               offset=(padding+16, padding+16, padding+16))
        self.shape = kernels.shape[1:]
        self.spectra = [torch.fft.rfftn(torch.as_tensor(k, device='cuda')) for k in kernels]

    def __call__(self, rho):
        t = self.t
        source = t.fft.rfftn(t.as_tensor(rho-1., device='cuda', dtype=t.float64), s=self.shape)
        crop = tuple(slice(0, n) for n in self.target_shape)
        tensor = np.stack([t.fft.irfftn(source*k, s=self.shape)[crop].cpu().numpy()
                           for k in self.spectra], axis=-1)
        return tensor.reshape(2, 16, 16, 16, 6)


def load(item):
    if c.sha256(item['path']) != item['sha256']:
        raise ValueError('Bound input changed')
    with np.load(item['path']) as f:
        return {k: f[k] for k in f.files}


def run(root, out, prior_path, draws=32, nfe=128, padding=16, limit=None):
    c.require_compute()
    from workflows.sbi import e2e_loa_field_sampling as sampling
    from workflows.sbi import e2e_global_sampler as sampler
    from workflows.sbi import e2e_cfm_pilot_evaluate as old
    from workflows.sbi import e2e_coupled_views as views
    from workflows.sbi.e2e_global_mock_evaluate import density_statistics
    from workflows.sbi.e2e_loa_field_tensor import consistent
    prepared = json.loads((root/'PREPARED.json').read_text())
    campaign.phase_guard(prepared['phase'], 'development')
    if draws < 4 or draws % 4 or padding not in (16, 32, 64):
        raise ValueError('Unregistered draw count or domain')
    out.mkdir(parents=True, exist_ok=True)
    fit = json.loads(prior_path.read_text())
    if fit['training_phases'] != list(c.TRAIN):
        raise ValueError('Spatial prior is not training-only')
    sampling.setup(True)
    models, hashes, normalizer = old.load_models(17, 13312, 26624)
    chart = views.load_chart(normalizer)
    source_shape = tuple(n+2*padding for n in (96, 48, 48))
    prior = spatial.CudaSpatialPrior(source_shape, fit, workers=4)
    operator = CoreTidal(source_shape, padding)
    binding = dict(draws=draws, nfe=nfe, padding=padding, methods=list(METHODS),
                   prior_sha256=c.sha256(prior_path), prepared_sha256=c.sha256(root/'PREPARED.json'),
                   checkpoints=hashes, normalizer=normalizer,
                   sources={Path(v.__file__).name: c.sha256(v.__file__) for v in (sampler, spatial, tidal)},
                   runner_sha256=c.sha256(__file__))
    bp = out/'BINDING.json'
    if bp.exists():
        if json.loads(bp.read_text()) != binding:
            raise ValueError('Changed run binding')
    else:
        c.atomic_json(bp, binding)
    completed = []
    for case in prepared['cases'][:limit]:
        campaign.phase_guard(case['phase'], 'development')
        directory = out/case['pair_id']; directory.mkdir(exist_ok=True)
        marker = directory/'COMPLETE.json'
        if marker.exists():
            record = json.loads(marker.read_text())
            if record['binding'] != c.digest(binding):
                raise ValueError('Changed completed-case binding')
            for item in record['outputs']:
                if c.sha256(item['path']) != item['sha256']:
                    raise ValueError('Completed output changed')
            completed.append(record)
            continue
        arrays = [load(item) for item in case['conditions']]
        support = np.concatenate((arrays[0]['support'], arrays[1]['support'][32:]), axis=0)
        rows = {name: [] for name in METHODS}; statistics = []; diagnostics = []; fields = []
        start = time.monotonic()
        for first in range(0, draws, 4):
            sample = sampler.generate(models, arrays, chart, case['pair_id'], first, 4, nfe)
            for i in range(4):
                rho = np.concatenate((sample['rho'][0][i], sample['rho'][1][i][32:]), axis=0)
                fields.append(rho.astype('f4'))
                statistics.append(density_statistics(sample['rho'][0][i]))
                for method in METHODS:
                    before = time.monotonic()
                    if method == 'periodic':
                        tensor = targets.owned_core_arrays(consistent(sample['rho'][0][i], sample['wide'][0][i]))
                        density = targets.owned_core_arrays(sample['rho'][0][i])
                        numerical = {}
                    else:
                        seed = int(c.digest(['spatial-exterior-v1', case['pair_id'], first+i])[:15], 16)
                        field, numerical = spatial.complete_density(rho, support, prior, np.random.default_rng(seed), method, padding)
                        tensor = operator(field)
                        crop = field[padding:padding+64, padding:padding+48, padding:padding+48]
                        density = targets.owned_core_arrays(crop)
                        trace_error = float(np.max(abs(tensor[..., [0, 3, 5]].sum(-1)-(density-1))))
                        if trace_error > 1e-9*max(1., float(np.max(density))):
                            raise ValueError('Completed-field free-space trace identity failed')
                        numerical['trace_maxabs'] = trace_error
                    rows[method].append(dict(tidal=tidal.eigensystem(tensor), density=density-1))
                    diagnostics.append(dict(numerical, draw=first+i, method=method, seconds=time.monotonic()-before))
            print(json.dumps(dict(pair=case['pair_id'], generated=first+4, seconds=time.monotonic()-start)), flush=True)
        outputs = []
        for method, values in rows.items():
            path = directory/(method+'.npz')
            np.savez_compressed(path, **{k: np.stack([r[k] for r in values]) for k in values[0]})
            outputs.append(c.file_record(path, content_hash=True))
        path = directory/'shared_density.npz'
        np.savez_compressed(path, rho=np.stack(fields), support=support,
                            **{k: np.stack([v[k] for v in statistics]) for k in statistics[0]})
        outputs.append(c.file_record(path, content_hash=True))
        record = dict(pair_id=case['pair_id'], phase=case['phase'], binding=c.digest(binding),
                      diagnostics=diagnostics, outputs=outputs, seconds=time.monotonic()-start,
                      job=os.environ['SLURM_JOB_ID'])
        c.atomic_json(marker, record); completed.append(record)
        print(json.dumps(dict(case_complete=case['pair_id'], seconds=record['seconds'])), flush=True)
    c.atomic_json(out/'RUN_COMPLETE.json', dict(cases=len(completed), total=len(prepared['cases']),
                  panel_complete=len(completed) == len(prepared['cases']), binding=c.digest(binding)))


def score(root, samples, out):
    c.require_compute()
    from workflows.sbi.e2e_global_mock_evaluate import metric_summary, density_statistics, controls
    from workflows.sbi.e2e_vdm_context_metrics import fair_energy, variogram_score
    from workflows.sbi.e2e_exterior_metrics import joint_rank, distance_masks
    prepared = json.loads((root/'PREPARED.json').read_text())
    campaign.phase_guard(prepared['phase'], 'development')
    binding = json.loads((samples/'BINDING.json').read_text())
    if binding['prepared_sha256'] != c.sha256(root/'PREPARED.json'):
        raise ValueError('Scoring inputs changed')
    results = []
    for case in prepared['cases']:
        marker = samples/case['pair_id']/'COMPLETE.json'
        if not marker.exists():
            continue
        record = json.loads(marker.read_text())
        if record['binding'] != c.digest(binding):
            raise ValueError('Scoring run changed')
        truth = load(case['truth'])
        eigen = tidal.eigensystem(truth['fullbox_tensor_cores'])
        density = targets.owned_core_arrays(truth['rho_joint'])-1
        support = load(case['conditions'][0])['support']
        masks = distance_masks(support)
        row = {k: case[k] for k in ('pair_id', 'phase', 'cap', 'shell', 'support_stratum')}
        row['methods'] = {}
        for item in record['outputs']:
            name = Path(item['path']).stem; data = load(item)
            if name == 'shared_density':
                features = density_statistics(truth['rho_joint'])
                row['shared_density'] = {k: metric_summary(data[k], features[k]) for k in features}
                row['shared_density']['seam_variogram'] = variogram_score(data['seam_probes'], features['seam_probes'])
                row['shared_density']['seam_energy'] = float(fair_energy(data['seam_probes'], features['seam_probes']))
                row['shared_density']['joint_regional_rank'] = joint_rank(data['regional'], features['regional'])
                shuffled = data['seam_probes'].copy()
                shuffled[:, 4:] = shuffled[np.random.default_rng(914).permutation(len(shuffled)), 4:]
                row['shared_density']['seam_variogram_shuffled'] = variogram_score(shuffled, features['seam_probes'])
                continue
            metrics = dict(density=metric_summary(data['density'], density),
                           tidal=[metric_summary(data['tidal'][..., j], eigen[..., j]) for j in range(3)])
            core_draws = data['density'].mean((2, 3, 4))
            core_truth = density.mean((1, 2, 3))
            tidal_regional = data['tidal'].mean((2, 3, 4)).reshape(len(data['tidal']), 6)
            tidal_truth = eigen.mean((1, 2, 3)).reshape(6)
            metrics['core_mass'] = metric_summary(core_draws, core_truth)
            metrics['joint_core_mass'] = joint_rank(core_draws, core_truth)
            metrics['joint_tidal_regional'] = joint_rank(tidal_regional, tidal_truth)
            metrics['core_mass_energy'] = float(fair_energy(core_draws, core_truth))
            metrics['core_mass_variogram'] = variogram_score(core_draws, core_truth)
            metrics['distance_splits'] = {label: dict(cells=int(mask.sum()),
                density=metric_summary(data['density'][:, mask], density[mask]),
                tidal=[metric_summary(data['tidal'][..., j][:, mask], eigen[..., j][mask]) for j in range(3)])
                for label, mask in masks.items() if mask.any()}
            for threshold in (0., .2):
                classes = (data['tidal'] > threshold).sum(-1)
                probabilities = np.stack([(classes == j).mean(0) for j in range(4)], axis=-1)
                labels = (eigen > threshold).sum(-1)
                metrics[f'brier_{threshold}'] = float(np.mean(np.sum((probabilities-np.eye(4)[labels])**2, axis=-1)))
            row['methods'][name] = metrics
        results.append(row)
    out.mkdir(parents=True, exist_ok=False)
    c.atomic_json(out/'SCORES.json', dict(cases=results, binding=binding, posterior_qualified=False,
                  controls=controls(binding['draws']), panel_complete=len(results) == len(prepared['cases'])))
    for method in METHODS:
        print(json.dumps(dict(method=method, cases=len(results), coverage90=[
            float(np.mean([r['methods'][method]['tidal'][j]['coverage']['0.9'] for r in results]))
            for j in range(3)])), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['prepare', 'run', 'score'])
    p.add_argument('--phase'); p.add_argument('--root', type=Path); p.add_argument('--samples', type=Path)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--prior', type=Path)
    p.add_argument('--draws', type=int, default=32); p.add_argument('--nfe', type=int, default=128)
    p.add_argument('--padding', type=int, default=16); p.add_argument('--limit', type=int)
    a = p.parse_args()
    if a.action == 'prepare': prepare(a.phase, a.out)
    elif a.action == 'run': run(a.root, a.out, a.prior, a.draws, a.nfe, a.padding, a.limit)
    else: score(a.root, a.samples, a.out)
