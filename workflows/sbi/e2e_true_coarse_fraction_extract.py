"""Push true-coarse parent fields through the unchanged Stage C cap operators."""
import argparse
import gc
import os
import sys
import time

import e2e_true_coarse_fraction_saved as io
SOURCE = io.BASE/'source_78156ac3417555ef269f'
sys.path.insert(0, str(SOURCE))
sys.path.insert(1, str(io.OLD/'source/workflows/sbi'))
import h5py
import numpy as np
import torch
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_spatial_exterior as spatial
from workflows.sbi.e2e_exterior_cap_tides import CapTidal
from workflows.sbi.e2e_loa_field_sampling import setup
import e2e_stage_b_prepare as prep
import e2e_density_only_reconstruction as reconstruction
b = prep.b


def run(phase, cap, first):
    io.compute_guard(); setup(True); torch.set_num_threads(4)
    if phase not in io.PHASES or cap not in ('NGC', 'SGC') or first not in range(0, 32, 4) or c.REPO != SOURCE:
        raise PermissionError('Registered development-only draw addresses required')
    started = time.monotonic(); dest = io.OUT/'cores'/phase/cap/f'd{first:03d}'
    dest.mkdir(parents=True, exist_ok=False)
    sources = [io.record(p) for p in (__file__, io.__file__, spatial.__file__, reconstruction.__file__,
        sys.modules[CapTidal.__module__].__file__, b.tidal.__file__)]
    field_path = io.OUT/'fields'/phase/cap/f'd{first:03d}/COMPLETE.json'
    field = io.read(field_path); io.record(field_path); io.verify(field['field'])
    if not field['passed'] or field['phase'] != phase or field['cap'] != cap or field['first'] != first or field['nfe'] != 128 or field['draws'] != 4 or field['parent_checkpoint']['sha256'] != io.FINE_SHA:
        raise ValueError('Matched B field contract failed')
    parent_path = io.OLD/'cores'/phase/cap/'parent/s0'/f'd{first:03d}/COMPLETE.json'
    parent = io.read(parent_path); io.record(parent_path); io.verify(parent['output']); io.verify(parent['binding'])
    with np.load(parent['output']['path']) as f:
        truth_eigen = f['truth_eigen']; weights = f['galaxy_counts']; ids = f['pair_id']
    rows = [r for r in b.auth.geometry(phase, b.FREEZE) if r['cap'] == cap]
    if list(ids) != [r['pair_id'] for r in rows] or len(rows) != 8 or np.any(weights.sum(-1) <= 0):
        raise ValueError('Stage C case/weight mismatch')
    truth_labels = (truth_eigen > .2).sum(-1)
    prior_path = io.BASE/'prior/PRIOR.json'; fit = io.read(prior_path); io.record(prior_path)
    if fit['training_phases'] != list(c.TRAIN):
        raise ValueError('Unregistered exterior prior')
    labels = np.empty((2, 8, 4, 2, 4096), dtype='u1')
    densitycores = np.empty((8, 4, 2, 4096)); diagnostics = []; maxtrace = 0.
    with h5py.File(field['field']['path']) as f:
        shape = f['rho'].shape[1:]; origin = np.asarray(f.attrs['origin_mpc_h']); support = f['support'][:]
        if f.attrs['phase'] != phase or f.attrs['cap'] != cap or f.attrs['first_draw'] != first:
            raise ValueError('Field HDF5 address mismatch')
        prior = spatial.CudaSpatialPrior(tuple(n+64 for n in shape), fit, workers=4)
        operator = CapTidal(shape, 32); coarseop = CapTidal((32, 16, 16), 32); fineop = CapTidal((64, 32, 32), 64)
        corecuts = [b.cuts(r, origin) for r in rows]
        patches = [tuple(slice(v.start-33, v.stop+33) for v in b.cuts(r, origin-32*6.766)) for r in rows]
        for draw in range(4):
            seed = int(c.digest(['spatial-exterior-cap-v1', phase, cap, first+draw])[:15], 16)
            completed, diag = spatial.complete_density(f['rho'][draw], support, prior, np.random.default_rng(seed), 'learned', 32)
            density = completed.astype('f4').astype(float)
            tensor = np.empty((8, 2, 16, 16, 16, 6))
            for k, component in enumerate(operator.components(density)):
                for j, core in enumerate(corecuts):
                    tensor[j, ..., k] = component[core].reshape(2, 16, 16, 16)
            for j, (core, patch) in enumerate(zip(corecuts, patches)):
                coarse = density[patch]; fine = reconstruction.reconstruct(coarse, 'linear')[2:-2, 2:-2, 2:-2]
                correction = np.empty_like(tensor[j])
                for k, (tc, tf) in enumerate(zip(coarseop.components(coarse[1:-1, 1:-1, 1:-1]), fineop.components(fine))):
                    correction[..., k] = (b.op.mean_pool(tf, 2)-tc).reshape(2, 16, 16, 16)
                maxtrace = max(maxtrace, float(abs(correction[..., [0, 3, 5]].sum(-1)).max()))
                for mi, t in enumerate((tensor[j], tensor[j]+correction)):
                    eigen = b.tidal.eigensystem(t).reshape(2, 4096, 3)
                    labels[mi, j, draw] = (eigen > .2).sum(-1)
                    np.testing.assert_allclose(eigen.sum(-1), f['rho'][(draw, *core)].astype(float).reshape(2, 4096)-1, atol=1e-9, rtol=0)
                densitycores[j, draw] = f['rho'][(draw, *core)].reshape(2, 4096)
            if maxtrace > 1e-9:
                raise ValueError('Correction trace identity failed')
            diag.update(draw=first+draw, seconds=time.monotonic()-started); diagnostics.append(diag)
            print(io.json.dumps(dict(stage='B_core_extract', phase=phase, cap=cap, draw=first+draw, seconds=time.monotonic()-started)), flush=True)
            del completed, density, tensor, component, coarse, fine, correction; gc.collect()
    fractions = np.stack([((labels == k)*weights[None, :, None]).sum(-1)/weights.sum(-1)[None, :, None] for k in range(4)], -1)
    truth = np.stack([((truth_labels == k)*weights).sum(-1)/weights.sum(-1) for k in range(4)], -1)
    payload = dest/'CORES.npz'
    with payload.open('xb') as stream:
        np.savez_compressed(stream, labels=labels, fractions=fractions, truth_fractions=truth,
            truth_labels=truth_labels, galaxy_counts=weights, density=densitycores, pair_id=ids)
    for rec in sources:
        if io.sha(rec['path']) != rec['sha256']:
            raise ValueError('Source changed during extraction')
    io.write(dest/'COMPLETE.json', dict(passed=True, phase=phase, cap=cap, first=first, draws=4,
        cases=parent['cases'], output=io.record(payload), sources=sources, inputs=list(io.CACHE.values()),
        diagnostics=diagnostics, correction_trace_maxabs=maxtrace,
        scope='one parent seed, two reused phases, diagnostic only', posterior_qualified=False,
        job=os.environ['SLURM_JOB_ID'], seconds=time.monotonic()-started))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--phase', choices=io.PHASES, required=True)
    p.add_argument('--cap', choices=('NGC', 'SGC'), required=True); p.add_argument('--first', type=int, required=True)
    a = p.parse_args(); run(a.phase, a.cap, a.first)
