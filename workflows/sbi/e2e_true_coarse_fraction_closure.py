"""Class-fraction truth closure with identical cap operators and supports.

Two controls: all-true source cuboid (deterministic) and true covered density
with the unchanged learned exterior completion and 32 original noise addresses.
No neural field generation, training, or changes to existing scripts.
"""
import argparse
import gc
import os
from pathlib import Path
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


def run(phase, cap):
    io.compute_guard(); setup(True); torch.set_num_threads(4)
    if phase not in io.PHASES or cap not in ('NGC', 'SGC') or c.REPO != SOURCE:
        raise PermissionError('Frozen development-only evaluator required')
    dest = io.OUT/'truth_closure'/phase/cap; dest.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = io.record(io.OUT/'MANIFEST.json')
    prepared_path = io.BASE/'verification/true_density_closure_20261005_v1'/phase/'PREPARED.json'
    prepared = io.read(prepared_path); io.record(prepared_path)
    if not prepared['passed'] or prepared['freeze_sha256'] != io.record(b.FREEZE)['sha256']:
        raise ValueError('Truth source binding changed')
    rec = next(r for r in prepared['records'] if r['cap'] == cap); io.verify(rec['source'])
    source_records = [io.record(q) for q in (__file__, io.__file__, spatial.__file__, reconstruction.__file__,
        sys.modules[CapTidal.__module__].__file__, b.tidal.__file__)]
    with h5py.File(rec['source']['path']) as f:
        true_source = f['rho'][:]; shape = tuple(map(int, f.attrs['target_shape']))
        origin = np.asarray(f.attrs['origin_mpc_h']); pad = int(f.attrs['padding'])
        if pad != 32 or f.attrs['phase'] != phase or f.attrs['cap'] != cap:
            raise ValueError('Truth source address changed')
    parent_path = io.BASE/'fullcap_density'/phase/cap/'d000/COMPLETE.json'
    parent = io.read(parent_path); io.record(parent_path); io.verify(parent['field'])
    with h5py.File(parent['field']['path']) as f:
        support = f['support'][:]; coverage = f['coverage'][:]
        if tuple(f['rho'].shape[1:]) != shape:
            raise ValueError('Cap geometry mismatch')
        np.testing.assert_array_equal(f.attrs['origin_mpc_h'], origin)
    rows = [r for r in b.auth.geometry(phase, b.FREEZE) if r['cap'] == cap]
    if len(rows) != 8:
        raise ValueError('Eight registered pairs per cap required')
    selection_path = prep.ROOT/'prepare/COMPLETE.json'
    selections = io.read(selection_path); io.record(selection_path)
    weights = []; target_labels = []; target_eigen = []
    for row in rows:
        truth, provenance = b.auth.truth(phase, row['pair_id'], b.FREEZE); io.verify(provenance)
        target = b.tidal.eigensystem(truth['fullbox_tensor_cores']).reshape(2, 4096, 3)
        selected = next(q for q in selections['cases'] if q['pair_id'] == row['pair_id']); io.verify(selected['output'])
        with np.load(selected['output']['path']) as f:
            weights.append(f['cell_counts'])
            np.testing.assert_array_equal((target > .2).sum(-1), f['truth_cell_labels'][0])
        target_eigen.append(target); target_labels.append((target > .2).sum(-1))
    weights = np.asarray(weights); target_labels = np.asarray(target_labels)
    corecuts = [b.cuts(row, origin) for row in rows]
    patches = [tuple(slice(v.start-33, v.stop+33) for v in b.cuts(row, origin-32*6.766)) for row in rows]
    operator = CapTidal(shape, 32)
    coarseop = CapTidal((32, 16, 16), 32); fineop = CapTidal((64, 32, 32), 64)
    maxtrace = 0.

    def labels(density):
        nonlocal maxtrace
        tensor = np.empty((8, 2, 16, 16, 16, 6))
        for k, component in enumerate(operator.components(density)):
            for j, core in enumerate(corecuts):
                tensor[j, ..., k] = component[core].reshape(2, 16, 16, 16)
        result = np.empty((2, 8, 2, 4096), dtype='u1')
        for j, (core, patch) in enumerate(zip(corecuts, patches)):
            coarse = density[patch]
            fine = reconstruction.reconstruct(coarse, 'linear')[2:-2, 2:-2, 2:-2]
            correction = np.empty_like(tensor[j])
            for k, (tc, tf) in enumerate(zip(coarseop.components(coarse[1:-1, 1:-1, 1:-1]), fineop.components(fine))):
                correction[..., k] = (b.op.mean_pool(tf, 2)-tc).reshape(2, 16, 16, 16)
            maxtrace = max(maxtrace, float(abs(correction[..., [0, 3, 5]].sum(-1)).max()))
            if maxtrace > 1e-9:
                raise ValueError('Correction trace identity failed')
            for mi, t in enumerate((tensor[j], tensor[j]+correction)):
                result[mi, j] = (b.tidal.eigensystem(t).reshape(2, 4096, 3) > .2).sum(-1)
        return result

    ideal = labels(true_source)
    fit_path = io.BASE/'prior/PRIOR.json'; fit = io.read(fit_path); io.record(fit_path)
    if fit['training_phases'] != list(c.TRAIN):
        raise ValueError('Exterior prior is not train-only')
    prior = spatial.CudaSpatialPrior(tuple(n+64 for n in shape), fit, workers=4)
    interior = true_source[32:-32, 32:-32, 32:-32].copy()
    interior[coverage == 0] = np.nan
    production = np.empty((2, 8, 32, 2, 4096), dtype='u1'); diagnostics = []
    for draw in range(32):
        seed = int(c.digest(['spatial-exterior-cap-v1', phase, cap, draw])[:15], 16)
        completed, diagnostic = spatial.complete_density(interior, support, prior, np.random.default_rng(seed), 'learned', 32)
        density = completed.astype('f4').astype(float)
        production[:, :, draw] = labels(density)
        diagnostic.update(draw=draw, seconds=time.monotonic()-started); diagnostics.append(diagnostic)
        print(io.json.dumps(dict(stage='truth_closure', phase=phase, cap=cap, draw=draw, seconds=time.monotonic()-started)), flush=True)
        del completed, density; gc.collect()
    def fractions(lab):
        w = weights[:, None] if lab.ndim == 4 else weights
        return np.stack([((lab == k)*w).sum(-1)/w.sum(-1) for k in range(4)], -1)
    target = fractions(target_labels)
    ideal_frac = np.stack([fractions(z) for z in ideal])
    prod_frac = np.stack([fractions(z) for z in production])
    output = dest/'CLOSURE.npz'
    with output.open('xb') as stream:
        np.savez_compressed(stream, ideal_labels=ideal, production_labels=production,
            truth_labels=target_labels, galaxy_counts=weights,
            ideal_fractions=ideal_frac, production_fractions=prod_frac, truth_fractions=target,
            pair_id=np.array([r['pair_id'] for r in rows]))
    for s in source_records:
        if io.sha(s['path']) != s['sha256']:
            raise ValueError('Source changed during closure')
    io.write(dest/'COMPLETE.json', dict(passed=True, phase=phase, cap=cap, draws=32,
        cases=[{k:row[k] for k in ('phase', 'cap', 'pair_id', 'shell', 'support_stratum')} for row in rows],
        output=io.record(output), manifest=manifest, sources=source_records, inputs=list(io.CACHE.values()),
        diagnostics=diagnostics, correction_trace_maxabs=maxtrace,
        ideal='true density throughout original padded source cuboid; original zero-contrast beyond',
        production='true covered density with original learned exterior, support, padding and noise addresses',
        truth='unchanged physical periodic full-box R7 labels',
        scope='one parent seed, two reused phases, diagnostic only', posterior_qualified=False,
        seconds=time.monotonic()-started, job=os.environ['SLURM_JOB_ID']))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--phase', choices=io.PHASES, required=True)
    p.add_argument('--cap', choices=('NGC', 'SGC'), required=True)
    args = p.parse_args(); run(args.phase, args.cap)
