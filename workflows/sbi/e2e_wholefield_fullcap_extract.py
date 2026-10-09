"""Source-bound six-phase extraction; no generation, fitting or gate decisions.

Run this file directly. Numerical dependencies come only from the immutable
generation snapshot. Extraction is staged separately and cannot modify it.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/wholefield_comparison_20261009_v1/fullcap_v1')
OUT = ROOT/'extraction_v1'
PHASES = tuple(f'ph{i:03d}' for i in range(12, 18))


def address(phase, cap, seed, first):
    if phase not in PHASES or cap not in ('NGC', 'SGC') or seed not in (17, 29):
        raise PermissionError('Only the approved six-phase/two-seed panel')
    if first not in range(0, 32, 4):
        raise ValueError('Only registered four-draw addresses')
    return f'{phase}/{cap}/seed{seed}/d{first:03d}'


def pool(x):
    a, b, c = x.shape
    return x.reshape(a//2, 2, b//2, 2, c//2, 2).mean((1, 3, 5))


def reconstruct(rho):
    """Exact linear branch of e2e_density_only_reconstruction.reconstruct."""
    rho = np.asarray(rho, dtype=np.float64)
    if rho.ndim != 3 or not np.isfinite(rho).all() or np.min(rho) <= 0:
        raise ValueError('Positive finite density required')
    slopes = []
    for axis in range(3):
        left = rho-np.roll(rho, 1, axis)
        right = np.roll(rho, -1, axis)-rho
        value = .5*(left+right)
        edge = [slice(None)]*3
        for index in (0, -1):
            edge[axis] = index
            value[tuple(edge)] = 0
        slopes.append(value)
    excursion = .25*sum(abs(v) for v in slopes)
    factor = np.minimum(1., .999*rho/np.maximum(excursion, 1e-300))
    fine = np.empty(tuple(2*n for n in rho.shape))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                fine[i::2, j::2, k::2] = rho+factor*sum(
                    (index-.5)*.5*v for index, v in zip((i, j, k), slopes))
    np.testing.assert_allclose(pool(fine), rho, atol=1e-14, rtol=1e-14)
    if fine.min() <= 0:
        raise ValueError('Reconstruction lost positivity')
    return fine


def cuts(row, origin):
    raw = np.asarray(row['center'])-16
    position = np.asarray(row['grid']['origin_mpc_h'])+3.383*raw
    offset = (position-origin)/6.766
    start = np.rint(offset).astype(int)
    if np.max(abs(offset-start)) > 1e-8:
        raise ValueError('Misaligned core')
    return tuple(slice(int(a), int(a+b)) for a, b in zip(start, (32, 16, 16)))


def summary(density, eigen):
    if density.shape[-2:] != (2, 4096) or eigen.shape != density.shape+(3,):
        raise ValueError('Expected two 16-cubed physical cores')
    if not np.isfinite(density).all() or not np.isfinite(eigen).all():
        raise ValueError('Nonfinite summary input')
    if np.min(np.diff(eigen, axis=-1)) < -1e-10:
        raise ValueError('Unordered eigenvalues')
    offset = density.reshape(density.shape[:-1]+(16, 16, 16))[..., 2:14, 2:14, 2:14].mean((-3, -2, -1))
    ev = eigen.mean(-2)
    shear1 = (ev[..., 2]-ev[..., 0])/np.sqrt(2)
    shear2 = (ev[..., 0]-2*ev[..., 1]+ev[..., 2])/np.sqrt(6)
    return np.stack((offset, shear1, shear2), -1).reshape(density.shape[:-2]+(6,))


def dependencies():
    sys.path.insert(0, str(ROOT/'source'))
    from workflows.sbi import e2e_wholefield_fullcap_run as generation
    generation.compute()
    if generation.REPO != ROOT/'source':
        raise ValueError('Nonfrozen dependency import')
    manifest = generation.read(ROOT/'MANIFEST.json')
    generation.verify_sources(manifest)
    return generation, manifest


def stage():
    gen, manifest = dependencies()
    OUT.mkdir(exist_ok=False)
    source = OUT/'extract.py'
    shutil.copy2(__file__, source)
    gen.write(OUT/'MANIFEST.json', dict(schema='wholefield-fullcap-extraction-v1',
              source=gen.record(source), generation=gen.record(ROOT/'MANIFEST.json'),
              phases=list(PHASES), seeds=[17, 29], operator=manifest['operator'],
              original_parity_atol=2e-7, correction_trace_atol=1e-9,
              corrected_parity_atol=1e-10, job=os.environ['SLURM_JOB_ID']))


def run(phase, cap, seed, first):
    key = address(phase, cap, seed, first)  # Before any source/target I/O.
    gen, manifest = dependencies()
    binding = gen.read(OUT/'MANIFEST.json')
    gen.verify_record(binding['source']); gen.verify_record(binding['generation'])
    if Path(__file__).resolve() != Path(binding['source']['path']).resolve():
        raise ValueError('Use staged extraction source')
    import h5py
    import torch
    from workflows.sbi import e2e_coupled_coordinates as coord
    from workflows.sbi import e2e_coupled_contract as contract
    from workflows.sbi.e2e_coupled_product_audit import read_target
    from workflows.sbi.e2e_coupled_target_products import owned_core_arrays
    from workflows.sbi.e2e_exterior_cap_tides import CapTidal
    from workflows.sbi.e2e_footprint_tidal_operator import eigensystem
    from workflows.sbi.e2e_loa_field_sampling import setup
    setup(True); torch.set_num_threads(4)
    tick = time.monotonic()
    task = next(t for t in manifest['tasks'] if t['key'] == key)
    reuse = gen.read(ROOT/'REUSE_VERIFIED.json'); gen.verify_record(reuse['manifest'])
    if key in reuse['tasks'] and reuse['passed']:
        density_record, tidal_record = (task[p+'_bound_receipt'] for p in ('density', 'tides'))
    else:
        complete = gen.read(ROOT/'tasks'/key/'COMPLETE.json')
        gen.verify_record(complete['manifest'])
        if not complete['passed'] or complete['key'] != key:
            raise ValueError('Unverified generation task')
        density_record, tidal_record = complete['density'], complete['tides']
    for item in (density_record, tidal_record):
        gen.verify_record(item)
    field = gen.read(density_record['path']); tides = gen.read(tidal_record['path'])
    for record, payload in ((field, 'field'), (tides, 'output')):
        if not record['passed'] or any(record[k] != v for k, v in
                dict(phase=phase, cap=cap, first=first, draws=4).items()):
            raise ValueError('Wrong source address')
        gen.verify_record(record[payload])
    if tides['method'] != 'learned':
        raise ValueError('Unregistered exterior completion')
    geometry_path = coord.ROOT/'geometry'/phase/'GEOMETRY_COMPLETE.json'
    geometry = coord.verify_receipt(geometry_path, payload=False)
    if geometry['phase'] != phase:
        raise ValueError('Wrong geometry phase')
    rows = [r for r in geometry['pairs'] if r['cap'] == cap]
    if len(rows) != 8 or len({r['pair_id'] for r in rows}) != 8:
        raise ValueError('Eight distinct registered pairs required')
    truth, truthrho, truth_sources = [], [], []
    for row in rows:
        target = coord.ROOT/'targets'/phase/(row['pair_id']+'.json')
        rec = coord.verify_receipt(target, payload=False)
        if rec['phase'] != phase or rec['pair_id'] != row['pair_id'] or len(rec['outputs']) != 1:
            raise ValueError('Wrong target identity')
        item = rec['outputs'][0]; path = contract.guarded(item['path'], phase)
        if path.parent != target.parent.resolve():
            raise ValueError('Target outside approved phase directory')
        gen.verify_record(item)
        values = read_target(path, phase, row['pair_id'])
        truth.append(eigensystem(values['fullbox_tensor_cores']).reshape(2, 4096, 3))
        truthrho.append(owned_core_arrays(values['rho_joint']).reshape(2, 4096))
        truth_sources.append(gen.record(target))
    truth, truthrho = np.asarray(truth), np.asarray(truthrho)
    np.testing.assert_allclose(truth.sum(-1), truthrho-1, rtol=0, atol=2e-6)
    dest = OUT/'cores'/key; dest.mkdir(parents=True, exist_ok=False)
    gen.write(dest/'BINDING.json', dict(extraction=gen.record(OUT/'MANIFEST.json'),
              density=density_record, tides=tidal_record, geometry=gen.record(geometry_path),
              targets=truth_sources, job=os.environ['SLURM_JOB_ID']))
    eigen = np.empty((2, 8, 4, 2, 4096, 3)); densitycores = np.empty((8, 4, 2, 4096))
    trace = parity = 0.
    with h5py.File(field['field']['path']) as f, h5py.File(tides['output']['path']) as saved:
        shape = f['rho'].shape[1:]; origin = np.asarray(f.attrs['origin_mpc_h'])
        for handle in (f, saved):
            if any(handle.attrs.get(k) != v for k, v in dict(phase=phase, cap=cap, first_draw=first).items()):
                raise ValueError('HDF5 source identity mismatch')
        if saved.attrs['padding_cells'] != 32 or saved.attrs['cell_mpc_h'] != 6.766:
            raise ValueError('Wrong operator geometry')
        np.testing.assert_array_equal(saved.attrs['origin_mpc_h'], origin)
        if f['rho'].shape[0] != 4 or saved['rho_completed'].shape != (4, *(n+64 for n in shape)):
            raise ValueError('Wrong complete field shape')
        operator = CapTidal(shape, 32)
        localcoarse = CapTidal((32, 16, 16), 32); localfine = CapTidal((64, 32, 32), 64)
        cores = [cuts(row, origin) for row in rows]
        patches = [tuple(slice(v.start-33, v.stop+33) for v in cuts(row, origin-32*6.766)) for row in rows]
        for core, patch in zip(cores, patches):
            if any(v.start < 0 or v.stop > n for v, n in zip(core, shape)) or any(
                    v.start < 0 or v.stop > n+64 for v, n in zip(patch, shape)):
                raise ValueError('Core/correction collar outside saved field')
        for draw in range(4):
            density = saved['rho_completed'][draw].astype(float)
            tensor = np.empty((8, 2, 16, 16, 16, 6))
            for k, component in enumerate(operator.components(density)):
                for j, core in enumerate(cores):
                    tensor[j, ..., k] = component[core].reshape(2, 16, 16, 16)
            for j, (core, patch) in enumerate(zip(cores, patches)):
                coarse = density[patch]; fine = reconstruct(coarse)[2:-2, 2:-2, 2:-2]
                correction = np.empty_like(tensor[j])
                for k, (tc, tf) in enumerate(zip(localcoarse.components(coarse[1:-1, 1:-1, 1:-1]), localfine.components(fine))):
                    correction[..., k] = (pool(tf)-tc).reshape(2, 16, 16, 16)
                trace = max(trace, float(abs(correction[..., [0, 3, 5]].sum(-1)).max()))
                for mi, t in enumerate((tensor[j], tensor[j]+correction)):
                    eigen[mi, j, draw] = eigensystem(t).reshape(2, 4096, 3)
                densitycores[j, draw] = f['rho'][(draw, *core)].reshape(2, 4096)
                np.testing.assert_allclose(eigen[:, j, draw].sum(-1),
                                          np.broadcast_to(densitycores[j, draw]-1, (2, 2, 4096)), rtol=0, atol=1e-9)
                old = saved['eigenvalues'][(draw, *core, slice(None))].reshape(2, 4096, 3)
                parity = max(parity, float(abs(old-eigen[0, j, draw]).max()))
            if trace > binding['correction_trace_atol'] or parity > binding['original_parity_atol']:
                raise ValueError('Tidal identity/replay failed')
            print(json.dumps(dict(stage='extract', key=key, draw=first+draw, seconds=time.monotonic()-tick)), flush=True)
            del density, tensor, component, coarse, fine, correction, tc, tf
            gc.collect()
    oldparity = None
    if seed == 17 and phase in ('ph012', 'ph013'):
        oldroot = ROOT.parent.parent/'cfm_conditioning_augmentation_20261006_v3/cores'/phase/cap/'parent/s0'/f'd{first:03d}'
        oldrec = gen.read(oldroot/'COMPLETE.json'); gen.verify_record(oldrec['output'])
        with np.load(oldrec['output']['path']) as old:
            np.testing.assert_array_equal(old['pair_id'], [r['pair_id'] for r in rows])
            oldparity = float(abs(old['eigen']-eigen).max())
            np.testing.assert_allclose(old['eigen'], eigen, rtol=0, atol=binding['corrected_parity_atol'])
            np.testing.assert_array_equal(old['density'], densitycores)
            np.testing.assert_array_equal(old['truth_eigen'], truth)
        truth_sources.append(gen.record(oldroot/'COMPLETE.json'))
    payload = dest/'CORES.npz'
    with payload.open('xb') as stream:
        np.savez_compressed(stream, eigen=eigen, density=densitycores, truth_eigen=truth,
                            truth_density=truthrho, draws=summary(densitycores, eigen[1]),
                            truth=summary(truthrho, truth), pair_id=np.array([r['pair_id'] for r in rows]))
    gen.write(dest/'COMPLETE.json', dict(passed=True, key=key, phase=phase, cap=cap, seed=seed,
              first=first, draws=4, output=gen.record(payload), binding=gen.record(dest/'BINDING.json'),
              original_parity_maxabs=parity, correction_trace_maxabs=trace, corrected_parity_maxabs=oldparity,
              sources=truth_sources, seconds=time.monotonic()-tick, job=os.environ['SLURM_JOB_ID'],
              peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30, posterior_qualified=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['stage', 'run'])
    parser.add_argument('--phase'); parser.add_argument('--cap')
    parser.add_argument('--seed', type=int); parser.add_argument('--first', type=int)
    args = parser.parse_args()
    if args.action == 'stage':
        stage()
    else:
        run(args.phase, args.cap, args.seed, args.first)
