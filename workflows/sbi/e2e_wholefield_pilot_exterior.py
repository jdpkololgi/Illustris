"""Cache common full-cap exterior tensors for local replacement diagnostics.

The exterior is a shared frozen reference draw, not reconditioned on each new
fine field. This makes the operator/reference comparison exact, but the new
posterior is still a modular finite-domain replacement, not a full-cap law.
"""
import argparse
import json
import time

import h5py
import numpy as np
import torch

from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_coordinates as coord
from workflows.sbi.e2e_exterior_cap_tides import CapTidal
from workflows.sbi.e2e_footprint_tidal_operator import eigensystem, kernel
from workflows.sbi.e2e_wholefield_fullcap_extract import cuts, reconstruct, pool
from workflows.sbi.e2e_wholefield_pilot_data import ROOT, EVAL, SHAPE
from workflows.sbi.e2e_wholefield_pilot_evaluate import BANK, read, verify, integer_crop, CORE
from workflows.sbi.e2e_wholefield_pilot_train import setup


def prepare(phase, cap, seed, first):
    if phase not in EVAL or cap not in ('NGC', 'SGC') or seed not in (17, 29) or first not in range(0, 32, 4):
        raise PermissionError('Wrong frozen exterior address')
    setup(seed)
    key = f'{phase}/{cap}/seed{seed}/d{first:03d}'
    out = ROOT/'exterior'/key; out.mkdir(parents=True, exist_ok=True)
    if (out/'COMPLETE.json').exists():
        return
    manifest = read(BANK/'MANIFEST.json'); reuse = read(BANK/'REUSE_VERIFIED.json')
    task = next(t for t in manifest['tasks'] if t['key'] == key)
    rec = task['tides_bound_receipt'] if key in reuse['tasks'] else read(BANK/'tasks'/key/'COMPLETE.json')['tides']
    verify(rec); tides = read(rec['path'])
    if not tides['passed'] or (tides['phase'], tides['cap'], tides['first']) != (phase, cap, first):
        raise ValueError('Wrong completed exterior')
    if __import__('pathlib').Path(tides['output']['path']).stat().st_size != tides['output']['bytes']:
        raise ValueError('Completed exterior size changed')
    geometry = read(coord.ROOT/'geometry'/phase/'GEOMETRY_COMPLETE.json')
    rows = [r for r in geometry['pairs'] if r['cap'] == cap]
    start = time.monotonic(); records = []
    with h5py.File(tides['output']['path']) as f:
        if (f.attrs['padding_cells'] != 32 or f.attrs['cell_mpc_h'] != 6.766
                or (f.attrs['phase'], f.attrs['cap'], f.attrs['first_draw']) != (phase, cap, first)):
            raise ValueError('Wrong completed field geometry/address')
        shape = f['eigenvalues'].shape[1:4]; origin = np.asarray(f.attrs['origin_mpc_h'])
        operator = CapTidal(shape, 32)
        cores = [cuts(row, origin) for row in rows]
        completed_cores = [cuts(row, origin-32*6.766) for row in rows]
        for draw in range(4):
            if all((out/f'{r["pair_id"]}_d{first+draw:03d}.npz').exists() for r in rows):
                continue
            density = f['rho_completed'][draw].astype(float)
            tensors = np.empty((8, 32, 16, 16, 6))
            for k, value in enumerate(operator.components(density)):
                for j, core in enumerate(cores):
                    tensors[j, ..., k] = value[core]
            for j, row in enumerate(rows):
                core = completed_cores[j]
                collar = tuple(slice(s.start-33, s.stop+33) for s in core)
                begin = np.array([s.start for s in core])-[32, 24, 24]
                domain = integer_crop(begin, SHAPE, density.shape)
                np.testing.assert_allclose(eigensystem(tensors[j]), f['eigenvalues'][(draw, *cores[j], slice(None))], atol=2e-7, rtol=0)
                path = out/f'{row["pair_id"]}_d{first+draw:03d}.npz'
                temporary = path.with_suffix('.tmp')
                with temporary.open('wb') as stream:
                    np.savez_compressed(stream, tensor=tensors[j], rho=density[domain].astype('f4'),
                                        collar=density[collar].astype('f4'))
                temporary.replace(path)
            print(json.dumps(dict(stage='exterior_cache', key=key, draw=first+draw, seconds=time.monotonic()-start)), flush=True)
    for row in rows:
        for draw in range(4):
            records.append(c.file_record(out/f'{row["pair_id"]}_d{first+draw:03d}.npz', content_hash=True))
    c.atomic_json(out/'COMPLETE.json', dict(passed=True, source=c.sha256(__file__), parent=rec,
                  key=key, files=records, seconds=time.monotonic()-start,
                  warning='Frozen exterior reused; not reconditioned on replacement fine field'))


class ReplacementTidal:
    def __init__(self):
        k = kernel(SHAPE, (32, 16, 16), offset=(32, 24, 24))
        self.shape = k.shape[1:]
        self.spectra = [torch.fft.rfftn(torch.as_tensor(a, device='cuda')) for a in k]
        self.coarse = CapTidal((32, 16, 16), 32)
        self.fine = CapTidal((64, 32, 32), 64)

    def tensor(self, rho, base):
        difference = np.asarray(rho, float)-base['rho'].astype(float)
        spectrum = torch.fft.rfftn(torch.as_tensor(difference, device='cuda'), s=self.shape)
        tensor = base['tensor'].copy()
        for k, value in enumerate(self.spectra):
            tensor[..., k] += torch.fft.irfftn(spectrum*value, s=self.shape)[:32, :16, :16].cpu().numpy()
        collar = base['collar'].astype(float).copy()
        collar[1:97, 9:73, 9:73] = rho
        reconstructed = reconstruct(collar)[2:-2, 2:-2, 2:-2]
        for k, (coarse, fine) in enumerate(zip(self.coarse.components(collar[1:-1, 1:-1, 1:-1]),
                                             self.fine.components(reconstructed))):
            tensor[..., k] += pool(fine)-coarse
        np.testing.assert_allclose(tensor[..., [0, 3, 5]].sum(-1), rho[CORE]-1, atol=2e-8, rtol=0)
        return tensor.reshape(2, 4096, 6)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', required=True, choices=EVAL)
    p.add_argument('--cap', required=True, choices=('NGC', 'SGC'))
    p.add_argument('--seed', type=int, required=True, choices=(17, 29))
    p.add_argument('--first', type=int, required=True)
    prepare(**vars(p.parse_args()))
