"""Bounded, development-only sensitivity for the approved whole-field campaign.

All numerical work and payload hashing run on an interactive compute node.
Saved products are read-only; generated receipts never overwrite prior results.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time

import numpy as np

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / 'configs/e2e_wholefield_gate2_v1.json'
ROOT = Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v3')
PARENT = ROOT / 'cfm_conditioning_augmentation_20261006_v3'
CAL = ROOT / 'calibration_investigation_20261006_v1/a2'
OUT = ROOT / 'wholefield_comparison_20261009_v1/gate2'
PHASES = ('ph012', 'ph013')


def guard():
    if not os.environ.get('SLURM_JOB_ID') or not socket.gethostname().startswith('nid'):
        raise RuntimeError('Interactive compute node required')
    if any(os.environ.get(k) for k in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONUSERBASE', 'LD_PRELOAD')):
        raise RuntimeError('Isolated runtime required')
    if os.environ.get('PYTHONNOUSERSITE') != '1':
        raise RuntimeError('PYTHONNOUSERSITE=1 required')


def phase_path(path):
    path = Path(path)
    for value in (str(path), str(path.resolve())):
        if not set(re.findall(r'ph\d{3}', value)) <= set(PHASES):
            raise PermissionError('Only development phases ph012/ph013 are allowed')
    return path


def record(path):
    path = phase_path(path)
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(16 << 20), b''):
            h.update(part)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=h.hexdigest())


def read(path):
    return json.loads(phase_path(path).read_text())


def verify(value):
    got = record(value['path'])
    if got['sha256'] != value['sha256']:
        raise ValueError('Input hash mismatch: ' + value['path'])
    return got


def write(path, obj):
    with Path(path).open('x') as stream:
        json.dump(obj, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def provenance():
    return dict(source=record(__file__), config=record(CONFIG),
                commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
                job=os.environ['SLURM_JOB_ID'], host=socket.gethostname(),
                created_utc=datetime.now(timezone.utc).isoformat())


def fine_vector(density, eigen):
    """No data-fitted scaling; trace-free tidal coordinates plus offset density."""
    density, eigen = np.asarray(density, float), np.asarray(eigen, float)
    if density.shape[-2:] != (2, 4096) or eigen.shape != density.shape + (3,):
        raise ValueError('Expected two 16-cubed cores with density/eigenvalue channels')
    if not np.isfinite(density).all() or not np.isfinite(eigen).all():
        raise ValueError('Nonfinite physical summaries')
    if np.min(np.diff(eigen, axis=-1)) < -1e-10:
        raise ValueError('Eigenvalues must be ordered')
    offset = density.reshape(density.shape[:-1] + (16, 16, 16))[..., 2:14, 2:14, 2:14].mean((-3, -2, -1))
    ev = eigen.mean(-2)
    shear1 = (ev[..., 2] - ev[..., 0]) / np.sqrt(2)
    shear2 = (ev[..., 0] - 2*ev[..., 1] + ev[..., 2]) / np.sqrt(6)
    fine = np.stack((offset, shear1, shear2), -1).reshape(density.shape[:-2] + (6,))
    full = np.concatenate((density.mean(-1)[..., None], ev), -1)
    return fine, full


def moments(draws, truth):
    """Equal-case trace Q/trace V, with finite-M noise correction."""
    x, y = np.asarray(draws, float), np.asarray(truth, float)
    if x.ndim != 3 or x.shape[0] != len(y) or x.shape[2:] != y.shape[1:] or x.shape[1] < 3:
        raise ValueError('Expected case/draw/vector and case/vector')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Nonfinite moment bank')
    error = x.mean(1) - y
    variance = x.var(1, ddof=1).sum(-1).mean()
    mse = (error**2).sum(-1).mean()
    if variance <= 1e-15:
        return dict(status='structurally_degenerate', variance=float(variance), mse=float(mse), ratio=None)
    return dict(status='finite', ratio=float(mse/variance - 1/x.shape[1]),
                variance=float(variance), mse=float(mse), corrected_mse=float(mse-variance/x.shape[1]),
                bias=error.mean(0).tolist(), cases=len(x), draws=x.shape[1], dimensions=x.shape[2])


def prepare():
    guard(); started = time.monotonic(); cfg = read(CONFIG)
    if cfg['phases'] != list(PHASES) or cfg['draws'] != 32:
        raise ValueError('Unregistered source panel')
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'PREPARED.json').exists() or (OUT/'PANEL.npz').exists():
        raise FileExistsError('Preserve existing prepared data; verify it instead of overwriting')
    prov = provenance()
    freeze = ROOT/'coherent_exterior_20261002_v1/SELECTION_FREEZE.json'
    frozen = read(freeze)
    if frozen['checkpoints']['fine'] != '88b2c7d1c5174e64dc98bb6f555bf555c876ae02496f237a71f7ee8c31e5cced':
        raise ValueError('Wrong frozen baseline')
    a2 = read(CAL/'COMPLETE.json')
    if not a2['passed'] or not a2['panel_complete'] or len(a2['cases']) != 32:
        raise ValueError('A2 comparison is incomplete')
    lookup = {r['pair_id']: r for r in a2['cases']}
    inputs = [record(freeze), record(CAL/'COMPLETE.json')]
    cases = []; banks = []; targets = []; fulls = []; fulltargets = []; errors = []
    for phase in PHASES:
        for cap in ('NGC', 'SGC'):
            chunks = []; fullchunks = []
            for first in range(0, 32, 4):
                path = PARENT/'cores'/phase/cap/'parent/s0'/f'd{first:03d}/COMPLETE.json'
                rec = read(path); inputs.append(record(path))
                if not rec['passed'] or rec.get('smoke', False) or (rec['phase'], rec['cap'], rec['first'], rec['draws']) != (phase, cap, first, 4):
                    raise ValueError('Source receipt scope mismatch')
                inputs.extend((verify(rec['binding']), verify(rec['output'])))
                with np.load(phase_path(rec['output']['path']), allow_pickle=False) as f:
                    eigen = f['eigen'][1]; rho = f['density']
                    if eigen.shape != (8, 4, 2, 4096, 3) or rho.shape != (8, 4, 2, 4096):
                        raise ValueError('Core shape mismatch')
                    x, full = fine_vector(rho, eigen)
                    truth, ft = fine_vector(f['truth_density'], f['truth_eigen'])
                    names = list(f['pair_id'])
                    if first == 0:
                        fixed_names, fixed_truth, fixed_ft = names, truth, ft
                        meta = rec['cases']
                        if names != [q['pair_id'] for q in meta]:
                            raise ValueError('Case identity mismatch')
                    else:
                        if names != fixed_names or meta != rec['cases']:
                            raise ValueError('Case ordering drift')
                        np.testing.assert_array_equal(truth, fixed_truth)
                        np.testing.assert_array_equal(ft, fixed_ft)
                    chunks.append(x); fullchunks.append(full)
            block = np.concatenate(chunks, 1); fullblock = np.concatenate(fullchunks, 1)
            for j, name in enumerate(fixed_names):
                prior = lookup[name]; inputs.append(verify(prior['summary']))
                with np.load(phase_path(prior['summary']['path']), allow_pickle=False) as f:
                    delta = float(np.max(abs(fullblock[j, ..., 1:] - f['corrected'])))
                    np.testing.assert_allclose(fullblock[j, ..., 1:], f['corrected'], rtol=0, atol=1e-12)
                    np.testing.assert_allclose(fixed_ft[j, ..., 1:], f['physical_truth'], rtol=0, atol=1e-12)
                    errors.append(delta)
            banks.append(block); targets.append(fixed_truth); fulls.append(fullblock); fulltargets.append(fixed_ft)
            cases.extend(meta)
            print(json.dumps(dict(stage='prepared_cap', phase=phase, cap=cap, seconds=time.monotonic()-started)), flush=True)
    x, y = np.concatenate(banks), np.concatenate(targets)
    full, ft = np.concatenate(fulls), np.concatenate(fulltargets)
    if x.shape != (32, 32, 6) or y.shape != (32, 6) or len({q['pair_id'] for q in cases}) != 32:
        raise ValueError('Complete 32-pair bank required')
    reports = []
    for phase in PHASES:
        for support in ('all', 'interior', 'boundary'):
            ix = [i for i, q in enumerate(cases) if q['phase'] == phase and (support == 'all' or q['support_stratum'] == support)]
            if len(ix) != (16 if support == 'all' else 8):
                raise ValueError('Registered support counts changed')
            reports.append(dict(phase=phase, support=support, fine=moments(x[ix], y[ix]),
                                mass_control=moments(full[ix, ..., 0], ft[ix, ..., 0])))
    with (OUT/'PANEL.npz').open('xb') as stream:
        np.savez_compressed(stream, draws=x, truth=y, full=full, full_truth=ft,
                            pair_id=np.array([q['pair_id'] for q in cases]))
    if record(__file__) != prov['source'] or record(CONFIG) != prov['config']:
        raise ValueError('Source/config changed during preparation')
    result = dict(**prov, passed=True, cases=cases, summaries=reports, arrays=record(OUT/'PANEL.npz'),
                  inputs=inputs, a2_parity_maxabs=max(errors), seconds=time.monotonic()-started,
                  normalization='identity physical units; no fitted metric normalization',
                  true_coarse_tidal_bank_available=False, posterior_qualified=False)
    write(OUT/'PREPARED.json', result)
    print(json.dumps(dict(stage='prepared', reports=reports, output=str(OUT/'PREPARED.json'))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare'])
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare()
