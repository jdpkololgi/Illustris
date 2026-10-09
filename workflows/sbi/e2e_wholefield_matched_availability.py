"""Read-only source audit for the requested six-phase/two-seed Gate 2 panel.

No model imports, field generation, tidal substitution, or sensitivity simulation.
The offset-density component is reported only as a diagnostic, never as the
six-coordinate primary statistic. Run on an interactive CPU compute node.
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

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[2]
BASE = Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v3')
PHASES = tuple(f'ph{i:03d}' for i in range(12, 18))
SEEDS = (17, 29)


def guard_path(path):
    path = Path(path)
    for text in (str(path), str(path.resolve())):
        if not set(re.findall(r'ph\d{3}', text)) <= set(PHASES):
            raise PermissionError('Only ph012--017 payloads may be read')
    return path


def sha(path):
    h = hashlib.sha256()
    with guard_path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(16 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(guard_path(path).read_text())


def offset(rho):
    rho = np.asarray(rho, dtype=float)
    if rho.shape[-3:] != (64, 48, 48) or not np.isfinite(rho).all():
        raise ValueError('Expected saved joint density rectangle')
    return np.stack([rho[..., start+2:start+14, 18:30, 18:30].mean((-3, -2, -1))
                     for start in (16, 32)], axis=-1)


def historical_regions(rho):
    # Independent reconstruction of the 18-column cache's historical windows.
    delta = np.asarray(rho) - 1
    boxes = [(s, 16, 16, 16) for s in (16, 32)]
    boxes += [(s, y, z, 24) for s in (8, 32) for y in (0, 24) for z in (0, 24)]
    boxes += [(s, y, z, 12) for s in (17, 33) for y in (17, 19) for z in (17, 19)]
    return np.stack([delta[..., s:s+n, y:y+n, z:z+n].mean((-3, -2, -1))
                     for s, y, z, n in boxes], axis=-1)


def moments(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim != 3 or y.shape != (len(x), x.shape[-1]):
        raise ValueError('Expected case/draw/coordinate arrays')
    variance = float(x.var(1, ddof=1).sum(-1).mean())
    mse = float(((x.mean(1)-y)**2).sum(-1).mean())
    if variance <= 0:
        raise ValueError('Nonpositive variance')
    return dict(ratio=mse/variance-1/x.shape[1], variance=variance,
                corrected_mse=mse-variance/x.shape[1], cases=len(x), draws=x.shape[1])


def main(out):
    if not os.environ.get('SLURM_JOB_ID') or not socket.gethostname().startswith('nid'):
        raise RuntimeError('Compute allocation required')
    if any(os.environ.get(k) for k in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONUSERBASE', 'LD_PRELOAD')):
        raise RuntimeError('Isolated environment required')
    if os.environ.get('PYTHONNOUSERSITE') != '1':
        raise RuntimeError('PYTHONNOUSERSITE=1 required')
    out.mkdir(parents=True, exist_ok=False)
    tick = time.monotonic()
    hashes = {str(Path(__file__)): sha(__file__)}

    def bind(path, expected=None):
        digest = sha(path)
        if expected is not None and digest != expected:
            raise ValueError('Changed source: '+str(path))
        hashes[str(path)] = digest

    cache = BASE/'cfm_additional_20260926_v1/regional_calibration_cache.npz'
    bind(cache)
    with np.load(cache, allow_pickle=False) as f:
        samples, truth, ids = (f[k] for k in ('samples', 'truth', 'identities'))
    assert samples.shape == (192, 32, 18) and truth.shape == (192, 18)
    identities = [tuple(v) for v in ids.tolist()]
    assert len(set(identities)) == 192
    assert set(ids[:, 0]) == set(PHASES) and set(ids[:, 1]) == {'17', '29'}
    receipts = {}
    for run in ('cfm_mixed_20260926_v1', 'cfm_additional_20260926_v1'):
        path = BASE/run/'COMPLETE.json'; bind(path)
        for name, digest in read(path)['input_hashes'].items():
            p = Path(name)
            if 'fine26624' not in name or '/nfe128/' not in name:
                continue
            guard_path(p); bind(p, digest); r = read(p)
            seed = re.search(r'seed(17|29)_', name).group(1)
            key = (r['phase'], seed, r['pair_id'])
            if key in receipts:
                raise ValueError('Duplicate source identity')
            receipts[key] = (p, r)
    assert set(receipts) == set(identities)
    bank, targets, rows = [], [], []
    cache_error = target_error = 0.
    chunks_checked = 0
    for index, identity in enumerate(identities):
        phase, seed, pair = identity
        p, r = receipts[identity]
        assert r['draws'] == 32 and r['nfe'] == 128
        assert set(r['chunks']) == {f'draw{i:03d}.npz' for i in range(0, 32, 2)}
        pieces, old = [], []
        for name, digest in sorted(r['chunks'].items()):
            chunk = p.parent/name; bind(chunk, digest)
            meta = chunk.with_suffix('.json'); bind(meta); m = read(meta)
            first = int(name[4:7])
            assert m['ids'] == [first, first+1] and m['sha256'] == digest
            assert m['phase'] == phase and m['pair_id'] == pair and m['nfe'] == 128
            assert m['binding'] == r['binding']
            with np.load(chunk, allow_pickle=False) as f:
                assert set(f.files) == {'rho', 'wide', 'mass_error'}
                assert f['rho'].shape == (2, 64, 48, 48)
                assert f['wide'].shape == (2, 48, 48, 48)
                pieces.append(offset(f['rho'])); old.append(historical_regions(f['rho']))
            chunks_checked += 1
        old = np.concatenate(old)
        np.testing.assert_allclose(old, samples[index], rtol=2e-5, atol=1e-7)
        cache_error = max(cache_error, float(abs(old-samples[index]).max()))
        target_receipt = BASE/'coupled_20260918_v1/cartesian_v2/targets'/phase/(pair+'.json')
        bind(target_receipt); tr = read(target_receipt)
        assert tr['phase'] == phase and tr['pair_id'] == pair and len(tr['outputs']) == 1
        item = tr['outputs'][0]; target_path = guard_path(item['path'])
        assert target_path.parent == target_receipt.parent.resolve()
        bind(target_path, item['sha256']); assert item['sha256'] == r['truth_sha256']
        with h5py.File(target_path, 'r') as f:
            assert f.attrs['phase'] == phase and f.attrs['pair_id'] == pair
            assert f.attrs['smoothing_mpc_h'] == 7. and f.attrs['smoothing_count'] == 1
            for key in f:
                assert isinstance(f.get(key, getlink=True), h5py.HardLink)
                assert not f[key].is_virtual and not f[key].external
            rho_truth = f['rho_joint'][:]
            y = offset(rho_truth)
            oldtruth = historical_regions(rho_truth)
        np.testing.assert_allclose(oldtruth, truth[index], rtol=2e-5, atol=1e-7)
        target_error = max(target_error, float(abs(oldtruth-truth[index]).max()))
        bank.append(np.concatenate(pieces)); targets.append(y)
        rows.append(dict(phase=phase, seed=int(seed), pair_id=pair, receipt=str(p),
                         full_continuous_ratio=None,
                         reason='No matched corrected full-cap tidal bank for pair-local draws'))
        if (index+1) % 16 == 0:
            print(json.dumps(dict(cases_checked=index+1, seconds=time.monotonic()-tick)), flush=True)
    bank, targets = np.asarray(bank), np.asarray(targets)
    baseline = []
    for phase in PHASES:
        for seed in SEEDS:
            use = (ids[:, 0] == phase) & (ids[:, 1] == str(seed))
            assert use.sum() == 16
            baseline.append(dict(phase=phase, seed=seed, full_continuous_ratio=None,
                                 offset_density_only=moments(bank[use], targets[use])))
    inventory = []
    root = BASE/'coherent_exterior_20261002_v1'
    for phase in PHASES:
        for cap in ('NGC', 'SGC'):
            for product in ('fullcap_density', 'fullcap_tides'):
                found, missing = [], []
                for first in range(0, 32, 4):
                    path = root/product/phase/cap/f'd{first:03d}/COMPLETE.json'
                    if not path.exists():
                        missing.append(first); continue
                    bind(path); r = read(path)
                    assert r['phase'] == phase and r['cap'] == cap and r['first'] == first
                    assert r['passed'] and r['draws'] == 4
                    item = r['field' if product == 'fullcap_density' else 'output']
                    exists = guard_path(item['path']).is_file()
                    found.append(dict(first=first, payload_exists=exists, receipt=str(path)))
                inventory.append(dict(phase=phase, cap=cap, product=product,
                                      receipts=found, missing_first_draws=missing,
                                      payload_hashes_verified=False))
    arrays = out/'DENSITY_ONLY.npz'
    with arrays.open('xb') as stream:
        np.savez_compressed(stream, samples=bank, truth=targets, identities=ids)
    result = dict(status='blocked_missing_matched_corrected_tidal_fields',
                  gate2_passed=None, sensitivity_simulated=False,
                  full_rule_false_advancement_calibrated=False,
                  panel_cases=192, chunks_checked=chunks_checked, baseline=baseline,
                  cache_maxabs=cache_error, target_cache_maxabs=target_error,
                  cache_columns='2 core + 8 block + 8 fine-region density means; no tidal values',
                  draw_schema=dict(rho=[2, 64, 48, 48], wide=[2, 48, 48, 48]),
                  density_only_is_not_primary=True, cases=rows, cap_inventory=inventory,
                  source_hashes=hashes, output_sha256=sha(arrays),
                  source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
                  job=os.environ['SLURM_JOB_ID'], host=socket.gethostname(),
                  created_utc=datetime.now(timezone.utc).isoformat(), seconds=time.monotonic()-tick)
    assert hashes[str(Path(__file__))] == sha(__file__)
    with (out/'AUDIT.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps(dict(status=result['status'], baseline=baseline,
                          chunks_checked=chunks_checked, output=str(out/'AUDIT.json'))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
