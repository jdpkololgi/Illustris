"""Independent verification of the saved-only matched-panel availability audit."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket

import h5py
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(root):
    assert os.environ.get('SLURM_JOB_ID') and socket.gethostname().startswith('nid')
    rec = json.loads((root/'AUDIT.json').read_text())
    assert rec['status'] == 'blocked_missing_matched_corrected_tidal_fields'
    assert rec['gate2_passed'] is None and not rec['sensitivity_simulated']
    assert rec['chunks_checked'] == 3072 and rec['panel_cases'] == 192
    assert sha(root/'DENSITY_ONLY.npz') == rec['output_sha256']
    with np.load(root/'DENSITY_ONLY.npz') as f:
        x, y, ids = f['samples'], f['truth'], f['identities']
    assert x.shape == (192, 32, 2) and y.shape == (192, 2)
    assert len({tuple(row) for row in ids}) == 192
    checks = 6
    for row in rec['baseline']:
        use = (ids[:, 0] == row['phase']) & (ids[:, 1] == str(row['seed']))
        xx, yy = x[use], y[use]
        assert xx.shape == (16, 32, 2)
        mu = np.sum(xx, axis=1)/32
        centered = xx-mu[:, None, :]
        trace = np.einsum('ijk,ijk->', centered, centered)/(31*16)
        mse = np.einsum('ij,ij->', mu-yy, mu-yy)/16
        np.testing.assert_allclose((mse-trace/32)/trace,
                                   row['offset_density_only']['ratio'], rtol=0, atol=1e-12)
        assert row['full_continuous_ratio'] is None
        checks += 3
    for phase in sorted(set(ids[:, 0])):
        a = {p: y[i] for i, (ph, seed, p) in enumerate(ids) if ph == phase and seed == '17'}
        b = {p: y[i] for i, (ph, seed, p) in enumerate(ids) if ph == phase and seed == '29'}
        assert set(a) == set(b) and len(a) == 16
        for pair in a:
            np.testing.assert_array_equal(a[pair], b[pair])
            checks += 1
    # One pre-specified first pair in each phase/seed; independently cut both
    # cores, both endpoints of the draw sequence, and matching HDF5 truth.
    for phase in sorted(set(ids[:, 0])):
        for seed in ('17', '29'):
            i = int(np.flatnonzero((ids[:, 0] == phase) & (ids[:, 1] == seed))[0])
            folder = Path(rec['cases'][i]['receipt']).parent
            for filename, draws in [('draw000.npz', (0, 1)), ('draw030.npz', (30, 31))]:
                path = folder/filename
                assert sha(path) == rec['source_hashes'][str(path)]
                with np.load(path) as f:
                    arr = f['rho'].astype(float)
                for core, start in enumerate((18, 34)):
                    expected = np.sum(arr[:, start:start+12, 18:30, 18:30], axis=(1, 2, 3))/1728
                    np.testing.assert_allclose(x[i, list(draws), core], expected, rtol=0, atol=1e-14)
                    checks += 1
            pair = ids[i, 2]
            paths = [Path(p) for p in rec['source_hashes'] if '/targets/' in p and p.endswith(pair+'.json')]
            assert len(paths) == 1
            target = json.loads(paths[0].read_text())['outputs'][0]
            with h5py.File(target['path'], 'r') as f:
                for core, start in enumerate((18, 34)):
                    expected = np.sum(f['rho_joint'][start:start+12, 18:30, 18:30].astype(float))/1728
                    np.testing.assert_allclose(y[i, core], expected, rtol=0, atol=1e-14)
                    checks += 1
    for row in rec['cap_inventory']:
        for first in row['missing_first_draws']:
            path = root.parents[1]/'coherent_exterior_20261002_v1'/row['product']/row['phase']/row['cap']/f'd{first:03d}/COMPLETE.json'
            assert not path.exists()
            checks += 1
    result = dict(passed=True, checks=checks, audit_sha256=sha(root/'AUDIT.json'),
                  verifier_sha256=sha(__file__), job=os.environ['SLURM_JOB_ID'],
                  all_ratios_recomputed=True, all_seed_truths_identical=True,
                  raw_reextraction_scope='first pair each phase/seed; draws 0,1,30,31 and truth',
                  missing_fullcap_paths_rechecked=True, gate2_passed=None)
    with (root/'VERIFIED.json').open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    main(p.parse_args().root)
