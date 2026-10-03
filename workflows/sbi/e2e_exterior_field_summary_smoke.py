"""Compute-node synthetic end-to-end readback for native field summaries."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import h5py
import numpy as np
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi.e2e_vdm_context_metrics import central_order_interval


def run(source, out):
    c.require_compute(); out.mkdir(parents=True, exist_ok=False)
    specification = importlib.util.spec_from_file_location('summary_under_test', source)
    summary = importlib.util.module_from_spec(specification); specification.loader.exec_module(summary)
    actual_root = summary.ROOT; freeze_path = actual_root/'SELECTION_FREEZE.json'
    selected = json.loads(freeze_path.read_text()); freeze_sha = summary.sha(freeze_path)
    summary.ROOT = out/'synthetic_fixture'
    eigen = (np.arange(32, dtype='f4')[:, None]/100 + np.array([0, .2, .5], dtype='f4'))
    rho = 1+eigen.sum(-1)
    for cap in ('NGC', 'SGC'):
        for first in range(0, 32, 4):
            directory = summary.ROOT/'fullcap_tides'/'Loa'/cap/f'd{first:03d}'
            directory.mkdir(parents=True)
            path = directory/'COMPLETED_FIELD.h5'
            with h5py.File(path, 'x') as f:
                f.create_dataset('support', data=np.ones((2, 2, 2), dtype='f4'))
                f.create_dataset('coverage', data=np.ones((2, 2, 2), dtype='u2'))
                f.create_dataset('eigenvalues', data=np.broadcast_to(eigen[first:first+4, None, None, None, :], (4, 2, 2, 2, 3)))
                density = f.create_dataset('rho_completed', shape=(4, 66, 66, 66), dtype='f4',
                                             fillvalue=1., chunks=(1, 2, 2, 2), compression='lzf')
                density[:, 32:34, 32:34, 32:34] = rho[first:first+4, None, None, None]
                f.attrs.update(origin_mpc_h=[0., 0., 0.], padding_cells=32, cell_mpc_h=6.766,
                               phase='Loa', cap=cap, synthetic_fixture=True)
            c.atomic_json(directory/'BINDING.json', dict(freeze_sha256=freeze_sha, synthetic_fixture=True))
            c.atomic_json(directory/'COMPLETE.json', dict(passed=True, posterior_qualified=False,
                phase='Loa', cap=cap, first=first, draws=4, method=selected['method'],
                binding=summary.sha(directory/'BINDING.json'), output=dict(path=str(path), sha256=summary.sha(path)),
                synthetic_fixture=True))
            c.atomic_json(directory/'VERIFIED.json', dict(passed=True,
                complete_receipt=dict(sha256=summary.sha(directory/'COMPLETE.json')),
                verifier_sha256=selected['sources']['workflows/sbi/e2e_exterior_cap_verify.py'],
                synthetic_fixture=True))
    summary.run(freeze_path, summary.sha(source))
    for cap in ('NGC', 'SGC'):
        with h5py.File(summary.ROOT/'loa_field_summary'/cap/'DENSITY_TIDAL_SUMMARY.h5') as f:
            np.testing.assert_allclose(f['eigenvalues/mean'][0, 0, 0], eigen.mean(0, dtype='f8'), rtol=1e-7)
            np.testing.assert_allclose(f['eigenvalues/std'][0, 0, 0], eigen.std(0, dtype='f8', ddof=1), rtol=1e-7)
            np.testing.assert_allclose(f['density/mean'][0, 0, 0], rho.mean(dtype='f8'), rtol=1e-7)
            np.testing.assert_allclose(f['density/std'][0, 0, 0], rho.std(dtype='f8', ddof=1), rtol=1e-7)
            for index, level in enumerate(summary.LEVELS):
                lo, hi, reference = central_order_interval(32, level)
                np.testing.assert_array_equal(f['eigenvalues/lower'][index, 0, 0, 0], eigen[lo])
                np.testing.assert_array_equal(f['eigenvalues/upper'][index, 0, 0, 0], eigen[hi])
                assert f.attrs['attainable_coverage'][index] == reference
            for index, threshold in enumerate((0., .2)):
                label = (eigen > threshold).sum(-1)
                expected = np.array([(label == cls).mean() for cls in range(4)])
                np.testing.assert_array_equal(f['environment_probability'][index, 0, 0, 0], expected)
                entropy = -np.sum(expected*np.log2(np.maximum(expected, 1/32)))/2
                np.testing.assert_allclose(f['environment_entropy'][index, 0, 0, 0], entropy, atol=1e-7)
    result = dict(passed=True, synthetic=True, source_sha256=summary.sha(source),
                  fixture='32 known monotonic draws, two tiny caps; no observed field or mock truth read',
                  checked=['mean', 'sample_std', 'four_interval_levels', 'classes', 'entropy', 'metadata'],
                  job=os.environ['SLURM_JOB_ID'])
    c.atomic_json(out/'SMOKE.json', result); print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True); a = p.parse_args(); run(a.source, a.out)
