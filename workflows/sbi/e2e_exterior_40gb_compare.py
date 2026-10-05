"""Read-only full-array replay comparison and targetless geometry inventory."""
import json
import os
from pathlib import Path
import sys

ROOT = Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/coherent_exterior_20261002_v1')
sys.path.insert(0, str(ROOT/'source_78156ac3417555ef269f'))
import h5py
import numpy as np
from scipy.fft import next_fast_len
from workflows.sbi import e2e_coupled_contract as c


def run():
    c.require_compute()
    result = dict(job=os.environ['SLURM_JOB_ID'], caps={}, input_geometries=[],
                  script_sha256=c.sha256(__file__), production_modified=False)
    for cap in ('NGC', 'SGC'):
        replay = ROOT/'verification/hbm40_20261005_v1'/cap/'tides'
        checked = json.loads((replay/'VERIFIED.json').read_text())
        assert checked['passed']
        assert checked['complete_receipt']['sha256'] == c.sha256(replay/'COMPLETE.json')
        reference = ROOT/'fullcap_tides/ph014'/cap/'d000'
        errors = {}
        with h5py.File(reference/'COMPLETED_FIELD.h5') as a, h5py.File(replay/'COMPLETED_FIELD.h5') as b:
            for key in ('rho_completed', 'eigenvalues'):
                assert a[key].shape == b[key].shape
                maximum, reference_max = 0., 0.
                for draw in range(a[key].shape[0]):
                    for lo in range(0, a[key].shape[1], 16):
                        x, y = a[key][draw, lo:lo+16], b[key][draw, lo:lo+16]
                        assert np.isfinite(x).all() and np.isfinite(y).all()
                        maximum = max(maximum, float(np.max(abs(x-y))))
                        reference_max = max(reference_max, float(np.max(abs(x))))
                assert maximum <= 2e-6*max(reference_max, 1.), (cap, key, maximum)
                errors[key] = dict(max_abs=maximum, reference_max=reference_max)
            for key in ('support', 'coverage'):
                assert np.array_equal(a[key][:], b[key][:])
        result['caps'][cap] = dict(errors=errors, checked=c.file_record(replay/'VERIFIED.json', content_hash=True))
        print(json.dumps(dict(cap=cap, **errors)), flush=True)
    freeze = json.loads((ROOT/'SELECTION_FREEZE.json').read_text())
    for phase in freeze['final_phases']+['Loa']:
        for cap in ('NGC', 'SGC'):
            root = ROOT/'fullcap_inputs'/phase/cap
            rec = json.loads((root/'PREPARED.json').read_text())
            plan = json.loads((root/'PLAN.json').read_text())
            with h5py.File(rec['outputs'][0]['path']) as f:
                shape = tuple(f['support'].shape)
            fft_shape = tuple(next_fast_len(2*n+63) for n in shape)
            result['input_geometries'].append(dict(phase=phase, cap=cap, shape=shape,
                windows=len(plan['midpoints']), fft_shape=fft_shape, fft_voxels=int(np.prod(fft_shape))))
    result['passed'] = True
    c.atomic_json(ROOT/'verification/hbm40_20261005_v1/REPLAY_COMPARISON.json', result)


if __name__ == '__main__':
    run()
