"""Verify indexed ph006 SGC output against the completed pilot, slab by slab."""
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from workflows.sbi import e2e_coupled_contract as c


def verify(draw, candidate):
    c.require_compute()
    assert draw in (0, 1)
    baseline = c.ROOT.parent / 'lowz_cap_draw_61582948/SGC'
    stats = {}
    for stage, key, data in [('density', 'field', 'rho'), ('tides', 'output', 'eigenvalues')]:
        refs = [json.loads((root/stage/'COMPLETE.json').read_text()) for root in (baseline, candidate)]
        for ref, idx in zip(refs, (0, draw)):
            assert ref['passed'] and (ref['phase'], ref['cap'], ref['first']) == ('ph006', 'SGC', idx)
            assert ref[key]['sha256'] == c.sha256(ref[key]['path'])
        with h5py.File(refs[0][key]['path']) as a, h5py.File(refs[1][key]['path']) as b:
            assert np.array_equal(a.attrs['origin_mpc_h'], b.attrs['origin_mpc_h'])
            assert a[data].shape == b[data].shape
            ss = 0.; n = 0; maximum = 0.
            for i in range(0, a[data].shape[1], 4):
                s = slice(i, i+4)
                mask = a['support'][s] > 0
                np.testing.assert_array_equal(mask, b['support'][s] > 0)
                x, y = a[data][0,s][mask], b[data][0,s][mask]
                assert np.isfinite(x).all() and np.isfinite(y).all()
                if draw == 0: np.testing.assert_allclose(x, y, atol=1e-5, rtol=1e-5)
                delta = x.astype('f8')-y
                ss += np.square(delta).sum(); n += delta.size
                maximum = max(maximum, float(np.max(abs(delta), initial=0)))
            rms = float((ss/n)**.5)
            if draw: assert rms > 1e-4
            stats[data] = dict(n=n, rms=rms, maximum=maximum)
    c.atomic_json(candidate/'SEED_GATE.json', dict(passed=True,draw=draw,comparison=stats))


if __name__ == '__main__':
    verify(int(sys.argv[1]), Path(sys.argv[2]))
