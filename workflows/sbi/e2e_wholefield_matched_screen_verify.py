"""Independent arithmetic/event audit; no simulation or decision-rule import."""
import hashlib
import json
import math
import os
from pathlib import Path
import socket

import numpy as np

OUT = Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/wholefield_comparison_20261009_v1/matched_gate2_20261013_v1')


def verify():
    if not os.environ.get('SLURM_JOB_ID') or not socket.gethostname().startswith('nid'):
        raise RuntimeError('Compute node required')
    result = json.loads((OUT/'COMPLETE.json').read_text())
    binding = json.loads((OUT/'BINDING.json').read_text())
    checks = 0
    for rec in [result['binding'], result['simulations'], binding['config'], binding['source'], binding['bank'], binding['panel']]:
        p = Path(rec['path'])
        assert p.stat().st_size == rec['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest() == rec['sha256']
        checks += 1
    cfg = json.loads(Path(binding['config']['path']).read_text())
    with np.load(binding['panel']['path'], allow_pickle=False) as f:
        # Compute in flattened phase-vector coordinates, independently of builder.
        z = f['draws'].transpose(0, 1, 3, 2, 4).reshape(6, 2, 32, 96)
        y = f['truth'].reshape(6, 2, 96)
        vv = np.mean(np.sum((z-z.mean(2, keepdims=True))**2, axis=-1), axis=2)*32/31
        qq = np.sum((z.mean(2)-y)**2, axis=-1)-vv/32
        rr = qq.sum(0)/vv.sum(0)
        np.testing.assert_allclose(rr, binding['seed_ratios'], rtol=1e-12)
        np.testing.assert_allclose(qq.sum()/vv.sum(), binding['baseline_ratio'], rtol=1e-12)
        floor = abs(abs(rr[0]-1)-abs(rr[1]-1))/np.sqrt(2)
        np.testing.assert_allclose(floor, binding['floor'], rtol=1e-12)
        checks += 3
    with np.load(result['simulations']['path'], allow_pickle=False) as f:
        rejects = 0
        for i, family in enumerate(cfg['families']):
            for j, corr in enumerate(cfg['correlations']):
                address = i*3+j
                a, b = f[f'{address}_base'], f[f'{address}_candidates']
                aa, bb = f[f'{address}_oracle_base'], f[f'{address}_oracle_candidates']
                assert a.shape == (4096,) and b.shape == (5, 2, 4, 4096)
                assert all(np.isfinite(t).all() for t in (a, b, aa, bb))
                checks += 2
                if corr == 1.:
                    np.testing.assert_allclose(b[0], np.broadcast_to(a, b[0].shape), atol=2e-14)
                    checks += 1
                for k, effect in enumerate(cfg['effects']):
                    row = result['rows'][address*5+k]
                    assert (row['family'], row['correlation'], row['effect']) == (family, corr, effect)
                    event = np.zeros(4096, bool)
                    for arm in range(2):
                        for checkpoint in range(4):
                            improvement = np.abs(a-1)-np.abs(b[k, arm, checkpoint]-1)
                            event |= (improvement >= .1*np.abs(a-1)) & (improvement >= floor)
                    oracle_gain = np.abs(aa-1)-np.abs(bb[k]-1)
                    oracle_event = (oracle_gain >= .1*np.abs(aa-1)) & (oracle_gain >= floor)
                    for label, values in (('finite', event), ('oracle', oracle_event)):
                        p = np.count_nonzero(values)/4096
                        radius = np.sqrt(np.log(120/.01)/8192)
                        assert row[label]['successes'] == np.count_nonzero(values)
                        np.testing.assert_allclose([row[label]['point'], row[label]['upper'], row[label]['lower']],
                                                   [p, min(1., p+radius), max(0., p-radius)], rtol=1e-13)
                        checks += 2
                    expected_target = 1+(1-effect)*(binding['baseline_ratio']-1)
                    np.testing.assert_allclose(row['target'], expected_target, rtol=1e-12)
                    if effect in (.2, .4) and row['finite']['upper'] < .8:
                        rejects += 1
                    checks += 2
    assert result['comparisons'] == 120 and len(result['rows']) == 60
    assert rejects == result['rejecting_rows']
    assert result['decision'] == ('reject_insufficient_sensitivity' if rejects else 'unresolved_requires_full_rule')
    assert not result['gate2_passed'] and not result['complete_rule_simulated'] and not result['posterior_qualified']
    checks += 4
    receipt = dict(passed=True, checks=checks, rejecting_rows=rejects,
                   independent_event_reconciliation=True, baseline_trace_reconciliation=True,
                   gate2_passed=False, job=os.environ['SLURM_JOB_ID'],
                   source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with (OUT/'VERIFIED.json').open('x') as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    verify()
