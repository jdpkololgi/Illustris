"""Independent label/fraction and scalar-moment reconciliation of completed A/B.

Per-case accounting is descriptive only. It never changes cases, truth, or gates.
"""
import os
import numpy as np
import e2e_true_coarse_fraction_saved as io


def scalar_moments(x, truth):
    x, truth = np.asarray(x), np.asarray(truth)
    if x.ndim != 3 or x.shape[1] != 32 or truth.shape != (len(x), x.shape[2]):
        raise ValueError('Expected case/32-draw/coordinate arrays')
    error = x.mean(axis=1)-truth
    per_case_variance = np.var(x, axis=1, ddof=1).sum(axis=1)
    adjusted = (error*error).sum(axis=1)-per_case_variance/32
    variance = per_case_variance.mean()
    bias = error.mean(axis=0)
    if variance <= 0:
        raise ValueError('Positive predictive variance required')
    return dict(ratio=float(adjusted.mean()/variance), bias=bias.tolist(),
        centered_ratio=float((((error-bias)**2).sum(axis=1).mean()-variance/32)/variance),
        trace_predictive_variance=float(variance)), error, adjusted, per_case_variance


def run():
    io.compute_guard()
    complete = io.read(io.OUT/'COMPLETE.json'); io.record(io.OUT/'COMPLETE.json')
    if not complete['passed'] or not complete['diagnostic_complete']:
        raise ValueError('Final summary required')
    saved = io.read(io.OUT/'SAVED_FIELDS_COMPLETE.json')
    io.record(io.OUT/'SAVED_FIELDS_COMPLETE.json'); io.verify(saved['arrays'])
    with np.load(saved['arrays']['path']) as f:
        a, truth, ids = [f[k] for k in ('parent_fractions', 'truth_fractions', 'parent_pair_id')]
    banks, cases, weights_all = [], [], []
    checks = 0
    for phase in io.PHASES:
        for cap in ('NGC', 'SGC'):
            chunks = []
            for first in range(0, 32, 4):
                path = io.OUT/'cores'/phase/cap/f'd{first:03d}/COMPLETE.json'
                rec = io.read(path); io.record(path); io.verify(rec['output'])
                if not rec['passed'] or (rec['phase'], rec['cap'], rec['first']) != (phase, cap, first):
                    raise ValueError('Invalid core receipt')
                with np.load(rec['output']['path']) as f:
                    w = f['galaxy_counts']; labels = f['labels']
                    fractions = np.empty_like(f['fractions'])
                    for mi in range(2):
                        for case in range(8):
                            for core in range(2):
                                for draw in range(4):
                                    fractions[mi, case, draw, core] = np.bincount(
                                        labels[mi, case, draw, core], weights=w[case, core],
                                        minlength=4)/w[case, core].sum()
                    np.testing.assert_array_equal(fractions, f['fractions']); checks += 1
                    np.testing.assert_array_equal(f['pair_id'], ids[len(cases):len(cases)+8])
                    np.testing.assert_array_equal(f['truth_fractions'], truth[len(cases):len(cases)+8])
                    chunks.append(fractions)
                    if first == 0:
                        weights = w.sum(axis=-1)
                    else:
                        np.testing.assert_array_equal(weights, w.sum(axis=-1))
            banks.append(np.concatenate(chunks, axis=2))
            cases.extend(rec['cases']); weights_all.extend(weights.tolist())
    b = np.concatenate(banks, axis=1)
    np.testing.assert_equal(a.shape, b.shape)
    rows = []
    for item in complete['primary']:
        mi = ('original', 'corrected').index(item['operator'])
        ix = [j for j, case in enumerate(cases) if
              (case['phase'], case['support_stratum']) == (item['phase'], item['group'])]
        if len(ix) != 8:
            raise ValueError('All eight registered pairs required')
        for arm, bank in (('A', a), ('B', b)):
            x = bank[mi, ix, :, :, :3].reshape(8, 32, 6)
            target = truth[ix, :, :3].reshape(8, 6)
            got, errors, q, variance = scalar_moments(x, target)
            for key, value in got.items():
                np.testing.assert_allclose(value, item[arm]['joint'][key], rtol=1e-12, atol=1e-13)
                checks += 1
            contribution = q/variance.sum()
            np.testing.assert_allclose(contribution.sum(), item[arm]['joint']['ratio'], atol=1e-13)
            for k, j in enumerate(ix):
                rows.append(dict(**cases[j], arm=arm, operator=item['operator'],
                    galaxy_counts=weights_all[j], mean_error=errors[k].tolist(),
                    adjusted_squared_error=float(q[k]), trace_predictive_variance=float(variance[k]),
                    contribution_to_phase_stratum_ratio=float(contribution[k])))
    attachment = io.OLD/'decision/SAMPLED_COARSE_POWER_ADDENDUM_20261008.json'
    local = io.OUT/'SAMPLED_COARSE_POWER_ADDENDUM.json'
    if io.record(attachment)['sha256'] != io.record(local)['sha256']:
        raise ValueError('Power attachment changed')
    io.write(io.OUT/'FINAL_CHECK.json', dict(passed=True, checks=checks, cases=rows,
        sources=[io.record(__file__), io.record(io.__file__)], inputs=list(io.CACHE.values()),
        scope=saved['scope'], job=os.environ['SLURM_JOB_ID'],
        caveat='Independent scalar and label accounting; no bootstrap redefinition, case removal, causal allocation, or gate change.'))
    print(io.json.dumps(dict(passed=True, checks=checks)), flush=True)


if __name__ == '__main__':
    run()
