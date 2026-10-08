"""Complete matched diagnostic only after all registered products validate."""
import os
import time

import numpy as np
import e2e_true_coarse_fraction_saved as io
import e2e_true_coarse_fraction_metrics as metrics


def load():
    saved = io.read(io.OUT/'SAVED_FIELDS_COMPLETE.json')
    io.record(io.OUT/'SAVED_FIELDS_COMPLETE.json'); io.verify(saved['arrays'])
    if not saved['passed']:
        raise ValueError('Saved-data stage incomplete')
    with np.load(saved['arrays']['path']) as f:
        a, truth, ids = [f[k] for k in ('parent_fractions', 'truth_fractions', 'parent_pair_id')]
    b, ideal, production, cases, names = [], [], [], [], []
    for phase in io.PHASES:
        for cap in ('NGC', 'SGC'):
            chunks = []
            for first in range(0, 32, 4):
                path = io.OUT/'cores'/phase/cap/f'd{first:03d}/COMPLETE.json'
                rec = io.read(path); io.record(path); io.verify(rec['output'])
                if not rec['passed'] or (rec['phase'], rec['cap'], rec['first'], rec['draws']) != (phase, cap, first, 4):
                    raise ValueError('Incomplete or misaddressed B product')
                for source in rec['sources']:
                    io.verify(source)
                with np.load(rec['output']['path']) as f:
                    chunks.append(f['fractions'])
                    if first == 0:
                        fixed = [f[k] for k in ('truth_fractions', 'galaxy_counts', 'pair_id')]
                        meta = rec['cases']
                    else:
                        for key, old in zip(('truth_fractions', 'galaxy_counts', 'pair_id'), fixed):
                            np.testing.assert_array_equal(f[key], old)
                        if meta != rec['cases']:
                            raise ValueError('B metadata changed across draws')
            np.testing.assert_array_equal(truth[len(cases):len(cases)+8], fixed[0])
            b.append(np.concatenate(chunks, axis=2)); cases.extend(meta); names.extend(fixed[2])
            path = io.OUT/'truth_closure'/phase/cap/'COMPLETE.json'
            rec = io.read(path); io.record(path); io.verify(rec['output'])
            if not rec['passed'] or (rec['phase'], rec['cap'], rec['draws']) != (phase, cap, 32):
                raise ValueError('Closure incomplete')
            with np.load(rec['output']['path']) as f:
                for key, old in zip(('truth_fractions', 'galaxy_counts', 'pair_id'), fixed):
                    np.testing.assert_array_equal(f[key], old)
                ideal.append(f['ideal_fractions']); production.append(f['production_fractions'])
    np.testing.assert_array_equal(names, ids)
    if len(cases) != 32 or len(set(names)) != 32:
        raise ValueError('Registered 32-pair panel required')
    b = np.concatenate(b, axis=1)
    ideal, production = np.concatenate(ideal, axis=1), np.concatenate(production, axis=1)
    if a.shape != b.shape or a.shape != (2, 32, 32, 2, 4):
        raise ValueError('Matched posterior dimensions required')
    return saved, a, b, truth, ideal, production, cases


def closure_error(prediction, truth):
    error = prediction-truth
    return dict(bias=error.mean((0, 1)).tolist(),
        rms=np.sqrt((error**2).mean((0, 1))).tolist(),
        maxabs=np.abs(error).max((0, 1)).tolist(),
        units='fraction, class order void/wall/filament/cluster')


def run():
    io.compute_guard(); started = time.monotonic()
    sources = [io.record(q) for q in (__file__, io.__file__, metrics.__file__, io.CONFIG)]
    saved, a, b, truth, ideal, production, cases = load()
    _, counts = metrics.resamples()
    rows = []; closure = []; sensitivity = []
    for phase in io.PHASES:
        for group in ('interior', 'boundary'):
            ix = [j for j, r in enumerate(cases) if r['phase'] == phase and r['support_stratum'] == group]
            if len(ix) != 8:
                raise ValueError('Eight pairs per phase/stratum required')
            for mi, operator in enumerate(('original', 'corrected')):
                ra, rb = [metrics.fractions_report(x[mi, ix], truth[ix], counts) for x in (a, b)]
                old = next(q for q in saved['A'] if (q['phase'], q['group'], q['operator']) == (phase, group, operator))
                np.testing.assert_allclose(ra['joint']['ratio'], old['joint']['ratio'], rtol=0, atol=1e-12)
                rows.append(dict(phase=phase, group=group, operator=operator, A=ra, B=rb,
                    trace_variance_A_minus_B=ra['joint']['trace_predictive_variance']-rb['joint']['trace_predictive_variance']))
                closure.append(dict(phase=phase, group=group, operator=operator,
                    ideal=closure_error(ideal[mi, ix], truth[ix]),
                    true_covered_mean=closure_error(production[mi, ix].mean(1), truth[ix]),
                    true_covered_moments=metrics.fractions_report(production[mi, ix], truth[ix], counts)))
                for name, target in (('ideal_full_source_cuboid', ideal[mi]),
                        ('mean_true_covered_with_original_exterior', production[mi].mean(1))):
                    sensitivity.append(dict(phase=phase, group=group, operator=operator, target=name,
                        A=metrics.fractions_report(a[mi, ix], target[ix], counts),
                        B=metrics.fractions_report(b[mi, ix], target[ix], counts)))
    primary = {q['phase']: q for q in rows if q['group'] == 'interior' and q['operator'] == 'corrected'}
    branch = metrics.branch({p:q['A']['joint'] for p,q in primary.items()},
        {p:q['B']['joint'] for p,q in primary.items()})
    meaning = dict(coarse_dominated='coarse-conditioning pathway contributes; not a unique coarse-factor cause',
        coarse_factor_not_implicated='defect persists with true coarse; not proof coarse has no effect',
        inconclusive='frozen rule does not distinguish the proposed mechanisms')[branch]
    for rec in sources:
        if io.sha(rec['path']) != rec['sha256']:
            raise ValueError('Source changed during summary')
    io.write(io.OUT/'COMPLETE.json', dict(passed=True, diagnostic_complete=True, scope=saved['scope'],
        posterior_qualified=False, model_selection=False, confirmation_access=False,
        primary=rows, closure=closure, operator_target_sensitivity=sensitivity,
        sensitivity_caveat='Changed reference targets, diagnostic only; never substituted into frozen primary rule. No causal bias fraction inferred.',
        interpretation_branch=branch, qualified_meaning=meaning, coarse_only=saved['C'],
        power_addendum=saved['power_addendum'], sources=sources, inputs=list(io.CACHE.values()),
        seconds=time.monotonic()-started, job=os.environ['SLURM_JOB_ID']))
    print(io.json.dumps(dict(passed=True, branch=branch, primary=primary)), flush=True)


if __name__ == '__main__':
    run()
