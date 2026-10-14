"""Saved eigenvalue threshold sensitivity; never chooses a threshold or gate."""
import os
import numpy as np
import e2e_true_coarse_fraction_saved as io
import e2e_true_coarse_fraction_metrics as metrics

OUT = io.PARENT/'threshold_canvas_diagnostics_20261008_v1'
PLAN = io.REPO/'docs/e2e_threshold_canvas_diagnostics_20261008.md'
THRESHOLDS = (0., .1, .15, .2, .25, .3, .4)
EDGES = (0., .01, .025, .05, .1, np.inf)


def fractions(labels, truth, weight):
    # posterior case/draw/core/cell, truth case/core/cell
    return (np.stack([((labels == k)*weight[:, None]).sum(-1)/weight.sum(-1)[:, None]
                      for k in range(4)], -1),
            np.stack([((truth == k)*weight).sum(-1)/weight.sum(-1) for k in range(4)], -1))


def scores(labels, truth):
    probability = np.stack([(labels == k).mean(1) for k in range(4)], -1)
    onehot = np.eye(4)[truth]
    disagreement = 1-(probability*onehot).sum(-1)
    brier = ((probability-onehot)**2).sum(-1)
    correction = (probability*(1-probability)).sum(-1)/(labels.shape[1]-1)
    return disagreement, brier, brier-correction


def margin_rows(labels, truth, target_eigen, weight, cases, threshold, arm, operator):
    disagree, brier, unbiased = scores(labels, truth)
    margin = np.min(abs(target_eigen-threshold), axis=-1)
    rows = []
    for group in ('interior', 'boundary'):
        selected = np.array([r['support_stratum'] == group for r in cases])
        for lo, hi in zip(EDGES[:-1], EDGES[1:]):
            w = weight*((margin >= lo) & (margin < hi))*selected[:, None, None]
            rows.append(dict(phase=cases[0]['phase'], group=group, arm=arm, operator=operator,
                threshold=threshold, margin_lower=lo, margin_upper=None if np.isinf(hi) else hi,
                galaxy_weight=float(w.sum()), disagreement_weight=float((w*disagree).sum()),
                brier_weight=float((w*brier).sum()), unbiased_brier_weight=float((w*unbiased).sum())))
    return rows


def run():
    io.compute_guard(); OUT.mkdir(exist_ok=True)
    sources = [io.record(p) for p in (__file__, io.__file__, metrics.__file__, PLAN)]
    final = io.read(io.OUT/'COMPLETE.json'); io.record(io.OUT/'COMPLETE.json')
    check = io.read(io.OUT/'FINAL_CHECK.json'); io.record(io.OUT/'FINAL_CHECK.json')
    assert final['diagnostic_complete'] and check['passed']
    banks = {t:[] for t in THRESHOLDS}; targets = {t:[] for t in THRESHOLDS}
    cases = []; margins = []; closure = []; case_rows = []
    for phase in io.PHASES:
        for cap in ('NGC', 'SGC'):
            eigen_chunks = []; b_chunks = []
            for first in range(0, 32, 4):
                path = io.OLD/'cores'/phase/cap/'parent/s0'/f'd{first:03d}/COMPLETE.json'
                rec = io.read(path); io.record(path); io.verify(rec['binding']); io.verify(rec['output'])
                assert rec['passed'] and (rec['phase'],rec['cap'],rec['first'],rec['draws']) == (phase,cap,first,4)
                with np.load(rec['output']['path']) as f:
                    eigen_chunks.append(f['eigen'])
                    current = [f[k] for k in ('truth_eigen', 'galaxy_counts', 'pair_id')]
                    if first == 0:
                        fixed = current; meta = rec['cases']
                    else:
                        assert rec['cases'] == meta
                        for new, old in zip(current, fixed): np.testing.assert_array_equal(new, old)
                path = io.OUT/'cores'/phase/cap/f'd{first:03d}/COMPLETE.json'
                rec = io.read(path); io.record(path); io.verify(rec['output'])
                assert rec['passed'] and rec['cases'] == meta
                with np.load(rec['output']['path']) as f:
                    b_chunks.append(f['labels'])
                    np.testing.assert_array_equal(f['galaxy_counts'], fixed[1])
                    np.testing.assert_array_equal(f['pair_id'], fixed[2])
                    np.testing.assert_array_equal(f['truth_labels'], (fixed[0] > .2).sum(-1))
            eigen = np.concatenate(eigen_chunks, axis=2)
            b = np.concatenate(b_chunks, axis=2)
            target, weight, names = fixed
            assert eigen.shape == (2,8,32,2,4096,3) and list(names) == [r['pair_id'] for r in meta]
            cases.extend(meta)
            for t in THRESHOLDS:
                lab = (eigen > t).sum(-1); tlab = (target > t).sum(-1)
                fs = []
                for mi, operator in enumerate(('original', 'corrected')):
                    frac, truth = fractions(lab[mi], tlab, weight); fs.append(frac)
                    margins.extend(margin_rows(lab[mi], tlab, target, weight, meta, t, 'A', operator))
                    if t == .2:
                        margins.extend(margin_rows(b[mi], tlab, target, weight, meta, t, 'B', operator))
                    for j, case in enumerate(meta):
                        variance = frac[j, :, :, :3].var(0, ddof=1).sum()
                        error = frac[j, :, :, :3].mean(0)-truth[j, :, :3]
                        case_rows.append(dict(**case, threshold=t, operator=operator,
                            galaxy_counts=weight[j].sum(-1).tolist(),
                            one_galaxy_fraction_increment=(1/weight[j].sum(-1)).tolist(),
                            adjusted_squared_error=float((error*error).sum()-variance/32),
                            trace_predictive_variance=float(variance), mean_error=error.tolist()))
                banks[t].append(np.stack(fs)); targets[t].append(truth)
            path = io.OUT/'truth_closure'/phase/cap/'COMPLETE.json'
            rec = io.read(path); io.record(path); io.verify(rec['output'])
            assert rec['passed'] and rec['cases'] == meta
            with np.load(rec['output']['path']) as f:
                np.testing.assert_array_equal(f['pair_id'], names)
                np.testing.assert_array_equal(f['galaxy_counts'], weight)
                tlab = (target > .2).sum(-1)
                np.testing.assert_array_equal(f['truth_labels'], tlab)
                for mi, operator in enumerate(('original', 'corrected')):
                    for arm, labels in [('ideal_closure', f['ideal_labels'][mi, :, None]),
                                        ('production_closure', f['production_labels'][mi])]:
                        # Duplicate the deterministic ideal labels solely for score helper dimensions.
                        if labels.shape[1] == 1: labels = np.repeat(labels, 2, axis=1)
                        closure.extend(margin_rows(labels, tlab, target, weight, meta, .2, arm, operator))
    assert len(cases) == 32 and len({r['pair_id'] for r in cases}) == 32
    _, counts = metrics.resamples(); rows = []; checks = 0
    for t in THRESHOLDS:
        bank = np.concatenate(banks[t], axis=1); truth = np.concatenate(targets[t])
        for phase in io.PHASES:
            for group in ('interior', 'boundary'):
                ix = [j for j, case in enumerate(cases) if (case['phase'],case['support_stratum']) == (phase,group)]
                assert len(ix) == 8
                for mi, operator in enumerate(('original','corrected')):
                    report = metrics.fractions_report(bank[mi,ix], truth[ix], counts)
                    if t == .2:
                        old = next(r for r in final['primary'] if (r['phase'],r['group'],r['operator']) == (phase,group,operator))
                        for field in ('ratio','bias','trace_predictive_variance','centered_ratio'):
                            np.testing.assert_allclose(report['joint'][field],old['A']['joint'][field],atol=1e-12,rtol=0)
                            checks += 1
                    rows.append(dict(threshold=t,phase=phase,group=group,operator=operator,**report))
    for source in sources:
        if io.sha(source['path']) != source['sha256']: raise ValueError('Source changed')
    io.write(OUT/'THRESHOLD_COMPLETE.json',dict(passed=True,checks=checks,rows=rows,margins=margins,
        closure_margins=closure,cases=case_rows,sources=sources,inputs=list(io.CACHE.values()),
        job=os.environ['SLURM_JOB_ID'],posterior_qualified=False,
        caveat='Development descriptive sensitivity, no threshold selection. Multi-threshold parent only; B and closure labels available only at 0.2. Margin bins use truth, not an observational selection.'))
    print(dict(passed=True,checks=checks,rows=len(rows)),flush=True)


if __name__ == '__main__':
    run()
