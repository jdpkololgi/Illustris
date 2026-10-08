"""Truth-only closure readout, independent of unfinished neural generation."""
import os

import numpy as np
import e2e_true_coarse_fraction_saved as io
import e2e_true_coarse_fraction_metrics as metrics
from e2e_true_coarse_fraction_summary import closure_error


def run():
    io.compute_guard()
    saved_path = io.OUT/'SAVED_FIELDS_COMPLETE.json'
    saved = io.read(saved_path); io.record(saved_path); io.verify(saved['arrays'])
    with np.load(saved['arrays']['path']) as f:
        parent, truth, ids = [f[k] for k in ('parent_fractions', 'truth_fractions', 'parent_pair_id')]
    ideal, production, names, cases = [], [], [], []
    for phase in io.PHASES:
        for cap in ('NGC', 'SGC'):
            path = io.OUT/'truth_closure'/phase/cap/'COMPLETE.json'
            rec = io.read(path); io.record(path); io.verify(rec['output'])
            if not rec['passed'] or (rec['phase'], rec['cap'], rec['draws']) != (phase, cap, 32):
                raise ValueError('Four complete registered closure banks required')
            with np.load(rec['output']['path']) as f:
                np.testing.assert_array_equal(truth[len(cases):len(cases)+8], f['truth_fractions'])
                ideal.append(f['ideal_fractions']); production.append(f['production_fractions'])
                names.extend(f['pair_id']); cases.extend(rec['cases'])
    np.testing.assert_array_equal(names, ids)
    ideal, production = np.concatenate(ideal, axis=1), np.concatenate(production, axis=1)
    _, counts = metrics.resamples(); reports = []
    for phase in io.PHASES:
        for group in ('interior', 'boundary'):
            ix = [j for j, r in enumerate(cases) if r['phase'] == phase and r['support_stratum'] == group]
            if len(ix) != 8:
                raise ValueError('Eight pairs per stratum required')
            for mi, operator in enumerate(('original', 'corrected')):
                reports.append(dict(phase=phase, group=group, operator=operator,
                    ideal_error=closure_error(ideal[mi, ix], truth[ix]),
                    true_covered_mean_error=closure_error(production[mi, ix].mean(1), truth[ix]),
                    true_covered_moments=metrics.fractions_report(production[mi, ix], truth[ix], counts),
                    A_physical=metrics.fractions_report(parent[mi, ix], truth[ix], counts),
                    A_ideal_reference=metrics.fractions_report(parent[mi, ix], ideal[mi, ix], counts),
                    A_true_covered_mean_reference=metrics.fractions_report(parent[mi, ix], production[mi, ix].mean(1), counts)))
    io.write(io.OUT/'TRUTH_CLOSURE_COMPLETE.json', dict(passed=True, closure_complete=True,
        reports=reports, scope=saved['scope'], sources=[io.record(__file__), io.record(metrics.__file__),
            io.record(io.__file__), io.record(io.REPO/'workflows/sbi/e2e_true_coarse_fraction_summary.py')],
        inputs=list(io.CACHE.values()), posterior_qualified=False, job=os.environ['SLURM_JOB_ID'],
        caveat='Changed references are sensitivities only; original physical truth and A/B rule remain primary. No fraction of bias is causally attributed.'))


if __name__ == '__main__':
    run()
