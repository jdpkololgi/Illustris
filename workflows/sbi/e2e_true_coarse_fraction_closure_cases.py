"""Resolve closure sensitivity by registered case, without changing selection."""
import os
import numpy as np
import e2e_true_coarse_fraction_saved as io


def run():
    io.compute_guard()
    saved = io.read(io.OUT/'SAVED_FIELDS_COMPLETE.json')
    io.record(io.OUT/'SAVED_FIELDS_COMPLETE.json'); io.verify(saved['arrays'])
    with np.load(saved['arrays']['path']) as f:
        a, truth, ids = [f[k] for k in ('parent_fractions', 'truth_fractions', 'parent_pair_id')]
    rows = []
    for phase in io.PHASES:
        for cap in ('NGC', 'SGC'):
            path = io.OUT/'truth_closure'/phase/cap/'COMPLETE.json'
            rec = io.read(path); io.record(path); io.verify(rec['output'])
            with np.load(rec['output']['path']) as f:
                for k, case in enumerate(rec['cases']):
                    j = list(ids).index(case['pair_id'])
                    weights = f['galaxy_counts'][k]
                    x = a[1, j, :, :, :3].reshape(32, 6)
                    variance = float(x.var(0, ddof=1).sum())
                    targets = dict(physical=truth[j], ideal=f['ideal_fractions'][1, k],
                        completed=f['production_fractions'][1, k].mean(0))
                    q = {}; errors = {}
                    for name, target in targets.items():
                        error = x.mean(0)-target[:, :3].reshape(6)
                        errors[name] = error.tolist()
                        q[name] = float(error@error-variance/32)
                    mismatch = (f['ideal_labels'][1, k] != f['truth_labels'][k])
                    rows.append(dict(**case, galaxy_counts=weights.sum(-1).tolist(),
                        ideal_changed_galaxy_weight=(mismatch*weights).sum(-1).tolist(),
                        physical_truth_fractions=truth[j].tolist(),
                        ideal_truth_fractions=targets['ideal'].tolist(),
                        completed_truth_fractions=targets['completed'].tolist(),
                        mean_error=errors, adjusted_squared_error=q, trace_predictive_variance=variance))
    for phase in io.PHASES:
        for group in ('interior', 'boundary'):
            selected = [r for r in rows if r['phase'] == phase and r['support_stratum'] == group]
            denominator = sum(r['trace_predictive_variance'] for r in selected)
            for row in selected:
                row['contribution_to_phase_stratum_ratio'] = {k:v/denominator for k,v in row['adjusted_squared_error'].items()}
                row['physical_minus_ideal_ratio_contribution'] = (row['adjusted_squared_error']['physical']-row['adjusted_squared_error']['ideal'])/denominator
    io.write(io.OUT/'TRUTH_CLOSURE_CASES.json', dict(passed=True, cases=rows,
        sources=[io.record(__file__), io.record(io.__file__)], inputs=list(io.CACHE.values()),
        scope=saved['scope'], job=os.environ['SLURM_JOB_ID'],
        caveat='Descriptive accounting, not causal attribution. All registered cases remain in the primary metric.'))


if __name__ == '__main__':
    run()
