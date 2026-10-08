"""Independent reconciliation of the saved-power addendum to historical spectra."""
import ast
import os
import numpy as np
import e2e_true_coarse_fraction_saved as io


def run():
    io.compute_guard()
    addendum_path = io.OUT/'SAMPLED_COARSE_POWER_ADDENDUM.json'
    addendum = io.read(addendum_path); io.record(addendum_path)
    saved_path = io.OUT/'SAVED_FIELDS_COMPLETE.json'; saved = io.read(saved_path); io.record(saved_path)
    if not saved['passed'] or not addendum['passed']:
        raise ValueError('Saved analysis incomplete')
    checks = 0
    for phase in io.PHASES:
        path = io.FACTOR/f'SCALES_{phase}.json'; reference = io.read(path); io.record(path)
        for target in (r for r in reference['summary'] if r['arm'] == 'canvas_raw'):
            name = 'coarse/legacy' if target['factor'] == 'coarse' else 'parent_s0/legacy'
            row = next(q for q in addendum['summary'] if q['phase'] == phase and q['setting_factor'] == name and q['k_min'] == target['k_min'] and q['k_max'] == target['k_max'])
            for field in ('truth_power', 'mean_power', 'sample_power', 'posterior_variance_power', 'mean_error_power', 'mean_error_minus_draw_noise', 'sample_truth_power_ratio'):
                np.testing.assert_allclose(row[field], target[field], rtol=1e-12, atol=1e-13)
                checks += 1
    comparisons = []
    for row in addendum['summary']:
        if not row['setting_factor'].endswith('/primary'):
            continue
        other = next(q for q in addendum['summary'] if q['phase'] == row['phase'] and q['setting_factor'] == row['setting_factor'].replace('/primary', '/legacy') and q['k_min'] == row['k_min'] and q['k_max'] == row['k_max'])
        comparisons.append(dict(phase=row['phase'], setting_factor=row['setting_factor'], k_min=row['k_min'],
            modes_difference=row['modes_per_window']-other['modes_per_window'],
            power_ratio_difference=row['sample_truth_power_ratio']-other['sample_truth_power_ratio']))
    parsed = []
    for name in ('metrics', 'saved', 'closure', 'prepare', 'sample', 'extract', 'verify'):
        path = io.REPO/'workflows/sbi'/f'e2e_true_coarse_fraction_{name}.py'
        ast.parse(path.read_text()); parsed.append(io.record(path))
    io.write(io.OUT/'VERIFICATION.json', dict(passed=True, historical_power_reconciliation_checks=checks,
        half_open_vs_legacy=comparisons, sources=parsed, inputs=list(io.CACHE.values()),
        scope='Saved-data addendum and syntax only; does not validate unexecuted generation', job=os.environ['SLURM_JOB_ID']))
    print(io.json.dumps(dict(passed=True, historical_power_reconciliation_checks=checks)), flush=True)


if __name__ == '__main__': run()
