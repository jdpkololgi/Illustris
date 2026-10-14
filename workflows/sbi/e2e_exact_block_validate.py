"""Independent receipt reconciliation; no new targets or gate changes."""
import hashlib
import json
import os
from pathlib import Path
import socket
import sys


def main(root):
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith('login'):
        raise RuntimeError('Compute allocation required')
    root = Path(root)
    receipt = json.loads((root/'COMPLETE.json').read_text())
    checks = 0

    def check(value):
        nonlocal checks
        if not value:
            raise AssertionError(f'Reconciliation check {checks+1} failed')
        checks += 1

    check(receipt['diagnostic_complete'] and receipt['passed'])
    check(len(receipt['targets']) == 18 and len(receipt['rows']) == 48)
    check(not any(receipt[k] for k in ('posterior_qualified','abacus_inputs','training')))
    check(receipt['runtime']['temporary_scratch_independent'])
    check('/pscratch/' not in receipt['runtime']['executable'])
    for source in receipt['sources']:
        check(hashlib.sha256(Path(source['path']).read_bytes()).hexdigest() == source['sha256'])
    keys = {(r['case'],r['order'],r['arm']) for r in receipt['rows']}
    check(len(keys) == 48)
    for target in receipt['targets']:
        key = target['case']
        part = json.loads((root/(key+'.json')).read_text())
        check(part['passed'] and part['target'] == target)
        check(part['sources'] == receipt['sources'] and part['runtime'] == receipt['runtime'])
        rows = [r for r in receipt['rows'] if r['case'] == key]
        check(part['rows'] == rows)
        orders = ('forward','reverse') if target['n'] == 4 else ('forward',)
        check({(r['order'],r['arm']) for r in rows} == {(o,a) for o in orders for a in ('local','local_global')})
        if target['n'] == 4:
            check(len(target['current_canvas_control']) == 2)
            check(next(r for r in target['current_canvas_control'] if r['width']==4)['covariance_relative'] <= .001)
        if target['constrained']:
            check(target['dimensions'] == 504 and 'domain_mean' in target['omitted_null_probes'])
        for row in rows:
            inv = row['invariance']
            check(inv['mean_relative'] <= 1e-8 and inv['covariance_relative'] <= 1e-8 and inv['spectral_radius'] < 1)
            check([c['sweeps'] for c in row['curve']] == [2**i for i in range(10)])
            for curve in row['curve']:
                maximum = max(abs(c-.9) for c in curve['coverage90'])
                check(abs(maximum-curve['maximum_coverage_error']) < 1e-14)
                expected = curve['worst_mean_contraction'] <= .01 and curve['worst_relative_covariance_error'] <= .01 and maximum <= .005
                check(curve['passed'] == expected)
            first = next((c['sweeps'] for c in row['curve'] if c['passed']),None)
            check(row['first_passing_sweeps'] == first and row['finite_budget_convergence_passed'] == (first is not None))
            check(abs(max(row['stationary_tau'])-row['worst_probe_tau']) < 1e-14)
            check(all(t > 0 for t in row['stationary_tau']))
            check(abs(row['median_sweep_seconds']*row['worst_probe_tau']-row['seconds_per_effective_draw_worst_probe']) < 1e-12)
            if target['constrained']:
                check(row['max_block_mean'] < 1e-12)
    check(len(receipt['actual_sampler_checks']) == 4)
    for row in receipt['actual_sampler_checks']:
        check(row['passed'] and max(row['max_mean_standard_errors'],row['max_covariance_standard_errors']) < 7)
    check(receipt['gaussian_invariance_passed'] == all(r['gaussian_invariance_passed'] for r in receipt['rows']))
    check(receipt['finite_budget_convergence_passed'] == all(r['finite_budget_convergence_passed'] for r in receipt['rows']))
    check(receipt['information_insufficiency']['passed'])
    result = dict(passed=True,checks=checks,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        complete_sha256=hashlib.sha256((root/'COMPLETE.json').read_bytes()).hexdigest(),
        job=os.environ['SLURM_JOB_ID'],node=socket.gethostname(),
        scope='Independent receipt/gate/source reconciliation; not a second implementation of the sampler.')
    with (root/'VALIDATION.json').open('x') as stream:
        json.dump(result,stream,indent=2)
    print(json.dumps(result))
    print('case | order | sweeps local/global | tau local/global | time-effective local/global')
    for row in receipt['rows']:
        if row['arm'] != 'local':
            continue
        other = next(r for r in receipt['rows'] if r['case']==row['case'] and r['order']==row['order'] and r['arm']=='local_global')
        print(row['case'],row['order'],row['first_passing_sweeps'],other['first_passing_sweeps'],
              row['worst_probe_tau'],other['worst_probe_tau'],
              row['seconds_per_effective_draw_worst_probe'],other['seconds_per_effective_draw_worst_probe'])
    print('max covariance invariance residual',max(r['invariance']['covariance_relative'] for r in receipt['rows']))
    print('runtime',receipt['seconds'],receipt['runtime'])
    print('sampler checks',receipt['actual_sampler_checks'])


if __name__ == '__main__':
    main(sys.argv[1])
