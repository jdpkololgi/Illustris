"""Independent saved-projection checks for the preregistered transformer pilot."""
import argparse
from pathlib import Path
import numpy as np
from scipy.stats import norm, chi2
import torch
import e2e_gaussian_3d_run as baseline
import e2e_gaussian_3d_validate as previous
import e2e_gaussian_transformer_run as experiment
import e2e_true_coarse_fraction_saved as io


def audit(root):
    io.compute_guard(); torch.set_num_threads(4)
    checks = []; rows = []; refinements = []
    check, close = previous.check, previous.close
    receipt = io.read(root/'COMPLETE.json')
    for item in receipt['sources']+receipt['baseline']+receipt['files']:
        io.verify(item); checks.append('Source or artifact hash')
    check(receipt['engineering_complete'] and not receipt['posterior_qualified'],
          'Engineering completion distinct from posterior qualification', checks)
    count = 256; probes = 30; maximum_evaluations = 28; alpha = .01
    tail = alpha/(maximum_evaluations*probes*2*2)
    mean_limit = .01+np.sqrt(1.01/count)*norm.ppf(1-tail)
    variance_limits = (.99*chi2.ppf(tail,count-1)/(count-1),
                       1.01*chi2.ppf(1-tail,count-1)/(count-1))
    for n in (8, 16):
        cases = baseline.targets(n)
        previous.independent_reference(n, cases, checks)
        old_oracle = None
        for architecture, location in (('unet', experiment.BASE/'run'), ('transformer', root)):
            training = io.read(location/f'train_n{n}_whole.json')
            check(training['updates'] == 4096 and training['fields'] == 32768,
                  'Matched update and field count', checks)
            check(training['supervised_voxels'] == 32768*n**3, 'Matched target-voxel supervision', checks)
            check(training['seed'] == baseline.SEED and len(training['losses']) == 4096,
                  'Frozen seed and complete loss history', checks)
            check(np.isfinite(training['losses']).all(), 'Finite training loss history', checks)
            data = io.read(location/f'evaluation_n{n}_whole.json')
            check(data['probes'] == cases[0]['names'], 'Identical fixed probe names', checks)
            close(data['target_correlation'], cases[0]['correlation'], 'Exact probe correlations', checks)
            with np.load(location/f'projections_n{n}_whole.npz') as projections:
                oracles = [projections[f'c{ci}_oracle'] for ci in range(2)]
                if old_oracle is None: old_oracle = oracles
                for ci, case in enumerate(cases):
                    close(oracles[ci], baseline.projected(case['exact'], case), 'Exact-control replay', checks)
                    close(oracles[ci], old_oracle[ci], 'Identical observations and exact controls across architectures', checks)
                for row in data['rows']:
                    ci = row['observation']
                    if row.get('integration_refinement'):
                        a = projections[f'c{ci}_whole_nfe32_s1']; b = projections[f'c{ci}_whole_nfe64_s1']
                        dm = abs((a-b).mean(0)).max()
                        dv = abs(a.var(0)-b.var(0)).max()
                        close(dm, row['maximum_probe_mean_change'], 'NFE paired mean refinement', checks)
                        close(dv, row['maximum_probe_variance_change'], 'NFE paired variance refinement', checks)
                        close(np.sqrt(np.mean((a-b)**2)), row['paired_probe_rms_change'], 'NFE paired RMS refinement', checks)
                        refinements.append(dict(n=n, architecture=architecture, observation=ci,
                            maximum_probe_mean_change=float(dm), maximum_probe_variance_change=float(dv),
                            paired_probe_rms_change=row['paired_probe_rms_change'],
                            tight_probe_refinement_pass=bool(dm <= .01 and dv <= .01)))
                        continue
                    oracle = row['arm'] == 'oracle_iid'
                    key = f'c{ci}_oracle' if oracle else f'c{ci}_whole_nfe{row["nfe"]}_s1'
                    values = projections[key]
                    check(values.shape == (count,probes) and np.isfinite(values).all(), '256 finite draws, 30 probes', checks)
                    previous.metrics(values,row['metrics'],cases[ci]['correlation'],checks)
                    # Recompute sums and centered sums of squares, not stored metric extrema.
                    mean = np.sum(values,axis=0)/count
                    var = np.sum((values-mean)**2,axis=0)/(count-1)
                    rejected = bool(np.any(abs(mean)>mean_limit) or np.any(var<variance_limits[0]) or np.any(var>variance_limits[1]))
                    # A second CDF-space decision independently checks the quantile threshold implementation.
                    mean_p = norm.sf(np.maximum(abs(mean)-.01,0)*np.sqrt(count/1.01))
                    low_p = chi2.cdf(var*(count-1)/.99,count-1)
                    high_p = chi2.sf(var*(count-1)/1.01,count-1)
                    check(rejected == bool(min(mean_p.min(),low_p.min(),high_p.min())<tail), 'CDF versus quantile null decisions', checks)
                    close(abs(mean).max(),row['metrics']['max_mean_error'],'Worst probe mean',checks)
                    if oracle:
                        check(not rejected,'Exact Gaussian control not rejected',checks)
                        if architecture == 'transformer': continue  # Count identical controls only once.
                    rows.append(dict(n=n, architecture='oracle_iid' if oracle else architecture,
                        observation=ci,nfe=row['nfe'],max_mean_error=float(abs(mean).max()),
                        worst_mean_probe=cases[ci]['names'][int(abs(mean).argmax())],
                        domain_mean_error=float(mean[0]),min_variance=float(var.min()),max_variance=float(var.max()),
                        null_rejected=rejected, status='failed' if rejected else 'inconclusive',qualified=False,
                        probe_covariance_relative=row['metrics']['probe_covariance_relative'],
                        maximum_coverage_error=row['metrics']['maximum_coverage_error']))
        del cases
    check(len(rows)==20,'Complete 16 learned and four independent control evaluations',checks)
    costs = io.read(root/'cost.json')['rows']
    check(len(costs)==4,'Two-size, two-architecture direct cost comparison',checks)
    for row in costs:
        check(row['repetitions']==3 and len(row['seconds'])==3,'Three timing repetitions',checks)
        check(np.isfinite(row['seconds']).all() and min(row['seconds'])>0 and row['peak_allocated_bytes']>0,
              'Finite positive timings and memory',checks)
        close(row['mean_seconds'],sum(row['seconds'])/3,'Independent mean runtime',checks)
        check(row['context_included'] and not row['trained_checkpoint'] and row['time_to_qualified_draw'] is None,
              'Cost benchmark scope preserved',checks)
    io.write(root/'VALIDATION.json',dict(passed=True,checks=len(checks),check_names=checks,summary=rows,
        refinements=refinements,resources=costs,posterior_qualified=False,
        null_test=dict(predeclared=True,familywise_alpha=alpha,maximum_evaluations=maximum_evaluations,
            actual_evaluations=len(rows),probes=probes,mean_absolute_limit=float(mean_limit),
            variance_limits=list(map(float,variance_limits)),
            scope='Gaussian-reference rejection screen with 1percent mean/variance slack; non-rejection is not qualification'),
        sources=[io.record(__file__),io.record(root/'COMPLETE.json')]))
    print('VALIDATION',len(checks),'checks passed',flush=True)
    for row in rows:
        if row['nfe']==32: print('COMPARISON',row,flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    audit(Path(parser.parse_args().root))
