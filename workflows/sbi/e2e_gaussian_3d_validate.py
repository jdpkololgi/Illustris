"""Independent saved-evidence audit; never promotes failed science gates."""
import argparse
from pathlib import Path
import json
import numpy as np
from scipy.linalg import solve_triangular
from scipy.sparse.linalg import LinearOperator, cg
from scipy.special import ndtr
from scipy.stats import norm, chi2
import torch
import e2e_gaussian_3d_run as run
import e2e_exterior_cfm as affine_cfm
import e2e_gaussian_block_sampler as block
import e2e_true_coarse_fraction_saved as io


def check(condition, message, checks):
    if not condition: raise AssertionError(message)
    checks.append(message)


def close(a, b, message, checks, atol=1e-10, rtol=1e-9):
    np.testing.assert_allclose(a,b,atol=atol,rtol=rtol);checks.append(message)


def hybrid(root, checks):
    receipt=io.read(root/'mean_covariance/COMPLETE.json')
    check(len(receipt['rows'])==128,'128 intervention rows',checks)
    for record in receipt['sources']: io.verify(record)
    data=affine_cfm.GaussianTrainingData();model=affine_cfm.TimeAffineCFM(8,8,data.context_dimensions)
    checkpoint=next(r['path'] for r in receipt['sources'] if r['path'].endswith('.pt'))
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model'])
    learned=affine_cfm.heun_affine(affine_cfm.learned_coefficients(model,512),8)
    exact=(np.linalg.cholesky(data.exact_covariance),
           np.concatenate((data.exact_response,np.zeros((8,8,1))),-1))
    # Independent dense row-update assembly, and direct fixed-point iterations
    # instead of the original bilinear Lyapunov solver.
    cases=data.heldout()
    for row in receipt['rows']:
        case=cases[row['observation']]
        endpoint=((exact if row['covariance']=='exact' else learned)[0],
                  (exact if row['mean']=='exact' else learned)[1])
        updates=affine_cfm.kernels(data,endpoint,case)
        if row['order']=='reverse':updates=updates[::-1]
        matrix=np.eye(64);offset=np.zeros(64);innovation=np.zeros((64,64))
        for u in updates:
            a=np.eye(64);a[u.indices]=u.gain
            b=np.zeros(64);b[u.indices]=u.offset
            q=np.zeros((64,64));q[np.ix_(u.indices,u.indices)]=u.covariance
            matrix,offset,innovation=a@matrix,a@offset+b,a@innovation@a.T+q
        if row['oracle_global']:
            modes=block.orthonormal_span(block.global_modes(4))
            covariance=np.linalg.inv(modes.T@case['precision']@modes)
            a=np.eye(64)-modes@covariance@modes.T@case['precision']
            b=(np.eye(64)-a)@case['mu'];q=modes@covariance@modes.T
            matrix,offset,innovation=a@matrix,a@offset+b,a@innovation@a.T+q
        mu=np.zeros(64);sigma=np.zeros((64,64))
        for iteration in range(10000):
            new_mu=matrix@mu+offset;new_sigma=matrix@sigma@matrix.T+innovation
            if max(np.max(abs(new_mu-mu)),np.max(abs(new_sigma-sigma)))<1e-13:
                mu,sigma=new_mu,new_sigma;break
            mu,sigma=new_mu,new_sigma
        else:
            radius=float(np.max(abs(np.linalg.eigvals(matrix))))
            residual=max(np.max(abs(matrix@mu+offset-mu)),
                         np.max(abs(matrix@sigma@matrix.T+innovation-sigma)))
            raise AssertionError('Independent fixed-point iteration did not converge: '+str(dict(
                mean=row['mean'],covariance=row['covariance'],observation=row['observation'],
                order=row['order'],oracle_global=row['oracle_global'],radius=radius,residual=residual)))
        print('FIXED POINT',row['mean'],row['covariance'],row['observation'],
              row['order'],row['oracle_global'],iteration+1,flush=True)
        white=solve_triangular(np.linalg.cholesky(case['sigma']),np.eye(64),lower=True)
        whitened=white@(sigma-case['sigma'])@white.T
        worst=np.max(abs(np.linalg.eigvalsh((whitened+whitened.T)/2)))
        close(worst,row['stationary']['worst_relative_covariance_error'],'intervention covariance',checks,atol=2e-10)
        close(np.linalg.norm(white@(mu-case['mu'])),row['stationary']['whitened_mean_error'],
              'intervention mean',checks,atol=2e-10)
    return receipt


def independent_reference(n, cases, checks):
    k=2*np.sin(np.pi*np.fft.fftfreq(n))
    lap=sum(v*v for v in np.meshgrid(k,k,k,indexing='ij'))
    power=(1+2.25*lap)**-2;power=.2*power/power.mean();power[0,0,0]+=.8*n**3
    c=cases[0];weight=c['mask']/c['std']**2
    def precision(x):
        return np.fft.ifftn(np.fft.fftn(x.reshape((n,)*3))/power).real.ravel()+weight*x
    op=LinearOperator((n**3,n**3),matvec=precision,dtype=np.float64)
    rng=np.random.default_rng(737+n)
    for _ in range(8):
        vector=rng.normal(size=n**3)
        close(precision(c['sigma']@vector),vector,'FFT precision covariance identity',checks,atol=2e-8)
    for case in cases:
        mean,info=cg(op,weight*case['y'],rtol=1e-11,atol=1e-12,maxiter=5000)
        check(info==0,'Independent FFT-CG mean converged',checks)
        close(mean,case['mu'],'FFT-CG posterior mean',checks,atol=2e-8)


def metrics(projected, result, correlation, checks):
    n=len(projected);mu=projected.sum(0)/n
    residual=projected-mu;variance=(residual*residual).sum(0)/(n-1)
    covariance=residual.T@residual/(n-1)
    mean_se=np.sqrt(variance/n)
    squares=residual**2;variance_se=np.sqrt(((squares-squares.mean(0))**2).sum(0)/(n*(n-1)))
    ml=np.maximum(abs(mu)-4*mean_se,0);vl=np.maximum(abs(variance-1)-4*variance_se,0)
    q=np.quantile(projected,[.05,.95],axis=0);coverage=ndtr(q[1])-ndtr(q[0])
    for key,value in (('mean',mu),('variance',variance),('covariance',covariance),('mean_se',mean_se),
        ('variance_se',variance_se),('exact_probability_in_empirical90',coverage),
        ('max_mean_error_lower4se',ml.max()),('max_variance_error_lower4se',vl.max()),
        ('probe_covariance_relative',np.linalg.norm(covariance-correlation)/np.linalg.norm(correlation)),
        ('maximum_coverage_error',abs(coverage-.9).max())):
        close(value,result[key],'projection '+key,checks)
    check(result['accuracy_status']==('failed' if max(ml.max(),vl.max())>.01 else 'inconclusive'),
          'No noisy promotion',checks)
    check(result['qualified'] is False,'No finite-ensemble posterior qualification',checks)


def run_validation(root):
    io.compute_guard();torch.set_num_threads(4);checks=[]
    intervention=hybrid(root,checks)
    out=root/'run';receipt=io.read(out/'COMPLETE.json')
    check(receipt['engineering_complete'] and not receipt['posterior_qualified'],'Engineering distinct from science',checks)
    for record in receipt['sources']+receipt['files']:io.verify(record);checks.append('Hash verified')
    check(len(receipt['resources'])==10,'Complete five-size two-arm cost grid',checks)
    actual_cost=io.read(root/'actual_sweep_cost.json')
    for record in actual_cost['sources']:io.verify(record)
    check(len(actual_cost['rows'])==10,'Direct entire-sweep cost grid complete',checks)
    for row in actual_cost['rows']:
        check(row['repetitions']==3 and len(row['seconds'])==3,'Three measured timing repetitions',checks)
        close(np.mean(row['seconds']),row['mean_seconds'],'Direct timing mean',checks)
        check(row['blocks']==(1 if row['arm']=='whole' else (row['n']//8)**3),'Correct full-sweep solve count',checks)
        check(row['peak_allocated_bytes']>0 and min(row['seconds'])>0,'Finite positive resource measurements',checks)
    summary=[]; oracle_screen_rejections=[]
    # Post-hoc audit prompted by one exact-null rejection of the approximate
    # four-SE screen. Preserve that screen unchanged. These exact Gaussian-null
    # bounds cover mean and variance, both tails, all30probes and28evaluations
    # (including repeated oracle controls), at familywise alpha0.01 by Bonferroni.
    alpha=.01; maximum_evaluations=28; probes=30; count=256
    tail=alpha/(maximum_evaluations*probes*2*2)
    mean_limit=.01+np.sqrt(1.01/count)*norm.ppf(1-tail)
    variance_limits=(.99*chi2.ppf(tail,count-1)/(count-1),
                     1.01*chi2.ppf(1-tail,count-1)/(count-1))
    for n in (8,16):
        cases=run.targets(n);independent_reference(n,cases,checks)
        for arm in (('whole',) if n==8 else ('whole','block')):
            train=io.read(out/f'train_n{n}_{arm}.json')
            check(train['updates']==4096 and train['fields']==32768,'Frozen matched training budget',checks)
            check(np.isfinite(train['losses']).all() and len(train['losses'])==4096,'Complete finite loss history',checks)
            data=io.read(out/f'evaluation_n{n}_{arm}.json')
            projections=np.load(out/f'projections_n{n}_{arm}.npz')
            close(data['target_correlation'],cases[0]['correlation'],'Reference probe correlations',checks)
            for row in data['rows']:
                if row.get('integration_refinement'):
                    ci=row['observation'];initialization='whole' if arm=='whole' else 'zero';s=1 if arm=='whole' else 4
                    a=projections[f'c{ci}_{initialization}_nfe32_s{s}']
                    b=projections[f'c{ci}_{initialization}_nfe64_s{s}']
                    close(np.sqrt(np.mean((a-b)**2)),row['paired_probe_rms_change'],'Paired NFE RMS',checks)
                    close(np.max(abs((a-b).mean(0))),row['maximum_probe_mean_change'],'Paired NFE mean',checks)
                    close(np.max(abs(a.var(0)-b.var(0))),row['maximum_probe_variance_change'],'Paired NFE variance',checks)
                    continue
                if row.get('mixing_diagnostic'):
                    trace=projections[f'c{row["observation"]}_chain_trace']
                    a=trace[8:-1];b=trace[9:];a=a-a.mean(1,keepdims=True);b=b-b.mean(1,keepdims=True)
                    lag=(a*b).mean((0,1))/np.sqrt((a*a).mean((0,1))*(b*b).mean((0,1)))
                    close(lag,row['late_lag1'],'Late chain lag correlation',checks)
                    continue
                if 'metrics' not in row:raise AssertionError('Unexpected evaluation row')
                ci=row['observation']
                key=f'c{ci}_oracle' if row['arm']=='oracle_iid' else f'c{ci}_{row["initialization"]}_nfe{row["nfe"]}_s{row["sweeps"]}'
                values=projections[key]
                check(values.shape==(256,30),'Matched256draws30probes',checks)
                metrics(values,row['metrics'],cases[ci]['correlation'],checks)
                if row['arm']=='oracle_iid':
                    close(values,run.projected(cases[ci]['exact'],cases[ci]),'Saved oracle projections',checks)
                    if row['metrics']['accuracy_status']=='failed':
                        oracle_screen_rejections.append(dict(n=n,fit=arm,observation=ci))
                mean=np.mean(values,axis=0);variance=np.var(values,axis=0,ddof=1)
                rejected=bool(np.any(abs(mean)>mean_limit) or np.any(variance<variance_limits[0]) or np.any(variance>variance_limits[1]))
                if row['arm']=='oracle_iid':check(not rejected,'Exact-null simultaneous control not rejected',checks)
                summary.append(dict(n=n,fit=arm,observation=ci,initialization=row.get('initialization','oracle_iid'),
                    nfe=row['nfe'],sweeps=row['sweeps'],status=row['metrics']['accuracy_status'],
                    max_mean_lower4se=row['metrics']['max_mean_error_lower4se'],
                    max_variance_lower4se=row['metrics']['max_variance_error_lower4se'],
                    simultaneous_gaussian_null_rejected=rejected,
                    maximum_absolute_mean=float(abs(mean).max()),minimum_variance=float(variance.min()),
                    maximum_variance=float(variance.max())))
    # Static source audit is supplemented by the paired-training and leakage tests.
    check(receipt['time_to_qualified_draw'] is None,'No speedup at unestablished matched accuracy',checks)
    io.write(root/'VALIDATION.json',dict(passed=True,checks=len(checks),check_names=checks,
        summary=summary,intervention_rows=len(intervention['rows']),
        original_four_se_screen_calibrated=False,original_oracle_screen_rejections=oracle_screen_rejections,
        simultaneous_gaussian_audit=dict(posthoc=True,reason='Original exact-null control falsely rejected',
            familywise_alpha=alpha,maximum_evaluations=maximum_evaluations,probes=probes,
            mean_absolute_limit=float(mean_limit),variance_limits=list(map(float,variance_limits)),
            scope='Reject Gaussian reference law allowing1percent mean and variance deviations; non-rejection is not qualification'),
        numerical_reference='Independent FFT precision action and conjugate-gradient means',
        intervention_reference='Independent dense row composition and fixed-point iteration',
        scope='Engineering and saved calculation checks, not scientific qualification',
        sources=[io.record(__file__),io.record(out/'COMPLETE.json'),io.record(root/'mean_covariance/COMPLETE.json'),
            io.record(io.REPO/'docs/e2e_matched_gaussian_3d_validation_addendum_20261009.md'),
            io.record(io.REPO/'tests/test_e2e_gaussian_3d_audit.py'),io.record(root/'actual_sweep_cost.json')]))
    print('VALIDATION',len(checks),'checks passed',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);run_validation(Path(p.parse_args().root))
