"""Prospective Gaussian block/global campaign; no Abacus inputs or training."""
import itertools
import importlib.metadata
import os
from pathlib import Path
import socket
import sys
import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve

import e2e_gaussian_block_sampler as block
import e2e_conditional_reference_math as reference
import e2e_true_coarse_fraction_saved as io

OUT = Path(os.environ.get('E2E_EXACT_BLOCK_OUTPUT', str(io.PARENT/'exact_block_sampler_20261009_v1')))
PLAN = io.REPO/'docs/e2e_exact_block_sampler_protocol_20261009.md'
SEED = 6100917


def problem(n, kind, observation, constrained, seed):
    covariance = reference.prior(n)[0]
    if kind == 'independent': covariance = np.eye(n**3)
    elif kind == 'long_range': covariance = .2*covariance+.8*np.ones_like(covariance)
    elif kind != 'spectral': raise ValueError(kind)
    if constrained:
        basis, indices = block.constrained_basis(n)
    else:
        basis = np.eye(n**3)
        indices = block.patch_indices(n,n//2,max(1,n//4))
    prior = block.symmetric(basis.T@covariance@basis)
    masks, stds = reference.observation_templates(n)
    mask, std = masks[observation].copy(), stds[observation].copy()
    if observation == 1:
        i,j,k = np.indices((n,)*3)
        mask *= ((i+2*j+3*k)%8==0).ravel()
        std *= 2
    rng = np.random.default_rng(seed)
    true = np.linalg.cholesky(prior)@rng.normal(size=len(prior))
    y = mask*(basis@true+std*rng.normal(size=n**3))
    weight = mask/std**2
    precision = cho_solve(cho_factor(prior),np.eye(len(prior)))+basis.T@(weight[:,None]*basis)
    sigma = block.symmetric(cho_solve(cho_factor(precision),np.eye(len(prior))))
    mu = sigma@(basis.T@(weight*y))
    # Independent covariance-conditioning identity in observation coordinates.
    h = basis[mask.astype(bool)]
    cross = prior@h.T
    obs_cov = h@cross+np.diag(std[mask.astype(bool)]**2)
    alternate = prior-cross@cho_solve(cho_factor(obs_cov),cross.T)
    alternate_mu = cross@cho_solve(cho_factor(obs_cov),y[mask.astype(bool)])
    np.testing.assert_allclose(sigma,alternate,atol=2e-10,rtol=2e-9)
    np.testing.assert_allclose(mu,alternate_mu,atol=2e-10,rtol=2e-9)
    return dict(mu=mu,sigma=sigma,precision=block.symmetric(precision),basis=basis,
        patches=indices,observed=int(mask.sum()),seed=seed)


def updates_for(case, n, reverse=False):
    identity = np.eye(len(case['mu']))
    patches = list(reversed(case['patches'])) if reverse else case['patches']
    local = [block.ConditionalUpdate.make(case['precision'],case['mu'],identity[:,ix]) for ix in patches]
    modes = block.orthonormal_span(case['basis'].T@block.global_modes(n))
    global_update = block.ConditionalUpdate.make(case['precision'],case['mu'],modes)
    return local, global_update


def information_check():
    """No available patch contains both opposite corners: marginals identical."""
    covariance = np.eye(64); opposite = np.zeros_like(covariance)
    opposite[0,-1] = opposite[-1,0] = .6
    positive, negative = covariance+opposite,covariance-opposite
    np.linalg.cholesky(positive); np.linalg.cholesky(negative)
    patches = block.patch_indices(4,3,1)
    for patch in patches:
        np.testing.assert_array_equal(positive[np.ix_(patch,patch)],negative[np.ix_(patch,patch)])
    return dict(passed=True,patches=len(patches),positive_cross_covariance=.6,
        negative_cross_covariance=-.6,scope='Unobserved opposite corners, identical full patch marginals; exterior information is necessary in general.')


def sample_check(case, local, global_update, probes):
    rows = []; count = 16384
    for arm, updates in [('local',local),('local_global',local+[global_update])]:
        for initialization in ('stationary','zero'):
            rng = np.random.default_rng(SEED+81)
            if initialization == 'stationary':
                state = case['mu']+rng.normal(size=(count,len(case['mu'])))@np.linalg.cholesky(case['sigma']).T
                expected_mean, expected_cov = case['mu'],case['sigma']
            else:
                state = np.zeros((count,len(case['mu'])))
                _,expected_mean,expected_cov = block.transition(updates)
            draws = block.sweep(updates,state,rng)@probes
            mu = expected_mean@probes; cov = probes.T@expected_cov@probes
            mean_z = (draws.mean(0)-mu)/np.sqrt(np.diag(cov)/count)
            sample_cov = np.cov(draws,rowvar=False)
            sd = np.sqrt((np.outer(np.diag(cov),np.diag(cov))+cov*cov)/(count-1))
            cov_z = (sample_cov-cov)/sd
            maximum = max(np.max(abs(mean_z)),np.max(abs(cov_z)))
            if maximum >= 7: raise AssertionError('Actual sampler disagrees with analytic moments')
            rows.append(dict(arm=arm,initialization=initialization,draws=count,
                max_mean_standard_errors=float(np.max(abs(mean_z))),
                max_covariance_standard_errors=float(np.max(abs(cov_z))),passed=True))
    return rows


def canvas_control(case):
    import e2e_exact_canvas_diagnostic as old
    old.torch.set_num_threads(1)
    values,vectors = np.linalg.eigh(case['sigma'])
    target = dict(mu=case['mu'],sigma=case['sigma'],values=values,vectors=vectors)
    rows = []
    for width in (3,4):
        geometry = old.geometry(4,width)
        indices = np.arange(64).reshape(4,4,4)
        patches = [old.marginal(target,indices[geometry.crop(i)[2:]].ravel()) for i in range(len(geometry.starts))]
        mean,covariance = old.pushforward(patches,geometry,512)
        got = old.measure(mean,covariance,target,4)
        if width == 4 and got['covariance_relative'] > 1e-3:
            raise AssertionError('Full-canvas finite-NFE positive control failed')
        rows.append(dict(width=width,nfe=512,**got))
    return rows


def run():
    io.compute_guard(); started = time.monotonic(); OUT.mkdir(exist_ok=True)
    source_paths = [__file__,block.__file__,reference.__file__,io.__file__,PLAN,
        io.REPO/'workflows/sbi/e2e_exact_canvas_diagnostic.py',
        io.REPO/'workflows/sbi/e2e_global_canvas.py',
        io.REPO/'workflows/sbi/e2e_conditional_reference.py',
        io.REPO/'workflows/sbi/e2e_direct_vdm.py',
        io.REPO/'workflows/sbi/e2e_vdm_context_models.py',
        io.REPO/'workflows/sbi/e2e_exact_block_step.sh',
        io.REPO/'tests/test_e2e_gaussian_block_sampler.py']
    sources = [io.record(p) for p in source_paths]
    runtime = dict(executable=sys.executable,python=sys.version,
        packages={name:importlib.metadata.version(name) for name in ('numpy','scipy','torch','h5py')},
        temporary_scratch_independent=os.environ.get('E2E_EXACT_BLOCK_TEMP_RUNTIME')=='1',
        output=str(OUT),module=os.environ.get('LOADEDMODULES',''))
    information = information_check(); rows = []; targets = []; sampling = []
    addresses = [(n,kind,obs,False) for n in (4,8) for kind in ('independent','spectral','long_range') for obs in range(2)]
    addresses += [(8,kind,obs,True) for kind in ('independent','spectral','long_range') for obs in range(2)]
    for ci,(n,kind,obs,constrained) in enumerate(addresses):
        key = f'n{n}_{kind}_o{obs}_'+('constrained' if constrained else 'full')
        setup = time.monotonic(); case = problem(n,kind,obs,constrained,SEED+ci)
        probes,names,omitted = block.probe_bank(n,case['basis'])
        case_report = dict(case=key,n=n,prior=kind,observation=obs,constrained=constrained,
            dimensions=len(case['mu']),observed=case['observed'],seed=case['seed'],
            probe_names=names,omitted_null_probes=omitted,target_setup_seconds=time.monotonic()-setup)
        if n == 4: case_report['current_canvas_control'] = canvas_control(case)
        targets.append(case_report)
        for order in (('forward','reverse') if n==4 else ('forward',)):
            kernel_started = time.monotonic()
            local,global_update = updates_for(case,n,order=='reverse')
            kernel_seconds = time.monotonic()-kernel_started
            if n==4 and kind=='spectral' and obs==0 and order=='forward':
                sampling = sample_check(case,local,global_update,probes)
            for arm,updates in [('local',local),('local_global',local+[global_update])]:
                affine = block.transition(updates)
                inv = block.invariance(affine,case['mu'],case['sigma'])
                passed = inv['mean_relative'] <= 1e-8 and inv['covariance_relative'] <= 1e-8 and inv['spectral_radius'] < 1
                if not passed: raise AssertionError((key,arm,inv))
                curve = block.convergence(affine,case['mu'],case['sigma'],probes)
                tau = block.autocorrelation(affine,case['sigma'],probes)
                durations = []
                for repetition in range(3):
                    rng = np.random.default_rng(SEED+92)
                    state = case['mu'][None].copy(); timer = time.monotonic()
                    for _ in range(32): state = block.sweep(updates,state,rng)
                    durations.append((time.monotonic()-timer)/32)
                if constrained:
                    physical = case['basis']@state[0]
                    max_block_mean = max(abs(physical[ix].mean()) for ix in block.patch_indices(n,4,4))
                    if max_block_mean > 1e-12: raise AssertionError('Block constraint drift')
                else: max_block_mean = None
                first = next((r['sweeps'] for r in curve if r['passed']),None)
                duration = float(np.median(durations))
                row = dict(case=key,order=order,arm=arm,local_updates=len(local),
                    global_rank=global_update.basis.shape[1] if arm=='local_global' else 0,
                    invariance=inv,gaussian_invariance_passed=passed,first_passing_sweeps=first,
                    finite_budget_convergence_passed=first is not None,curve=curve,
                    stationary_tau=tau.tolist(),worst_probe_tau=float(tau.max()),
                    median_sweep_seconds=duration,seconds_per_effective_draw_worst_probe=float(duration*tau.max()),
                    kernel_setup_seconds=kernel_seconds,max_block_mean=max_block_mean)
                rows.append(row)
                print(io.json.dumps(dict(case=key,order=order,arm=arm,first_passing_sweeps=first,
                    worst_probe_tau=row['worst_probe_tau'],invariance=inv)),flush=True)
        io.write(OUT/(key+'.json'),dict(passed=True,target=case_report,
            rows=[r for r in rows if r['case']==key],sources=sources,runtime=runtime,job=os.environ['SLURM_JOB_ID']))
    if len(targets)!=18 or len(rows)!=48: raise AssertionError('Incomplete registered panel')
    for source in sources:
        if io.sha(source['path']) != source['sha256']: raise ValueError('Source changed during campaign')
    io.write(OUT/'COMPLETE.json',dict(passed=True,diagnostic_complete=True,targets=targets,rows=rows,
        information_insufficiency=information,actual_sampler_checks=sampling,sources=sources,runtime=runtime,
        gaussian_invariance_passed=all(r['gaussian_invariance_passed'] for r in rows),
        finite_budget_convergence_passed=all(r['finite_budget_convergence_passed'] for r in rows),
        posterior_qualified=False,abacus_inputs=False,training=False,job=os.environ['SLURM_JOB_ID'],
        node=socket.gethostname(),seconds=time.monotonic()-started,
        limitation='Exact dense global precision oracle; not a learned boundary conditional or DESI-scale implementation.'))
    print('COMPLETE '+str(OUT/'COMPLETE.json'),flush=True)


if __name__ == '__main__':
    run()
