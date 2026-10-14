"""Frozen restricted-context Gaussian panel; compute-node execution only."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import sys
import time

import numpy as np

import e2e_gaussian_block_sampler as block
import e2e_restricted_context as context
import e2e_true_coarse_fraction_saved as io

PLAN = io.REPO/'docs/e2e_restricted_context_cfm_protocol_20261009.md'
SEED = 61009421


def runtime():
    return dict(python=sys.version,executable=sys.executable,
        packages={n:importlib.metadata.version(n) for n in ('numpy','scipy','torch','h5py')},
        temporary_scratch_independent=True)


def run(out):
    io.compute_guard()
    out = Path(out); out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):
        raise RuntimeError('Output directory must be empty; no implicit resume/overwrite')
    paths = [__file__,context.__file__,block.__file__,context.reference.__file__,io.__file__,PLAN,
        io.REPO/'tests/test_e2e_restricted_context.py']
    sources = [io.record(p) for p in paths]
    run_environment = runtime()
    io.write(out/'MANIFEST.json',dict(sources=sources,runtime=run_environment,
        job=os.environ['SLURM_JOB_ID'],node=socket.gethostname(),seed=SEED))
    addresses = [(n,k,o) for n in (4,6,8) for k in ('independent','spectral','long_range') for o in (0,1)]
    addresses += [(6,'open',o) for o in (0,1)]
    rows = []; targets = []; started = time.monotonic()
    for ci,(n,kind,regime) in enumerate(addresses):
        case = context.problem(n,kind,regime,SEED+ci)
        key = f'n{n}_{kind}_o{regime}'
        probes,names,omitted = block.probe_bank(n,np.eye(n**3))
        target = dict(case=key,n=n,prior=kind,observation=regime,observed=int(case['mask'].sum()),
            seed=SEED+ci,probe_names=names,omitted_probes=omitted,patch_width=2,
            correlation_length_cells=1.5,periodic=kind!='open')
        targets.append(target)
        geometries = [(0,'forward')]
        if n==8 and kind=='long_range' and regime==1:
            geometries += [(0,'reverse'),(1,'forward')]
        modes = block.orthonormal_span(block.global_modes(n))
        global_update = block.ConditionalUpdate.make(case['precision'],case['mu'],modes)
        for shift,order in geometries:
            patches = context.patches(n,shift=shift)
            if order == 'reverse':
                patches = list(reversed(patches))
            full = [context.conditional(case,ix,'full',kind!='open') for ix in patches]
            for name in context.CONTEXTS:
                build_started = time.monotonic()
                updates = full if name=='full' else [context.conditional(case,ix,name,kind!='open') for ix in patches]
                losses = [context.information_loss(u,f) for u,f in zip(updates,full)]
                build_seconds = time.monotonic()-build_started
                local_transition = block.transition(updates)
                for arm in ('local','local_global'):
                    affine = local_transition if arm=='local' else global_update.compose(*local_transition)
                    inv = block.invariance(affine,case['mu'],case['sigma'])
                    exact_pass = inv['mean_relative']<=1e-8 and inv['covariance_relative']<=1e-8 and inv['spectral_radius']<1
                    if (name=='full' or kind=='independent') and not exact_pass:
                        raise AssertionError((key,name,arm,'Exact positive control failed',inv))
                    result,stationary_mean,stationary_cov = context.stationary(affine,case,probes)
                    tau = block.autocorrelation(affine,stationary_cov,probes) if result['stable'] else None
                    # Only test target burn-in when stationary target accuracy passes.
                    curve = block.convergence(affine,case['mu'],case['sigma'],probes) if result['accuracy_passed'] else []
                    first = next((r['sweeps'] for r in curve if r['passed']),None)
                    actual_updates = updates if arm=='local' else updates+[global_update]
                    durations = []
                    for repeat in range(3 if result['stable'] else 0):
                        rng = np.random.default_rng(SEED+50)
                        state = case['mu'][None].copy(); tic = time.monotonic()
                        for _ in range(16):
                            state = block.sweep(actual_updates,state,rng)
                        durations.append((time.monotonic()-tic)/16)
                    sweep_seconds = float(np.median(durations)) if durations else None
                    worst_tau = float(tau.max()) if tau is not None else None
                    row = dict(case=key,shift=shift,order=order,context=name,arm=arm,
                        updates=len(updates),context_rank_min=min(u.context_rank for u in updates),
                        context_rank_max=max(u.context_rank for u in updates),
                        expected_conditional_kl_mean=float(np.mean(losses)),
                        expected_conditional_kl_max=float(max(losses)),
                        invariance=inv,exact_invariance_passed=exact_pass,stationary=result,
                        stationary_tau=tau.tolist() if tau is not None else None,worst_probe_tau=worst_tau,
                        tau_reference='Actual chain stationary distribution; not target accuracy',
                        build_seconds=build_seconds,median_sweep_seconds=sweep_seconds,
                        seconds_per_effective_draw_worst_probe=sweep_seconds*worst_tau if worst_tau is not None else None,
                        curve=curve,first_target_passing_sweeps=first)
                    rows.append(row)
                    print(json.dumps(dict(case=key,shift=shift,order=order,context=name,arm=arm,
                        exact=exact_pass,accuracy=result['accuracy_passed'],
                        worst_covariance=result.get('worst_relative_covariance_error'),
                        stable=result['stable'],spectral_radius=result['spectral_radius'],
                        information_loss=float(np.mean(losses)))),flush=True)
        io.write(out/(key+'.json'),dict(target=target,rows=[r for r in rows if r['case']==key],
            sources=sources,runtime=run_environment,passed=True))
    if len(targets)!=20 or len(rows)!=264:
        raise AssertionError('Incomplete frozen panel')
    for source in sources:
        if io.sha(source['path']) != source['sha256']:
            raise AssertionError('Source changed during run')
    screening = {}
    for name in context.CONTEXTS:
        selected = [r for r in rows if r['context']==name]
        screening[name] = dict(total=len(selected),
            exact_passes=sum(r['exact_invariance_passed'] for r in selected),
            stationary_accuracy_passes=sum(r['stationary']['accuracy_passed'] for r in selected),
            all_stationary_accurate=all(r['stationary']['accuracy_passed'] for r in selected))
    io.write(out/'COMPLETE.json',dict(passed=True,diagnostic_complete=True,targets=targets,rows=rows,
        screening=screening,sources=sources,runtime=run_environment,seconds=time.monotonic()-started,
        job=os.environ['SLURM_JOB_ID'],node=socket.gethostname(),posterior_qualified=False,
        training=False,abacus_inputs=False,
        limitation='Analytic Gaussian context comparison; all observations remain available; dense global covariance oracle.'))
    print('COMPLETE '+str(out/'COMPLETE.json'),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    run(parser.parse_args().output)
