"""Frozen one-seed exterior-conditioned Gaussian CFM training and exact QA."""
import argparse
import json
import math
import os
from pathlib import Path
import socket
import time

import numpy as np
import torch

import e2e_exterior_cfm as cfm
import e2e_gaussian_block_sampler as block
import e2e_restricted_context as context
import e2e_restricted_context_diagnostic as oracle
import e2e_true_coarse_fraction_saved as io

SEED = 61009431
UPDATES = 20000
CHECKPOINTS = (0,5000,10000,20000)
NFES = (128,256,512)


def evaluate(model,data,cases,update,out,sources):
    rows = []; components = []; endpoint_comparisons = []
    probes,names,_ = block.probe_bank(4,np.eye(64))
    previous = None
    for nfe in NFES:
        endpoint = cfm.heun_affine(cfm.learned_coefficients(model,nfe),data.width)
        component = cfm.conditional_metrics(data,endpoint)
        components.append(dict(nfe=nfe,blocks=component,
            all_component_accuracy_passed=all(r['component_accuracy_passed'] for r in component)))
        if previous is not None:
            covariance = endpoint[0]@endpoint[0].transpose(0,2,1)
            old_covariance = previous[0]@previous[0].transpose(0,2,1)
            endpoint_comparisons.append(dict(nfe=nfe,
                covariance_relative_change=float(np.linalg.norm(covariance-old_covariance)/np.linalg.norm(covariance)),
                mean_response_relative_change=float(np.linalg.norm(endpoint[1]-previous[1])/max(1.,np.linalg.norm(endpoint[1])))))
        previous = endpoint
        for ci,case in enumerate(cases):
            local = cfm.kernels(data,endpoint,case)
            local_affine = block.transition(local)
            modes = block.orthonormal_span(block.global_modes(4))
            global_update = block.ConditionalUpdate.make(case['precision'],case['mu'],modes)
            for arm in ('learned_local','learned_local_oracle_global'):
                affine = local_affine if arm=='learned_local' else global_update.compose(*local_affine)
                result,mean,covariance = context.stationary(affine,case,probes)
                inv = block.invariance(affine,case['mu'],case['sigma'])
                tau = block.autocorrelation(affine,covariance,probes) if result['stable'] else None
                curve = block.convergence(affine,case['mu'],case['sigma'],probes) if result['accuracy_passed'] else []
                rows.append(dict(observation_case=ci,nfe=nfe,arm=arm,invariance=inv,
                    stationary=result,stationary_tau=tau.tolist() if tau is not None else None,
                    first_target_passing_sweeps=next((r['sweeps'] for r in curve if r['passed']),None),curve=curve))
    actual = cfm.actual_model_check(model,data,128)
    refined = endpoint_comparisons[-1]['covariance_relative_change']<=.001 and endpoint_comparisons[-1]['mean_response_relative_change']<=.001
    # The pilot's non-oracle learned-local result is the primary claim.
    finest = [r for r in rows if r['nfe']==512 and r['arm']=='learned_local']
    result = dict(update=update,rows=rows,conditional_components=components,
        endpoint_refinement=endpoint_comparisons,refinement_passed=refined,
        actual_model_check=actual,probe_names=names,
        learned_local_stationary_passed=all(r['stationary']['accuracy_passed'] for r in finest),
        learned_components_passed=components[-1]['all_component_accuracy_passed'],
        pilot_accuracy_passed=refined and all(r['stationary']['accuracy_passed'] for r in finest) and components[-1]['all_component_accuracy_passed'],
        sources=sources,heldout_observation_seeds=list(range(61019500,61019508)))
    io.write(out/f'evaluation_{update}.json',result)
    summary = dict(update=update,passed=result['pilot_accuracy_passed'],refinement=refined,
        component_mean=max(r['mean_whitened_rms'] for r in component),
        component_covariance=max(r['worst_relative_covariance_error'] for r in component),
        worst_stationary_covariance=max((r['stationary'].get('worst_relative_covariance_error',float('inf')) for r in finest)),
        worst_stationary_mean=max((r['stationary'].get('whitened_mean_error',float('inf')) for r in finest)))
    print('EVALUATED '+json.dumps(summary),flush=True)
    return result


def controls(data,cases,out,sources):
    rows = []
    probes,_,_ = block.probe_bank(4,np.eye(64))
    exact_draw = (np.linalg.cholesky(data.exact_covariance),
        np.concatenate([data.exact_response,np.zeros((len(data.patches),data.width,1))],axis=-1))
    for name,nfe in [('direct',0)]+[('cfm',n) for n in NFES]:
        endpoint = exact_draw if name=='direct' else cfm.heun_affine(cfm.exact_coefficients(data,nfe),data.width)
        components = cfm.conditional_metrics(data,endpoint)
        for ci,case in enumerate(cases):
            updates = cfm.kernels(data,endpoint,case)
            affine = block.transition(updates)
            result,_,_ = context.stationary(affine,case,probes)
            if name=='direct' and block.invariance(affine,case['mu'],case['sigma'])['covariance_relative'] > 1e-8:
                raise AssertionError('Exact conditional-draw positive control failed')
            if nfe==512 and (not result['accuracy_passed'] or max(r['worst_relative_covariance_error'] for r in components)>.001):
                raise AssertionError('Exact conditional-CFM positive control failed')
            rows.append(dict(method=name,nfe=nfe,observation_case=ci,stationary=result,components=components))
    io.write(out/'CONTROLS.json',dict(passed=True,rows=rows,sources=sources))


def save_checkpoint(model,optimizer,generator,update,out,sources):
    destination = out/f'model_{update}.pt'
    with destination.open('xb') as stream:
        torch.save(dict(update=update,model=model.state_dict(),optimizer=optimizer.state_dict(),
            generator=generator.get_state(),torch_cpu_rng=torch.get_rng_state(),
            torch_cuda_rng=torch.cuda.get_rng_state_all(),sources=sources),stream)
    return io.record(destination)


def run(args):
    io.compute_guard()
    if not torch.cuda.is_available():
        raise RuntimeError('This pilot requires the approved single GPU allocation')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = Path(args.output); out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):
        raise RuntimeError('Fresh output directory required; no implicit resume or overwrite')
    paths = [__file__,cfm.__file__,context.__file__,block.__file__,oracle.__file__,
        context.reference.__file__,io.__file__,oracle.PLAN,
        io.REPO/'tests/test_e2e_exterior_cfm.py',
        io.REPO/'workflows/sbi/e2e_exterior_cfm_step.sh']
    sources = [io.record(p) for p in paths]
    run_environment = oracle.runtime()
    run_environment.update(gpu=torch.cuda.get_device_name(),cuda=torch.version.cuda)
    manifest = dict(sources=sources,runtime=run_environment,mode=args.mode,
        job=os.environ['SLURM_JOB_ID'],node=socket.gethostname(),seed=SEED,
        updates=UPDATES,batch=2048,time_groups=32,checkpoints=CHECKPOINTS,nfes=NFES,
        train_conditions='Whitened full exterior and fixed-mask noisy observations; no posterior objects',
        architecture='128x128 time MLP, eight affine-in-state/context block heads')
    io.write(out/'MANIFEST.json',manifest)
    data = cfm.GaussianTrainingData()
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    model = cfm.TimeAffineCFM(len(data.patches),data.width,data.context_dimensions).cuda()
    optimizer = torch.optim.Adam(model.parameters(),lr=.0003)
    generator = torch.Generator(device='cuda').manual_seed(SEED+1)
    if args.mode=='smoke':
        started = time.monotonic()
        for _ in range(64):
            loss = cfm.train_step(model,optimizer,data,generator)
        torch.cuda.synchronize()
        seconds_per_update = (time.monotonic()-started)/64
        actual = cfm.actual_model_check(model,data,32)
        io.write(out/'SMOKE.json',dict(passed=True,seconds_per_update=seconds_per_update,
            estimated_training_seconds=seconds_per_update*UPDATES,
            parameters=sum(p.numel() for p in model.parameters()),loss=loss,
            peak_gpu_bytes=torch.cuda.max_memory_allocated(),actual_model_check=actual,
            sources=sources,runtime=run_environment))
        print('SMOKE '+json.dumps(dict(seconds_per_update=seconds_per_update,
            estimated_training_seconds=seconds_per_update*UPDATES,loss=loss)),flush=True)
        return
    prerequisite = io.read(args.oracle_receipt)
    if not prerequisite['diagnostic_complete'] or not prerequisite['screening']['full']['all_stationary_accurate']:
        raise RuntimeError('Passing full-context oracle prerequisite required')
    smoke = io.read(args.smoke_receipt)
    if not smoke['passed'] or smoke['sources'] != sources:
        raise RuntimeError('Matching passed smoke required')
    if smoke['estimated_training_seconds'] > 2400:
        raise RuntimeError('Pilot does not fit the bounded allocation; no automatic budget extension')
    io.write(out/'PREREQUISITES.json',dict(oracle=io.record(args.oracle_receipt),smoke=io.record(args.smoke_receipt)))
    cases = data.heldout()
    controls(data,cases,out,sources)
    records = []; summaries = []; losses = []; started = time.monotonic()
    records.append(save_checkpoint(model,optimizer,generator,0,out,sources))
    summaries.append(evaluate(model,data,cases,0,out,sources))
    for update in range(1,UPDATES+1):
        if time.monotonic()-started > 3000:
            raise TimeoutError('Bounded pilot time limit; no automatic continuation')
        lr = .00003+.5*(.0003-.00003)*(1+math.cos(math.pi*(update-1)/UPDATES))
        for group in optimizer.param_groups:
            group['lr'] = lr
        loss = cfm.train_step(model,optimizer,data,generator)
        losses.append(loss)
        if update%500==0:
            print('TRAIN '+json.dumps(dict(update=update,loss_mean_last500=float(np.mean(losses[-500:])),
                learning_rate=lr,seconds=time.monotonic()-started)),flush=True)
        if update in CHECKPOINTS:
            records.append(save_checkpoint(model,optimizer,generator,update,out,sources))
            summaries.append(evaluate(model,data,cases,update,out,sources))
    for source in sources:
        if io.sha(source['path']) != source['sha256']:
            raise AssertionError('Source changed during pilot')
    final = summaries[-1]
    io.write(out/'COMPLETE.json',dict(passed=True,engineering_complete=True,
        pilot_accuracy_passed=final['pilot_accuracy_passed'],
        learned_local_stationary_passed=final['learned_local_stationary_passed'],
        learned_components_passed=final['learned_components_passed'],refinement_passed=final['refinement_passed'],
        updates=UPDATES,seed=SEED,checkpoints=records,losses=losses,sources=sources,runtime=run_environment,
        evaluations=[io.record(out/f'evaluation_{u}.json') for u in CHECKPOINTS],
        seconds=time.monotonic()-started,job=os.environ['SLURM_JOB_ID'],node=socket.gethostname(),
        posterior_qualified=False,abacus_inputs=False,
        limitation='One structured Gaussian CFM fit, fixed prior and mask; not the existing 3D model or a fully learned global update.'))
    print('COMPLETE '+str(out/'COMPLETE.json'),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',choices=('smoke','train'),required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--oracle-receipt')
    parser.add_argument('--smoke-receipt')
    run(parser.parse_args())
