"""Frozen matched neural CFM diagnostic with explicit finite-ensemble limits."""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import socket
import time
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import ndtr
import torch
import e2e_gaussian_3d_cfm as g
import e2e_gaussian_block_sampler as block
import e2e_true_coarse_fraction_saved as io

SEED = 61009492
UPDATES = 4096
BATCH = 8
DRAWS = 256
SAMPLE_BATCH = 32
PLAN = io.REPO/'docs/e2e_matched_gaussian_3d_protocol_20261009.md'


def sync(): torch.cuda.synchronize()


def sources():
    from workflows.sbi import e2e_coupled_benchmark_models as parent
    from workflows.sbi import e2e_direct_vdm as backbone
    return [io.record(p) for p in (__file__, g.__file__, g.context.__file__, block.__file__,
        parent.__file__, backbone.__file__, io.__file__, PLAN,
        io.REPO/'tests/test_e2e_gaussian_3d_cfm.py')]


def initialize():
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    return g.GaussianBackbone().cuda()


def generators():
    return (torch.Generator(device='cuda').manual_seed(SEED+1),
            torch.Generator(device='cuda').manual_seed(SEED+2))


def resource(model, n, arm):
    """Batch1 end-to-end update/draw cost; context counted once per solve."""
    data = g.JointGenerator(n); gen, bridge = generators()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    model.train()
    for step in range(2): g.train_step(model, optimizer, data, gen, arm, step, 1, bridge)
    sync(); torch.cuda.reset_peak_memory_stats(); started = time.monotonic()
    for step in range(3): g.train_step(model, optimizer, data, gen, arm, step, 1, bridge)
    sync(); train_seconds = (time.monotonic()-started)/3
    train_peak = torch.cuda.max_memory_allocated()
    del optimizer; model.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache()
    model.eval(); x, obs = data.batch(1, gen); width = n if arm == 'whole' else 8
    noise = torch.randn((1, 1, width, width, width), device='cuda', generator=bridge)
    g.draw(model, obs, x, (0,)*3, width, noise, 4)
    sync(); torch.cuda.reset_peak_memory_stats(); started = time.monotonic()
    for _ in range(3): g.draw(model, obs, x, (0,)*3, width, noise, 32)
    sync(); seconds = (time.monotonic()-started)/3
    return dict(n=n, arm=arm, batch=1, train_seconds=train_seconds, train_peak_bytes=train_peak,
        draw_or_update_seconds=seconds, sample_peak_bytes=torch.cuda.max_memory_allocated(),
        nfe=32, blocks=(n//8)**3 if arm == 'block' else 1,
        projected_sweep_seconds=seconds*((n//8)**3 if arm == 'block' else 1),
        accuracy_measured=False, trained_checkpoint=False)


def smoke(out, records):
    rows = []
    for n, arm in ((8, 'whole'), (16, 'whole'), (16, 'block')):
        model = initialize(); data = g.JointGenerator(n); gen, bridge = generators()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        for u in range(4): g.train_step(model, optimizer, data, gen, arm, u, BATCH, bridge)
        sync(); torch.cuda.reset_peak_memory_stats(); started = time.monotonic()
        for u in range(32): loss = g.train_step(model, optimizer, data, gen, arm, u, BATCH, bridge)
        sync(); seconds = (time.monotonic()-started)/32
        row = dict(n=n, arm=arm, seconds_per_update=seconds,
            projected_training_seconds=seconds*UPDATES, loss=loss,
            peak_bytes=torch.cuda.max_memory_allocated(), parameters=sum(p.numel() for p in model.parameters()))
        rows.append(row); print('SMOKE '+json.dumps(row), flush=True)
        del optimizer, model, data; gc.collect(); torch.cuda.empty_cache()
    model = initialize()
    costs = [resource(model, n, arm) for n, arm in ((8, 'whole'), (16, 'whole'), (16, 'block'))]
    # Actual batch32 throughput is measured, not inferred linearly from batch1.
    data = g.JointGenerator(16); gen, _ = generators(); x, obs = data.batch(SAMPLE_BATCH, gen)
    noise = torch.randn((SAMPLE_BATCH, 1, 8, 8, 8), device='cuda', generator=gen)
    g.draw(model, obs, x, (0,)*3, 8, noise, 4); sync(); started = time.monotonic()
    g.draw(model, obs, x, (0,)*3, 8, noise, 32); sync()
    block_solve_seconds = time.monotonic()-started
    # 16 primary sweeps +4 refined sweeps at double NFE +1 invariance sweep,
    # two observations, eight chain batches, eight blocks. Conservative 20% margin.
    estimated_evaluation = block_solve_seconds*(16+8+1)*2*(DRAWS//SAMPLE_BATCH)*8*1.2+300
    estimated_total = sum(r['projected_training_seconds'] for r in rows)+estimated_evaluation+300
    io.write(out/'SMOKE.json', dict(passed=estimated_total < 6000, rows=rows, resource=costs,
        block_solve_batch32_seconds=block_solve_seconds, estimated_evaluation_seconds=estimated_evaluation,
        estimated_total_seconds=estimated_total, sources=records))
    print('SMOKE TOTAL '+str(estimated_total), flush=True)


def train(n, arm, out, records):
    model = initialize(); data = g.JointGenerator(n); gen, bridge = generators()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    losses = []; summaries = []; sync(); started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for u in range(UPDATES):
        lr = 1e-5+.5*(1e-4-1e-5)*(1+math.cos(math.pi*u/UPDATES))
        for group in optimizer.param_groups: group['lr'] = lr
        losses.append(g.train_step(model, optimizer, data, gen, arm, u, BATCH, bridge))
        if (u+1) % 256 == 0:
            row = dict(n=n, arm=arm, update=u+1, loss=float(np.mean(losses[-256:])), seconds=time.monotonic()-started)
            summaries.append(row); print('TRAIN '+json.dumps(row), flush=True)
    sync(); seconds = time.monotonic()-started
    path = out/f'model_n{n}_{arm}.pt'
    with path.open('xb') as stream:
        torch.save(dict(model=model.state_dict(), n=n, arm=arm, updates=UPDATES, seed=SEED, sources=records), stream)
    io.write(out/f'train_n{n}_{arm}.json', dict(n=n, arm=arm, seed=SEED, updates=UPDATES, fields=UPDATES*BATCH,
        supervised_voxels=UPDATES*BATCH*(n**3 if arm == 'whole' else 8**3),
        seconds=seconds, peak_bytes=torch.cuda.max_memory_allocated(), losses=losses, summaries=summaries,
        checkpoint=io.record(path), sources=records))
    del optimizer; model.zero_grad(set_to_none=True); model.eval(); gc.collect(); torch.cuda.empty_cache()
    return model


def targets(n):
    prior = g.context.prior(n, 'long_range'); mask, std = g.context.observations(n, 1)
    observed = np.flatnonzero(mask); cross = prior[:, observed]
    obs_cov = prior[np.ix_(observed, observed)]+np.diag(std[observed]**2)
    factor = cho_factor(obs_cov); gain = cho_solve(factor, cross.T).T
    sigma = block.symmetric(prior-gain@cross.T); chol = np.linalg.cholesky(sigma)
    if n == 8:
        precision = cho_solve(cho_factor(prior), np.eye(n**3))+np.diag(mask/std**2)
        np.testing.assert_allclose(precision@sigma, np.eye(n**3), atol=2e-10)
    probes, names, _ = block.probe_bank(n, np.eye(n**3))
    variance = np.einsum('ij,ij->j', probes, sigma@probes)
    normalized = probes/np.sqrt(variance)
    target_correlation = normalized.T@sigma@normalized
    # FFT generation shares precisely the same law, with independent held-out seed.
    k = 2*np.sin(np.pi*np.fft.fftfreq(n)); lap = sum(v*v for v in np.meshgrid(k,k,k,indexing='ij'))
    power = (1+1.5**2*lap)**-2; power /= power.mean()
    cases = []
    for ci in range(2):
        rng = np.random.default_rng(61029500+100*n+ci)
        truth = (np.fft.ifftn(np.fft.fftn(rng.normal(size=(n,)*3))*np.sqrt(.2*power)).real
                 +np.sqrt(.8)*rng.normal()).ravel()
        y = mask*(truth+std*rng.normal(size=n**3)); mu = gain@y[observed]
        exact = rng.normal(size=(DRAWS,n**3))@chol.T+mu
        cases.append(dict(n=n, ci=ci, mu=mu, sigma=sigma, chol=chol, y=y, mask=mask, std=std,
            probes=normalized, names=names, correlation=target_correlation, exact=exact))
    return cases


def scores(projected, correlation):
    count = len(projected); mean = projected.mean(0); centered = projected-mean
    variance = np.var(projected, axis=0, ddof=1)
    mean_se = np.sqrt(variance/count)
    variance_se = np.sqrt(np.var(centered**2, axis=0, ddof=1)/count)
    mean_lower = np.maximum(abs(mean)-4*mean_se,0)
    variance_lower = np.maximum(abs(variance-1)-4*variance_se,0)
    quantiles = np.quantile(projected, (.05,.95), axis=0)
    coverage = ndtr(quantiles[1])-ndtr(quantiles[0])
    covariance = np.cov(projected, rowvar=False)
    # A single fixed probe outside tolerance suffices to reject the stronger gate.
    rejected = bool(np.max(mean_lower)>.01 or np.max(variance_lower)>.01)
    return dict(draws=count, mean=mean.tolist(), variance=variance.tolist(), mean_se=mean_se.tolist(),
        variance_se=variance_se.tolist(), covariance=covariance.tolist(),
        max_mean_error=float(np.max(abs(mean))), max_variance_error=float(np.max(abs(variance-1))),
        max_mean_error_lower4se=float(np.max(mean_lower)),
        max_variance_error_lower4se=float(np.max(variance_lower)),
        probe_covariance_relative=float(np.linalg.norm(covariance-correlation)/np.linalg.norm(correlation)),
        exact_probability_in_empirical90=coverage.tolist(),
        maximum_coverage_error=float(np.max(abs(coverage-.9))),
        accuracy_status='failed' if rejected else 'inconclusive', qualified=False,
        reason='Four-SE diagnostic rejection' if rejected else 'Finite ensemble cannot certify full joint tolerance')


def projected(state, case):
    return (state-case['mu'])@case['probes']


def observation_tensor(case, count):
    n = case['n']; raw = np.stack((case['y'],case['mask'],case['std'])).reshape(1,3,n,n,n)
    return torch.tensor(raw,device='cuda',dtype=torch.float32).expand(count,-1,-1,-1,-1)


@torch.no_grad()
def evaluate(model, arm, cases, out):
    n = cases[0]['n']; rows = []; samples = {}
    for case in cases:
        ci = case['ci']; oracle = projected(case['exact'], case)
        rows.append(dict(n=n, observation=ci, arm='oracle_iid', nfe=0, sweeps=0, seconds=0,
                         metrics=scores(oracle,case['correlation'])))
        samples[f'c{ci}_oracle'] = oracle
        configurations = [('whole',32,1),('whole',64,1)] if arm=='whole' else [('zero',32,16),('zero',64,4),('oracle_init',32,1)]
        outputs = {}
        for initialization,nfe,nsweeps in configurations:
            snapshots = (1,4,8,16) if nsweeps==16 else (nsweeps,)
            batches = {s:[] for s in snapshots}; paths = []; sync(); started=time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            for first in range(0,DRAWS,SAMPLE_BATCH):
                count = min(SAMPLE_BATCH,DRAWS-first); obs=observation_tensor(case,count)
                gen=torch.Generator(device='cuda').manual_seed(61039500+ci*10000+first)
                state=torch.zeros((count,1,n,n,n),device='cuda')
                if initialization=='oracle_init':
                    state.copy_(torch.tensor(case['exact'][first:first+count].reshape(count,1,n,n,n),device='cuda',dtype=torch.float32))
                trace=[]
                for s in range(1,nsweeps+1):
                    if arm=='whole':
                        noise=torch.randn(state.shape,device='cuda',generator=gen)
                        state=g.draw(model,obs,state,(0,)*3,n,noise,nfe)
                    else: state=g.sweep(model,obs,state,gen,nfe)
                    array=state.flatten(1).cpu().numpy().astype(np.float64)
                    values=projected(array,case);trace.append(values)
                    if s in snapshots: batches[s].append(values)
                paths.append(np.stack(trace))
            sync(); elapsed=time.monotonic()-started
            for s, chunks in batches.items():
                values=np.concatenate(chunks)
                key=f'c{ci}_{initialization}_nfe{nfe}_s{s}'; samples[key]=values;outputs[(initialization,nfe,s)]=values
                rows.append(dict(n=n,observation=ci,arm=arm,initialization=initialization,nfe=nfe,sweeps=s,
                    seconds_total_configuration=elapsed,peak_bytes=torch.cuda.max_memory_allocated(),
                    metrics=scores(values,case['correlation'])))
                print('EVAL '+json.dumps({k:v for k,v in rows[-1].items() if k!='metrics'})+' '+rows[-1]['metrics']['accuracy_status'],flush=True)
            if nsweeps==16:
                trace=np.concatenate(paths,axis=1);samples[f'c{ci}_chain_trace']=trace
                a=trace[8:-1];b=trace[9:];a=a-a.mean(1,keepdims=True);b=b-b.mean(1,keepdims=True)
                lag=(a*b).mean((0,1))/np.sqrt((a*a).mean((0,1))*(b*b).mean((0,1)))
                rows.append(dict(n=n,observation=ci,arm=arm,mixing_diagnostic=True,late_lag1=lag.tolist(),
                    ess_qualified=False,reason='Only16sweeps; no reliable stationary ESS'))
        if arm=='whole': a,b=outputs[('whole',32,1)],outputs[('whole',64,1)]
        else: a,b=outputs[('zero',32,4)],outputs[('zero',64,4)]
        rows.append(dict(n=n,observation=ci,arm=arm,integration_refinement=True,
            paired_probe_rms_change=float(np.sqrt(np.mean((a-b)**2))),
            maximum_probe_mean_change=float(np.max(abs((a-b).mean(0)))),
            maximum_probe_variance_change=float(np.max(abs(a.var(0)-b.var(0))))))
    path=out/f'projections_n{n}_{arm}.npz'
    with path.open('xb') as stream: np.savez_compressed(stream,**samples)
    io.write(out/f'evaluation_n{n}_{arm}.json',dict(rows=rows,n=n,arm=arm,probes=cases[0]['names'],
        target_correlation=cases[0]['correlation'].tolist(),projection_file=io.record(path),qualified=False))
    return rows


def run(args):
    io.compute_guard();torch.set_num_threads(4)
    if not torch.cuda.is_available():raise RuntimeError('Approved GPU required')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False);records=sources()
    io.write(out/'MANIFEST.json',dict(sources=records,job=os.environ['SLURM_JOB_ID'],node=socket.gethostname(),
        gpu=torch.cuda.get_device_name(),torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,
        updates=UPDATES,batch=BATCH,draws=DRAWS,seed=SEED,mode=args.mode))
    if args.mode=='smoke':smoke(out,records);return
    receipt=io.read(args.smoke)
    if not receipt['passed'] or receipt['sources']!=records:raise RuntimeError('Matching passed smoke required')
    started=time.monotonic();evaluations=[];costs=[]
    for n in (8,16):
        cases=targets(n)
        for arm in (('whole',) if n==8 else ('whole','block')):
            model=train(n,arm,out,records)
            evaluations.extend(evaluate(model,arm,cases,out))
            del model;gc.collect();torch.cuda.empty_cache()
    for n in (8,16,24,32,48):
        for arm in ('whole','block'):
            model=initialize();row=resource(model,n,arm);costs.append(row)
            print('RESOURCE '+json.dumps(row),flush=True)
            del model;gc.collect();torch.cuda.empty_cache()
    for record in records:
        if io.sha(record['path'])!=record['sha256']:raise AssertionError('Source changed during run')
    files=[io.record(p) for p in sorted(out.iterdir()) if p.is_file()]
    if sum(p['bytes'] for p in files)>2**30:raise RuntimeError('Bounded Home output limit exceeded')
    io.write(out/'COMPLETE.json',dict(engineering_complete=True,posterior_qualified=False,sources=records,
        files=files,resources=costs,evaluations=evaluations,seconds=time.monotonic()-started,
        single_block_control='n8 whole and block identical; paired training equality tested',
        time_to_qualified_draw=None,abacus_inputs=False,job=os.environ['SLURM_JOB_ID']))
    print('COMPLETE',out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('smoke','run'),required=True)
    p.add_argument('--output',required=True);p.add_argument('--smoke');run(p.parse_args())
