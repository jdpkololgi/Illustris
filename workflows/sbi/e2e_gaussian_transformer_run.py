"""Bounded transformer-only fits against the immutable whole-field baseline."""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import socket
import time
import numpy as np
import torch
import e2e_gaussian_3d_cfm as g
import e2e_gaussian_3d_run as baseline
import e2e_gaussian_transformer as transformer
import e2e_true_coarse_fraction_saved as io

PLAN = io.REPO/'docs/e2e_gaussian_transformer_protocol_20261009.md'
BASE = Path('/global/u2/d/dkololgi/diagnostics/matched_gaussian_3d_20261009_v1')


def sources():
    return baseline.sources()+[io.record(p) for p in (
        __file__, transformer.__file__, PLAN,
        io.REPO/'workflows/sbi/e2e_gaussian_transformer_validate.py',
        io.REPO/'workflows/sbi/e2e_gaussian_transformer_step.sh',
        io.REPO/'tests/test_e2e_gaussian_transformer.py')]


def initialize():
    torch.manual_seed(baseline.SEED)
    torch.cuda.manual_seed_all(baseline.SEED)
    return transformer.GaussianTransformer().cuda()


def check_baseline():
    receipt = io.read(BASE/'run/COMPLETE.json')
    audit = io.read(BASE/'VALIDATION.json')
    if not receipt['engineering_complete'] or not audit['passed']:
        raise RuntimeError('Verified baseline required')
    for item in receipt['sources']+receipt['files']+audit['sources']:
        io.verify(item)
    return [io.record(BASE/'run/COMPLETE.json'), io.record(BASE/'VALIDATION.json')]+[
        io.record(BASE/'run'/f'{prefix}_n{n}_whole.{extension}')
        for n in (8, 16) for prefix, extension in (
            ('train', 'json'), ('evaluation', 'json'), ('projections', 'npz'))]


def smoke(out, records):
    rows = []
    for n in (8, 16):
        model = initialize()
        data = g.JointGenerator(n)
        data_gen, bridge_gen = baseline.generators()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        for update in range(4): g.train_step(model, opt, data, data_gen, 'whole', update, 8, bridge_gen)
        baseline.sync(); torch.cuda.reset_peak_memory_stats(); started = time.monotonic()
        for update in range(32): loss = g.train_step(model, opt, data, data_gen, 'whole', update, 8, bridge_gen)
        baseline.sync(); seconds = (time.monotonic()-started)/32
        train_peak = torch.cuda.max_memory_allocated()
        del opt
        model.zero_grad(set_to_none=True); model.eval(); gc.collect(); torch.cuda.empty_cache()
        x, obs = data.batch(baseline.SAMPLE_BATCH, data_gen)
        noise = torch.randn(x.shape, device='cuda', generator=bridge_gen)
        g.draw(model, obs, x, (0, 0, 0), n, noise, 4)
        baseline.sync(); started = time.monotonic()
        g.draw(model, obs, x, (0, 0, 0), n, noise, 32)
        baseline.sync(); sampling_seconds = time.monotonic()-started
        row = dict(n=n, seconds_per_update=seconds, train_peak_bytes=train_peak, loss=loss,
                   batch32_nfe32_seconds=sampling_seconds,
                   parameters=sum(p.numel() for p in model.parameters()))
        rows.append(row); print('SMOKE '+json.dumps(row), flush=True)
        del model, data, x, obs, noise
        gc.collect(); torch.cuda.empty_cache()
    # Two observations, eight batches, NFE32+64; allowance for dense reference,
    # independent audit, tests and timings. No adaptive smaller scientific run.
    projected = 1.3*sum(r['seconds_per_update']*4096+r['batch32_nfe32_seconds']*48 for r in rows)+420
    passed = bool(projected < 2400 and max(r['train_peak_bytes'] for r in rows) < 30*2**30)
    io.write(out/'SMOKE.json', dict(passed=passed, rows=rows, estimated_seconds=projected, sources=records))
    print('SMOKE_GATE', passed, projected, flush=True)
    if not passed: raise RuntimeError('Fixed experiment does not fit bounded allocation')


def train(n, out, records):
    model = initialize()
    data = g.JointGenerator(n)
    data_gen, bridge_gen = baseline.generators()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    losses = []; summaries = []
    baseline.sync(); torch.cuda.reset_peak_memory_stats(); started = time.monotonic()
    for update in range(baseline.UPDATES):
        lr = 1e-5+.5*(1e-4-1e-5)*(1+math.cos(math.pi*update/baseline.UPDATES))
        for group in opt.param_groups: group['lr'] = lr
        losses.append(g.train_step(model, opt, data, data_gen, 'whole', update, baseline.BATCH, bridge_gen))
        if (update+1) % 256 == 0:
            row = dict(n=n, update=update+1, loss=float(np.mean(losses[-256:])), seconds=time.monotonic()-started)
            summaries.append(row); print('TRAIN '+json.dumps(row), flush=True)
    baseline.sync(); seconds = time.monotonic()-started
    path = out/f'model_n{n}_whole.pt'
    with path.open('xb') as stream:
        torch.save(dict(model=model.state_dict(), n=n, architecture=model.architecture,
                        seed=baseline.SEED, updates=baseline.UPDATES, sources=records), stream)
    io.write(out/f'train_n{n}_whole.json', dict(n=n, architecture=model.architecture,
        parameters=sum(p.numel() for p in model.parameters()), seed=baseline.SEED,
        updates=baseline.UPDATES, fields=baseline.UPDATES*baseline.BATCH,
        supervised_voxels=baseline.UPDATES*baseline.BATCH*n**3,
        seconds=seconds, peak_bytes=torch.cuda.max_memory_allocated(), losses=losses,
        summaries=summaries, checkpoint=io.record(path), sources=records))
    del opt
    model.zero_grad(set_to_none=True); model.eval(); gc.collect(); torch.cuda.empty_cache()
    return model


@torch.no_grad()
def costs():
    rows = []
    for n in (8, 16):
        for architecture in ('unet', 'transformer'):
            gc.collect(); torch.cuda.empty_cache()
            model = (baseline.initialize() if architecture == 'unet' else initialize()).eval()
            data = g.JointGenerator(n); gen, bridge = baseline.generators()
            x, obs = data.batch(1, gen)
            noise = torch.randn(x.shape, device='cuda', generator=bridge)
            g.draw(model, obs, x, (0, 0, 0), n, noise, 32)
            baseline.sync(); torch.cuda.reset_peak_memory_stats(); timings = []
            for _ in range(3):
                started = time.monotonic()
                g.draw(model, obs, x, (0, 0, 0), n, noise, 32)
                baseline.sync(); timings.append(time.monotonic()-started)
            row = dict(n=n, architecture=architecture, batch=1, nfe=32, repetitions=3,
                seconds=timings, mean_seconds=float(np.mean(timings)),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                parameters=sum(p.numel() for p in model.parameters()), trained_checkpoint=False,
                context_included=True, time_to_qualified_draw=None)
            rows.append(row); print('COST '+json.dumps(row), flush=True)
            del model, data, x, obs, noise
    gc.collect(); torch.cuda.empty_cache()
    return rows


def main(args):
    io.compute_guard(); torch.set_num_threads(4)
    if not torch.cuda.is_available(): raise RuntimeError('Approved GPU required')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    records = sources(); comparison = check_baseline()
    io.write(out/'MANIFEST.json', dict(sources=records, baseline=comparison,
        job=os.environ['SLURM_JOB_ID'], node=socket.gethostname(), gpu=torch.cuda.get_device_name(),
        torch=torch.__version__, numpy=np.__version__, cuda=torch.version.cuda, mode=args.mode,
        updates=baseline.UPDATES, batch=baseline.BATCH, draws=baseline.DRAWS, seed=baseline.SEED))
    if args.mode == 'smoke':
        smoke(out, records)
        return
    receipt = io.read(args.smoke)
    if not receipt['passed'] or receipt['sources'] != records:
        raise RuntimeError('Matching passed smoke required')
    started = time.monotonic()
    for n in (8, 16):
        model = train(n, out, records)
        cases = baseline.targets(n)
        baseline.evaluate(model, 'whole', cases, out)
        del model, cases
        gc.collect(); torch.cuda.empty_cache()
    resources = costs()
    io.write(out/'cost.json', dict(rows=resources, sources=records, scope='Untrained inference-only cost'))
    for item in records+comparison: io.verify(item)
    files = [io.record(p) for p in sorted(out.iterdir()) if p.is_file()]
    if sum(f['bytes'] for f in files) > 2**30: raise RuntimeError('Output budget exceeded')
    io.write(out/'COMPLETE.json', dict(engineering_complete=True, posterior_qualified=False,
        seconds=time.monotonic()-started, sources=records, baseline=comparison, files=files,
        resources=resources, time_to_qualified_draw=None,
        note='Legacy four-SE screen retained by shared evaluator; predeclared Gaussian-null audit controls interpretation'))
    print('RUN_COMPLETE', time.monotonic()-started, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('smoke', 'run'), required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--smoke')
    main(parser.parse_args())
