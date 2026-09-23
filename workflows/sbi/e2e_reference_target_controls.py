"""Matched Gaussian CFM controls; exact posterior access is explicitly privileged."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import socket
import time

import numpy as np
import torch
from torch import nn

from workflows.sbi.e2e_conditional_reference import Data, atomic_json, atomic_checkpoint, evaluate
from workflows.sbi.e2e_conditional_reference_continue import restore_training_state, digest
from workflows.sbi.e2e_conditional_reference_math import null_thresholds
from workflows.sbi.e2e_direct_vdm import ConditionalVDM, LinearSchedule


class Teacher(nn.Module):
    def __init__(self, data):
        super().__init__()
        device = data.chol.device
        tensor = lambda a: torch.as_tensor(np.asarray(a), device=device, dtype=torch.float32)
        cases = data.cases[:2]
        self.register_buffer('vectors', tensor([c['vectors'] for c in cases]))
        self.register_buffer('values', tensor([c['values'] for c in cases]))
        self.register_buffer('gain', tensor([c['sigma'] * (c['mask']/c['std']**2)[None, :] for c in cases]))
        self.register_buffer('masks', data.mask.clone())

    def which(self, condition):
        mask = condition[:, 1].flatten(1)
        matches = (mask[:, None, :] == self.masks[None]).all(-1)
        if not torch.all(matches.sum(1) == 1):
            raise ValueError('unrecognized observation operator')
        return matches.to(torch.int64).argmax(1)

    def moments(self, condition):
        which = self.which(condition)
        y = condition[:, 0].flatten(1)
        mu = torch.empty_like(y)
        for k in range(2):
            idx = which == k
            mu[idx] = y[idx] @ self.gain[k].T
        return mu, which

    @torch.no_grad()
    def forward(self, z, t, condition):
        mu, which = self.moments(condition)
        return affine_velocity(z, t, mu, which, self.vectors, self.values)


def affine_velocity(z, t, mu, which, vectors, values):
    flat = z.flatten(1)
    result = torch.empty_like(flat)
    for k in range(2):
        idx = which == k
        tt = t[idx, None]
        lam = values[k]
        a = (tt*lam-(1-tt))/(tt.square()*lam+(1-tt).square())
        residual = (flat[idx]-tt*mu[idx]) @ vectors[k]
        result[idx] = mu[idx] + (residual*a) @ vectors[k].T
    return result.reshape_as(z)


class LearnedAffine(nn.Module):
    """Known eigenbasis, learned eigenvalues AND learned mean map; not an oracle."""
    def __init__(self, teacher, fixed=None):
        super().__init__()
        self.schedule = LinearSchedule(learned=False)
        self.fixed = fixed
        self.register_buffer('vectors', teacher.vectors.clone())
        self.register_buffer('masks', teacher.masks.clone())
        d = self.vectors.shape[-1]
        self.log_values = nn.Parameter(torch.zeros(2, d, device=self.vectors.device))
        if fixed is None:
            self.mean_map = nn.Parameter(torch.zeros(2, d, d, device=self.vectors.device))
        else:
            self.mean = nn.Parameter(torch.zeros(d, device=self.vectors.device))

    def forward(self, z, gamma, condition):
        t = (gamma-self.schedule.low)/self.schedule.slope.abs()
        masks = condition[:, 1].flatten(1)
        which = (masks[:, None] == self.masks[None]).all(-1).long().argmax(1)
        if self.fixed is not None:
            mu = self.mean.expand(len(z), -1)
        else:
            y = condition[:, 0].flatten(1)
            mu = torch.empty_like(y)
            for k in range(2):
                idx = which == k
                mu[idx] = y[idx] @ self.mean_map[k].T
        return affine_velocity(z, t, mu, which, self.vectors, self.log_values.exp())


def bridge(x, generator):
    t = (torch.arange(len(x), device=x.device)+torch.rand((), device=x.device, generator=generator))/len(x)
    noise = torch.randn(x.shape, device=x.device, generator=generator)
    tt = t[:, None, None, None, None]
    return (1-tt)*noise+tt*x, t, x-noise


def step(model, optimizer, teacher, x, condition, generator, exact):
    optimizer.zero_grad(set_to_none=True)
    z, t, stochastic = bridge(x, generator)
    target = teacher(z, t, condition) if exact else stochastic
    prediction = model(z, model.schedule(t), condition)
    loss = (prediction-target).square().mean()
    if not torch.isfinite(loss):
        raise FloatingPointError('nonfinite loss')
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def learning_rate(branch, update, total):
    if branch == 'decay':
        return 3e-6 + (3e-4-3e-6)*.5*(1+math.cos(math.pi*update/total))
    return 3e-4


def items():
    result = []
    for seed in [17, 29]:
        for fixed in [None, 0, 1]:
            mode = 'amortised' if fixed is None else f'fixed{fixed}'
            for branch in ['baseline', 'decay', 'teacher', 'affine_stochastic', 'affine_teacher']:
                result.append(dict(seed=seed, fixed=fixed, branch=branch,
                                   name=f'{branch}_seed{seed}_{mode}', parent=f'cfm_seed{seed}_{mode}'))
    return result


def prepare(args):
    root = Path(args.output).resolve()
    repo = Path(__file__).resolve().parents[2]
    parent = Path(args.parent).resolve()
    if root.exists():
        raise ValueError('new experiment requires a fresh output directory')
    cfg = json.loads((repo/'configs/e2e_conditional_reference_v1.json').read_text())
    names = ['e2e_reference_target_controls.py', 'e2e_conditional_reference.py',
             'e2e_conditional_reference_math.py', 'e2e_conditional_reference_continue.py', 'e2e_direct_vdm.py']
    files = [repo/'workflows/sbi'/n for n in names]
    files += [repo/'configs/e2e_conditional_reference_v1.json']
    files += [repo/'tests/test_e2e_reference_target_controls.py']
    sources = {str(p.relative_to(repo)): digest(p) for p in files}
    parents = {}
    parent_manifest = json.loads((parent/'manifest.json').read_text())
    for item in items():
        path = parent/'results'/item['parent']/'checkpoint_65536.pt'
        parents[item['parent']] = digest(path)
    root.mkdir(parents=True)
    for p in files:
        dest = root/'source'/p.relative_to(repo)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
    atomic_json(root/'manifest.json', dict(config=cfg, sources=sources, parent=str(parent),
        parent_sources=parent_manifest['sources'], parents=parents, items=items(), updates=16384,
        checkpoints=[8192, 16384], workers=4, created=time.time()))


@torch.no_grad()
def excess_risk(model, teacher, data, fixed, seed):
    gen = torch.Generator(device='cuda').manual_seed(700000+seed)
    mode = model.training
    model.eval()
    values = []
    for _ in range(128):
        x, c = data.batch(32, fixed, gen)
        z, t, _ = bridge(x, gen)
        values.append(float((model(z, model.schedule(t), c)-teacher(z, t, c)).square().mean()))
    model.train(mode)
    return dict(exact_target_mse=float(np.mean(values)), batch_standard_error=float(np.std(values, ddof=1)/np.sqrt(len(values))),
                examples=4096, diagnostic_seed=700000+seed)


def worker(args):
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('GPU Slurm allocation required')
    torch.set_num_threads(4)
    root = Path(args.output)
    manifest = json.loads((root/'manifest.json').read_text())
    for relative, expected in manifest['sources'].items():
        if digest(root/'source'/relative) != expected:
            raise ValueError('snapshot changed')
    rank = int(os.environ.get('SLURM_PROCID', '0'))
    cfg = manifest['config']
    data = Data(cfg, torch.device('cuda'))
    teacher = Teacher(data)
    precise = copy.copy(data)
    precise.nulls = [null_thresholds(c, data.q, 2048, cfg['null_repeats']) for c in data.cases]
    started = time.monotonic()
    def guard():
        if time.monotonic()-started > 6800:
            raise TimeoutError('checkpointed development run time limit')
    owned = manifest['items'][rank::manifest['workers']]
    if args.smoke:
        owned = [manifest['items'][0], manifest['items'][2], manifest['items'][3], manifest['items'][4]]
    timings = []
    for item in owned:
        guard()
        branch, seed, fixed = item['branch'], item['seed'], item['fixed']
        out = root/'results'/item['name']
        if not args.smoke:
            out.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(seed)
        affine = branch.startswith('affine')
        model = (LearnedAffine(teacher, fixed) if affine else ConditionalVDM(3, 8, 2, False)).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        generator = torch.Generator(device='cuda').manual_seed(seed+3000)
        first = 0
        if not affine:
            path = Path(manifest['parent'])/'results'/item['parent']/'checkpoint_65536.pt'
            if digest(path) != manifest['parents'][item['parent']]:
                raise ValueError('parent changed')
            saved = torch.load(path, map_location='cuda', weights_only=False)
            if saved['update'] != 65536 or saved['sources'] != manifest['parent_sources']:
                raise ValueError('parent ancestry mismatch')
            restore_training_state(model, optimizer, generator, saved)
        latest = out/'latest.pt'
        if not args.smoke and latest.exists():
            saved = torch.load(latest, map_location='cuda', weights_only=False)
            if saved['sources'] != manifest['sources'] or saved['item'] != item:
                raise ValueError('resume provenance mismatch')
            restore_training_state(model, optimizer, generator, saved)
            first = saved['update']
        tic = time.monotonic()
        losses = []
        total = 64 if args.smoke else manifest['updates']
        for update in range(first+1, total+1):
            guard()
            for group in optimizer.param_groups:
                group['lr'] = learning_rate(branch, update, manifest['updates'])
            x, c = data.batch(cfg['batch'], fixed, generator)
            losses.append(step(model, optimizer, teacher, x, c, generator, branch in ['teacher', 'affine_teacher']))
            if not args.smoke and update % 1024 == 0:
                state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                    generator=generator.get_state(), cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                    update=update, sources=manifest['sources'], item=item)
                atomic_checkpoint(latest, state)
                with (out/'learning.jsonl').open('a') as f:
                    f.write(json.dumps(dict(update=update, loss=float(np.mean(losses)), lr=optimizer.param_groups[0]['lr']))+'\n')
                print('TRAIN', rank, item['name'], update, float(np.mean(losses)), flush=True)
                losses = []
            if not args.smoke and update in manifest['checkpoints']:
                atomic_checkpoint(out/f'checkpoint_{update}.pt', state)
                cases = range(cfg['cases']) if fixed is None else [fixed]
                evaluate(model, 'cfm', data, cases, cfg|{'nfe':[128]}, seed, update, out, guard)
        if args.smoke:
            torch.cuda.synchronize()
            timings.append(dict(branch=branch, seconds_per_update=(time.monotonic()-tic)/total, loss=losses[-1]))
            continue
        final = out/'precision'
        final.mkdir(exist_ok=True)
        cases = range(cfg['cases']) if fixed is None else [fixed]
        evaluate(model, 'cfm', precise, cases, cfg|{'draws':2048, 'nfe':[128,256]}, seed, total, final, guard)
        atomic_json(final/'nulls.json', precise.nulls)
        atomic_json(final/'risk.json', excess_risk(model, teacher, data, fixed, seed))
    atomic_json(root/(f'SMOKE.json' if args.smoke else f'worker_{rank}_COMPLETE.json'),
        dict(seconds=time.monotonic()-started, job=os.environ['SLURM_JOB_ID'], node=socket.gethostname(),
             sources=manifest['sources'], items=owned, timings=timings, peak_gpu_bytes=torch.cuda.max_memory_allocated()))


def collect(args):
    root = Path(args.output)
    m = json.loads((root/'manifest.json').read_text())
    workers = [json.loads((root/f'worker_{r}_COMPLETE.json').read_text()) for r in range(m['workers'])]
    if any(w['sources'] != m['sources'] for w in workers):
        raise ValueError('worker source mismatch')
    rows = []
    for item in m['items']:
        base = root/'results'/item['name']/'precision'
        for c in (range(4) if item['fixed'] is None else [item['fixed']]):
            for nfe in [128,256]:
                r = json.loads((base/f'evaluation_16384_{c}_{nfe}.json').read_text())
                arr = np.load(base/f'draws_16384_{c}_{nfe}.npy', mmap_mode='r')
                if arr.shape != (2048,512) or not np.isfinite(arr).all():
                    raise ValueError('bad draw artifact')
                if (r['case'],r['nfe'],r['seed'],r['update']) != (c,nfe,item['seed'],16384):
                    raise ValueError('evaluation identity mismatch')
                rows.append(r | item | {'power_pass':all(.9 <= p <= 1.1 for p in r['power_ratio']),
                    'risk':json.loads((base/'risk.json').read_text())})
    atomic_json(root/'COMPLETE.json', dict(rows=rows, workers=workers, sources=m['sources']))
    for branch in ['baseline','decay','teacher','affine_stochastic','affine_teacher']:
        for mode in ['fixed','amortised']:
            subset = [r for r in rows if r['branch']==branch and r['nfe']==256 and (r['fixed'] is None)==(mode=='amortised')]
            print(branch, mode, len(subset), {k:round(float(np.mean([r[k] for r in subset])),5) for k in ['mean_rms','covariance_relative','variance_ratio','octant_coverage']},
                  'high_k', round(float(np.mean([r['power_ratio'][-1] for r in subset])),5),
                  'joint_pass',sum(r['passed'] and r['power_pass'] for r in subset), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['prepare','worker','collect'])
    p.add_argument('--output', required=True)
    p.add_argument('--parent')
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    globals()[args.command](args)
