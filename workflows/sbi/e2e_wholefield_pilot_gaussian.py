"""Small four-arm Gaussian diagnostic, not a statistical permission gate."""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from workflows.sbi import e2e_coupled_contract as c
import e2e_gaussian_3d_cfm as g
import e2e_gaussian_3d_run as reference
import e2e_gaussian_transformer as transformer

ROOT = Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/wholefield_pilot_20261013_v1/gaussian')


def endpoint_velocity(endpoint, state, t, epsilon=1/1024):
    denominator = (1-t).clamp_min(epsilon).reshape(-1, 1, 1, 1, 1)
    return (endpoint-state)/denominator


class Parameterized(nn.Module):
    def __init__(self, architecture, prediction):
        super().__init__()
        if architecture not in ('unet', 'transformer') or prediction not in ('v', 'x'):
            raise ValueError('Unknown candidate')
        self.network = g.GaussianBackbone() if architecture == 'unet' else transformer.GaussianTransformer()
        self.prediction = prediction

    def prepare(self, *args):
        return self.network.prepare(*args)

    def velocity(self, z, t, prepared):
        result = self.network.velocity(z, t, prepared)
        return result if self.prediction == 'v' else endpoint_velocity(result, z, t)


@torch.no_grad()
def evaluate(model, cases, path, seed, draws=128):
    rows = []; saved = {}
    model.eval()
    for case in cases:
        ci = case['ci']; noise_gen = torch.Generator(device='cuda').manual_seed(2026101300+ci)
        noise = torch.randn((draws, 1, 16, 16, 16), device='cuda', generator=noise_gen)
        projections = {}
        for nfe in (64, 128):
            values = []
            for first in range(0, draws, 16):
                z = noise[first:first+16]
                obs = reference.observation_tensor(case, len(z))
                result = g.draw(model, obs, torch.zeros_like(z), (0, 0, 0), 16, z, nfe)
                values.append(reference.projected(result[:, 0].cpu().numpy().reshape(len(z), -1), case))
            q = np.concatenate(values); mean = q.mean(0); covariance = np.cov(q, rowvar=False)
            metrics = dict(mean_error=float(np.max(abs(mean))),
                           covariance_relative=float(np.linalg.norm(covariance-case['correlation'])/np.linalg.norm(case['correlation'])),
                           variance_min=float(np.diag(covariance).min()), variance_max=float(np.diag(covariance).max()))
            rows.append(dict(observation=ci, nfe=nfe, draws=draws, **metrics))
            saved[f'c{ci}_nfe{nfe}'] = q; projections[nfe] = q
        rows[-1]['paired_refinement_rms'] = float(np.sqrt(np.mean((projections[128]-projections[64])**2)))
    with (path/'PROJECTIONS.npz').open('xb') as stream:
        np.savez_compressed(stream, **saved)
    score = float(np.mean([r['mean_error']+r['covariance_relative'] for r in rows if r['nfe'] == 128]))
    result = dict(rows=rows, development_score=score, seed=seed, calibrated=False,
                  selection_use='parameterization diagnostic only, not posterior qualification or training prerequisite')
    c.atomic_json(path/'EVALUATION.json', result)
    return result


def run(steps, seconds):
    c.require_compute()
    if torch.cuda.device_count() != 1:
        raise RuntimeError('One visible GPU')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ROOT.mkdir(parents=True, exist_ok=True)
    sources = {str(Path(m.__file__).resolve()): c.sha256(m.__file__) for m in (g, reference, transformer)}
    sources[str(Path(__file__).resolve())] = c.sha256(__file__)
    binding = dict(steps=steps, batch=8, seeds=[17, 29, 41], n=16, sources=sources,
                   data='existing correlated Gaussian with its actual observation law',
                   denominator_floor=1/1024, checkpoint_model='EMA .999',
                   scope='nonblocking short diagnostic; not the old full 65536-update gate',
                   no_analytic_teacher=True, common_training_data=True)
    if not (ROOT/'BINDING.json').exists():
        c.atomic_json(ROOT/'BINDING.json', binding)
    elif json.loads((ROOT/'BINDING.json').read_text()) != binding:
        raise ValueError('Diagnostic binding changed')
    start = time.monotonic(); cases = reference.targets(16)
    for seed in (17, 29, 41):
        for architecture in ('unet', 'transformer'):
            for prediction in ('v', 'x'):
                dest = ROOT/f'{architecture}_{prediction}_seed{seed}'
                if (dest/'EVALUATION.json').exists():
                    continue
                dest.mkdir(exist_ok=True)
                torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                model = Parameterized(architecture, prediction).cuda()
                ema = copy.deepcopy(model).eval().requires_grad_(False)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
                data = g.JointGenerator(16)
                data_gen = torch.Generator(device='cuda').manual_seed(seed+100)
                bridge = torch.Generator(device='cuda').manual_seed(seed+200)
                losses = []; begin = time.monotonic(); previous = 0
                if (dest/'LATEST.pt').exists():
                    state = torch.load(dest/'LATEST.pt', map_location='cuda', weights_only=False)
                    model.load_state_dict(state['model']); ema.load_state_dict(state['ema'])
                    optimizer.load_state_dict(state['optimizer'])
                    data_gen.set_state(state['data_rng'].cpu()); bridge.set_state(state['bridge_rng'].cpu())
                    previous = state['step']
                def save(step):
                    tmp = dest/'LATEST.tmp'
                    torch.save(dict(step=step, model=model.state_dict(), ema=ema.state_dict(),
                         optimizer=optimizer.state_dict(), data_rng=data_gen.get_state(), bridge_rng=bridge.get_state(),
                         binding=binding, architecture=architecture, prediction=prediction, seed=seed), tmp)
                    os.replace(tmp, dest/'LATEST.pt')
                for step in range(previous, steps):
                    # Same CFM objective and time law; only parameterization differs.
                    optimizer.zero_grad(set_to_none=True)
                    x, obs = data.batch(8, data_gen)
                    prepared = model.prepare(obs, torch.zeros_like(x), (0, 0, 0), 16)
                    noise = torch.randn(x.shape, device='cuda', generator=bridge)
                    t = (torch.arange(8, device='cuda')+torch.rand((), device='cuda', generator=bridge))/8
                    tt = t[:, None, None, None, None]
                    pred = model.velocity((1-tt)*noise+tt*x, t, prepared)
                    loss = (pred-(x-noise)).square().mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite diagnostic loss')
                    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), .5, error_if_nonfinite=True)
                    optimizer.step()
                    with torch.no_grad():
                        for ep, p in zip(ema.parameters(), model.parameters()):
                            ep.lerp_(p, .001)
                    losses.append(float(loss.detach()))
                    if (step+1) % 256 == 0 or step == 0:
                        row = dict(architecture=architecture, prediction=prediction, seed=seed, step=step+1,
                                   loss=float(np.mean(losses[-256:])), seconds=time.monotonic()-begin)
                        print(json.dumps(row), flush=True)
                        with (dest/'TRAIN.jsonl').open('a') as stream:
                            stream.write(json.dumps(row)+'\n')
                        save(step+1)
                        if time.monotonic()-start > seconds-240:
                            print('DIAGNOSTIC_SEGMENT_SAVED', flush=True)
                            return
                save(steps)
                result = evaluate(ema, cases, dest, seed)
                print(json.dumps(dict(stage='evaluated', architecture=architecture, prediction=prediction, seed=seed,
                                      score=result['development_score'])), flush=True)
                del model, ema, optimizer, data, prepared, pred, loss, x, obs, noise
                gc.collect(); torch.cuda.empty_cache()
    summary = {}
    for architecture in ('unet', 'transformer'):
        values = {}
        for prediction in ('v', 'x'):
            values[prediction] = [json.loads((ROOT/f'{architecture}_{prediction}_seed{s}'/'EVALUATION.json').read_text())['development_score']
                                 for s in (17, 29, 41)]
        selected = min(values, key=lambda p: (np.mean(values[p]), p != 'v'))
        summary[architecture] = dict(selected_prediction=selected, scores=values)
    for path, sha in sources.items():
        if c.sha256(path) != sha:
            raise ValueError('Source changed during diagnostic')
    c.atomic_json(ROOT/'COMPLETE.json', dict(passed=True, selection=summary, statistical_gate=False,
                  gaussian_calibrated=False, binding=binding, seconds=time.monotonic()-start))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps', type=int, default=4096)
    p.add_argument('--seconds', type=int, default=6600)
    a = p.parse_args(); run(a.steps, a.seconds)
