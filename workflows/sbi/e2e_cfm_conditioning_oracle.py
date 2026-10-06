"""Frozen true-coarse diagnostic: paired local windows, never full-cap scores.

Run this extension against the immutable training snapshot, without modifying
checkpoint-bound modules. Preserve the historical common uncovered-cell fill.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

RUN = Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/cfm_conditioning_augmentation_20261006_v3')
sys.path.insert(0, str(RUN/'source'))
sys.path.insert(1, str(RUN/'source/workflows/sbi'))
import h5py
import numpy as np
import torch
from scipy import fft
from workflows.sbi import e2e_cfm_conditioning_train as train
from workflows.sbi import e2e_cfm_conditioning_sample as sample
from workflows.sbi import e2e_coupled_views as views
from workflows.sbi.e2e_cfm_conditioning_model import AugmentedCondition, corrupt_endpoint
from workflows.sbi.e2e_vdm_context_models import replicate, decode_density, block_mean
from workflows.sbi.e2e_global_canvas import Geometry, integrate
import e2e_sampler_factor_diagnostic as factor
from e2e_checkpoint_shear import LocalOperator

c = train.c
ARMS = [('control', 0.)]+[('augmented', s) for s in (0., .025, .05, .1, .2)]
EDGES = (0., .04, .08, .16, .32, np.inf)
CROP = (slice(16,32), slice(18,30), slice(18,30))


def read(path):
    return json.loads(Path(path).read_text())


def verified(record):
    if c.sha256(record['path']) != record['sha256']:
        raise ValueError('Changed source: '+record['path'])


def powers(rho, wide):
    residual = np.asarray(rho, float)-factor.b.op.lift(np.asarray(wide, float)[CROP])
    axes = np.meshgrid(*[2*np.pi*fft.fftfreq(n, d=6.766) for n in residual.shape], indexing='ij', sparse=True)
    k = np.sqrt(sum(a*a for a in axes))
    p = abs(fft.fftn(residual, norm='ortho', workers=4))**2
    return [float(p[(k>lo)&(k<=hi)].mean()) for lo, hi in zip(EDGES[:-1], EDGES[1:])]


def width(operator, rho, wide):
    eigen = factor.b.tidal.eigensystem(operator.tensor(rho, wide)).mean((1,2,3))
    return float((eigen[:,2]-eigen[:,0]).mean())


def load_model(arm):
    state = train.parent_state(); model = train.initialize(state).eval()
    checkpoint = train.PARENT
    if arm != 'parent':
        checkpoint = RUN/arm/'CHECKPOINT_003328.pt'
        fit = torch.load(checkpoint, map_location='cuda', weights_only=False)
        if (fit['step'] != 3328 or fit['binding']['arm'] != arm
                or fit['binding']['profile'] != train.profile()
                or fit['binding']['sources'] != train.source_records()
                or fit['binding']['parent_sha256'] != c.sha256(train.PARENT)):
            raise ValueError('Continuation checkpoint binding mismatch')
        model.load_state_dict(fit['ema'])
    return model, c.file_record(checkpoint, content_hash=True), state['binding']['normalizer']


def run(case_index, smoke=False):
    train.setup()
    from workflows.sbi.e2e_loa_field_sampling import setup
    setup(True)
    if c.REPO != RUN/'source':
        raise PermissionError('Frozen dependency snapshot required')
    prepared_path = factor.OUT/'PREPARED.json'
    prepared = read(prepared_path)
    if not prepared['passed'] or prepared['freeze_sha256'] != c.sha256(factor.b.FREEZE):
        raise ValueError('Oracle panel binding changed')
    rows = prepared['cases']
    if len(rows) != 32 or not 0 <= case_index < 32:
        raise ValueError('Registered 32-pair panel required')
    row = rows[case_index]; phase, cap, pid = (row[k] for k in ('phase','cap','pair_id'))
    sample.guard(phase, cap, 0, 'parent', 0.)
    dest = RUN/('oracle_smoke' if smoke else 'oracle')/pid
    dest.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); verified(row['file'])
    with h5py.File(row['input']) as f:
        arrays = {k:f[k][:] for k in ('joint','wide','true_wide','true_rho','canvas_rho')}
    baseline_receipt = factor.OUT/'results'/(pid+'.json')
    baseline = read(baseline_receipt); verified(baseline['fields'])
    if baseline['draws'] != 32 or baseline['pair_id'] != pid:
        raise ValueError('Oracle baseline address mismatch')
    operator = LocalOperator(); core = np.asarray(row['core_start'])
    operator.cores = [factor.slices(core+[16*side,0,0], (16,16,16)) for side in (0,1)]
    truthp = powers(arrays['true_rho'], arrays['true_wide'])
    truthw = width(operator, arrays['true_rho'], arrays['true_wide'])
    parentp = []; parentw = []
    with h5py.File(baseline['fields']['path']) as f:
        for r, w in zip(f['oracle_coarse_local_fine/rho'], f['oracle_coarse_local_fine/wide']):
            parentp.append(powers(r, w)); parentw.append(width(operator, r, w))
        replay_expected = f['oracle_coarse_local_fine/rho'][:4]
    plan = read(sample.BASE/'fullcap_inputs'/phase/cap/'PLAN.json')
    cg, fg = [Geometry.make(plan['midpoints'], s) for s in ('coarse','fine')]
    single = Geometry.make([row['midpoint']], 'fine')
    count = 4 if smoke else 32
    valid = np.isfinite(arrays['canvas_rho'][0])
    models = {}; checkpoints = {}
    for arm in ('parent','control','augmented'):
        models[arm], checkpoints[arm], normalizer = load_model(arm)
    chart = views.load_chart(normalizer)
    targetz = (np.log(arrays['true_wide'])-chart['coarse_logrho']['mean'][0])/chart['coarse_logrho']['std'][0]
    values = {f'{arm}/s{s:g}':dict(power=[], width=[]) for arm,s in ARMS}
    maxmass = 0.; parity = 0.
    with torch.no_grad():
        for first in range(0,count,4):
            tensor = lambda x: torch.as_tensor(x, device='cuda', dtype=torch.float32)[None].expand(4,*x.shape)
            endpoint = tensor(targetz[None])
            eta = sample.canvas_noise(cg, phase, cap, first, 'conditioning')[cg.crop(row['window_index'])].clone()
            nf = factor.noise_crop(fg, row['window_index'], phase, cap, 'fine', first, 4)
            physical = torch.exp(endpoint.double()*chart['coarse_logrho']['std'][0]+chart['coarse_logrho']['mean'][0])
            mass = physical[(slice(None),slice(None),*CROP)]
            grid = [('parent', 0.)]+ARMS if first==0 else ARMS
            for arm,s in grid:
                strength = endpoint.new_full((4,),s)
                noisy = corrupt_endpoint(endpoint, strength, eta)
                condition = AugmentedCondition(tensor(views.scale_channels(arrays['joint'],chart['joint'])),
                    tensor(views.scale_channels(arrays['wide'],chart['wide'])),tensor(np.zeros(3,'f4')),
                    'joint',replicate(noisy[(slice(None),slice(None),*CROP)]),noisy,'oracle_diagnostic',strength)
                u,_ = integrate(models[arm], 'fine', [condition], nf.clone(), single, 128)
                rho = decode_density(mass,u.double()*chart['fine_residual']['std'][0])
                if not torch.isfinite(rho).all() or not (rho>0).all():
                    raise ValueError('Invalid oracle density')
                maxmass = max(maxmass,float(((block_mean(rho)-mass)/mass).abs().max()))
                rr,ww = rho[:,0].cpu().numpy(),physical[:,0].cpu().numpy()
                for j,(r,w) in enumerate(zip(rr,ww)):
                    r = np.where(valid,r,factor.b.op.lift(w[CROP])).astype('f4')
                    w = w.astype('f4')
                    if arm=='parent':
                        parity = max(parity,float(abs(r-replay_expected[j]).max()))
                        if not np.array_equal(r,replay_expected[j]):
                            raise ValueError('Oracle parent replay is not exact')
                    else:
                        q=values[f'{arm}/s{s:g}'];q['power'].append(powers(r,w));q['width'].append(width(operator,r,w))
                print(json.dumps(dict(case=case_index,pair_id=pid,arm=arm,strength=s,draws_done=first+4,
                    seconds=time.monotonic()-started)),flush=True)
    if maxmass > train.profile()['mass_relative_tolerance']:
        raise ValueError('Oracle mass failure')
    result=dict(passed=True,smoke=smoke,phase=phase,cap=cap,pair_id=pid,draws=count,nfe=128,
        source=c.file_record(__file__,content_hash=True),frozen_sources=train.source_records(),
        prepared=c.file_record(prepared_path,content_hash=True),input=row['file'],
        baseline=c.file_record(baseline_receipt,content_hash=True),checkpoints=checkpoints,
        config=c.file_record(train.CONFIG,content_hash=True),truth_power=truthp,truth_width=truthw,
        parent=dict(power=parentp,width=parentw),arms=values,exact_parent_replay=True,
        parent_maxabs=parity,max_mass_error=maxmass,seconds=time.monotonic()-started,
        job=os.environ['SLURM_JOB_ID'],posterior_qualified=False)
    c.atomic_json(dest/'COMPLETE.json',result)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--case',type=int,required=True);ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args();run(args.case,args.smoke)
