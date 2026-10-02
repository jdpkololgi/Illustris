"""Two-window shared-state engine; no claim of posterior qualification.

Same Heun/overlap rule as e2e_global_overlap_probe, factored for mock evaluation.
Inference accepts observations and frozen models only, never target products.
"""
from dataclasses import replace
import numpy as np
import torch
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_operators as op
from workflows.sbi import e2e_coupled_views as views
from workflows.sbi.e2e_coupled_benchmark_models import FieldCondition
from workflows.sbi.e2e_vdm_context_models import project, replicate, decode_density, block_mean
from workflows.sbi.e2e_global_overlap_probe import slices, overlap


def addressed_noise(stage, address, first, count, side=None, device='cuda'):
    shape = (56,48,48) if stage == 'coarse' else (96,48,48)
    if side is not None:
        shape = (48,48,48) if stage == 'coarse' else (64,48,48)
    rows = []
    for draw in range(first, first+count):
        seed = int(c.digest(['global-mock-v1', address, stage, draw, side])[:15], 16)
        rng = torch.Generator(device=device).manual_seed(seed)
        rows.append(torch.randn((1,1,*shape), device=device, generator=rng))
    value = torch.cat(rows)
    return project(value) if stage == 'fine' else value


@torch.no_grad()
def integrate(model, stage, conditions, state, nfe=128, fuse=True):
    if nfe < 2 or nfe % 2:
        raise ValueError('Positive even NFE required')
    projector = project if stage == 'fine' else lambda x: x
    state = state.clone() if fuse else [v.clone() for v in state]

    def velocity(z, t):
        chunks = [z[slices(stage,i)] for i in range(2)] if fuse else z
        values = [model(x, model.schedule(x.new_full((len(x),), t)), condition)
                  for x, condition in zip(chunks, conditions)]
        if not fuse:
            return values
        total, weight = torch.zeros_like(z), torch.zeros_like(z[:1])
        for i, v in enumerate(values):
            total[slices(stage,i)] += v
            weight[slices(stage,i)] += 1
        return projector(total/weight)

    for step in range(nfe//2):
        t, dt = step/(nfe//2), 2/nfe
        first = velocity(state, t)
        if fuse:
            proposal = projector(state + dt*first)
            state = projector(state + .5*dt*(first + velocity(proposal, t+dt)))
        else:
            proposal = [projector(z + dt*v) for z,v in zip(state,first)]
            second = velocity(proposal, t+dt)
            state = [projector(z + .5*dt*(v+w)) for z,v,w in zip(state,first,second)]
    return [state[slices(stage,i)] for i in range(2)] if fuse else state


@torch.no_grad()
def generate(models, arrays, chart, address, first=0, count=4, nfe=128,
             variant='shared_state', base_noise=None):
    if variant not in ('shared_state','independent','shared_noise') or len(arrays) != 2:
        raise ValueError('Registered two-window variant required')
    bases = []
    t = lambda x: torch.as_tensor(x, device='cuda', dtype=torch.float32)[None].expand(count,*x.shape)
    for values in arrays:
        bases.append(FieldCondition(t(views.scale_channels(values['joint'],chart['joint'])),
                     t(views.scale_channels(values['wide'],chart['wide'])),
                     t(np.zeros(3,'f4')), 'wide'))
    def noise(stage):
        z = addressed_noise(stage,address,first,count) if base_noise is None else base_noise[stage]
        if variant == 'shared_state':
            return z
        right = (addressed_noise(stage,address,first,count,side=1) if variant == 'independent'
                 else z[slices(stage,1)])
        return [z[slices(stage,0)], right]
    fuse = variant == 'shared_state'
    zparts = integrate(models['coarse'],'coarse',bases,noise('coarse'),nfe,fuse)
    crop = tuple(slice(*v) for v in op.layout()['joint_coarse_crop_in_wide'])
    fine = [replace(b, region='joint', coarse_joint=replicate(z[(slice(None),slice(None),*crop)]),
                    coarse_wide=z, coarse_source='sampled') for b,z in zip(bases,zparts)]
    uparts = integrate(models['fine'],'fine',fine,noise('fine'),nfe,fuse)
    rho, wide, mass = [], [], []
    for z,u in zip(zparts,uparts):
        w = torch.exp(z.double()*chart['coarse_logrho']['std'][0]+chart['coarse_logrho']['mean'][0])
        coarse = w[(slice(None),slice(None),*crop)]
        r = decode_density(coarse,u.double()*chart['fine_residual']['std'][0])
        if not torch.isfinite(r).all() or not (r > 0).all():
            raise FloatingPointError('Invalid density')
        mass.append(float(((block_mean(r)-coarse)/coarse).abs().max()))
        rho.append(r[:,0].cpu().numpy())
        wide.append(w[:,0].cpu().numpy())
    difference = rho[0][:,32:] - rho[1][:,:-32]
    overlap_maxabs = float(np.max(abs(difference)))
    if max(mass) > 2e-6 or (fuse and overlap_maxabs > 1e-10):
        raise ValueError('Mass conservation or coherent overlap failed')
    return dict(rho=rho, wide=wide, max_mass_error=max(mass),
                overlap_maxabs=overlap_maxabs, overlap_rmse=float(np.sqrt(np.mean(difference**2))))
