"""Fresh seed-17 wide/coarse CFM on the corrected release. No evaluation or old weights."""
import argparse
import copy
import json
import math
import os
import time

import numpy as np
import torch
from workflows.sbi import e2e_coupled_cfm_pilot as pilot
from workflows.sbi import lowz_unet_fit as fine
from workflows.sbi import e2e_coupled_benchmark_models as coarse_models
from workflows.sbi import e2e_wholefield_pilot_models as fine_models
from workflows.sbi.e2e_wholefield_pilot_data import COARSE_CROP, SHAPE
from workflows.sbi.e2e_vdm_context_models import decode_density


@torch.no_grad()
def chain_check(ema, fine_root, normalizer):
    """One training address, observations only, tiny NFE for interface QA, not evaluation."""
    phase, address = json.loads((fine_root/'BINDING.json').read_text())['domains'][0]
    if phase not in pilot.c.TRAIN:
        raise PermissionError('Technical chain check requires a training address')
    obs = pilot.views.load_observations(phase,address,normalizer)
    ten = lambda a: torch.as_tensor(a,device='cuda',dtype=torch.float32)[None]
    cond = pilot.FieldCondition(ten(obs['joint']),ten(obs['wide']),
                                ten(obs['joint_center_from_wide_mpc_h']),'wide')
    z = coarse_models.sample(ema,cond,'cfm',2,[19031])
    chart = pilot.views.load_chart(normalizer)
    with np.load(fine.t.ROOT/'data'/phase/address/'OBSERVATIONS.npz') as f:
        wide = pilot.views.scale_channels(f['wide'],chart['wide'])
        np.testing.assert_allclose(wide,obs['wide'],rtol=0,atol=1e-5)
        joint = ten(pilot.views.scale_channels(f['joint'],chart['joint']))
    model = fine.t.FineModel('unet','x').cuda()
    model.load_state_dict(torch.load(fine_root/'LATEST.pt',map_location='cpu',weights_only=False)['ema'])
    residual = fine_models.sample(model,fine_models.Condition(joint,ten(wide),z,'sampled'),
                                  torch.randn((1,1,*SHAPE),device='cuda'),nfe=4)
    cs = chart['coarse_logrho']
    mass = torch.exp(z.double()*cs['std'][0]+cs['mean'][0])[(...,*COARSE_CROP)]
    rho = decode_density(mass,residual.double()*chart['fine_residual']['std'][0])
    if not torch.isfinite(rho).all() or rho.min()<=0:
        raise ValueError('Invalid decoded density')
    return dict(phase=phase,address=address,shape=list(rho.shape),truth_condition=False,
                scope='single finite-domain interface check; no scientific score')


def panel():
    binding, normalizer, _ = fine.inputs()
    c = pilot.c
    root = c.ROOT.parent / 'lowz_unet_endpoint_20261118_seed17'
    completed = json.loads((root/'COMPLETE.json').read_text())
    record = json.loads((root/'BINDING.json').read_text())
    if (record['data'] != binding or record['normalizer'] != normalizer
            or completed['steps'] != 15360 or completed['seed'] != 17
            or c.sha256(root/'LATEST.pt') != completed['latest_sha256']):
        raise ValueError('Fine checkpoint or low-z provenance mismatch')
    ids = pilot.training_ids()
    if len(ids) != 2560 or len(set(ids)) != 2560:
        raise ValueError('Expected 128 paired coarse domains per training phase')
    return ids, normalizer['sha256'], root, completed['latest_sha256']


def run(preflight):
    fine.t.setup(17)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    ids, normalizer, fine_root, fine_hash = panel()
    epoch_steps = len(ids)//2
    # Historical primary coarse checkpoint: epoch 16 of a 32-epoch LR schedule.
    schedule = 32*epoch_steps
    end = 24 if preflight else 16*epoch_steps
    root = pilot.c.ROOT.parent / ('lowz_coarse_preflight_' + os.environ['SLURM_JOB_ID']
                                  if preflight else 'lowz_coarse_velocity_seed17_v1')
    root.mkdir(exist_ok=False)
    profile = [opts[k] for p in pilot.c.TRAIN
               for opts in [[i for i in ids if i[0] == p]] for k in (0,64)]
    power = 0.
    for i in profile:
        _, target = pilot.pair(i, normalizer)
        power += abs(np.fft.fftn(target['coarse_logrho'][0], norm='ortho'))**2/len(profile)
    axes = np.meshgrid(*[np.fft.fftfreq(n) for n in power.shape], indexing='ij', sparse=True)
    shell = np.rint(np.sqrt(sum(k*k for k in axes))*min(power.shape)).astype(int)
    radial = np.bincount(shell.ravel(), weights=power.ravel())/np.bincount(shell.ravel())
    weight = torch.as_tensor(pilot.spectral_weights(radial[shell], .30),device='cuda',dtype=torch.float32)
    bound = dict(seed=17, stage='coarse', domain='wide', prediction='velocity',
                 preflight=preflight, config=pilot.c.sha256(pilot.c.CONFIG), normalizer=normalizer,
                 fine_path=str(fine_root/'LATEST.pt'), fine_sha256=fine_hash,
                 train_ids=ids, spectral_ids=profile, schedule_steps=schedule, stop_step=end,
                 runner=pilot.c.sha256(__file__), pilot=pilot.c.sha256(pilot.__file__))
    pilot.c.atomic_json(root/'BINDING.json', bound)
    model = pilot.CoupledBackbone('coarse','wide',base=24,levels=3).cuda()
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-5)
    rng = torch.Generator(device='cuda').manual_seed(1017)
    start = time.monotonic()
    for i in range(end):
        epoch, pos = divmod(i,epoch_steps)
        if pos == 0:
            order = np.random.default_rng(17+10000*epoch).permutation(len(ids))
        batch = pilot.batch_pair([pilot.pair(ids[k],normalizer) for k in order[2*pos:2*pos+2]])
        x, cond = pilot.condition(batch,'coarse','cuda')
        t = ((torch.arange(2,device='cuda')+torch.rand((),device='cuda',generator=rng))/2)[:,None,None,None,None]
        eps = torch.randn(x.shape,device='cuda',generator=rng)
        pred = model((1-t)*eps+t*x,model.schedule(t.flatten()),cond)
        loss = pilot.weighted_loss(pred-(x-eps),weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),.5,error_if_nonfinite=True)
        opt.param_groups[0]['lr'] = 1e-5+.5*(1e-4-1e-5)*(1+math.cos(math.pi*i/schedule))
        opt.step()
        with torch.no_grad():
            for a,b in zip(ema.parameters(),model.parameters()): a.lerp_(b,.001)
        step = i+1
        if step == 1 or step%32 == 0 or step == end:
            print(json.dumps(dict(step=step,loss=float(loss),seconds=time.monotonic()-start)),flush=True)
        if step%epoch_steps == 0 or step == end:
            pilot.save(root/'LATEST.pt',dict(step=step,model=model.state_dict(),ema=ema.state_dict(),
                       optimizer=opt.state_dict(),generator=rng.get_state(),weight=weight,binding=bound))
    pilot.c.atomic_json(root/('PREFLIGHT.json' if preflight else 'COARSE_FIT_COMPLETE.json'),
                        dict(steps=end,seconds=time.monotonic()-start,
                             peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,
                             coarse_sha256=pilot.c.sha256(root/'LATEST.pt'), fine_sha256=fine_hash,
                             chain_check=chain_check(ema,fine_root,normalizer)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('preflight','production'),required=True)
    run(parser.parse_args().mode=='preflight')
