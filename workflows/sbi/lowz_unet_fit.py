"""Single fresh low-z fine-field fit; never reuse or overwrite a historical fit."""
import copy
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from workflows.sbi import e2e_wholefield_pilot_train as t


def validate_panel(ids):
    phases = set(t.c.TRAIN)
    if (len(phases) != 20 or len(ids) != 960 or len(set(ids)) != 960
            or Counter(p for p, _ in ids) != dict.fromkeys(phases, 48)
            or t.EPOCH_STEPS != 480 or t.STEPS != 15360):
        raise ValueError("Incorrect phase membership, exposure or schedule")


def spectral_weight(ids, chart):
    power = np.zeros(t.SHAPE)
    addresses = []
    for phase in t.c.TRAIN:
        options = [a for a in ids if a[0] == phase]
        for index in (0, 24):
            address = options[index]
            with np.load(t.ROOT/'data'/phase/address[1]/'TARGETS.npz') as f:
                residual = f['residual']/chart['fine_residual']['std'][0]
            power += abs(np.fft.fftn(residual, norm='ortho'))**2
            addresses.append(address)
    power /= len(addresses)
    axes = np.meshgrid(*[np.fft.fftfreq(n) for n in t.SHAPE], indexing='ij', sparse=True)
    shell = np.rint(np.sqrt(sum(k*k for k in axes))*min(t.SHAPE)).astype(int)
    radial = np.bincount(shell.ravel(), weights=power.ravel())/np.bincount(shell.ravel())
    return torch.as_tensor(t.spectral_weights(radial[shell], .3), device='cuda', dtype=torch.float32), addresses


def inputs():
    binding = json.loads((t.c.REPO/'configs/lowz_retraining_binding_20261118.json').read_text())['cfm_fine']
    if t.c.sha256(t.c.CONFIG) != binding['config_sha256'] or t.c.sha256(binding['loader_receipt']) != binding['loader_receipt_sha256']:
        raise ValueError("Low-z data binding changed")
    receipt = json.loads(Path(binding['loader_receipt']).read_text())
    normalizer = receipt['normalizer']
    if not receipt['pass'] or receipt['loader_source_sha256'] != t.c.sha256(t.__file__) or t.c.sha256(normalizer['path']) != normalizer['sha256']:
        raise ValueError("Loader or normalization binding changed")
    ids = t.panel(t.c.TRAIN)
    validate_panel(ids)
    return binding, normalizer, ids


def main():
    t.setup(17)
    binding, normalizer, ids = inputs()
    root = t.c.ROOT.parent/'lowz_unet_endpoint_20261118_seed17'
    root.mkdir(exist_ok=False)
    chart = t.views.load_chart(normalizer['sha256'])
    weight, addresses = spectral_weight(ids, chart)
    t.c.atomic_json(root/'BINDING.json', dict(data=binding, normalizer=normalizer, domains=ids,
        spectral_samples=addresses, seed=17, epochs=32, batch=2, sources=t.sources(),
        git_revision=os.environ['FIT_REVISION'], source_digest=os.environ['FIT_DIGEST'],
        launcher=t.c.sha256(__file__), job=os.environ['SLURM_JOB_ID']))
    model=t.FineModel('unet','x').cuda()
    ema=copy.deepcopy(model).eval().requires_grad_(False)
    optimizer=torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    for i in range(t.STEPS):
        x, cond=t.batch(t.selected(ids,17,i),chart,cache=None)
        row=t.update(model,ema,optimizer,weight,x,cond,17,i)
        step=i+1
        if step==1 or step%16==0:
            print(json.dumps(dict(step=step,**row)),flush=True)
        if step==1 or step%480==0:
            state=dict(step=step,model=model.state_dict(),ema=ema.state_dict(),
                       optimizer=optimizer.state_dict(),weight=weight,
                       binding_sha256=t.c.sha256(root/'BINDING.json'),
                       cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state())
            t.save(root/'LATEST.pt',state)
            if step in t.CHECKPOINTS:
                t.save(root/f'CHECKPOINT_{step:06d}.pt',state)
                result=t.development(ema,chart,weight,cache=None)
                if not result['available'] or not np.isfinite(result['fixed_noise_cfm_loss']):
                    raise ValueError("Development check failed")
                t.c.atomic_json(root/f'DEVELOPMENT_{step:06d}.json',result)
    t.c.atomic_json(root/'COMPLETE.json',dict(epochs=32,steps=step,seed=17,
        latest_sha256=t.c.sha256(root/'LATEST.pt'),posterior_qualified=False))


if __name__=='__main__':
    main()
