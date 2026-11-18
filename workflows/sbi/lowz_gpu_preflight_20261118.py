"""Disposable timing only. One GPU, two independent finite preflights."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from workflows.sbi import e2e_wholefield_pilot_train as t

B = json.loads(Path('configs/lowz_retraining_binding_20261118.json').read_text())
OUT = Path(os.environ['PREFLIGHT_DIR'])


def verified(path, digest):
    raw = Path(path).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == digest
    return json.loads(raw)


def bench(label, model, opt, update, extra, steps, epoch_steps):
    def timed(i):
        torch.cuda.synchronize()
        start = time.perf_counter()
        loss = update(i)
        torch.cuda.synchronize()
        assert np.isfinite(loss)
        return time.perf_counter()-start, loss
    timed(0)
    snapshot = OUT/(label+'.pt')
    objects = [model, opt, *extra]
    torch.save([x.state_dict() for x in objects], snapshot)
    _, reference = timed(1)
    weights = [p.detach().cpu().clone() for p in model.parameters()]
    for obj, state in zip(objects, torch.load(snapshot,weights_only=True)):
        obj.load_state_dict(state)
    _, replay = timed(1)
    difference = max(float((p.detach().cpu()-v).abs().max()) for p,v in zip(model.parameters(),weights))
    assert difference <= 2e-6 and abs(reference-replay)<=1e-6
    durations = [timed(i)[0] for i in range(2,2+steps)]
    result = dict(seconds=durations,mean_seconds=float(np.mean(durations)),epoch_steps=epoch_steps,
                  gpu_gib=torch.cuda.max_memory_allocated()/2**30,replay_maxabs=difference)
    print(label,json.dumps(result),flush=True)
    (OUT/(label+'.json')).write_text(json.dumps(result,indent=2))


def cfm():
    b=B['cfm_fine']
    verified(b['config'],b['config_sha256'])
    proof=verified(b['loader_receipt'],b['loader_receipt_sha256'])
    chart=t.views.load_chart(proof['normalizer']['sha256'])
    ids=t.panel(t.c.TRAIN)
    assert len(ids)==960 and t.EPOCH_STEPS==480
    model=t.FineModel('unet','x').cuda()
    ema=copy.deepcopy(model).eval().requires_grad_(False)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-5)
    weight,_=t.spectrum(ids,chart)
    def update(i):
        x,cond=t.batch(t.selected(ids,17,i),chart)
        return t.update(model,ema,opt,weight,x,cond,17,i)['loss']
    bench('cfm',model,opt,update,[ema],48,t.EPOCH_STEPS)


def p12():
    from workflows.abacus_tweb import p10_train_arm_a as a
    u=a.unet_impl
    b=B['p12a_encoder']
    contracts=verified(b['manifest'],b['sha256'])['contracts']
    epoch={}
    for k,v in contracts.items():
        root=Path(v['path'])
        roles=verified(root/'TRAINING_LOADER_READY.json',v['sha256'])['roles']
        epoch[k]=sum(len(np.load(root/'phases'/ph/'training_core_id.npy',mmap_mode='r')) for ph in roles['training'])
    loader=a.P10PhaseBalancedLoader(contracts['full']['path'])
    ph=loader.training_phases[0]
    state=a.prepare_phase_runtime(loader,ph,loader.target_scaler,training=True)
    adapter=loader.field_adapter(ph)
    cores=np.random.default_rng(42).permutation(list(state.expected_core_weight))[:40]
    norm=loader.field_normalization
    model=u.UPatch(base=24,latent_channels=32).cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0001)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epoch['full'])
    def inputs(i,halo):
        patch=adapter.extract(int(cores[i]),halo,u.CHANNELS,alignment_voxels=u.ALIGNMENT_VOXELS)
        return patch.authoritative_parent_id,u.model_inputs(patch,norm,'cuda')
    def update(i):
        parent,(x,points)=inputs(i,u.HALO_VOXELS)
        pred=model(x,points)
        target=torch.as_tensor(state.target_scaled[parent],device='cuda')
        w=torch.as_tensor(state.parent_weight[parent],device='cuda')
        loss=a.phase_equal_patch_objective((w*(pred-target).square().mean(1)).sum(),
            phase_weight_denominator=state.weight_denominator,phase_objective_scale=epoch['full']/len(loader.training_phases))
        opt.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True)
        opt.step();sched.step()
        return float(loss.detach())
    bench('p12',model,opt,update,[sched],24,epoch)
    model.eval();torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        durations=[]
        for i in range(5):
            start=time.perf_counter()
            _,(x,points)=inputs(26+i,48)
            latent=model.sample_latent(x,points);model.head(latent).cpu()
            torch.cuda.synchronize();durations.append(time.perf_counter()-start)
    (OUT/'halo48.json').write_text(json.dumps(dict(seconds=durations,gpu_gib=torch.cuda.max_memory_allocated()/2**30)))


if __name__=='__main__':
    key=sys.argv[1]
    t.setup(17 if key=='cfm' else 42)
    assert 'A100' in torch.cuda.get_device_name() and torch.cuda.get_device_properties(0).total_memory>75*2**30
    {'cfm':cfm,'p12':p12}[key]()
