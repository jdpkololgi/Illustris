"""Bounded train-only coupled CFM pilot; never opens confirmation payloads."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import torch
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_coordinates as coord
from workflows.sbi import e2e_coupled_views as views
from workflows.sbi.e2e_coupled_benchmark_models import CoupledBackbone, FieldCondition
from workflows.sbi.e2e_vdm_context_models import project, replicate


def spectral_weights(power,alpha):
    if not 0<=alpha<=.5 or not np.isfinite(power).all() or np.min(power)<0:
        raise ValueError('invalid training spectrum/exponent')
    positive=power[power>0]
    if not len(positive):raise ValueError('degenerate target spectrum')
    reference=float(np.median(positive));floor=reference*1e-3
    weight=(np.maximum(power,floor)/reference)**(-alpha)
    return weight/np.sqrt(np.mean(weight**2))


def weighted_loss(error,weight):
    return (torch.fft.fftn(error,dim=(-3,-2,-1),norm='ortho').abs().square()*weight.square()).mean()


def training_ids():
    result=[]
    for phase in c.TRAIN:
        receipt=coord.verify_receipt(coord.ROOT/'conditions'/phase/'CONDITIONS_COMPLETE.json',payload=False)
        ids=sorted(Path(item['path']).stem for item in receipt['pair_receipts'])
        if len(ids)!=128 or len(set(ids))!=128:raise ValueError('training panel incomplete')
        result.extend((phase,pair) for pair in ids)
    return result


def pair(index,normalizer):
    phase,pair_id=index
    if phase not in c.TRAIN:raise PermissionError('pilot fitting is train-only')
    return views.load_training_pair(phase,pair_id,normalizer,offset=(0,0,0))


def condition(batch,stage,device):
    obs,target=batch
    t=lambda x:torch.as_tensor(x,device=device,dtype=torch.float32)
    coarse=t(target['coarse_logrho'])
    crop=views.crop_slices(c.TRAIN[0],(0,0,0))[1]
    cond=FieldCondition(t(obs['joint']),t(obs['wide']),t(obs['joint_center_from_wide_mpc_h']),
        'wide' if stage=='coarse' else 'joint',
        None if stage=='coarse' else replicate(coarse[(slice(None),slice(None),*crop)]),
        None if stage=='coarse' else coarse,None if stage=='coarse' else 'training_truth')
    return (coarse if stage=='coarse' else t(target['fine_residual'])),cond


def batch_pair(items):
    return tuple({k:np.stack([item[i][k] for item in items]) for k in items[0][i]} for i in (0,1))


def save(path,state):
    temporary=path.with_suffix('.tmp');torch.save(state,temporary);os.replace(temporary,path)


def run(a):
    c.require_compute()
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('one visible GPU required per factor')
    if a.stage not in ('coarse','fine') or a.seed not in (17,29):raise ValueError('unregistered factor')
    torch.set_num_threads(4);torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    normalizer=c.sha256(coord.ROOT/'normalization/NORMALIZATION_COMPLETE.json')
    views.load_chart(normalizer)
    release=coord.ROOT/'data_release/DATA_PRODUCTS_QUALIFIED.json'
    if c.sha256(release)!='cf26b07c16ec0714fcd6d0c318c77b0ab3799497a00eba4195f48a2a42889ced':
        raise ValueError('prepared data release changed')
    ids=training_ids();domain='wide' if a.stage=='coarse' else 'joint'
    model=CoupledBackbone(a.stage,domain,base=24,levels=3).cuda()
    ema=copy.deepcopy(model).eval().requires_grad_(False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-5)
    generator=torch.Generator(device='cuda').manual_seed(a.seed+1000)
    bound=dict(stage=a.stage,seed=a.seed,alpha=a.alpha,steps=26624,normalizer=normalizer,
        runner=c.sha256(__file__),model=c.sha256(__import__('inspect').getfile(CoupledBackbone)),
        train_ids=ids,context_offset=[0,0,0],ema=.999,lr=[1e-4,1e-5],batch=2,
        source_profile='26 train-only pair spectra; no development selection yet',
        confirmation_access=False)
    c.atomic_json(root/'BINDING.json',bound) if not (root/'BINDING.json').exists() else None
    if json.loads((root/'BINDING.json').read_text())!=json.loads(json.dumps(bound)):
        raise ValueError('resume binding differs')
    start=time.monotonic();latest=root/'LATEST.pt';step=0
    if latest.exists():
        state=torch.load(latest,map_location='cuda',weights_only=False)
        if state['binding']!=bound:raise ValueError('checkpoint binding differs')
        model.load_state_dict(state['model']);ema.load_state_dict(state['ema'])
        optimizer.load_state_dict(state['optimizer']);generator.set_state(state['generator'].cpu())
        weight=state['weight'].cuda();step=state['step']
    else:
        power=None;profile_ids=[]
        key='coarse_logrho' if a.stage=='coarse' else 'fine_residual'
        for phase in c.TRAIN:
            selected=[v for v in ids if v[0]==phase]
            for index in (0,64):
                _,target=pair(selected[index],normalizer)
                p=abs(np.fft.fftn(target[key][0],norm='ortho'))**2
                power=p/26 if power is None else power+p/26;profile_ids.append(selected[index])
        # Radial bins use physical box lengths, including the rectangular fine axis.
        axes=np.meshgrid(*[np.fft.fftfreq(n) for n in power.shape],indexing='ij',sparse=True)
        shell=np.rint(np.sqrt(sum(k*k for k in axes))*min(power.shape)).astype(int)
        spectrum=np.bincount(shell.ravel(),weights=power.ravel())/np.bincount(shell.ravel())
        weight=torch.as_tensor(spectral_weights(spectrum[shell],a.alpha),device='cuda',dtype=torch.float32)
        c.atomic_json(root/'SPECTRAL_PROFILE.json',dict(ids=profile_ids,power=spectrum.tolist(),alpha=a.alpha,
            interpretation='loss weighting only; physical bridge/base noise unchanged'))
    def checkpoint():
        state=dict(binding=bound,step=step,model=model.state_dict(),ema=ema.state_dict(),
            optimizer=optimizer.state_dict(),generator=generator.get_state(),weight=weight,
            seconds=time.monotonic()-start)
        save(latest,state)
        if step in (832,1664,3328,6656,13312,26624):save(root/f'CHECKPOINT_{step:06d}.pt',state)
        c.atomic_json(root/'STATUS.json',dict(step=step,seconds=time.monotonic()-start,
            job=os.environ.get('SLURM_JOB_ID'),peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,
            training_complete=step==26624,confirmation_opened=False),replace=True)
    projector=project if a.stage=='fine' else lambda x:x
    while step<26624 and time.monotonic()-start<a.seconds:
        epoch=step//832;position=step%832
        order=np.random.default_rng(a.seed+epoch*10000).permutation(len(ids))
        tick=time.monotonic();items=[pair(ids[i],normalizer) for i in order[2*position:2*position+2]]
        x,cond=condition(batch_pair(items),a.stage,'cuda');model.train();optimizer.zero_grad(set_to_none=True)
        t=(torch.arange(2,device='cuda')+torch.rand((),device='cuda',generator=generator))/2
        t=t[:,None,None,None,None]
        eps=projector(torch.randn(x.shape,device='cuda',generator=generator))
        z=(1-t)*eps+t*x;pred=model(z,model.schedule(t.flatten()),cond)
        error=projector(pred-(x-eps));loss=weighted_loss(error,weight)
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite training loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),.5,error_if_nonfinite=True)
        lr=1e-5+.5*(1e-4-1e-5)*(1+math.cos(math.pi*step/26624))
        for group in optimizer.param_groups:group['lr']=lr
        optimizer.step()
        with torch.no_grad():
            for ep,p in zip(ema.parameters(),model.parameters()):ep.lerp_(p,.001)
        step+=1
        if step%32==0 or step==1:
            torch.cuda.synchronize()
            epower=torch.fft.fftn(error.detach(),dim=(-3,-2,-1),norm='ortho').abs().square().mean((0,1)).cpu().numpy()
            axes=np.meshgrid(*[np.fft.fftfreq(n) for n in epower.shape],indexing='ij',sparse=True)
            radius=np.sqrt(sum(k*k for k in axes))
            shells=[float(epower[(radius>=lo)&(radius<hi)].mean()) for lo,hi in
                zip((0,.05,.1,.2,.3,.4),(.05,.1,.2,.3,.4,1.))]
            record=dict(step=step,weighted_loss=float(loss),mse=float(error.square().mean()),
                dc_error=float(error.mean((2,3,4)).square().mean()),grad_norm=float(norm),lr=lr,
                velocity_error_shells=shells,diagnostic_scope='stochastic training velocity, not posterior calibration',
                step_seconds=time.monotonic()-tick,elapsed=time.monotonic()-start)
            with (root/'TRAIN.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
            print(json.dumps(record),flush=True)
        if step==1 or step%256==0 or step in (832,1664,3328,6656,13312,26624):checkpoint()
    checkpoint();print('SEGMENT_COMPLETE',step,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--output',required=True);p.add_argument('--alpha',type=float,default=.30)
    p.add_argument('--seconds',type=int,default=6600);run(p.parse_args())
