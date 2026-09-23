"""Train-prior-only spectral controls with explicit base-noise contracts."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
import torch
from torch import nn
from workflows.sbi.e2e_conditional_reference import Data,atomic_json,atomic_checkpoint
from workflows.sbi.e2e_conditional_reference_math import metrics,qualify,null_thresholds,oracle_moments
from workflows.sbi.e2e_conditional_reference_continue import digest,restore_training_state
from workflows.sbi.e2e_reference_target_controls import Teacher,affine_velocity,bridge
from workflows.sbi.e2e_direct_vdm import ConditionalVDM
from workflows.sbi.e2e_spectral_absorption import draw_diagnostic, shell_power

ARMS=['physical','weighted','coordinates','white_bridge']


class Spectral(nn.Module):
    def __init__(self,power,exponent=.5):
        super().__init__()
        p=torch.as_tensor(power,dtype=torch.float32)
        p=p.clamp_min(p.max()*1e-6)
        self.register_buffer('scale',p.rsqrt() if exponent==.5 else p.pow(-exponent))
    def forward(self,x,inverse=False):
        scale=self.scale.reciprocal() if inverse else self.scale
        return torch.fft.ifftn(torch.fft.fftn(x,dim=(-3,-2,-1))*scale,dim=(-3,-2,-1)).real


class WhitenedTeacher(Teacher):
    def __init__(self,data,transform):
        super().__init__(data)
        self.transform=transform
        d=data.n**3
        w=transform(torch.eye(d,device=data.chol.device).reshape(d,1,data.n,data.n,data.n)).flatten(1).double()
        cov=torch.stack([torch.as_tensor(c['sigma'],device=w.device) for c in data.cases[:2]])
        val,vec=torch.linalg.eigh(w@cov@w.T)
        self.values.copy_(val.float());self.vectors.copy_(vec.float())
    def moments(self,condition):
        mu,which=super().moments(condition)
        n=condition.shape[-1]
        return self.transform(mu.reshape(-1,1,n,n,n)).flatten(1),which


def targets(x,c,gen,arm,exact,transform,teacher,white_teacher):
    # Identical random consumption in every branch.
    z,t,v=bridge(x,gen)
    if arm=='white_bridge':
        tt=t[:,None,None,None,None]
        # Recover the SAME physical epsilon from x-v; use it as white u-noise.
        eps=x-v;u=transform(x)
        z=(1-tt)*eps+tt*u
        v=white_teacher(z,t,c) if exact else u-eps
    else:
        if exact:v=teacher(z,t,c)
        if arm=='coordinates':z,v=transform(z),transform(v)
    return z,t,v


def train_step(model,opt,x,c,gen,arm,exact,transform,teacher,white_teacher):
    opt.zero_grad(set_to_none=True)
    z,t,v=targets(x,c,gen,arm,exact,transform,teacher,white_teacher)
    error=model(z,model.schedule(t),c)-v
    if arm=='weighted':error=transform(error)
    loss=error.square().mean()
    if not torch.isfinite(loss):raise FloatingPointError('nonfinite spectral loss')
    loss.backward();opt.step()
    return float(loss.detach())


@torch.no_grad()
def sample(model,c,nfe,gen,arm,transform):
    if nfe<2 or nfe%2:raise ValueError('even Heun NFE required')
    mode=model.training;model.eval()
    z=torch.randn(c[:,:1].shape,device=c.device,generator=gen)
    if arm=='coordinates':z=transform(z)
    dt=2/nfe
    try:
        for i in range(nfe//2):
            t=z.new_full((len(z),),i*dt);tt=t+dt
            first=model(z,model.schedule(t),c)
            second=model(z+dt*first,model.schedule(tt),c)
            z=z+dt/2*(first+second)
        if arm in ['coordinates','white_bridge']:z=transform(z,inverse=True)
        if not torch.isfinite(z).all():raise FloatingPointError('nonfinite draws')
        return z
    finally:model.train(mode)


def items():
    return [dict(seed=s,fixed=f,exact=e,arm=a,name=f'{a}_{"exact" if e else "stochastic"}_seed{s}_{"amortised" if f is None else "fixed0"}')
            for s in [17,29] for f in [None,0] for e in [False,True] for a in ARMS]


def prepare(a):
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():raise RuntimeError('GPU allocation required')
    root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
    repo=Path(__file__).resolve().parents[2]
    cfg=json.loads((repo/'configs/e2e_conditional_reference_v1.json').read_text())
    data=Data(cfg,torch.device('cuda'));gen=torch.Generator(device='cuda').manual_seed(912731)
    total=torch.zeros((8,8,8),device='cuda',dtype=torch.float64)
    for _ in range(32):
        eps=torch.randn((256,512),device='cuda',generator=gen)
        x=(eps@data.chol.T).reshape(-1,8,8,8)
        total+=torch.fft.fftn(x,dim=(-3,-2,-1),norm='ortho').abs().square().double().sum(0)
    power=(total/8192).cpu().numpy();np.save(root/'train_prior_power.npy',power)
    transform=Spectral(power).cuda();white=WhitenedTeacher(data,transform)
    # Analytic physical-coordinate sampler checks for the changed bridge.
    d=512;inv=transform(torch.eye(d,device='cuda').reshape(d,1,8,8,8),inverse=True).flatten(1).cpu().double().numpy()
    controls=[]
    for j,case in enumerate(data.cases):
        mu=transform(data.means[j].reshape(1,1,8,8,8)).flatten().cpu().numpy()
        vec=white.vectors[j%2].cpu().numpy().astype(float);val=white.values[j%2].cpu().numpy().astype(float)
        for nfe in [128,256]:
            r=oracle_moments(dict(mu=mu,vectors=vec,values=val),'cfm',nfe)
            sigma=inv@((vec*r['variance'])@vec.T)@inv.T
            err=float(np.linalg.norm(sigma-case['sigma'])/np.linalg.norm(case['sigma']))
            power_ratio=shell_power(r['variance'],inv@vec,data.radius)/shell_power(case['values'],case['vectors'],data.radius)
            mean_error=float(np.linalg.norm(inv@vec@r['mean']-case['mu'])/np.sqrt(np.trace(case['sigma'])))
            if max(err,mean_error,float(np.max(np.abs(power_ratio-1))))>.01:raise ValueError('white bridge oracle needs more NFE')
            controls.append(dict(case=j,nfe=nfe,physical_covariance_error=err,physical_mean_error=mean_error,power_ratio=power_ratio.tolist()))
    atomic_json(root/'oracle_controls.json',controls)
    files=[repo/'workflows/sbi'/n for n in ['e2e_spectral_preconditioning.py','e2e_spectral_absorption.py',
        'e2e_conditional_reference.py','e2e_conditional_reference_math.py','e2e_conditional_reference_continue.py',
        'e2e_reference_target_controls.py','e2e_direct_vdm.py']]
    files += [repo/'configs/e2e_conditional_reference_v1.json']
    files += [repo/'docs/e2e_spectral_preconditioning_20260923.md',repo/'tests/test_e2e_spectral_preconditioning.py']
    sources={str(p.relative_to(repo)):digest(p) for p in files}
    for p in files:
        q=root/'source'/p.relative_to(repo);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
    atomic_json(root/'manifest.json',dict(sources=sources,config=cfg,items=items(),updates=32768,
        checkpoints=[8192,32768],power_sha256=digest(root/'train_prior_power.npy'),
        prior_fit_seed=912731,prior_fit_count=8192,workers=4,job=os.environ['SLURM_JOB_ID'],
        git_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()))


@torch.no_grad()
def velocity_risk(model,data,item,transform,teacher,white):
    gen=torch.Generator(device='cuda').manual_seed(991182)
    physical=[];weighted=[]
    for _ in range(16):
        x,c=data.batch(32,item['fixed'],gen)
        z,t,v=targets(x,c,gen,item['arm'],True,transform,teacher,white)
        error=model(z,model.schedule(t),c)-v
        if item['arm'] in ['coordinates','white_bridge']:error=transform(error,inverse=True)
        physical.append(float(error.square().mean()));weighted.append(float(transform(error).square().mean()))
    return dict(physical_mse=float(np.mean(physical)),spectral_mse=float(np.mean(weighted)),
        examples=512,seed=991182,bridge='changed' if item['arm']=='white_bridge' else 'original')


@torch.no_grad()
def evaluate(model,data,item,cfg,transform,out,update,draws,nfes,indices=None):
    out.mkdir(parents=True,exist_ok=True)
    for case in (indices if indices is not None else (range(4) if item['fixed'] is None else [0])):
        for nfe in nfes:
            p=out/f'evaluation_{update}_{case}_{nfe}.json'
            if p.exists():continue
            gen=torch.Generator(device='cuda').manual_seed(200000+item['seed']*100+case)
            samples=[]
            for first in range(0,draws,64):
                c=data.conditions[case].expand(min(64,draws-first),-1,-1,-1,-1)
                samples.append(sample(model,c,nfe,gen,item['arm'],transform).flatten(1).cpu().numpy())
            arr=np.concatenate(samples);r=metrics(arr,data.cases[case],data.q,data.radius)
            null=null_thresholds(data.cases[case],data.q,draws,128)
            np.save(out/f'draws_{update}_{case}_{nfe}.npy',arr)
            atomic_json(p,item|dict(case=case,nfe=nfe,update=update,draws=draws,**r,**qualify(r,null,cfg),
                power_pass=all(.9<=v<=1.1 for v in r['power_ratio'])))
            if draws==2048 and nfe==256:
                atomic_json(out/f'absorption_{case}.json',draw_diagnostic(arr,data.cases[case],data.radius))
            print('EVAL',item['name'],update,case,nfe,r['mean_rms'],r['power_ratio'][-1],flush=True)


def worker(a):
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():raise RuntimeError('GPU allocation required')
    torch.set_num_threads(4);root=Path(a.output);m=json.loads((root/'manifest.json').read_text())
    for p,h in m['sources'].items():
        if digest(root/'source'/p)!=h:raise ValueError('source drift')
    if digest(root/'train_prior_power.npy')!=m['power_sha256']:raise ValueError('transform drift')
    cfg=m['config'];data=Data(cfg,torch.device('cuda'));teacher=Teacher(data)
    transform=Spectral(np.load(root/'train_prior_power.npy')).cuda();white=WhitenedTeacher(data,transform)
    rank=int(os.environ.get('SLURM_PROCID','0'));owned=m['items'][rank::4]
    if a.smoke:owned=m['items'][:8]
    started=time.monotonic();timings=[]
    for item in owned:
        torch.manual_seed(item['seed']);model=ConditionalVDM(3,8,2,False).cuda()
        opt=torch.optim.Adam(model.parameters(),lr=3e-4);gen=torch.Generator(device='cuda').manual_seed(item['seed']+3000)
        out=root/'results'/item['name'];first=0
        if not a.smoke:
            out.mkdir(parents=True,exist_ok=True)
            if (out/'latest.pt').exists():
                saved=torch.load(out/'latest.pt',map_location='cuda',weights_only=False)
                if saved['sources']!=m['sources'] or saved['item']!=item or saved['power_sha256']!=m['power_sha256']:raise ValueError('resume drift')
                restore_training_state(model,opt,gen,saved);first=saved['update']
        tic=time.monotonic();losses=[];total=64 if a.smoke else m['updates']
        for update in range(first+1,total+1):
            if time.monotonic()-started>10500:raise TimeoutError('checkpointed bounded panel')
            x,c=data.batch(32,item['fixed'],gen)
            losses.append(train_step(model,opt,x,c,gen,item['arm'],item['exact'],transform,teacher,white))
            if not a.smoke and update%1024==0:
                state=dict(model=model.state_dict(),optimizer=opt.state_dict(),generator=gen.get_state(),
                    cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),sources=m['sources'],
                    power_sha256=m['power_sha256'],item=item,update=update)
                atomic_checkpoint(out/'latest.pt',state)
                with (out/'learning.jsonl').open('a') as f:f.write(json.dumps(dict(update=update,loss=float(np.mean(losses))))+'\n')
                print('TRAIN',item['name'],update,float(np.mean(losses)),flush=True);losses=[]
            if not a.smoke and update in m['checkpoints']:
                atomic_checkpoint(out/f'checkpoint_{update}.pt',state)
                evaluate(model,data,item,cfg,transform,out,update,512,[128])
        if a.smoke:
            torch.cuda.synchronize();timings.append(dict(item=item,seconds_per_update=(time.monotonic()-tic)/64))
        else:
            evaluate(model,data,item,cfg,transform,out,total,512,[128])
            evaluate(model,data,item,cfg,transform,out/'precision',total,2048,[128,256])
            atomic_json(out/'velocity_risk.json',velocity_risk(model,data,item,transform,teacher,white))
    atomic_json(root/('SMOKE.json' if a.smoke else f'worker_{rank}_COMPLETE.json'),dict(items=owned,
        sources=m['sources'],power_sha256=m['power_sha256'],job=os.environ['SLURM_JOB_ID'],
        seconds=time.monotonic()-started,timings=timings))


def collect(a):
    root=Path(a.output);m=json.loads((root/'manifest.json').read_text());rows=[]
    for rank in range(4):
        w=json.loads((root/f'worker_{rank}_COMPLETE.json').read_text())
        if w['sources']!=m['sources'] or w['power_sha256']!=m['power_sha256']:raise ValueError('worker drift')
    for item in m['items']:
        for case in (range(4) if item['fixed'] is None else [0]):
            for nfe in [128,256]:
                d=root/'results'/item['name']/'precision';p=d/f'evaluation_32768_{case}_{nfe}.json'
                r=json.loads(p.read_text());arr=np.load(d/f'draws_32768_{case}_{nfe}.npy',mmap_mode='r')
                if arr.shape!=(2048,512) or not np.isfinite(arr).all() or any(r[k]!=v for k,v in item.items()):raise ValueError('invalid result')
                if (r['case'],r['nfe'],r['update'])!=(case,nfe,32768):raise ValueError('wrong ensemble')
                rows.append(r)
    atomic_json(root/'COMPLETE.json',dict(rows=rows,sources=m['sources'],power_sha256=m['power_sha256']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','worker','collect'])
    p.add_argument('--output',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();globals()[a.command](a)
