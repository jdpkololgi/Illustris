"""Development-only constant/decayed LR x raw/EMA Gaussian controls."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
import torch
from workflows.sbi.e2e_conditional_reference import Data,atomic_json,atomic_checkpoint
from workflows.sbi.e2e_conditional_reference_continue import digest,restore_training_state
from workflows.sbi.e2e_direct_vdm import ConditionalVDM
from workflows.sbi.e2e_reference_target_controls import Teacher
from workflows.sbi.e2e_spectral_preconditioning import Spectral,WhitenedTeacher,train_step,evaluate,velocity_risk
from workflows.sbi.e2e_partial_whitening import context,diagnostics
from workflows.sbi.e2e_alpha025_continue import dc_error


def learning_rate(mode,step):
    if mode=='constant':return 3e-4
    if mode!='decay' or not 0<=step<=32768:raise ValueError('invalid LR schedule')
    return 3e-6+(3e-4-3e-6)*.5*(1+math.cos(math.pi*step/32768))


@torch.no_grad()
def update_ema(ema,model,beta=.999):
    for p,q in zip(ema.parameters(),model.parameters()):p.lerp_(q,1-beta)
    for p,q in zip(ema.buffers(),model.buffers()):p.copy_(q)


def offset_scatter(rows):
    result=[]
    for template in [0,1]:
        rr=[r for r in rows if r['case']%2==template]
        x=np.array([r['signed_error_posterior_sd'] for r in rr]);v=np.array([r['mc_standard_error_posterior_sd']**2 for r in rr])
        if not len(x):raise ValueError('missing template')
        offset=float(x.mean());scatter=float(np.mean((x-offset)**2));mse=float(np.mean(x*x))
        vm=float(v.sum()/len(x)**2)
        result.append(dict(template=template,cases=len(x),offset=offset,scatter_sd=math.sqrt(scatter),
            raw_mse=mse,offset_share=offset**2/mse if mse else None,
            mc_corrected_offset_squared=offset**2-vm,
            mc_corrected_scatter_squared=scatter-float(v.mean())+vm,
            caveat='MC correction assumes independent case ensembles; different cases use distinct seeds'))
    return result


def prepare(a):
    root=Path(a.output);parent=Path(a.parent);old=json.loads((parent/'manifest.json').read_text())
    root.mkdir(parents=True,exist_ok=False);repo=Path(__file__).resolve().parents[2]
    panel=[];parents={}
    for i in old['items']:
        if i['fixed'] is not None:continue
        path=parent/'results'/i['name']/'checkpoint_65536.pt';parents[str(path)]=digest(path)
        for mode in ['constant','decay']:
            panel.append(i|dict(name=i['name']+'_'+mode,parent_name=i['name'],lr_mode=mode))
    if len(panel)!=8:raise ValueError('expected four amortised parents')
    shutil.copy2(parent/'train_prior_power.npy',root/'train_prior_power.npy')
    if digest(root/'train_prior_power.npy')!=old['power_sha256']:raise ValueError('power drift')
    paths=list(old['sources'])+['workflows/sbi/e2e_stabilization.py','workflows/sbi/e2e_stabilization_step.sh',
        'tests/test_e2e_stabilization.py','docs/e2e_stabilization_20260924.md']
    sources={}
    for name in paths:
        src=repo/name;dst=root/'source'/name;dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,dst);sources[name]=digest(src)
    atomic_json(root/'manifest.json',dict(parent=str(parent),parents=parents,parent_sources=old['sources'],
        sources=sources,config=old['config']|dict(cases=4),items=panel,power_sha256=old['power_sha256'],
        ema_beta=.999,checkpoints=[81920,98304],scope='development_only',
        git_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()))


def load_fit(root,m,item):
    torch.manual_seed(item['seed']);model=ConditionalVDM(3,8,2,False).cuda()
    opt=torch.optim.Adam(model.parameters(),lr=3e-4);gen=torch.Generator(device='cuda')
    out=root/'results'/item['name'];resume=out/'latest.pt'
    if resume.exists():
        state=torch.load(resume,map_location='cuda',weights_only=False)
        if state['item']!=item or state['sources']!=m['sources'] or state['power_sha256']!=m['power_sha256']:raise ValueError('resume drift')
    else:
        path=Path(m['parent'])/'results'/item['parent_name']/'checkpoint_65536.pt'
        if digest(path)!=m['parents'][str(path)]:raise ValueError('parent drift')
        state=torch.load(path,map_location='cuda',weights_only=False)
        if state['update']!=65536 or state['sources']!=m['parent_sources'] or state['power_sha256']!=m['power_sha256']:raise ValueError('wrong parent')
        for key in ['alpha','seed','exact','fixed','arm']:
            if state['item'][key]!=item[key]:raise ValueError('parent mismatch')
    restore_training_state(model,opt,gen,state);ema=copy.deepcopy(model).requires_grad_(False)
    if resume.exists():ema.load_state_dict(state['ema'])
    return model,ema,opt,gen,state['update']


def checkpoint(model,ema,opt,gen,m,item,update):
    return dict(model=model.state_dict(),ema=ema.state_dict(),optimizer=opt.state_dict(),generator=gen.get_state(),
        cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),sources=m['sources'],item=item,
        update=update,power_sha256=m['power_sha256'],ema_beta=m['ema_beta'])


def assess(model,ema,data,teacher,white,transform,root,m,item,step):
    for view,net in [('raw',model),('ema',ema)]:
        out=root/'results'/item['name']/view/f'checkpoint_{step}'
        evaluate(net,data,item,m['config'],transform,out,step,2048,[128,256] if step==98304 else [256])
        diagnostics(net,data,item,transform,out,step,range(4))
        dc=[];absorption=[]
        for case in range(4):
            path=out/f'draws_{step}_{case}_256.npy'
            dc.append(dict(case=case,draw_sha256=digest(path),**dc_error(np.load(path),data.cases[case])))
            prediction=json.loads((out/f'bridge_absorption_{case}.json').read_text())
            metric=json.loads((out/f'evaluation_{step}_{case}_256.json').read_text())
            absorption.append(dict(case=case,observed_power_ratio=metric['power_ratio'],
                predicted_power_ratio=prediction['predicted_shell_ratio'],dc_99_null_ratio=prediction['dc_99_null_ratio']))
        atomic_json(out/'BATTERY.json',dict(dc=dc,offset_scatter=offset_scatter(dc),
            absorption=absorption,velocity_risk=velocity_risk(net,data,item,transform,teacher,white),view=view,step=step))


def worker(a):
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():raise RuntimeError('GPU allocation required')
    if a.smoke:
        torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True
    torch.set_num_threads(4);root=Path(a.output);m=context(root);data=Data(m['config'],torch.device('cuda'))
    teacher=Teacher(data);transform=Spectral(np.load(root/'train_prior_power.npy'),.25).cuda();white=WhitenedTeacher(data,transform)
    rank=int(os.environ.get('SLURM_PROCID',0));owned=m['items'][rank::4]
    if a.smoke:owned=m['items'][:4]
    started=time.monotonic()
    for item in owned:
        model,ema,opt,gen,first=load_fit(root,m,item);out=root/'results'/item['name']
        if not a.smoke:
            out.mkdir(parents=True,exist_ok=True)
            if first in m['checkpoints'] and not (out/f'checkpoint_{first}.pt').exists():
                atomic_checkpoint(out/f'checkpoint_{first}.pt',checkpoint(model,ema,opt,gen,m,item,first))
        stop=first+64 if a.smoke else 98304;losses=[]
        for step in range(first+1,stop+1):
            for group in opt.param_groups:group['lr']=learning_rate(item['lr_mode'],step-65536)
            x,c=data.batch(32,None,gen)
            losses.append(train_step(model,opt,x,c,gen,'white_bridge',item['exact'],transform,teacher,white))
            update_ema(ema,model,m['ema_beta'])
            if not a.smoke and step%1024==0:
                state=checkpoint(model,ema,opt,gen,m,item,step);atomic_checkpoint(out/'latest.pt',state)
                with (out/'learning.jsonl').open('a') as f:f.write(json.dumps(dict(update=step,loss=float(np.mean(losses)),lr=opt.param_groups[0]['lr']))+'\n')
                print('TRAIN',item['name'],step,float(np.mean(losses)),flush=True);losses=[]
            if not a.smoke and step in m['checkpoints']:
                atomic_checkpoint(out/f'checkpoint_{step}.pt',state)
                assess(model,ema,data,teacher,white,transform,root,m,item,step)
        if a.smoke:
            saved=copy.deepcopy(checkpoint(model,ema,opt,gen,m,item,stop))
            x,c=data.batch(32,None,gen);train_step(model,opt,x,c,gen,'white_bridge',item['exact'],transform,teacher,white);update_ema(ema,model)
            replay=copy.deepcopy(model);re=copy.deepcopy(ema);ropt=torch.optim.Adam(replay.parameters(),lr=3e-4);rg=torch.Generator(device='cuda')
            restore_training_state(replay,ropt,rg,saved);re.load_state_dict(saved['ema'])
            xx,cc=data.batch(32,None,rg);train_step(replay,ropt,xx,cc,rg,'white_bridge',item['exact'],transform,teacher,white);update_ema(re,replay)
            for aa,bb in [(model,replay),(ema,re)]:
                for p,q in zip(aa.parameters(),bb.parameters()):torch.testing.assert_close(p,q,rtol=1e-6,atol=1e-6)
            if not torch.equal(gen.get_state(),rg.get_state()):raise ValueError('RNG replay failed')
        else:
            # Idempotently finish interrupted checkpoint assessments on resume.
            for step in m['checkpoints']:
                state=torch.load(out/f'checkpoint_{step}.pt',map_location='cuda',weights_only=False)
                model.load_state_dict(state['model']);ema.load_state_dict(state['ema'])
                assess(model,ema,data,teacher,white,transform,root,m,item,step)
    atomic_json(root/('SMOKE.json' if a.smoke else f'worker_{rank}_COMPLETE.json'),dict(sources=m['sources'],items=owned,
        seconds=time.monotonic()-started,replay_checked=a.smoke))


def collect(a):
    root=Path(a.output);m=context(root);rows=[];battery=[]
    for rank in range(4):
        if json.loads((root/f'worker_{rank}_COMPLETE.json').read_text())['sources']!=m['sources']:raise ValueError('worker drift')
    for item in m['items']:
        for view in ['raw','ema']:
            for step in m['checkpoints']:
                out=root/'results'/item['name']/view/f'checkpoint_{step}'
                battery.append(item|json.loads((out/'BATTERY.json').read_text()))
                for case in range(4):
                    for nfe in ([128,256] if step==98304 else [256]):
                        r=json.loads((out/f'evaluation_{step}_{case}_{nfe}.json').read_text())
                        arr=np.load(out/f'draws_{step}_{case}_{nfe}.npy',mmap_mode='r')
                        if arr.shape!=(2048,512) or not np.isfinite(arr).all() or any(r[k]!=v for k,v in item.items()):raise ValueError('invalid ensemble')
                        rows.append(r|dict(view=view))
    stable=[]
    for exact in [False,True]:
        for mode in ['constant','decay']:
            for view in ['raw','ema']:
                rr=[r for r in rows if r['exact']==exact and r['lr_mode']==mode and r['view']==view and r['nfe']==256]
                stable.append(dict(exact=exact,lr_mode=mode,view=view,cells=len(rr),
                    all_gates_both_checkpoints=len(rr)==16 and all(r['passed'] and r['power_pass'] for r in rr)))
    if len(rows)!=192:raise ValueError('incomplete panel')
    lookup={(r['name'],r['view'],r['case'],r['update'],r['nfe']):r for r in rows}
    nfe_changes=[dict(name=r['name'],view=r['view'],case=r['case'],
        max_shell_difference=max(abs(x-y) for x,y in zip(r['power_ratio'],
            lookup[(r['name'],r['view'],r['case'],98304,128)]['power_ratio'])))
        for r in rows if r['update']==98304 and r['nfe']==256]
    atomic_json(root/'COMPLETE.json',dict(rows=rows,battery=battery,stability=stable,
        nfe_changes=nfe_changes,sources=m['sources'],scope='development_only'))
    lines=['# Stabilization: exploratory development readout','',
        'Four development observations, two seeds. No confirmation or promotion claim.',
        'Covariance is16-probe error; power ratios should be1. All gates include every shell.','',
        '| Target | LR | Weights | Update | Mean | Covariance | DC power | Top power | Full gates |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for ex in [False,True]:
        for mode in ['constant','decay']:
            for view in ['raw','ema']:
                for step in m['checkpoints']:
                    rr=[r for r in rows if (r['exact'],r['lr_mode'],r['view'],r['update'],r['nfe'])==(ex,mode,view,step,256)]
                    avg=lambda key:float(np.mean([r[key] for r in rr]))
                    lines.append(f'| {"exact" if ex else "stochastic"} | {mode} | {view} | {step} | {avg("mean_rms"):.5f} | {avg("covariance_relative"):.5f} | {np.mean([r["power_ratio"][0] for r in rr]):.5f} | {np.mean([r["power_ratio"][-1] for r in rr]):.5f} | {sum(r["passed"] and r["power_pass"] for r in rr)}/{len(rr)} |')
    lines+=['','## Stability across both checkpoints','']
    lines += [f'- target_exact={r["exact"]}, LR={r["lr_mode"]}, weights={r["view"]}: {r["all_gates_both_checkpoints"]}' for r in stable]
    lines+=['','## DC decomposition (posterior-sd units)','',
        '| Fit | Weights | Update | Template | Offset | Scatter | Offset share |',
        '|---|---|---:|---:|---:|---:|---:|']
    for b in battery:
        for d in b['offset_scatter']:
            lines.append(f'| {b["name"]} | {b["view"]} | {b["step"]} | {d["template"]} | {d["offset"]:.5f} | {d["scatter_sd"]:.5f} | {d["offset_share"]} |')
    lines+=['','Only two observations/template; scatter estimates are weakly replicated.',
        f'Maximum paired128/256NFE shell change: {max(r["max_shell_difference"] for r in nfe_changes):.6g}.',
        'COMPLETE.json also contains MC terms, observed/predicted absorption, DC nulls and exact-target velocity risks.']
    (root/'SUMMARY.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','worker','collect'])
    p.add_argument('--output',required=True);p.add_argument('--parent');p.add_argument('--smoke',action='store_true')
    a=p.parse_args();globals()[a.command](a)
