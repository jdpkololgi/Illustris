"""Matched fresh-start development alpha sweep; no confirmation access."""
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
from workflows.sbi.e2e_stabilization import (Data, Teacher, Spectral, WhitenedTeacher,
    ConditionalVDM, train_step, update_ema, checkpoint, assess, atomic_json,
    atomic_checkpoint, digest, restore_training_state, learning_rate)
from workflows.sbi.e2e_partial_whitening import context, selection_score
from workflows.sbi.e2e_conditional_reference_math import oracle_moments
from workflows.sbi.e2e_spectral_bridge_diagnostics import transformed_case
from workflows.sbi.e2e_spectral_absorption import shell_power


def items():
    return [dict(alpha=a,seed=s,exact=e,fixed=None,arm='white_bridge',lr_mode='decay',
                 name=f'alpha{a:.2f}_{"exact" if e else "stochastic"}_seed{s}')
            for a in [.25,.30,.35,.40] for s in [17,29] for e in [False,True]]


def lr(step):
    return 3e-4 if step<=65536 else learning_rate('decay',step-65536)


def choose(rows):
    scores=[]
    for a in [.25,.30,.35,.40]:
        rr=[r for r in rows if r['alpha']==a and not r['exact'] and r['view']=='ema' and r['nfe']==256]
        keys={(r['seed'],r['case'],r['update']) for r in rr}
        expected={(s,c,u) for s in [17,29] for c in range(4) for u in [81920,98304]}
        if len(rr)!=16 or keys!=expected:raise ValueError('incomplete selection panel')
        scores.append(dict(alpha=a,worst=max(map(selection_score,rr)),
            average=float(np.mean(list(map(selection_score,rr)))),
            stable_pass=all(r['passed'] and r['power_pass'] for r in rr)))
    return dict(candidate=min(scores,key=lambda s:(s['worst'],s['average'],s['alpha'])),scores=scores,
                scope='development only; candidate is not confirmation or promotion')


def prepare(a):
    root=Path(a.output);parent=Path(a.parent);old=context(parent)
    repo=Path(__file__).resolve().parents[2];root.mkdir(parents=True,exist_ok=False)
    shutil.copy2(parent/'train_prior_power.npy',root/'train_prior_power.npy')
    paths=list(old['sources'])+['workflows/sbi/e2e_stable_alpha.py',
        'workflows/sbi/e2e_reference_usecase.py','workflows/sbi/e2e_stable_alpha_step.sh',
        'tests/test_e2e_stable_alpha.py','docs/e2e_stable_alpha_20260924.md']
    sources={}
    for name in paths:
        dst=root/'source'/name;dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(repo/name,dst);sources[name]=digest(dst)
    atomic_json(root/'manifest.json',dict(parent=str(parent),sources=sources,
        power_sha256=old['power_sha256'],config=old['config']|dict(cases=4),items=items(),
        ema_beta=.999,checkpoints=[81920,98304],updates=98304,scope='development_only',
        initialization='fresh matched per seed; no warm starts or coordinate switching',
        git_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()))


def worker(a):
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():raise RuntimeError('compute allocation required')
    torch.set_num_threads(4)
    if a.smoke:
        torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True
    root=Path(a.output);m=context(root);data=Data(m['config'],torch.device('cuda'));teacher=Teacher(data)
    rank=int(os.environ.get('SLURM_PROCID',0));owned=m['items'][rank::4]
    if a.smoke:owned=[i for i in m['items'] if i['seed']==17]
    started=time.monotonic();oracles=[]
    for item in owned:
        transform=Spectral(np.load(root/'train_prior_power.npy'),item['alpha']).cuda();white=WhitenedTeacher(data,transform)
        if a.smoke:
            for case in data.cases[:2]:
                cc,basis,w=transformed_case(case,transform.scale.cpu().numpy())
                result=oracle_moments(cc,'cfm',256)
                ratios=shell_power(result['variance'],basis,data.radius)/shell_power(case['values'],case['vectors'],data.radius)
                if max(abs(ratios-1))>.01:raise ValueError('oracle sampler floor')
                oracles.append(dict(alpha=item['alpha'],power_ratio=ratios.tolist()))
        torch.manual_seed(item['seed']);model=ConditionalVDM(3,8,2,False).cuda()
        opt=torch.optim.Adam(model.parameters(),lr=3e-4)
        gen=torch.Generator(device='cuda').manual_seed(item['seed']+3000)
        ema=copy.deepcopy(model).requires_grad_(False);first=0;out=root/'results'/item['name']
        if not a.smoke:
            out.mkdir(parents=True,exist_ok=True)
            if (out/'latest.pt').exists():
                state=torch.load(out/'latest.pt',map_location='cuda',weights_only=False)
                if state['sources']!=m['sources'] or state['item']!=item or state['power_sha256']!=m['power_sha256']:raise ValueError('resume drift')
                restore_training_state(model,opt,gen,state);ema.load_state_dict(state['ema']);first=state['update']
                if first in m['checkpoints'] and not (out/f'checkpoint_{first}.pt').exists():atomic_checkpoint(out/f'checkpoint_{first}.pt',state)
        losses=[];stop=64 if a.smoke else m['updates']
        for step in range(first+1,stop+1):
            for group in opt.param_groups:group['lr']=lr(step)
            x,c=data.batch(32,None,gen)
            losses.append(train_step(model,opt,x,c,gen,'white_bridge',item['exact'],transform,teacher,white))
            update_ema(ema,model)
            if not a.smoke and step%1024==0:
                state=checkpoint(model,ema,opt,gen,m,item,step);atomic_checkpoint(out/'latest.pt',state)
                with (out/'learning.jsonl').open('a') as f:f.write(json.dumps(dict(update=step,loss=float(np.mean(losses)),lr=lr(step)))+'\n')
                print('TRAIN',item['name'],step,float(np.mean(losses)),flush=True);losses=[]
                if step in m['checkpoints']:
                    atomic_checkpoint(out/f'checkpoint_{step}.pt',state)
                    assess(model,ema,data,teacher,white,transform,root,m,item,step)
        if a.smoke:
            state=copy.deepcopy(checkpoint(model,ema,opt,gen,m,item,stop))
            x,c=data.batch(32,None,gen);train_step(model,opt,x,c,gen,'white_bridge',item['exact'],transform,teacher,white);update_ema(ema,model)
            replay=copy.deepcopy(model);shadow=copy.deepcopy(ema);ropt=torch.optim.Adam(replay.parameters(),lr=3e-4);rg=torch.Generator(device='cuda')
            restore_training_state(replay,ropt,rg,state);shadow.load_state_dict(state['ema'])
            x,c=data.batch(32,None,rg);train_step(replay,ropt,x,c,rg,'white_bridge',item['exact'],transform,teacher,white);update_ema(shadow,replay)
            for one,two in [(model,replay),(ema,shadow)]:
                for p,q in zip(one.parameters(),two.parameters()):torch.testing.assert_close(p,q,rtol=1e-6,atol=1e-6)
            if not torch.equal(gen.get_state(),rg.get_state()):raise ValueError('RNG replay failed')
        else:
            for step in m['checkpoints']:
                if (out/'ema'/f'checkpoint_{step}'/'BATTERY.json').exists() and (out/'raw'/f'checkpoint_{step}'/'BATTERY.json').exists():continue
                state=torch.load(out/f'checkpoint_{step}.pt',map_location='cuda',weights_only=False)
                model.load_state_dict(state['model']);ema.load_state_dict(state['ema'])
                assess(model,ema,data,teacher,white,transform,root,m,item,step)
    atomic_json(root/('SMOKE.json' if a.smoke else f'worker_{rank}_COMPLETE.json'),dict(sources=m['sources'],items=owned,seconds=time.monotonic()-started,oracles=oracles,replay_checked=a.smoke))


def collect(a):
    root=Path(a.output);m=context(root);rows=[];battery=[]
    for rank in range(4):
        receipt=json.loads((root/f'worker_{rank}_COMPLETE.json').read_text())
        if receipt['sources']!=m['sources'] or receipt['items']!=m['items'][rank::4]:raise ValueError('worker mismatch')
    for item in m['items']:
        for view in ['raw','ema']:
            for step in m['checkpoints']:
                out=root/'results'/item['name']/view/f'checkpoint_{step}'
                battery.append(item|json.loads((out/'BATTERY.json').read_text()))
                for case in range(4):
                    for nfe in ([128,256] if step==98304 else [256]):
                        r=json.loads((out/f'evaluation_{step}_{case}_{nfe}.json').read_text())
                        arr=np.load(out/f'draws_{step}_{case}_{nfe}.npy',mmap_mode='r')
                        if arr.shape!=(2048,512) or not np.isfinite(arr).all() or any(r[k]!=v for k,v in item.items()):raise ValueError('ensemble mismatch')
                        rows.append(r|dict(view=view))
    if len(rows)!=384:raise ValueError('incomplete panel')
    selection=choose(rows)
    lookup={(r['name'],r['view'],r['case'],r['update'],r['nfe']):r for r in rows}
    nfe_changes=[dict(name=r['name'],view=r['view'],case=r['case'],
        max_shell_difference=max(abs(x-y) for x,y in zip(r['power_ratio'],lookup[(r['name'],r['view'],r['case'],98304,128)]['power_ratio'])))
        for r in rows if r['update']==98304 and r['nfe']==256]
    atomic_json(root/'COMPLETE.json',dict(rows=rows,battery=battery,selection=selection,nfe_changes=nfe_changes,sources=m['sources'],scope=m['scope']))
    lines=['# Stabilized alpha sweep: development only','',
        'Matched fresh starts; no confirmation. Both checkpoints and seeds enter selection.',
        '', '| Target | Alpha | Weights | Update | Mean | Covariance | DC | Top | Full gates |',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for ex in [False,True]:
        for alpha in [.25,.30,.35,.40]:
            for view in ['raw','ema']:
                for step in m['checkpoints']:
                    rr=[r for r in rows if (r['exact'],r['alpha'],r['view'],r['update'],r['nfe'])==(ex,alpha,view,step,256)]
                    avg=lambda k:float(np.mean([r[k] for r in rr]))
                    lines.append(f'| {"exact" if ex else "stochastic"} | {alpha} | {view} | {step} | {avg("mean_rms"):.5f} | {avg("covariance_relative"):.5f} | {np.mean([r["power_ratio"][0] for r in rr]):.5f} | {np.mean([r["power_ratio"][-1] for r in rr]):.5f} | {sum(r["passed"] and r["power_pass"] for r in rr)}/8 |')
    lines+=['','Selection: '+json.dumps(selection),
        'Maximum paired NFE shell difference: '+str(max(r['max_shell_difference'] for r in nfe_changes)),
        '','If no stable pass: stop alpha/optimizer tuning; do not automatically extend.']
    (root/'SUMMARY.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','worker','collect'])
    p.add_argument('--output',required=True);p.add_argument('--parent');p.add_argument('--smoke',action='store_true')
    a=p.parse_args();globals()[a.command](a)
