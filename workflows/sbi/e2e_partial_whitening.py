"""Registered alpha sweep, unchanged continuation, and sealed confirmation."""
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
from scipy.stats import chi2
from workflows.sbi.e2e_conditional_reference import Data,atomic_json,atomic_checkpoint
from workflows.sbi.e2e_conditional_reference_continue import digest,restore_training_state
from workflows.sbi.e2e_reference_target_controls import Teacher
from workflows.sbi.e2e_direct_vdm import ConditionalVDM
from workflows.sbi.e2e_spectral_preconditioning import Spectral,WhitenedTeacher,train_step,evaluate,velocity_risk
from workflows.sbi.e2e_spectral_absorption import draw_diagnostic,shell_power
from workflows.sbi.e2e_conditional_reference_math import oracle_moments
from workflows.sbi.e2e_spectral_bridge_diagnostics import transformed_case


def items():
    result=[]
    for alpha in [.2,.25,.3,.35,.5]:
        for seed in [17,29]:
            for fixed in [None,0]:
                for exact in [False,True]:
                    old=f'white_bridge_{"exact" if exact else "stochastic"}_seed{seed}_{"amortised" if fixed is None else "fixed0"}'
                    result.append(dict(alpha=alpha,seed=seed,fixed=fixed,exact=exact,arm='white_bridge',
                        name=f'alpha{alpha:.2f}_{old}',parent_name=old if alpha==.5 else None))
    return result


def selection_score(row):
    return max(row['mean_rms']/.1,row['covariance_relative']/.15,
        abs(row['variance_ratio']-1)/.1,abs(row['octant_coverage']-.9)/.05,
        max(abs(v-1) for v in row['power_ratio'])/.1)


def choose(rows):
    scores=[]
    for alpha in [.2,.25,.3,.35]:
        rr=[r for r in rows if r['alpha']==alpha and not r['exact'] and r['fixed'] is None and r['nfe']==256]
        if len(rr)!=8 or len({(r['seed'],r['case']) for r in rr})!=8:raise ValueError('incomplete development panel')
        v=[selection_score(r) for r in rr]
        scores.append(dict(alpha=alpha,worst=max(v),average=float(np.mean(v))))
    winner=min(scores,key=lambda r:(r['worst'],r['average'],r['alpha']))
    return dict(alpha=winner['alpha'],scores=scores,rule='stochastic amortised development only; no confirmation reselection')


def context(root):
    m=json.loads((root/'manifest.json').read_text())
    for p,h in m['sources'].items():
        if digest(root/'source'/p)!=h:raise ValueError('frozen source drift')
    if digest(root/'train_prior_power.npy')!=m['power_sha256']:raise ValueError('power drift')
    return m


def prepare(a):
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():raise RuntimeError('GPU allocation required')
    root=Path(a.output);parent=Path(a.parent);root.mkdir(parents=True,exist_ok=False)
    old=json.loads((parent/'manifest.json').read_text());repo=Path(__file__).resolve().parents[2]
    shutil.copy2(parent/'train_prior_power.npy',root/'train_prior_power.npy')
    if digest(root/'train_prior_power.npy')!=old['power_sha256']:raise ValueError('parent power mismatch')
    parents={}
    for name in {i['parent_name'] for i in items() if i['parent_name']}:
        path=parent/'results'/name/'checkpoint_32768.pt';parents[str(path)]=digest(path)
    paths=list(old['sources'])+['workflows/sbi/e2e_partial_whitening.py',
        'workflows/sbi/e2e_spectral_bridge_diagnostics.py','tests/test_e2e_partial_whitening.py',
        'docs/e2e_partial_whitening_20260923.md','workflows/sbi/e2e_partial_whitening_step.sh']
    sources={}
    for name in paths:
        src=repo/name;dst=root/'source'/name;dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,dst);sources[name]=digest(src)
    cfg=old['config']|dict(cases=12)
    atomic_json(root/'manifest.json',dict(parent=str(parent),parents=parents,parent_sources=old['sources'],
        sources=sources,config=cfg,items=items(),power_sha256=old['power_sha256'],
        development_cases=list(range(4)),confirmation_cases=list(range(4,12)),
        git_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()))
    data=Data(cfg,torch.device('cuda'));controls=[]
    power=np.load(root/'train_prior_power.npy')
    for alpha in [.2,.25,.3,.35,.5]:
        transform=Spectral(power,alpha).cuda()
        for index,c in enumerate(data.cases[:2]):
            cc,basis,w=transformed_case(c,transform.scale.cpu().numpy())
            for nfe in [128,256]:
                r=oracle_moments(cc,'cfm',nfe)
                ratios=shell_power(r['variance'],basis,data.radius)/shell_power(c['values'],c['vectors'],data.radius)
                if np.max(np.abs(ratios-1))>.01:raise ValueError('oracle shell floor exceeds1%')
                controls.append(dict(alpha=alpha,case=index,nfe=nfe,power_ratio=ratios.tolist()))
    atomic_json(root/'ORACLE.json',controls)


def parent_state(m,item):
    path=Path(m['parent'])/'results'/item['parent_name']/'checkpoint_32768.pt'
    if digest(path)!=m['parents'][str(path)]:raise ValueError('parent changed')
    s=torch.load(path,map_location='cuda',weights_only=False)
    if s['sources']!=m['parent_sources'] or s['power_sha256']!=m['power_sha256'] or s['update']!=32768:raise ValueError('parent provenance')
    for key in ['seed','fixed','exact','arm']:
        if s['item'][key]!=item[key]:raise ValueError('wrong parent')
    return s


def model_state(m,item,root,allow_resume=True):
    torch.manual_seed(item['seed']);model=ConditionalVDM(3,8,2,False).cuda()
    opt=torch.optim.Adam(model.parameters(),lr=3e-4)
    gen=torch.Generator(device='cuda').manual_seed(item['seed']+3000);start=0
    path=root/'results'/item['name']/'latest.pt'
    if allow_resume and path.exists():
        s=torch.load(path,map_location='cuda',weights_only=False)
        if s['sources']!=m['sources'] or s['item']!=item or s['power_sha256']!=m['power_sha256']:raise ValueError('resume mismatch')
        restore_training_state(model,opt,gen,s);start=s['update']
    elif item['parent_name']:
        s=parent_state(m,item);restore_training_state(model,opt,gen,s);start=32768
    return model,opt,gen,start


def diagnostics(model,data,item,transform,out,update,indices):
    for index in indices:
        path=out/f'draws_{update}_{index}_256.npy';draws=np.load(path)
        c,basis,w=transformed_case(data.cases[index],transform.scale.cpu().numpy())
        result=draw_diagnostic(draws@w.T,c,data.radius,basis)
        result['draw_sha256']=digest(path)
        result['dc_99_null_ratio']=(chi2.ppf([.005,.995],len(draws)-1)/(len(draws)-1)).tolist()
        atomic_json(out/f'bridge_absorption_{index}.json',result)


def worker(a):
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():raise RuntimeError('GPU allocation required')
    torch.set_num_threads(4);root=Path(a.output);m=context(root);cfg=m['config']
    data=Data(cfg,torch.device('cuda'));teacher=Teacher(data);power=np.load(root/'train_prior_power.npy')
    rank=int(os.environ.get('SLURM_PROCID',0))
    owned=[item for k,item in enumerate(m['items']) if (k//4+k%4)%4==rank]
    if a.smoke:owned=[i for i in m['items'] if i['seed']==17 and i['fixed'] is None]
    started=time.monotonic();timings=[]
    for item in owned:
        transform=Spectral(power,item['alpha']).cuda();white=WhitenedTeacher(data,transform)
        model,opt,gen,start=model_state(m,item,root,not a.smoke)
        total=65536 if item['parent_name'] else 32768
        points=[49152,65536] if item['parent_name'] else [8192,32768]
        out=root/'results'/item['name']
        if not a.smoke:out.mkdir(parents=True,exist_ok=True)
        tick=time.monotonic();losses=[]
        for update in range(start+1,(start+64 if a.smoke else total)+1):
            x,c=data.batch(32,item['fixed'],gen)
            losses.append(train_step(model,opt,x,c,gen,'white_bridge',item['exact'],transform,teacher,white))
            if not a.smoke and update%1024==0:
                s=dict(model=model.state_dict(),optimizer=opt.state_dict(),generator=gen.get_state(),
                    cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),sources=m['sources'],
                    power_sha256=m['power_sha256'],item=item,update=update)
                atomic_checkpoint(out/'latest.pt',s)
                with (out/'learning.jsonl').open('a') as f:f.write(json.dumps(dict(update=update,loss=float(np.mean(losses))))+'\n')
                print('TRAIN',item['name'],update,float(np.mean(losses)),flush=True);losses=[]
            if not a.smoke and update in points:
                atomic_checkpoint(out/f'checkpoint_{update}.pt',s)
                evaluate(model,data,item,cfg,transform,out,update,512,[128])
        if a.smoke:
            timings.append(dict(item=item,seconds_per_update=(time.monotonic()-tick)/64))
            if not all(torch.isfinite(p).all() for p in model.parameters()):raise ValueError('nonfinite smoke')
            if item['parent_name']:
                state=copy.deepcopy(dict(model=model.state_dict(),optimizer=opt.state_dict(),generator=gen.get_state(),
                    cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()))
                x,c=data.batch(32,item['fixed'],gen)
                loss=train_step(model,opt,x,c,gen,'white_bridge',item['exact'],transform,teacher,white)
                replay=copy.deepcopy(model);ropt=torch.optim.Adam(replay.parameters(),lr=3e-4)
                rgen=torch.Generator(device='cuda');restore_training_state(replay,ropt,rgen,state)
                xx,cc=data.batch(32,item['fixed'],rgen)
                rloss=train_step(replay,ropt,xx,cc,rgen,'white_bridge',item['exact'],transform,teacher,white)
                if abs(loss-rloss)>1e-6 or not torch.equal(gen.get_state(),rgen.get_state()):raise ValueError('restart replay diverged')
                for p,q in zip(model.parameters(),replay.parameters()):torch.testing.assert_close(p,q,atol=1e-6,rtol=1e-6)
                timings[-1]['restart_replay_pass']=True
        else:
            evaluate(model,data,item,cfg,transform,out,total,512,[128])
            evaluate(model,data,item,cfg,transform,out/'precision',total,2048,[128,256])
            diagnostics(model,data,item,transform,out/'precision',total,range(4) if item['fixed'] is None else [0])
            atomic_json(out/'velocity_risk.json',velocity_risk(model,data,item,transform,teacher,white))
    atomic_json(root/('SMOKE.json' if a.smoke else f'worker_{rank}_COMPLETE.json'),dict(items=owned,
        sources=m['sources'],power_sha256=m['power_sha256'],seconds=time.monotonic()-started,timings=timings))


def select(a):
    root=Path(a.output);m=context(root);rows=[]
    for rank in range(4):
        w=json.loads((root/f'worker_{rank}_COMPLETE.json').read_text())
        if w['sources']!=m['sources']:raise ValueError('worker drift')
    for item in m['items']:
        update=65536 if item['parent_name'] else 32768
        for index in (range(4) if item['fixed'] is None else [0]):
            for nfe in [128,256]:
                p=root/'results'/item['name']/'precision'/f'evaluation_{update}_{index}_{nfe}.json'
                r=json.loads(p.read_text())
                if any(r[k]!=v for k,v in item.items()) or r['draws']!=2048:raise ValueError('wrong result')
                rows.append(r)
    atomic_json(root/'DEVELOPMENT.json',dict(rows=rows,sources=m['sources']))
    result=choose(rows)|dict(development_sha256=digest(root/'DEVELOPMENT.json'))
    if (root/'SELECTION.json').exists() and json.loads((root/'SELECTION.json').read_text())!=result:raise ValueError('selection changed')
    atomic_json(root/'SELECTION.json',result)


def confirmation(a):
    torch.set_num_threads(4);root=Path(a.output);m=context(root)
    selected=json.loads((root/'SELECTION.json').read_text());alpha=selected['alpha']
    if selected['development_sha256']!=digest(root/'DEVELOPMENT.json'):raise ValueError('development drift')
    data=Data(m['config'],torch.device('cuda'));power=np.load(root/'train_prior_power.npy')
    panel=[]
    for i in m['items']:
        if i['fixed'] is None and i['alpha'] in [alpha,.5]:
            panel.append((i,65536 if i['alpha']==.5 else 32768,False))
            if i['alpha']==.5:panel.append((i,32768,True))
    rank=int(os.environ.get('SLURM_PROCID',0))
    for item,update,old in panel[rank::4]:
        model,opt,gen,_=model_state(m,item,root)
        if old:model.load_state_dict(parent_state(m,item)['model'])
        transform=Spectral(power,item['alpha']).cuda()
        out=root/'confirmation'/f'{item["name"]}_u{update}'
        evaluate(model,data,item,m['config'],transform,out,update,2048,[128,256],range(4,12))
        diagnostics(model,data,item,transform,out,update,range(4,12))
    atomic_json(root/f'confirmation_{rank}_COMPLETE.json',dict(selection_sha256=digest(root/'SELECTION.json'),sources=m['sources']))


def collect(a):
    root=Path(a.output);m=context(root);rows=[]
    for rank in range(4):
        r=json.loads((root/f'confirmation_{rank}_COMPLETE.json').read_text())
        if r['sources']!=m['sources'] or r['selection_sha256']!=digest(root/'SELECTION.json'):raise ValueError('confirmation drift')
    for path in sorted((root/'confirmation').glob('*/evaluation_*.json')):
        r=json.loads(path.read_text());arr=np.load(path.with_name(path.name.replace('evaluation_','draws_')).with_suffix('.npy'),mmap_mode='r')
        if arr.shape!=(2048,512) or not np.isfinite(arr).all():raise ValueError('invalid confirmation draws')
        rows.append(r)
    if len(rows)!=192:raise ValueError(f'incomplete confirmation {len(rows)}')
    atomic_json(root/'COMPLETE.json',dict(confirmation=rows,sources=m['sources'],selection=json.loads((root/'SELECTION.json').read_text())))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','worker','select','confirmation','collect'])
    p.add_argument('--output',required=True);p.add_argument('--parent');p.add_argument('--smoke',action='store_true')
    a=p.parse_args();globals()[a.command](a)
