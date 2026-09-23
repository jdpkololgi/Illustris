"""Adaptive, separately archived convergence check of the four amortised affine fits."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from workflows.sbi.e2e_reference_target_controls import Teacher, LearnedAffine, step, excess_risk
from workflows.sbi.e2e_conditional_reference import Data, atomic_json, atomic_checkpoint, evaluate
from workflows.sbi.e2e_conditional_reference_continue import digest, restore_training_state
from workflows.sbi.e2e_conditional_reference_math import null_thresholds


def prepare(a):
    root=Path(a.output); parent=Path(a.parent)
    m=json.loads((parent/'manifest.json').read_text())
    selected=[i for i in m['items'] if i['fixed'] is None and i['branch'].startswith('affine')]
    assert len(selected)==4
    parents={i['name']:digest(parent/'results'/i['name']/'checkpoint_16384.pt') for i in selected}
    root.mkdir(parents=True,exist_ok=False)
    shutil.copytree(parent/'source',root/'source')
    this=Path(__file__); dest=root/'source/workflows/sbi'/this.name
    shutil.copy2(this,dest)
    sources=m['sources']|{f'workflows/sbi/{this.name}':digest(this)}
    atomic_json(root/'manifest.json',dict(parent=str(parent.resolve()),parent_sources=m['sources'],
        parents=parents,sources=sources,items=selected,config=m['config'],start=16384,stop=65536,
        adaptive=True,reason='Avoid interpreting unconverged fresh affine fits as representation failures.'))


def worker(a):
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('approved GPU allocation required')
    torch.set_num_threads(4)
    root=Path(a.output);m=json.loads((root/'manifest.json').read_text())
    for relative,h in m['sources'].items():
        if digest(root/'source'/relative)!=h: raise ValueError('source drift')
    rank=int(os.environ.get('SLURM_PROCID','0'))
    item=m['items'][rank];cfg=m['config']
    data=Data(cfg,torch.device('cuda'));teacher=Teacher(data)
    model=LearnedAffine(teacher).cuda();opt=torch.optim.Adam(model.parameters(),lr=3e-4)
    gen=torch.Generator(device='cuda')
    p=Path(m['parent'])/'results'/item['name']/'checkpoint_16384.pt'
    if digest(p)!=m['parents'][item['name']]: raise ValueError('parent drift')
    saved=torch.load(p,map_location='cuda',weights_only=False)
    if saved['update']!=16384 or saved['sources']!=m['parent_sources'] or saved['item']!=item:
        raise ValueError('parent mismatch')
    out=root/'results'/item['name'];out.mkdir(parents=True,exist_ok=True)
    if (out/'latest.pt').exists():
        saved=torch.load(out/'latest.pt',map_location='cuda',weights_only=False)
        if saved['sources']!=m['sources'] or saved['item']!=item: raise ValueError('resume mismatch')
    restore_training_state(model,opt,gen,saved)
    started=time.monotonic()
    def guard():
        if time.monotonic()-started>1800: raise TimeoutError('adaptive check bound')
    for update in range(saved['update']+1,m['stop']+1):
        guard();x,c=data.batch(32,None,gen)
        loss=step(model,opt,teacher,x,c,gen,item['branch']=='affine_teacher')
        if update%1024==0:
            state=dict(model=model.state_dict(),optimizer=opt.state_dict(),generator=gen.get_state(),
                cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),update=update,
                sources=m['sources'],item=item,parent_sha256=m['parents'][item['name']])
            atomic_checkpoint(out/'latest.pt',state)
            print('TRAIN',item['name'],update,loss,flush=True)
        if update in [32768,65536]:
            atomic_checkpoint(out/f'checkpoint_{update}.pt',state)
            evaluate(model,'cfm',data,range(4),cfg|{'nfe':[128]},item['seed'],update,out,guard)
    precise=copy.copy(data)
    precise.nulls=[null_thresholds(c,data.q,2048,128) for c in data.cases]
    final=out/'precision';final.mkdir(exist_ok=True)
    evaluate(model,'cfm',precise,range(4),cfg|{'draws':2048,'nfe':[256]},item['seed'],65536,final,guard)
    atomic_json(final/'risk.json',excess_risk(model,teacher,data,None,item['seed']))
    atomic_json(final/'nulls.json',precise.nulls)
    atomic_json(root/f'worker_{rank}_COMPLETE.json',dict(item=item,sources=m['sources'],
        job=os.environ['SLURM_JOB_ID'],seconds=time.monotonic()-started))


def collect(a):
    root=Path(a.output);m=json.loads((root/'manifest.json').read_text())
    rows=[]
    for rank,item in enumerate(m['items']):
        w=json.loads((root/f'worker_{rank}_COMPLETE.json').read_text())
        if w['sources']!=m['sources'] or w['item']!=item: raise ValueError('worker mismatch')
        d=root/'results'/item['name']/'precision'
        for case in range(4):
            r=json.loads((d/f'evaluation_65536_{case}_256.json').read_text())
            if (r['case'],r['update'],r['nfe'],r['seed'])!=(case,65536,256,item['seed']):
                raise ValueError('evaluation mismatch')
            arr=np.load(d/f'draws_65536_{case}_256.npy',mmap_mode='r')
            if arr.shape!=(2048,512) or not np.isfinite(arr).all(): raise ValueError('invalid draws')
            rows.append(r|item|dict(power_pass=all(.9<=v<=1.1 for v in r['power_ratio']),
                risk=json.loads((d/'risk.json').read_text())))
    atomic_json(root/'COMPLETE.json',dict(rows=rows,sources=m['sources'],adaptive=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','worker','collect'])
    p.add_argument('--parent');p.add_argument('--output',required=True)
    a=p.parse_args();globals()[a.command](a)
