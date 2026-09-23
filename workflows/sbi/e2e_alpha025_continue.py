"""Plain continuation of the selected alpha, with physical DC error receipts."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import numpy as np
import torch
from workflows.sbi.e2e_conditional_reference import Data,atomic_json
from workflows.sbi.e2e_conditional_reference_continue import digest
from workflows.sbi.e2e_spectral_preconditioning import Spectral,evaluate
from workflows.sbi.e2e_partial_whitening import context,model_state,worker,diagnostics


def dc_error(draws,case):
    draws=np.asarray(draws,dtype=float)
    if draws.ndim!=2 or len(draws)<2 or not np.isfinite(draws).all():raise ValueError('invalid ensemble')
    dc=draws.mean(axis=1);truth=float(np.mean(case['mu']))
    variance=float(np.sum(case['sigma'])/draws.shape[1]**2)
    if variance<=0:raise ValueError('nonpositive DC variance')
    error=(float(dc.mean())-truth)/np.sqrt(variance)
    mcvar=float(dc.var(ddof=1)/len(dc)/variance)
    return dict(signed_error_posterior_sd=error,mc_standard_error_posterior_sd=np.sqrt(mcvar),
        raw_squared_error=error**2,mc_corrected_squared_error=error**2-mcvar,
        dc_power_ratio=float(dc.var(ddof=1)/variance),draws=len(draws))


def selected_items(manifest):
    rows=[i|dict(parent_name=i['name']) for i in manifest['items'] if i['alpha']==.25]
    if len(rows)!=8 or len({i['name'] for i in rows})!=8:raise ValueError('expected eight alpha=.25 parents')
    return rows


def prepare(a):
    root=Path(a.output);parent=Path(a.parent);old=json.loads((parent/'manifest.json').read_text())
    selection=json.loads((parent/'SELECTION.json').read_text())
    if selection['alpha']!=.25:raise ValueError('selected alpha differs')
    root.mkdir(parents=True,exist_ok=False);repo=Path(__file__).resolve().parents[2]
    panel=selected_items(old);parents={}
    for i in panel:
        p=parent/'results'/i['name']/'checkpoint_32768.pt';parents[str(p)]=digest(p)
    shutil.copy2(parent/'train_prior_power.npy',root/'train_prior_power.npy')
    if digest(root/'train_prior_power.npy')!=old['power_sha256']:raise ValueError('power mismatch')
    paths=list(old['sources'])+['workflows/sbi/e2e_alpha025_continue.py',
        'workflows/sbi/e2e_alpha025_continue_step.sh','tests/test_e2e_alpha025_continue.py',
        'docs/e2e_alpha025_continuation_20260923.md']
    sources={}
    for name in paths:
        src=repo/name;dst=root/'source'/name;dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,dst);sources[name]=digest(src)
    atomic_json(root/'manifest.json',dict(parent=str(parent),parents=parents,parent_sources=old['sources'],
        sources=sources,config=old['config'],items=panel,power_sha256=old['power_sha256'],
        selection_sha256=digest(parent/'SELECTION.json'),updates=[32768,49152,65536],
        evaluation_note='cases4-11 are reused longitudinal evaluation, not new confirmation',
        git_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()))


def assess(a):
    torch.set_num_threads(4);root=Path(a.output);m=context(root);data=Data(m['config'],torch.device('cuda'))
    transform=Spectral(np.load(root/'train_prior_power.npy'),.25).cuda()
    rank=int(os.environ.get('SLURM_PROCID',0));owned=m['items'][rank::4]
    for item in owned:
        model,_,_,update=model_state(m,item,root)
        if update!=65536:raise ValueError('incomplete continuation')
        out=root/'results'/item['name']/'longitudinal'
        if item['fixed'] is None:
            evaluate(model,data,item,m['config'],transform,out,update,2048,[128,256],range(4,12))
            diagnostics(model,data,item,transform,out,update,range(4,12))
        rows=[]
        for case in (range(12) if item['fixed'] is None else [0]):
            newdir=root/'results'/item['name']/('precision' if case<4 else 'longitudinal')
            if case<4:olddir=Path(m['parent'])/'results'/item['name']/'precision'
            else:olddir=Path(m['parent'])/'confirmation'/f'{item["name"]}_u32768'
            for step,folder in [(32768,olddir),(65536,newdir)]:
                path=folder/f'draws_{step}_{case}_256.npy'
                rows.append(dict(case=case,update=step,draw_sha256=digest(path),**dc_error(np.load(path),data.cases[case])))
        atomic_json(root/'results'/item['name']/'DC_MEAN.json',dict(rows=rows,item=item))
    atomic_json(root/f'assess_{rank}_COMPLETE.json',dict(sources=m['sources'],items=owned))


def collect(a):
    root=Path(a.output);m=context(root);rows=[];dc=[]
    for rank in range(4):
        for label in ['worker','assess']:
            receipt=json.loads((root/f'{label}_{rank}_COMPLETE.json').read_text())
            if receipt['sources']!=m['sources']:raise ValueError('receipt drift')
    for item in m['items']:
        for case in (range(12) if item['fixed'] is None else [0]):
            folder=root/'results'/item['name']/('precision' if case<4 else 'longitudinal')
            for nfe in [128,256]:
                p=folder/f'evaluation_65536_{case}_{nfe}.json';r=json.loads(p.read_text())
                arr=np.load(folder/f'draws_65536_{case}_{nfe}.npy',mmap_mode='r')
                if arr.shape!=(2048,512) or not np.isfinite(arr).all():raise ValueError('bad ensemble')
                if any(r[k]!=v for k,v in item.items()) or r['update']!=65536:raise ValueError('wrong metadata')
                rows.append(r)
        dc.append(json.loads((root/'results'/item['name']/'DC_MEAN.json').read_text()))
    if len(rows)!=104:raise ValueError('missing final ensembles')
    atomic_json(root/'COMPLETE.json',dict(rows=rows,dc=dc,sources=m['sources'],power_sha256=m['power_sha256']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','worker','assess','collect'])
    p.add_argument('--output',required=True);p.add_argument('--parent');p.add_argument('--smoke',action='store_true')
    p.add_argument('--deterministic-smoke',action='store_true');a=p.parse_args();globals()[a.command](a)
