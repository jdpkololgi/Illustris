"""Observation-only posterior draws with durable shared parent realizations.

Scientific draw identities, checkpoints, arrays and source bindings are checked
on every resume. Truth enters only the separately labelled oracle attribution.
"""
from dataclasses import replace
import argparse
import os
from pathlib import Path
import signal
import socket
import time

import numpy as np
import torch

from workflows.sbi import e2e_durable as durable, e2e_wide_pipeline as existing
from workflows.sbi.e2e_vdm_assessment import atomic_npz
from workflows.sbi.e2e_vdm_context_data import read_json, spec, output_root, ROLES
from workflows.sbi.e2e_vdm_context_dataset import Products, coarse_local_crop
from workflows.sbi.e2e_vdm_context_models import FineCondition, replicate, decode_density, coupled_sample
from workflows.sbi.e2e_vdm_context_train import verify_launch, new_model
from workflows.sbi.e2e_vdm_context_tasks import draw_tasks, task_seed, coarse_cache_key

MICROBATCH = 8
STOP = False


def stop_requested(signum,frame):
    global STOP
    STOP=True


def expand_condition(condition,size):
    expand=lambda x: None if x is None else x.expand(size,*x.shape[1:])
    return FineCondition(expand(condition.local),expand(condition.wide),expand(condition.offset_mpc_h),
                         expand(condition.coarse_local),expand(condition.coarse_wide),condition.coarse_source)


def with_coarse(condition,rho,chart,offset_raw,source):
    if rho.ndim!=5 or rho.shape[1:]!=(1,48,48,48) or not torch.isfinite(rho).all() or (rho<=0).any():
        raise ValueError('positive physical shared coarse cube required')
    condition=expand_condition(condition,len(rho))
    # Match training: convert physical coarse to FP32 before charting.
    standardized=(rho.float().log()-chart['coarse']['mean'])/chart['coarse']['std']
    sl=coarse_local_crop(offset_raw,(0,0,0))
    local=replicate(standardized[(slice(None),slice(None),*sl)])
    return replace(condition,coarse_local=local,coarse_wide=standardized,coarse_source=source)


def read_array_receipt(folder,receipt,binding,array_name):
    record=read_json(receipt)
    path=(folder/record['file']).resolve()
    if path.parent!=folder.resolve() or record['binding']!=binding or existing.sha256(path)!=record['sha256']:
        raise ValueError('array receipt/payload/binding mismatch')
    with np.load(path,allow_pickle=False) as values:
        array=values[array_name]
    if not np.isfinite(array).all():
        raise ValueError('nonfinite saved draws')
    return array,record


def load_model(root,arm,replica,checkpoint,factor,device):
    branch=root/'models'/f'{arm}_{factor}_seed{replica}'
    pointer=read_json(branch/f'CHECKPOINT_{checkpoint:06d}.json')
    path=(branch/pointer['path']).resolve()
    if branch.resolve() not in path.parents or existing.sha256(path)!=pointer['sha256']:
        raise ValueError('checkpoint reference drift')
    if read_json(path.parent/'COMMITTED.json')!=pointer:
        raise ValueError('uncommitted model checkpoint')
    receipt=read_json(path.with_suffix('.json'))
    state=torch.load(path,map_location='cpu',weights_only=False)
    binding=state['binding']
    if (state['stage']!=factor or state['method']!='vdm' or state['step']!=checkpoint
            or binding['arm']!=arm or binding['factor']!=factor or binding['seed']!=replica
            or binding['manifest_sha256']!=existing.sha256(root/'MANIFEST.json')
            or receipt['checkpoint_sha256']!=pointer['sha256']
            or receipt['binding_sha256']!=existing.digest(binding)):
        raise ValueError('model/step/source binding mismatch')
    model,optimizer,generator=new_model(spec(),replica,arm,factor,device)
    del optimizer,generator
    model.load_state_dict(state['model'])
    model.eval()
    return model,pointer['sha256']


class SharedParents:
    def __init__(self,root,observations,model,model_sha,device,oracle=None):
        if observations.targets:
            raise PermissionError('shared-parent generator needs an observation-only reader')
        self.root,self.observations,self.model,self.sha,self.device,self.oracle = root,observations,model,model_sha,device,oracle

    def sampled(self,task,ids):
        """Canonical batches prevent request order from changing numerical draws."""
        if task['arm']!='D' or not ids:
            raise ValueError('D draw identities required')
        arrays={}
        for first in sorted({(i//MICROBATCH)*MICROBATCH for i in ids}):
            draw_ids=list(range(first,first+MICROBATCH))
            key=coarse_cache_key('D',task['replica'],task['checkpoint'],task['domain'],first,task['steps'],task['purpose'])
            folder=self.root/'parents'/str(Path(key).parent)
            receipt=folder/f'{first:06d}.json'
            binding=dict(manifest_sha256=existing.sha256(self.root/'MANIFEST.json'),checkpoint_sha256=self.sha,
                domain=task['domain'],replica=task['replica'],checkpoint=task['checkpoint'],
                steps=task['steps'],purpose=task['purpose'],ids=draw_ids)
            with durable.single_writer(folder):
                if receipt.exists():
                    rho,_=read_array_receipt(folder,receipt,binding,'rho')
                else:
                    if STOP:
                        raise SystemExit(75)
                    started=time.monotonic()
                    condition=self.observations.condition(task['domain'],device=self.device).wide
                    z=coupled_sample(self.model,condition.expand(MICROBATCH,-1,-1,-1,-1),
                                     task['steps'],[task_seed(task,i,'coarse') for i in draw_ids])
                    chart=self.observations.chart['coarse']
                    rho=torch.exp(z.double()*chart['std']+chart['mean'])[:,0].cpu().numpy()
                    if not np.isfinite(rho).all() or np.min(rho)<=0:
                        raise FloatingPointError('nonfinite/nonpositive sampled coarse density')
                    path=atomic_npz(folder,f'{first:06d}',dict(rho=rho))
                    durable.publish_json(receipt,dict(file=path.name,sha256=existing.sha256(path),binding=binding,
                        seconds=time.monotonic()-started,job=os.environ.get('SLURM_JOB_ID'),node=socket.gethostname()))
                if rho.shape!=(MICROBATCH,48,48,48):
                    raise ValueError('cached parent batch shape drift')
                arrays.update(zip(draw_ids,rho))
        return torch.as_tensor(np.stack([arrays[i] for i in ids]),device=self.device)[:,None]

    def get(self,task,ids):
        mode=task['coarse_mode']
        if mode=='sampled':
            return self.sampled(task,ids)
        if mode=='fixed_mean':
            # Use exactly the32 registered joint parent draws, not extra fits/draws.
            reference=dict(task,purpose='joint',coarse_mode='sampled')
            rho=self.sampled(reference,list(range(spec()['draws_joint'])))
            return rho.mean(0,keepdim=True).expand(len(ids),-1,-1,-1,-1)
        if mode=='oracle_diagnostic':
            if self.oracle is None or not self.oracle.targets:
                raise PermissionError('explicit diagnostic-only truth reader required')
            rho=self.oracle.raw_targets(task['domain'])['coarse']
            return torch.as_tensor(rho,device=self.device)[None,None].expand(len(ids),-1,-1,-1,-1)
        raise ValueError('unregistered coarse source')


def case(root,task,model,checkpoint_sha,observations,parents=None,limit=None):
    if observations.targets:
        raise PermissionError('normal posterior generation cannot receive truth reader')
    folder=root/'draws'/task['task_id']
    count=task['count'] if limit is None else min(task['count'],limit)
    if count%MICROBATCH or task['start']%MICROBATCH:
        raise ValueError('registered draws use canonical eight-draw chunks')
    device=next(model.parameters()).device
    chart=observations.chart
    base=observations.condition(task['anchor'],device=device)
    offset=observations.rows[task['anchor']].get('core_offset_raw',[0,0,0])
    with durable.single_writer(folder):
        for first in range(task['start'],task['start']+count,MICROBATCH):
            ids=list(range(first,first+MICROBATCH))
            receipt=folder/f'{first:06d}.json'
            binding=dict(manifest_sha256=existing.sha256(root/'MANIFEST.json'),task=task,ids=ids,
                         checkpoint_sha256=checkpoint_sha,coarse_checkpoint_sha256=None if parents is None else parents.sha)
            if receipt.exists():
                values,_=read_array_receipt(folder,receipt,binding,'delta')
                if values.shape!=(MICROBATCH,48,48,48):
                    raise ValueError('saved fine batch shape drift')
                continue
            if STOP:
                raise SystemExit(75)
            started=time.monotonic()
            condition=expand_condition(base,MICROBATCH)
            if task['arm']=='D':
                rho_coarse=parents.get(task,ids)
                condition=with_coarse(base,rho_coarse,chart,offset,task['coarse_mode'])
            z=coupled_sample(model,condition,task['steps'],[task_seed(task,i,'fine') for i in ids])
            if task['arm']=='D':
                sl=coarse_local_crop(offset,(0,0,0))
                rho=decode_density(rho_coarse[(slice(None),slice(None),*sl)],z.double()*chart['residual']['std'])
            else:
                rho=torch.exp(z.double()*chart['fine']['std']+chart['fine']['mean'])
            if not torch.isfinite(rho).all() or (rho<=0).any():
                raise FloatingPointError('nonfinite/nonpositive sampled fine density')
            delta=(rho[:,0]-1).cpu().numpy()
            path=atomic_npz(folder,f'{first:06d}',dict(delta=delta))
            durable.publish_json(receipt,dict(file=path.name,sha256=existing.sha256(path),binding=binding,
                seconds=time.monotonic()-started,seeds=[task_seed(task,i,'fine') for i in ids],
                job=os.environ.get('SLURM_JOB_ID'),node=socket.gethostname()))
            print('DRAW_CHUNK',task['task_id'],first,time.monotonic()-started,flush=True)
        if count==task['count'] and not (folder/'COMPLETE.json').exists():
            receipts={str(i):existing.sha256(folder/f'{i:06d}.json') for i in range(task['start'],task['start']+count,MICROBATCH)}
            durable.publish_json(folder/'COMPLETE.json',dict(task=task,chunks=receipts,
                manifest_sha256=existing.sha256(root/'MANIFEST.json')))


def selected_tasks(ledger,arm,replica,mode):
    tasks=[t for t in ledger['tasks'] if t['arm']==arm and t['replica']==replica]
    if mode=='refinement':
        return [(t,8 if t['steps']==250 else None) for t in tasks if t['checkpoint']==spec()['updates']
            and t['purpose']=='main' and t['anchor'] in ledger['panels']['refinement']]
    if mode!='all':
        raise ValueError('unregistered sample mode')
    return [(t,None) for t in tasks]


def run(root,arm,replica,mode):
    global STOP
    STOP=False
    root=output_root(root)
    device=existing.runtime()
    verify_launch(root)
    ledger=read_json(root/'DRAW_LEDGER.json')
    expected=draw_tasks(read_json(root/'data/GEOMETRY.json')['rows'])
    if ledger!=expected:
        raise ValueError('draw ledger drift')
    release=root/'MODELS_FROZEN.json'
    if not release.exists():
        raise PermissionError('all ten model factors must be frozen before assessment')
    from workflows.sbi.e2e_vdm_context_control import verify_models_frozen
    verify_models_frozen(root)
    if mode=='all':
        refinement=read_json(root/'analysis/SAMPLER_GATE.json')
        if not refinement['passed'] or refinement['manifest_sha256']!=existing.sha256(root/'MANIFEST.json'):
            raise PermissionError('new-checkpoint sampler gate not passed')
    signal.signal(signal.SIGUSR1,stop_requested)
    signal.signal(signal.SIGTERM,stop_requested)
    phases=['ph004'] if mode=='refinement' else list(ROLES)
    observations=Products(root,phases,targets=False)
    oracle=None if mode=='refinement' or arm!='D' else Products(root,['ph004','ph005'],targets=True,confirmation_receipt=release)
    plan=selected_tasks(ledger,arm,replica,mode)
    active=None
    model=parents=None
    for task,limit in plan:
        checkpoint=task['checkpoint']
        if checkpoint!=active:
            if model is not None:
                del model,parents
            model,checksum=load_model(root,arm,replica,checkpoint,'fine',device)
            parents=None
            if arm=='D':
                coarse,coarse_sha=load_model(root,arm,replica,checkpoint,'coarse',device)
                parents=SharedParents(root,observations,coarse,coarse_sha,device,oracle)
            active=checkpoint
        case(root,task,model,checksum,observations,parents,limit)
    folder=root/'sampling'
    folder.mkdir(exist_ok=True)
    receipt=folder/f'{arm}_seed{replica}_{mode.upper()}_COMPLETE.json'
    if not receipt.exists():
        durable.publish_json(receipt,dict(manifest_sha256=existing.sha256(root/'MANIFEST.json'),
            arm=arm,replica=replica,mode=mode,tasks=len(plan)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True,type=Path)
    p.add_argument('--arm',required=True,choices=list('ABCD'))
    p.add_argument('--replica',required=True,type=int,choices=[0,1])
    p.add_argument('--mode',required=True,choices=['refinement','all'])
    a=p.parse_args()
    run(a.root,a.arm,a.replica,a.mode)


if __name__=='__main__':
    main()
