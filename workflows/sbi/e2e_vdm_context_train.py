"""Restartable fixed-exposure fits for the approved four-arm VDM matrix.

The main entrypoint remains gated by full-size GPU smoke and physical receipts.
No submission, data selection, evaluation-driven stopping, or budget decisions.
"""
from dataclasses import fields
import argparse
import json
import os
import random
from pathlib import Path
import signal
import socket
import time

import h5py
import numpy as np
import torch

from workflows.sbi import e2e_durable as durable
from workflows.sbi import e2e_wide_pipeline as existing
from workflows.sbi.e2e_direct_vdm import ConditionalVDM
from workflows.sbi.e2e_vdm_context_data import spec, CONFIG, output_root, read_json, address
from workflows.sbi.e2e_vdm_context_dataset import Products, context_crop, coarse_local_crop
from workflows.sbi.e2e_vdm_context_models import ContextVDM, FineCondition, encode_density, replicate, loss

STOP = False


def request_stop(signum, frame):
    global STOP
    STOP = True


def new_model(c, seed, arm, factor, device):
    if seed not in c['replicas'] or arm not in c['arms'] or factor not in ('fine','coarse'):
        raise ValueError('unregistered model branch')
    if factor == 'coarse' and arm != 'D':
        raise ValueError('only D has a coarse factor')
    # Same fine initialization for all information arms; separate coarse factor.
    initialization=address(c['seed'],seed,factor,'initialization')
    random.seed(initialization)
    np.random.seed(initialization%(2**32))
    torch.manual_seed(initialization)
    model = (ContextVDM(arm,base=c['unet_base'],levels=c['unet_levels']) if factor=='fine'
             else ConditionalVDM(condition_channels=12,base=c['unet_base'],levels=c['unet_levels'],learned=False))
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(),lr=c['learning_rate'],weight_decay=c['weight_decay'])
    gen = torch.Generator(device=device).manual_seed(address(c['seed'],seed,factor,'objective'))
    return model,opt,gen


class TrainingCache:
    """Raw train-only CPU arrays, not seven duplicate normalized GPU datasets."""
    def __init__(self, root, arm):
        self.c, self.arm = spec(), arm
        phases = ['ph000','ph002'] if arm=='A' else ['ph000','ph002','ph003']
        ds = Products(root,phases,targets=True)
        if ds.chart is None:
            raise ValueError('normalization missing')
        self.chart,self.receipts = ds.chart,ds.receipts
        self.ids = sorted(k for k,r in ds.rows.items() if arm!='A' or r['small_train'])
        if len(self.ids) != (32 if arm=='A' else 384):
            raise ValueError('training diversity quota mismatch')
        self.items = {}
        for anchor in self.ids:
            row = ds.rows[anchor]
            if row['role'] != 'train':
                raise PermissionError('nontraining field in fit cache')
            with h5py.File(ds.root/'data'/row['phase']/f"{row['cap']}_observations.h5",'r') as f:
                local = torch.from_numpy(f[anchor]['local'][:])
                wide = torch.from_numpy(f[anchor]['wide_extended'][:])
            with h5py.File(ds.root/'data'/row['phase']/'targets.h5','r') as f:
                rho = torch.from_numpy(f[anchor]['rho_parent'][:])[None,None]
                coarse = torch.from_numpy(f[anchor]['coarse_rho_extended'][:]).float()
            _,u = encode_density(rho)
            self.items[anchor] = dict(local=local,wide=wide,logrho=rho.log()[0].float(),
                                      u=u[0].float(),coarse=coarse)
        self.host_bytes = sum(x.numel()*x.element_size() for item in self.items.values() for x in item.values())

    def batch(self, update, seed, factor, device):
        x,local,wide,offsets,cl,cw = [],[],[],[],[],[]
        n,c = self.chart,self.c
        mean = lambda key: torch.tensor(n[key]['mean'])[:,None,None,None]
        std = lambda key: torch.tensor(n[key]['std'])[:,None,None,None]
        lm,ls,wm,ws = mean('local'),std('local'),mean('wide'),std('wide')
        sm,ss = mean('summary'),std('summary')
        for j in range(c['batch_size']):
            presentation = update*c['batch_size']+j
            index = existing.example_index(presentation,len(self.ids),c['seed']+seed)
            offset = c['context_offsets_raw'][address(c['seed'],seed,presentation,'context-offset')%7]
            item = self.items[self.ids[index]]
            wc = item['wide'][(slice(None),*context_crop(offset))]
            wide.append((wc-wm)/ws)
            offsets.append(-np.asarray(offset)*c['raw_cell_mpc']*c['coordinate_h'])
            if factor=='coarse':
                density = item['coarse'][context_crop(offset)]
                x.append(((density.log()-n['coarse']['mean'])/n['coarse']['std'])[None])
                continue
            summary = (wc.mean((1,2,3),keepdim=True)-sm)/ss
            local.append(torch.cat([(item['local']-lm)/ls,summary.expand(-1,48,48,48)]))
            if self.arm=='D':
                density = item['coarse'][context_crop(offset)]
                standardized = (density.log()-n['coarse']['mean'])/n['coarse']['std']
                cw.append(standardized[None])
                part = standardized[coarse_local_crop((0,0,0),offset)]
                cl.append(replicate(part[None,None])[0])
                x.append(item['u']/n['residual']['std'])
            else:
                x.append((item['logrho']-n['fine']['mean'])/n['fine']['std'])
        batch = lambda values: torch.stack(values).to(device)
        targets = batch(x)
        if factor=='coarse':
            return targets,batch(wide)
        cond = FineCondition(batch(local),batch(wide),torch.as_tensor(np.array(offsets),device=device,dtype=torch.float32),
            batch(cl) if cl else None,batch(cw) if cw else None,'training_truth' if cl else None)
        return targets,cond


def update_model(model,opt,gen,x,condition,c,update):
    model.train()
    opt.zero_grad(set_to_none=True)
    value,terms = loss(model,x,condition,gen,c['decoder_std'])
    if not torch.isfinite(value):
        raise FloatingPointError('nonfinite VLB')
    value.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(),c['gradient_clip'],error_if_nonfinite=True)
    opt.step()
    return dict(update=update+1,loss=float(value.detach()),gradient_norm=float(norm),
        **{k:float(v.detach()) if isinstance(v,torch.Tensor) else float(v) for k,v in terms.items()})


def restore(branch,binding,model,opt,gen,factor):
    receipt = read_json(branch/'LATEST.json')
    path = (branch/receipt['path']).resolve()
    if branch.resolve() not in path.parents or path.name != 'state.pt':
        raise ValueError('checkpoint path escapes branch')
    if read_json(path.parent/'COMMITTED.json') != receipt or existing.sha256(path) != receipt['sha256']:
        raise ValueError('uncommitted/corrupt latest checkpoint; no silent fallback')
    state = existing.load_checkpoint(path,binding,stage=factor,method='vdm')
    if state['step'] != receipt['step'] or len(state['history']) != state['step']:
        raise ValueError('checkpoint history/step mismatch')
    model.load_state_dict(state['model'])
    opt.load_state_dict(state['optimizer'])
    existing.restore_rng(state['rng'],gen)
    return state['step'],state['history'],receipt


def checkpoint(branch,model,opt,gen,binding,factor,step,history):
    return durable.save(branch,model=model,optimizer=opt,generator=gen,binding=binding,
                        stage=factor,method='vdm',step=step,history=history)


def verify_manifest(root):
    manifest = read_json(root/'MANIFEST.json')
    if manifest.get('schema') != 'e2e-vdm-context-run-v1' or manifest['config_sha256'] != existing.sha256(CONFIG):
        raise ValueError('frozen full-run manifest missing/mismatched')
    source = Path(manifest['source']).resolve()
    if source != Path(__file__).resolve().parents[2]:
        raise ValueError('run only from full frozen source')
    for name,digest in manifest['source_sha256'].items():
        if existing.sha256(source/name) != digest:
            raise ValueError('source drift: '+name)
    for file,expected in manifest['data_receipts'].items():
        if existing.sha256(root/file) != expected:
            raise ValueError('data receipt drift')
    from workflows.sbi.e2e_vdm_context_physics import verify_representation_release
    verify_representation_release(root)
    return manifest


def verify_launch(root):
    manifest = verify_manifest(root)
    smoke = read_json(root/'SMOKE.json')
    if (not smoke['passed']
            or smoke['manifest_sha256'] != existing.sha256(root/'MANIFEST.json')
            or smoke['forecast_gpu_hours'] > spec()['budget']['gpu_hours']):
        raise PermissionError('physical/replay/cost smoke gate not passed')
    restart=read_json(root/'RESTART_TEST.json')
    if not restart['passed'] or not restart['exact'] or restart['manifest_sha256']!=existing.sha256(root/'MANIFEST.json'):
        raise PermissionError('actual process signal/restart gate not passed')
    return manifest


def train(root,arm,seed,factor,restart_test=None):
    global STOP
    STOP = False
    root = output_root(root)
    device = existing.runtime()
    if restart_test is None:
        verify_launch(root)
    else:
        verify_manifest(root)
        if restart_test not in ('baseline','pause','resume') or (arm,seed,factor)!=('A',0,'fine'):
            raise ValueError('only registered A0 twelve-update restart smoke allowed')
    c = spec()
    if restart_test is not None:
        c=dict(c,updates=12,checkpoint_every=4,checkpoint_updates=[4,8,12])
    data = TrainingCache(root,arm)
    model,opt,gen = new_model(c,seed,arm,factor,device)
    branch = root/'models'/f'{arm}_{factor}_seed{seed}'
    if restart_test is not None:
        branch=root/'restart_smoke'/('baseline' if restart_test=='baseline' else 'interrupted')
    binding = dict(manifest_sha256=existing.sha256(root/'MANIFEST.json'),arm=arm,factor=factor,
        seed=seed,normalization_sha256=existing.sha256(root/'data/NORMALIZATION.json'),products=data.receipts)
    if restart_test is not None:
        binding['technical_restart_test']=True
    signal.signal(signal.SIGUSR1,request_stop)
    signal.signal(signal.SIGTERM,request_stop)
    started = time.monotonic()
    with durable.single_writer(branch):
        step,history = 0,[]
        if (branch/'LATEST.json').exists():
            step,history,last = restore(branch,binding,model,opt,gen,factor)
        elif any(p.name != 'WRITER.lock' for p in branch.iterdir()):
            raise ValueError('uncommitted branch state preserved; recovery review required')
        else:
            # Commit exact initialization, allowing a restart even before update256.
            last = checkpoint(branch,model,opt,gen,binding,factor,0,history)
        if (branch/'COMPLETE.json').exists():
            complete = read_json(branch/'COMPLETE.json')
            if step != c['updates'] or complete['binding'] != binding or complete['checkpoint'] != last:
                raise ValueError('completion/checkpoint mismatch')
            print('VERIFIED_COMPLETE',arm,factor,seed,flush=True)
            return
        if STOP:
            durable.publish_json(branch/f'PAUSE_{step:06d}_{os.environ["SLURM_JOB_ID"]}.json',
                dict(checkpoint=last,binding=binding,clean=True))
            raise SystemExit(75)
        while step < c['updates']:
            x,condition = data.batch(step,seed,factor,device)
            history.append(update_model(model,opt,gen,x,condition,c,step))
            step += 1
            if step == 1 or step%128==0:
                print('TRAIN',arm,factor,seed,history[-1],'seconds',time.monotonic()-started,flush=True)
            if step%c['checkpoint_every']==0 or step in c['checkpoint_updates'] or STOP:
                last = checkpoint(branch,model,opt,gen,binding,factor,step,history)
                if step in c['checkpoint_updates']:
                    durable.publish_json(branch/f'CHECKPOINT_{step:06d}.json',last)
            if restart_test=='pause' and step==4:
                durable.publish_json(branch/'SIGNAL_READY.json',dict(pid=os.getpid(),checkpoint=last))
                deadline=time.monotonic()+60
                while not STOP and time.monotonic()<deadline:
                    time.sleep(.05)
                if not STOP:
                    raise RuntimeError('restart smoke did not receive the actual signal')
            if STOP and step < c['updates']:
                durable.publish_json(branch/f'PAUSE_{step:06d}_{os.environ["SLURM_JOB_ID"]}.json',
                    dict(checkpoint=last,binding=binding,clean=True))
                raise SystemExit(75)
        durable.publish_json(branch/'COMPLETE.json',dict(binding=binding,checkpoint=last,updates=step,
            examples_seen=step*c['batch_size'],distinct_patches=len(data.ids),
            independent_training_phases=len(data.receipts),parameters=sum(p.numel() for p in model.parameters()),
            segment_seconds=time.monotonic()-started,job=os.environ['SLURM_JOB_ID'],node=socket.gethostname(),
            production_ready=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True,type=Path)
    p.add_argument('--arm',choices=list('ABCD'),required=True)
    p.add_argument('--seed',type=int,choices=[0,1],required=True)
    p.add_argument('--factor',choices=['fine','coarse'],required=True)
    p.add_argument('--restart-test',choices=['baseline','pause','resume'])
    a=p.parse_args()
    train(a.root,a.arm,a.seed,a.factor,a.restart_test)


if __name__=='__main__':
    main()
