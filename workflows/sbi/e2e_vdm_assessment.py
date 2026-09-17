"""Frozen checkpoint/solver/posterior assessment with resumable addressed draws."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import tempfile
import time
import h5py
import numpy as np
import torch
from workflows.sbi import e2e_wide_pipeline as p, e2e_durable as durable
from workflows.sbi.e2e_direct_vdm import ConditionalVDM
from workflows.sbi.e2e_direct_experiment import density,observation
from workflows.sbi.e2e_diversity_norm import overlap
from workflows.sbi.e2e_clean_limit_launch import snapshot_paths,SCRATCH
from workflows.sbi.e2e_wide_denoising_audit import Bands
from workflows.sbi.e2e_vdm_assessment_math import (coupled_sample,pool2,probes,tidal_features,
    tensor_features,regional_density,summarize_calibration)

CONFIG='configs/e2e_vdm_assessment_v1.json'
STOP=False
NAMES=['delta','lambda1','lambda2','lambda3','gap21','gap32']


def stop_requested(signum,frame):
    global STOP
    STOP=True


def atomic_npz(folder,prefix,arrays):
    fd,name=tempfile.mkstemp(prefix=prefix+'-',suffix='.npz',dir=folder)
    with os.fdopen(fd,'wb') as f:
        np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
    durable.sync_dir(folder)
    return Path(name)


def stage(root):
    s=json.loads((p.REPO/CONFIG).read_text());old=Path(s['parent_root']);raw=Path(s['raw_root'])
    if root.resolve().parent!=SCRATCH or not root.name.startswith('vdm_assessment_'):
        raise ValueError('new registered Scratch child required')
    if subprocess.check_output(['git','status','--porcelain'],cwd=p.REPO,text=True).strip():
        raise ValueError('commit before staging')
    if p.sha256(old/'MANIFEST.json')!=s['parent_manifest_sha256']:raise ValueError('parent drift')
    done=json.loads((old/'MATRIX_COMPLETE.json').read_text());parents={}
    if not done['complete'] or done['manifest_sha256']!=s['parent_manifest_sha256']:raise ValueError('incomplete parent')
    for branch in s['branches']:
        b=old/branch
        if p.sha256(b/'COMPLETE.json')!=done['branches'][branch]:raise ValueError('branch drift')
        complete=json.loads((b/'COMPLETE.json').read_text());chart=json.loads((b/'CHART.json').read_text())
        if any(a in chart['fit_ids'] for a in s['anchors']):raise ValueError('evaluation field fitted')
        for step in s['checkpoints']:
            path=b/f'update_{step:06d}.pt';receipt=json.loads(path.with_suffix('.json').read_text())
            if receipt['update']!=step or receipt['binding']!=complete['binding'] or p.sha256(path)!=receipt['sha256']:
                raise ValueError('checkpoint binding/hash mismatch')
            parents[f'{branch}/{step}']=dict(path=str(path),sha256=receipt['sha256'],binding=receipt['binding'])
    prepared=json.loads((raw/'PREPARED.json').read_text());rows={r['anchor_id']:r for g in ['train','transfer'] for r in prepared['selection'][g]}
    panel=[rows[a] for a in s['anchors']]
    if any(r['phase']!='ph003' for r in panel) or overlap(*panel):raise ValueError('phase/footprint panel mismatch')
    rev=subprocess.check_output(['git','rev-parse','HEAD'],cwd=p.REPO,text=True).strip()
    names=snapshot_paths(subprocess.check_output(['git','ls-files','-z'],cwd=p.REPO).decode().split('\0'))
    names+=['docs/e2e_vdm_assessment_v1.md']
    root.mkdir(exist_ok=False);src=root/'source';src.mkdir();(root/'logs').mkdir()
    proc=subprocess.Popen(['git','archive',rev,'--',*names],cwd=p.REPO,stdout=subprocess.PIPE)
    result=subprocess.run(['tar','-xf','-','-C',str(src)],stdin=proc.stdout);proc.stdout.close()
    if proc.wait() or result.returncode:raise RuntimeError('partial archive preserved')
    durable.publish_json(root/'MANIFEST.json',dict(spec=s,source=str(src),revision=rev,parents=parents,panel=panel,
        raw_prepared_sha256=p.sha256(raw/'PREPARED.json'),parent_complete_sha256=p.sha256(old/'MATRIX_COMPLETE.json'),
        source_sha256={n:p.sha256(src/n) for n in names if (src/n).is_file()},heldout_ph001_access=False))
    print('STAGED',root,rev,flush=True)


def verify(root):
    m=json.loads((root/'MANIFEST.json').read_text());s=m['spec'];old=Path(s['parent_root'])
    if Path(m['source']).resolve()!=p.REPO.resolve() or s!=json.loads((p.REPO/CONFIG).read_text()):raise ValueError('wrong snapshot')
    for n,h in m['source_sha256'].items():
        if p.sha256(p.REPO/n)!=h:raise ValueError('source drift '+n)
    if p.sha256(old/'MANIFEST.json')!=s['parent_manifest_sha256'] or p.sha256(old/'MATRIX_COMPLETE.json')!=m['parent_complete_sha256']:
        raise ValueError('parent drift')
    if p.sha256(Path(s['raw_root'])/'PREPARED.json')!=m['raw_prepared_sha256']:raise ValueError('raw metadata drift')
    return m


def prepare(root):
    device=p.runtime();m=verify(root);s=m['spec'];raw=Path(s['raw_root']);old=Path(s['parent_root'])
    prepared=json.loads((raw/'PREPARED.json').read_text())
    if p.sha256(raw/'cache.h5')!=prepared['cache_sha256']:raise ValueError('cache drift')
    charts=[json.loads((old/b/'CHART.json').read_text()) for b in s['branches']]
    if any(c!=charts[0] for c in charts):raise ValueError('different normalization charts')
    chart=charts[0];meta=prepared['base_binding'];wide=Path(meta['config']['products_root'])/'WIDE_COARSE_PRODUCTS_COMPLETE.json'
    if p.sha256(wide)!=meta['config']['products_manifest_sha256']:raise ValueError('wide manifest drift')
    rows={r['anchor_id']:r for r in json.loads(wide.read_text())['parents']};arrays={};references={}
    for anchor in s['anchors']:
        row=rows[anchor];shard=Path(row['shard'])
        if p.sha256(shard)!=meta['data_binding']['payloads'][str(shard)]:raise ValueError('fullbox reference drift')
        with h5py.File(raw/'cache.h5','r') as f:
            g=f[anchor];item={k:p.tensor(g[name][:],device) for k,name in [('target','target'),('condition','condition'),('wide','wide')]}
        delta=pool2(density(item,prepared)[0,0].cpu().numpy())
        cond=observation(item).cpu().numpy()[0]
        cm=np.array(chart['condition_mean'],np.float32)[:,None,None,None];cs=np.array(chart['condition_std'],np.float32)[:,None,None,None]
        cond=(cond-cm)/cs
        support=pool2(item['condition'][0,1].cpu().numpy())>=.5
        with h5py.File(shard,'r') as f:
            g=f[row['group']];tensor=pool2(g['tensor_spectral'][:]);native_delta=pool2(g['delta_r7_gaussian'][:])
        if not np.allclose(delta,native_delta,atol=2e-5,rtol=2e-5):raise ValueError('target reconstruction mismatch')
        if not np.allclose(tensor[...,[0,3,5]].sum(-1),delta,atol=2e-5,rtol=2e-5):raise ValueError('fullbox tensor trace mismatch')
        prefix=anchor+'__';arrays[prefix+'condition']=cond;arrays[prefix+'truth']=delta
        arrays[prefix+'support']=support[probes(48,s['core_side'],s['probe_stride'])].flatten()
        arrays[prefix+'fullbox']=tensor_features(tensor,delta,s['core_side'],s['probe_stride'])
        arrays[prefix+'regional']=regional_density(delta)
        for b in s['boundaries']:
            arrays[prefix+b]=tidal_features(delta,b,core=s['core_side'],stride=s['probe_stride'])
        references[anchor]=dict(shard=str(shard),sha256=meta['data_binding']['payloads'][str(shard)],
            oracle_boundary_rms={b:np.sqrt(np.mean((arrays[prefix+b]-arrays[prefix+'fullbox'])**2,axis=0)).tolist() for b in s['boundaries']})
    path=atomic_npz(root,'prepared',arrays)
    durable.publish_json(root/'PREPARED.json',dict(file=path.name,sha256=p.sha256(path),chart=chart,references=references,
        manifest_sha256=p.sha256(root/'MANIFEST.json'),job=os.environ['SLURM_JOB_ID'],node=socket.gethostname(),
        scope='same-phase development-heldout grid probes; fullbox tensor averaged BEFORE eigendecomposition; no ph001'))
    print('PREPARED',json.dumps(references),flush=True)


def load_data(root,m,device):
    prep=json.loads((root/'PREPARED.json').read_text());path=root/prep['file']
    if prep['manifest_sha256']!=p.sha256(root/'MANIFEST.json') or p.sha256(path)!=prep['sha256']:raise ValueError('prepared drift')
    with np.load(path) as f:arrays={k:f[k] for k in f.files}
    data={a:dict(condition=p.tensor(arrays[a+'__condition'],device),truth=arrays[a+'__truth']) for a in m['spec']['anchors']}
    return prep,arrays,data


def load_model(m,branch,step,device):
    rec=m['parents'][f'{branch}/{step}'];path=Path(rec['path'])
    if p.sha256(path)!=rec['sha256']:raise ValueError('checkpoint drift')
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state['binding']!=rec['binding'] or len(state['history'])!=step:raise ValueError('checkpoint metadata drift')
    model=ConditionalVDM(learned=branch.startswith('learned')).to(device)
    model.load_state_dict(state['model']);model.eval();return model


def features(fields,truth,s):
    bands=Bands(48,6.766,[0,.08,.16,.32,np.inf]);out={b:[] for b in s['boundaries']}
    out.update(power=[],correlation=[],regional=[],onepoint=[])
    for delta in fields:
        for b in s['boundaries']:out[b].append(tidal_features(delta,b,core=s['core_side'],stride=s['probe_stride']))
        metric=bands.compare(delta,truth);out['power'].append(metric['prediction_power']);out['correlation'].append(metric['correlation'])
        out['regional'].append(regional_density(delta))
        out['onepoint'].append([delta.mean(),delta.std(),*np.quantile(delta,[.001,.01,.1,.5,.9,.99,.999]),*[(delta<=v).mean() for v in [-.8,-.5,0,.5,1,2,4]]])
    return {k:np.asarray(v) for k,v in out.items()}


def binding(root,m,branch,step,anchor,steps,ids):
    return dict(manifest_sha256=p.sha256(root/'MANIFEST.json'),prepared_sha256=p.sha256(root/'PREPARED.json'),
        checkpoint_sha256=m['parents'][f'{branch}/{step}']['sha256'],branch=branch,update=step,anchor=anchor,steps=steps,ids=ids)


def read_chunk(folder,receipt,expected):
    saved=json.loads(receipt.read_text());path=folder/saved['file']
    if saved['binding']!=expected or p.sha256(path)!=saved['sha256']:raise ValueError('resumed chunk drift')
    return saved


def case(root,m,model,prep,item,branch,step,anchor,steps,count,device):
    s=m['spec'];folder=root/branch/f'u{step}_s{steps}'/anchor;folder.mkdir(parents=True,exist_ok=True)
    for first in range(0,count,s['microbatch']):
        ids=list(range(first,min(count,first+s['microbatch'])));receipt=folder/f'{first:06d}.json'
        bind=binding(root,m,branch,step,anchor,steps,ids)
        if receipt.exists():read_chunk(folder,receipt,bind);continue
        if STOP:raise SystemExit(75)
        seeds=[p.seed_for(s['seed'],anchor,i,'assessment') for i in ids];start=time.monotonic()
        c=item['condition'].expand(len(ids),-1,-1,-1,-1)
        z=coupled_sample(model,c,steps,seeds,s['noise_grid'])
        fields=torch.expm1(z*prep['chart']['std']+prep['chart']['mean'])[:,0].cpu().numpy()
        if not np.isfinite(fields).all():raise FloatingPointError('nonfinite density sample')
        arrays=dict(delta=fields,**features(fields,item['truth'],s))
        path=atomic_npz(folder,f'{first:06d}',arrays)
        durable.publish_json(receipt,dict(file=path.name,sha256=p.sha256(path),binding=bind,
            seconds=time.monotonic()-start,seeds=seeds,job=os.environ['SLURM_JOB_ID'],node=socket.gethostname()))
        print('CHUNK',branch,step,anchor,steps,first,round(time.monotonic()-start,2),flush=True)
    complete=folder/'COMPLETE.json'
    if not complete.exists():durable.publish_json(complete,dict(count=count,chunks={str(i):p.sha256(folder/f'{i:06d}.json') for i in range(0,count,s['microbatch'])}))


def run(root,branch,mode='all'):
    device=p.runtime();m=verify(root);s=m['spec'];prep,arrays,data=load_data(root,m,device)
    smoke=json.loads((root/'SMOKE.json').read_text())
    if not smoke['passed'] or smoke['manifest_sha256']!=p.sha256(root/'MANIFEST.json'):
        raise ValueError('matching smoke required')
    if branch not in s['branches']:raise ValueError('unregistered branch')
    signal.signal(signal.SIGUSR1,stop_requested);signal.signal(signal.SIGTERM,stop_requested)
    with durable.single_writer(root/branch):
        steps_to_do=s['checkpoints'] if mode in ('all','checkpoints') else [5120]
        for step in steps_to_do:
            model=load_model(m,branch,step,device)
            if mode in ('all','checkpoints'):
                for anchor in s['anchors']:case(root,m,model,prep,data[anchor],branch,step,anchor,s['checkpoint_steps'],s['checkpoint_draws'],device)
            if step==5120 and mode in ('all','refinement'):
                for steps in s['refinement_steps']:
                    for anchor in s['anchors']:case(root,m,model,prep,data[anchor],branch,step,anchor,steps,s['refinement_draws'],device)
            del model
        verify(root)
        path=root/branch/(mode.upper()+'_COMPLETE.json')
        if not path.exists():durable.publish_json(path,dict(complete=True,mode=mode,manifest_sha256=p.sha256(root/'MANIFEST.json')))


def smoke(root):
    device=p.runtime();m=verify(root);s=m['spec'];prep,arrays,data=load_data(root,m,device)
    model=load_model(m,s['branches'][0],512,device);item=data[s['anchors'][0]]
    seeds=[p.seed_for(s['seed'],s['anchors'][0],i,'smoke') for i in range(s['microbatch'])]
    start=time.monotonic();z=coupled_sample(model,item['condition'].expand(len(seeds),-1,-1,-1,-1),250,seeds)
    torch.cuda.synchronize();seconds=time.monotonic()-start
    # Actual-network scalar vs batched marginal equality to floating-point tolerance.
    single=coupled_sample(model,item['condition'],250,seeds[:1])
    rel=float((z[:1]-single).square().mean().sqrt()/single.square().mean().sqrt().clamp_min(1e-10))
    if rel>2e-4:raise ValueError('batch partition changes trajectory beyond tolerance')
    delta=torch.expm1(z*prep['chart']['std']+prep['chart']['mean'])[:,0].cpu().numpy()
    start=time.monotonic();f=features(delta,item['truth'],s);metric_seconds=time.monotonic()-start
    folder=root/'smoke';folder.mkdir(exist_ok=False);path=atomic_npz(folder,'chunk',dict(delta=delta,**f))
    receipt=folder/'000000.json';bind=binding(root,m,s['branches'][0],512,s['anchors'][0],250,list(range(len(seeds))))
    durable.publish_json(receipt,dict(binding=bind,file=path.name,sha256=p.sha256(path)))
    read_chunk(folder,receipt,bind)
    # Resume consumes committed draw identities, never restores/advances a shared RNG.
    if not np.array_equal(np.load(path)['delta'],delta):raise ValueError('saved chunk replay mismatch')
    durable.publish_json(root/'SMOKE.json',dict(passed=True,manifest_sha256=p.sha256(root/'MANIFEST.json'),
        batch=s['microbatch'],steps=250,sampling_seconds=seconds,metric_seconds=metric_seconds,
        scalar_batch_relative_rms=rel,committed_chunk_replay=True,peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        job=os.environ['SLURM_JOB_ID'],node=socket.gethostname()))
    print('SMOKE',seconds,metric_seconds,'batch relative rms',rel,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['stage','prepare','smoke','run'])
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--branch')
    parser.add_argument('--panel',choices=['all','checkpoints','refinement'],default='all');a=parser.parse_args()
    if a.mode=='stage':stage(a.root)
    elif a.mode=='prepare':prepare(a.root)
    elif a.mode=='smoke':smoke(a.root)
    else:run(a.root,a.branch,a.panel)


if __name__=='__main__':main()
