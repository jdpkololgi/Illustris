"""Bounded direct-density pilot and paired frozen coarse-to-fine inference."""
import argparse
import json
import os
from pathlib import Path
import socket
import signal
import subprocess
import time
import numpy as np
import torch
from torch.nn import functional as F
from workflows.sbi import e2e_wide_pipeline as p
from workflows.sbi import e2e_durable as durable
from workflows.sbi import e2e_frozen_controls as frozen
from workflows.sbi.e2e_direct_vdm import ConditionalVDM, vlb, sample
from workflows.sbi.e2e_clean_limit_launch import snapshot_paths, SCRATCH
from workflows.sbi.e2e_wide_denoising_audit import Bands, TRAIN384
from workflows.sbi.e2e_oracle_solver import sample_vp_heun

CONFIG = 'configs/e2e_direct_vdm_v1.json'
STOP = False


def request_stop(signum, frame):
    global STOP
    STOP = True


def stage(root):
    root = root.resolve(); spec = json.loads((p.REPO/CONFIG).read_text())
    if root.parent != SCRATCH or not root.name.startswith('direct_vdm_'):
        raise ValueError('new direct_vdm_ Scratch child required')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=p.REPO, text=True).strip():
        raise ValueError('commit source before staging')
    parent = Path(spec['frozen_root'])/'MANIFEST.json'
    if p.sha256(parent) != spec['frozen_manifest_sha256']:
        raise ValueError('frozen parent drift')
    rev = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=p.REPO, text=True).strip()
    names = snapshot_paths(subprocess.check_output(['git','ls-files','-z'], cwd=p.REPO).decode().split('\0'))
    names += ['docs/e2e_direct_vdm_v1.md', 'docs/evidence/e2e_field_v2/wide_gpu_smoke_20260911/SMOKE_COMPLETE.json']
    root.mkdir(exist_ok=False); source = root/'source'; source.mkdir(); (root/'logs').mkdir()
    archive = subprocess.Popen(['git','archive',rev,'--',*names], cwd=p.REPO, stdout=subprocess.PIPE)
    unpack = subprocess.run(['tar','-xf','-','-C',str(source)], stdin=archive.stdout)
    archive.stdout.close()
    if archive.wait() or unpack.returncode:
        raise RuntimeError('partial snapshot preserved')
    durable.publish_json(root/'MANIFEST.json', dict(spec=spec, git_revision=rev, source=str(source),
        source_sha256={n:p.sha256(source/n) for n in names if (source/n).is_file()},
        heldout_ph001_access=False, production_ready=False))
    print('STAGED',root,rev,flush=True)


def verify(root):
    m = json.loads((root/'MANIFEST.json').read_text())
    if Path(m['source']).resolve() != p.REPO.resolve() or m['spec'] != json.loads((p.REPO/CONFIG).read_text()):
        raise ValueError('wrong source snapshot')
    for name, digest in m['source_sha256'].items():
        if p.sha256(p.REPO/name) != digest:
            raise ValueError('source drift: '+name)
    parent = Path(m['spec']['frozen_root'])/'MANIFEST.json'
    if p.sha256(parent) != m['spec']['frozen_manifest_sha256']:
        raise ValueError('parent manifest drift')
    return m, json.loads(parent.read_text())


def density(item, prepared):
    f, c = (prepared['original_normalization']['targets'][k] for k in ('fine','coarse'))
    return item['target']*f['std']+f['mean']+item['condition'][:,-1:]*c['std']+c['mean']


def observation(item):
    """Only 12 local observables + 12 broadcast wide means, NEVER coarse truth."""
    local = F.avg_pool3d(item['condition'][:,:12], 2)
    wide = item['wide'].mean((2,3,4), keepdim=True).expand(-1,-1,*local.shape[2:])
    return torch.cat([local, wide], dim=1)


def prepare(prepared, items, spec):
    train_ids = [r['anchor_id'] for r in prepared['selection']['train'] if r['phase'] in spec['train_phases']]
    if len(train_ids) != 10 or any(items[k]['phase'] == spec['development_phase'] for k in train_ids):
        raise ValueError('expected ten fit cutouts from two phases')
    data = {}
    for anchor, item in items.items():
        delta = F.avg_pool3d(density(item, prepared), 2)
        if not torch.isfinite(delta).all() or (delta <= -1).any():
            raise ValueError('log target requires strictly positive physical density')
        data[anchor] = dict(x=torch.log1p(delta), condition=observation(item), truth=delta[0,0].cpu().numpy(),
                            phase=item['phase'], group='fit' if anchor in train_ids else 'development')
    # Fit every affine statistic ONLY on the ten fit fields, never ph003/SGC.
    xs = torch.cat([data[k]['x'] for k in train_ids]); cs = torch.cat([data[k]['condition'] for k in train_ids])
    xm, xstd = xs.mean(), xs.std(unbiased=False)
    cm = cs.mean((0,2,3,4),keepdim=True); cstd = cs.std((0,2,3,4),unbiased=False,keepdim=True).clamp_min(1e-6)
    for item in data.values():
        item['x'] = (item['x']-xm)/xstd; item['condition'] = (item['condition']-cm)/cstd
    chart = dict(mean=float(xm),std=float(xstd),condition_mean=cm.flatten().tolist(),condition_std=cstd.flatten().tolist(),
                 fit_ids=train_ids,grid=48,cell_mpc_h=6.766,
                 target='2x2x2 voxel average of delta_R7 at z0.2; log1p then train-only affine',
                 condition='12 local channels pooled2 + 12 broadcast wide-channel means; no target-derived coarse')
    return data, chart


def metrics(pred, truth, bands):
    if not np.isfinite(pred).all():
        raise FloatingPointError('nonfinite physical density')
    n = pred.shape[0]
    mass = lambda x: (x+1).reshape(2,n//2,2,n//2,2,n//2).mean((1,3,5)).flatten().tolist()
    return dict(spectrum=bands.compare(pred,truth),mean=float(pred.mean()),std=float(pred.std()),
        rmse=float(np.sqrt(np.mean((pred-truth)**2))),below_minus_one=float((pred < -1).mean()),
        quantiles=np.quantile(pred,[.001,.01,.1,.5,.9,.99,.999]).tolist(),
        regional_density=mass(pred),truth_regional_density=mass(truth))


def selected(prepared):
    return frozen.panel(prepared['selection'], ['ph000','ph002','ph003'])


@torch.no_grad()
def evaluate(model, data, chart, rows, spec, folder, seed, update):
    folder.mkdir(exist_ok=False); results = []; bands = Bands(48,6.766,[0,.08,.16,.32,np.inf])
    for row in rows:
        anchor = row['anchor_id']; item = data[anchor]
        for draw in range(spec['draws']):
            address = p.seed_for(spec['seed'],anchor,draw,'direct-draw')
            z = sample(model,item['condition'],spec['sample_steps'],torch.Generator(device=item['x'].device).manual_seed(address))
            pred = torch.expm1(z*chart['std']+chart['mean'])[0,0].cpu().numpy()
            path = folder/f'{anchor}_{draw}.npz'; np.savez_compressed(path,delta=pred)
            results.append(dict(anchor_id=anchor,phase=row['phase'],cap=row['cap'],group=item['group'],draw=draw,
                seed=seed,update=update,noise_seed=address,metrics=metrics(pred,item['truth'],bands),
                path=str(path),sha256=p.sha256(path)))
        print('DIRECT SAMPLES',seed,update,anchor,flush=True)
    # One bounded condition-use check: same random path, transfer ph003, donor fit ph000.
    row = [r for r in rows if r['phase']=='ph003' and r['cap']=='SGC'][0]
    anchor = row['anchor_id']; item = data[anchor]; donor = data[chart['fit_ids'][0]]
    address = p.seed_for(spec['seed'],anchor,0,'direct-draw')
    wrong = sample(model,donor['condition'],spec['sample_steps'],torch.Generator(device=item['x'].device).manual_seed(address))
    wrong = torch.expm1(wrong*chart['std']+chart['mean'])[0,0].cpu().numpy()
    path = folder/'condition_swapped.npz'; np.savez_compressed(path,delta=wrong)
    durable.publish_json(folder/'RESULTS.json',dict(records=results,condition_swap=dict(anchor=anchor,
        donor=chart['fit_ids'][0],metrics=metrics(wrong,item['truth'],bands),path=str(path),sha256=p.sha256(path)),
        caveat='four draws per anchor cannot establish coverage; ph003 development only, not sealed ph001'))
    return results


def new_model(spec, seed, arm, device):
    torch.manual_seed(spec['seed']+seed)
    model = ConditionalVDM(base=spec['base_channels'],levels=spec['levels'],learned=arm=='learned_vlb').to(device)
    opt = torch.optim.AdamW(model.parameters(),lr=spec['learning_rate'],weight_decay=spec['weight_decay'])
    gen = torch.Generator(device=device).manual_seed(spec['seed']+100+seed)
    return model,opt,gen


def update_model(model,opt,gen,data,chart,spec,update,seed):
    ids = chart['fit_ids']; index = [p.example_index(update*spec['batch_size']+j,len(ids),spec['seed']+seed) for j in range(spec['batch_size'])]
    x = torch.cat([data[ids[i]]['x'] for i in index]); c = torch.cat([data[ids[i]]['condition'] for i in index])
    model.train(); opt.zero_grad(set_to_none=True)
    loss, terms = vlb(model,x,c,gen,spec['decoder_std'])
    if not torch.isfinite(loss):
        raise FloatingPointError('nonfinite VLB')
    loss.backward(); norm = torch.nn.utils.clip_grad_norm_(model.parameters(),spec['clip'],error_if_nonfinite=True); opt.step()
    return dict(update=update+1,loss=float(loss.detach()),grad_norm=float(norm),
                **{k:float(v.detach()) for k,v in terms.items()})


def save_checkpoint(path,model,opt,gen,history,binding):
    temp = path.with_suffix('.tmp')
    if path.exists() or temp.exists():
        raise FileExistsError(path)
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=gen.get_state(),
                    history=history,binding=binding),temp)
    with temp.open('rb') as f: os.fsync(f.fileno())
    os.replace(temp,path)
    durable.publish_json(path.with_suffix('.json'),dict(sha256=p.sha256(path),update=len(history),binding=binding))


def train(root, seed, arm, smoke=False):
    device=p.runtime(); m,parent=verify(root); spec=m['spec']; prepared,items=frozen.load_data(parent,device)
    data,chart=prepare(prepared,items,spec); del items
    if seed not in spec['replicas'] or arm not in spec['arms']:
        raise ValueError('unregistered branch')
    branch=root/('smoke' if smoke else f'{arm}_seed{seed}'); branch.mkdir(exist_ok=False)
    durable.publish_json(branch/'CHART.json',chart)
    model,opt,gen=new_model(spec,seed,arm,device); start=time.monotonic(); history=[]
    binding=dict(manifest_sha256=p.sha256(root/'MANIFEST.json'),chart_sha256=p.sha256(branch/'CHART.json'),seed=seed,arm=arm)
    stop=8 if smoke else spec['updates']
    signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    for step in range(stop):
        history.append(update_model(model,opt,gen,data,chart,spec,step,seed))
        if step==0 or (step+1)%128==0:
            print('TRAIN',arm,seed,history[-1],'seconds',time.monotonic()-start,flush=True)
        if not smoke and ((step+1)%256==0 or step+1 in spec['checkpoints'] or STOP):
            save_checkpoint(branch/f'update_{step+1:06d}.pt',model,opt,gen,history,binding)
        if STOP:
            durable.publish_json(branch/'INTERRUPTED.json',dict(update=step+1,binding=binding,complete=False))
            raise SystemExit(75)
    if smoke:
        # Exact state/RNG continuation; dropout-free model makes global RNG unused.
        path=branch/'update_000008.pt'; save_checkpoint(path,model,opt,gen,history,binding)
        expected=update_model(model,opt,gen,data,chart,spec,8,seed)
        restored,o,g=new_model(spec,seed,arm,device)
        state=torch.load(path,map_location=device,weights_only=False)
        restored.load_state_dict(state['model']); o.load_state_dict(state['optimizer']); g.set_state(state['rng'].cpu())
        actual=update_model(restored,o,g,data,chart,spec,8,seed)
        from workflows.sbi.e2e_wide_continue import equal_state
        if expected != actual or not equal_state(model.state_dict(),restored.state_dict()) or not equal_state(opt.state_dict(),o.state_dict()):
            raise ValueError('checkpoint replay mismatch')
        first=data[chart['fit_ids'][0]]
        z=sample(model,first['condition'],8,torch.Generator(device=device).manual_seed(3))
        if not torch.isfinite(torch.expm1(z*chart['std']+chart['mean'])).all():
            raise FloatingPointError('smoke density nonfinite')
        durable.publish_json(root/'SMOKE.json',dict(passed=True,exact_replay=True,manifest_sha256=binding['manifest_sha256'],
            seconds=time.monotonic()-start,parameters=sum(x.numel() for x in model.parameters()),
            peak_gpu_bytes=torch.cuda.max_memory_allocated(),node=socket.gethostname(),job=os.environ['SLURM_JOB_ID']))
    else:
        if not (branch/f'update_{stop:06d}.pt').exists():
            save_checkpoint(branch/f'update_{stop:06d}.pt',model,opt,gen,history,binding)
        results=evaluate(model,data,chart,selected(prepared),spec,branch/'draws',seed,stop)
        verify(root)
        durable.publish_json(branch/'COMPLETE.json',dict(complete=True,binding=binding,update=stop,history=history,
            gamma_endpoints=[float(model.schedule.low),float(model.schedule.low+model.schedule.slope.abs())],
            seconds=time.monotonic()-start,records=len(results),node=socket.gethostname(),job=os.environ['SLURM_JOB_ID'],
            parameters=sum(x.numel() for x in model.parameters()),production_ready=False))
    print('DONE',branch,time.monotonic()-start,flush=True)


def submit(root):
    m,_=verify(root)
    smoke=json.loads((root/'SMOKE.json').read_text())
    if not smoke['passed'] or not smoke['exact_replay'] or smoke['manifest_sha256']!=p.sha256(root/'MANIFEST.json'):
        raise ValueError('matching smoke required')
    command=['sbatch','--parsable','--nodes=1','--ntasks=1','--cpus-per-task=32',
        '--constraint=gpu&hbm80g','--gpus=1','--qos=shared','--account=desi_g','--time=01:30:00',
        '--licenses=scratch','--no-requeue','--signal=USR1@180','--job-name=e2e-direct-vdm',
        '--chdir='+str(root/'source'),'--output='+str(root/'logs/train_%j.out'),
        '--error='+str(root/'logs/train_%j.err'),str(root/'source/workflows/sbi/submit_e2e_direct_vdm.slurm'),str(root)]
    durable.publish_json(root/'SUBMISSION_INTENT.json',dict(command=command,manifest_sha256=p.sha256(root/'MANIFEST.json')))
    result=subprocess.run(command,text=True,capture_output=True)
    durable.publish_json(root/'SUBMISSION.json',dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
    if result.returncode: raise RuntimeError('submission failed; no automatic retry')
    print('SUBMITTED',result.stdout.strip(),flush=True)


def matrix(root):
    m,_=verify(root)
    for seed in m['spec']['replicas']:
        for arm in m['spec']['arms']:
            train(root,seed,arm)
    durable.publish_json(root/'MATRIX_COMPLETE.json',dict(complete=True,
        manifest_sha256=p.sha256(root/'MANIFEST.json'),production_ready=False,
        branches={f'{a}_seed{s}':p.sha256(root/f'{a}_seed{s}/COMPLETE.json')
                  for s in m['spec']['replicas'] for a in m['spec']['arms']}))


@torch.no_grad()
def paired(root):
    from workflows.sbi.e2e_wide_research_canary import preflight
    from workflows.sbi.e2e_wide_continue import checked_binding
    device=p.runtime();m,parent=verify(root);prepared,items=frozen.load_data(parent,device)
    c,ds,_,_=preflight(); binding=checked_binding(TRAIN384,p.provenance(c,ds))
    coarsepath=TRAIN384/'diffusion_coarse/step_000384.pt'
    state=p.load_checkpoint(coarsepath,binding,'coarse','diffusion')
    coarse=p.build_model(c,'coarse',device).eval();coarse.load_state_dict(state['model']);del state
    folder=root/'paired';folder.mkdir(exist_ok=False);records=[];bands=Bands(96,3.383,[0,.08,.16,.32,np.inf])
    finechart=prepared['original_normalization']['targets']['fine'];coarsechart=prepared['original_normalization']['targets']['coarse']
    for seed in m['spec']['replicas']:
        _,model,opt,gen,state=frozen.load_parent(parent,prepared,seed,device)
        for row in selected(prepared):
            anchor=row['anchor_id'];item=items[anchor];truth=density(item,prepared)[0,0].cpu().numpy()
            truecoarse=item['condition'][0,-1].cpu().numpy()*coarsechart['std']+coarsechart['mean']
            for draw in range(2):
                coarse_seed=p.seed_for(916471,anchor,draw,'coarse')
                cg=sample_vp_heun(coarse,item['wide'],64,torch.Generator(device=device).manual_seed(coarse_seed))
                up=p.coarse_to_fine(cg[0,0].cpu().numpy()*coarsechart['std']+coarsechart['mean'])
                arrays={};case={}
                for mode,local in [('true_coarse',truecoarse),('generated_coarse',up)]:
                    cond=item['condition'].clone();cond[:,-1]=torch.as_tensor((local-coarsechart['mean'])/coarsechart['std'],device=device)
                    fg=sample_vp_heun(model,cond,64,torch.Generator(device=device).manual_seed(p.seed_for(916471,anchor,draw,'fine')),wide_condition=item['wide'])
                    pred=fg[0,0].cpu().numpy()*finechart['std']+finechart['mean']+local
                    arrays[mode]=pred;case[mode]=metrics(pred,truth,bands)
                path=folder/f'{seed}_{anchor}_{draw}.npz';np.savez_compressed(path,**arrays)
                records.append(dict(seed=seed,anchor_id=anchor,phase=row['phase'],group=row['panel_group'],draw=draw,
                    metrics=case,coarse_local_rmse=float(np.sqrt(np.mean((up-truecoarse)**2))),path=str(path),sha256=p.sha256(path)))
            print('PAIRED',seed,anchor,flush=True)
        frozen.check_restored(model,opt,state);del model,opt,gen,state
    verify(root)
    durable.publish_json(folder/'RESULTS.json',dict(records=records,coarse_checkpoint_sha256=p.sha256(coarsepath),
        manifest_sha256=p.sha256(root/'MANIFEST.json'),nfe=128,job=os.environ['SLURM_JOB_ID'],node=socket.gethostname(),
        caveat='paired full-density patch inference; no full-cap coherence/calibration; coarse checkpoint only384updates'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['stage','smoke','train','paired','submit','matrix'])
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--arm',choices=['fixed_vlb','learned_vlb'],default='learned_vlb');args=parser.parse_args()
    if args.mode=='stage':stage(args.root)
    elif args.mode=='submit':submit(args.root)
    elif args.mode=='matrix':matrix(args.root)
    elif args.mode=='paired':paired(args.root)
    else:train(args.root,args.seed,args.arm,args.mode=='smoke')


if __name__=='__main__':main()
