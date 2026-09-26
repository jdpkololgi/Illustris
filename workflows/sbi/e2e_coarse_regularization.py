"""Matched coarse-only continuation; EMA assessment on a frozen small panel."""
import argparse
import copy
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_views as views
from workflows.sbi.e2e_coupled_cfm_pilot import training_ids,pair,batch_pair,condition,weighted_loss,save
from workflows.sbi.e2e_coupled_benchmark_models import CoupledBackbone,FieldCondition,sample
from workflows.sbi.e2e_coarse_controls import BASE,TRAIN_ROOT,OPEN,panel,target,regions,scored,draw_seed

def evaluation_panel():
    available=panel();chosen=[]
    for phase in ('ph000','ph002','ph020','ph024'):
        chosen.extend([x for x in available if x[0]==phase][:2])
    for i,phase in enumerate(OPEN):
        values=[x for x in available if x[0]==phase];chosen.extend(values[i::4])
    if len(chosen)!=24:raise ValueError('panel mismatch')
    return chosen

def lr_at(step):return 1e-5+.5*9e-5*(1+math.cos(math.pi*step/26624))

def evaluate(model,seed,step,folder,normalizer):
    folder.mkdir(exist_ok=True);chart=views.load_chart(normalizer);model.eval()
    for phase,pid in evaluation_panel():
        path=folder/f'{pid}.json'
        if path.exists():continue
        ob=views.load_observations(phase,pid,normalizer)
        t=lambda x:torch.as_tensor(x,device='cuda',dtype=torch.float32)[None].expand(2,*x.shape)
        cond=FieldCondition(t(ob['joint']),t(ob['wide']),t(ob['joint_center_from_wide_mpc_h']),'wide')
        draws=[]
        for start in range(0,32,2):
            z=sample(model,cond,'cfm',64,[draw_seed(seed,phase,pid,i,'coarse') for i in (start,start+1)]).cpu().numpy()[:,0]
            rho=np.exp(z.astype(float)*chart['coarse_logrho']['std'][0]+chart['coarse_logrho']['mean'][0])
            if not np.isfinite(rho).all():raise ValueError('nonfinite draws')
            draws.extend(regions(v,phase) for v in rho)
        _,truth,sha=target(phase,pid);draws=np.asarray(draws)
        c.atomic_json(path,dict(seed=seed,step=step,phase=phase,pair=pid,truth_sha256=sha,draws=draws.tolist(),**scored(draws,truth)))
        print('EVAL',step,pid,flush=True)

def run(a):
    c.require_compute();torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    if torch.cuda.device_count()!=1:raise RuntimeError('one GPU per worker')
    root=Path(a.root)/f'{a.arm}_{a.seed}';root.mkdir(parents=True,exist_ok=True)
    parent=TRAIN_ROOT/f'coarse_{a.seed}'/'CHECKPOINT_013312.pt'
    original=torch.load(parent,map_location='cpu',weights_only=False);b=original['binding']
    import inspect
    if b['model']!=c.sha256(inspect.getfile(CoupledBackbone)) or b['stage']!='coarse' or b['seed']!=a.seed:raise ValueError('parent binding')
    normalizer=b['normalizer'];ids=training_ids()
    if b['train_ids']!=ids:raise ValueError('training identities changed')
    wd=1e-5 if a.arm=='baseline' else .1
    binding=dict(parent=c.sha256(parent),runner=c.sha256(__file__),arm=a.arm,seed=a.seed,weight_decay=wd,
        normalizer=normalizer,panel=evaluation_panel(),steps=26624,confirmation_access=False)
    marker=root/'BINDING.json'
    if marker.exists() and json.loads(marker.read_text())!=json.loads(json.dumps(binding)):raise ValueError('resume binding')
    if not marker.exists():c.atomic_json(marker,binding)
    latest=root/'LATEST.pt';state=torch.load(latest,map_location='cpu',weights_only=False) if latest.exists() else original
    if latest.exists() and state['binding']!=binding:raise ValueError('checkpoint provenance')
    model=CoupledBackbone('coarse','wide',24,3).cuda();model.load_state_dict(state['model'])
    ema=copy.deepcopy(model).eval().requires_grad_(False);ema.load_state_dict(state['ema'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=wd);optimizer.load_state_dict(state['optimizer'])
    for group in optimizer.param_groups:group['weight_decay']=wd
    generator=torch.Generator(device='cuda');generator.set_state(state['generator'].cpu())
    weight=state['weight'].cuda();step=state['step'];start=time.monotonic()
    def checkpoint():
        value=dict(binding=binding,step=step,model=model.state_dict(),ema=ema.state_dict(),optimizer=optimizer.state_dict(),
            generator=generator.get_state(),weight=weight)
        save(latest,value)
        if step%1664==0:save(root/f'CHECKPOINT_{step:06d}.pt',value)
    # Evaluation never consumes the dedicated training generator.
    while step<26624:
        epoch=step//832;position=step%832;order=np.random.default_rng(a.seed+epoch*10000).permutation(len(ids))
        items=[pair(ids[i],normalizer) for i in order[2*position:2*position+2]]
        x,cond=condition(batch_pair(items),'coarse','cuda');model.train();optimizer.zero_grad(set_to_none=True)
        t=((torch.arange(2,device='cuda')+torch.rand((),device='cuda',generator=generator))/2)[:,None,None,None,None]
        eps=torch.randn(x.shape,device='cuda',generator=generator);z=(1-t)*eps+t*x
        error=model(z,model.schedule(t.flatten()),cond)-(x-eps);loss=weighted_loss(error,weight)
        if not torch.isfinite(loss):raise ValueError('nonfinite loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),.5,error_if_nonfinite=True)
        for group in optimizer.param_groups:group['lr']=lr_at(step)
        optimizer.step()
        with torch.no_grad():
            for ep,p in zip(ema.parameters(),model.parameters()):ep.lerp_(p,.001)
        step+=1
        if step%64==0:print('TRAIN',a.arm,a.seed,step,float(loss),'elapsed',time.monotonic()-start,flush=True)
        if step%256==0 or step%1664==0:checkpoint()
        if step in (19968,26624):
            checkpoint();evaluate(ema,a.seed,step,root/f'eval{step}',normalizer)
    # A resume after checkpoint-save/before-evaluation also completes missing assessments.
    for number in (19968,26624):
        saved=torch.load(root/f'CHECKPOINT_{number:06d}.pt',map_location='cpu',weights_only=False)
        ema.load_state_dict(saved['ema']);evaluate(ema,a.seed,number,root/f'eval{number}',normalizer)
    replay=None
    if a.arm=='baseline':
        reference=torch.load(TRAIN_ROOT/f'coarse_{a.seed}'/'CHECKPOINT_026624.pt',map_location='cpu',weights_only=False)
        replay=max(float((v.detach().cpu()-reference['model'][k]).abs().max()) for k,v in model.state_dict().items())
    c.atomic_json(root/'COMPLETE.json',dict(binding=c.digest(binding),steps=step,seconds=time.monotonic()-start,
        baseline_max_parameter_difference=replay),replace=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--arm',choices=('baseline','decay'),required=True)
    p.add_argument('--seed',type=int,choices=(17,29),required=True);run(p.parse_args())
