"""Bounded synthetic GPU cost/replay measurement; never a scientific fit.

Seven factors account for14later fits across two seeds, with I/J sharing coarse.
Independent fine cases charge TWO48-cubed evaluations per paired presentation.
Each checkpoint replay runs in a fresh process. No allocation requests here.
"""
import argparse
from dataclasses import fields, is_dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_coordinates as coord
from workflows.sbi import e2e_coupled_benchmark_models as models
from workflows.sbi import e2e_direct_vdm as direct
from workflows.sbi import e2e_vdm_context_models as legacy
from workflows.sbi.e2e_wide_continue import equal_state
from workflows.sbi.e2e_coupled_prepare_ops import accounting
from workflows.sbi.e2e_coupled_stage_b import used_bytes

CONFIG=c.REPO/'configs/e2e_coupled_gpu_benchmark_v1.json'
CASES=('D_coarse','D_fine','IJ_coarse','I_fine','J_fine','CFM_coarse','CFM_fine')


def config():
    cfg=json.loads(CONFIG.read_text())
    if (cfg['schema']!='e2e-coupled-gpu-benchmark-v1' or not cfg['synthetic_only']
            or cfg['scientific_training_authorized'] or cfg['cases']!=list(CASES)
            or cfg['unet_base']!=24 or cfg['unet_levels']!=3 or cfg['batch_pairs']!=2
            or cfg['warmup_updates']+1+cfg['timed_updates']>12
            or cfg['max_run_seconds']>3000
            or any(n<2 or n%2 or cfg['vdm_noise_grid']%n for n in cfg['sample_nfe'])):
        raise ValueError('unregistered or excessive technical benchmark')
    return cfg


def binding():
    return dict(config_sha256=c.sha256(CONFIG),source_hashes={
        Path(module.__file__).name:c.sha256(module.__file__)
        for module in (models,direct,legacy,sys.modules[equal_state.__module__])},
        runner_sha256=c.sha256(__file__),synthetic_only=True,scientific_fit=False)


def runtime():
    c.require_compute(); c.config()
    for item in json.loads((c.REPO/'SOURCE.json').read_text())['files']:
        if c.sha256(c.REPO/item['relative'])!=item['sha256']:
            raise ValueError('technical GPU source snapshot changed')
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('exactly one allocated visible GPU required')
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':
        raise RuntimeError('deterministic CUBLAS workspace must be set before launching Python')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    return 'cuda'


def make_case(name,cfg,device):
    if name not in CASES: raise ValueError('unregistered benchmark case')
    torch.manual_seed(cfg['model_seed']); random.seed(cfg['model_seed']); np.random.seed(cfg['model_seed'])
    base,levels=cfg['unet_base'],cfg['unet_levels']; batch=cfg['batch_pairs']
    kind='cfm' if name.startswith('CFM') else 'vdm'
    stage='coarse' if name.endswith('coarse') else 'fine'
    if name=='D_coarse': model=direct.ConditionalVDM(12,base,levels,learned=False)
    elif name=='D_fine': model=legacy.ContextVDM('D',base,levels)
    else:
        domain='wide' if stage=='coarse' else ('parent' if name=='I_fine' else 'joint')
        model=models.CoupledBackbone(stage,domain,base,levels)
    model=model.to(device)
    generator=torch.Generator(device='cpu').manual_seed(cfg['fixture_seed'])
    def draw(shape):
        return (torch.empty(shape,device='meta') if device=='meta'
                else torch.randn(shape,generator=generator).to(device))
    joint=draw((batch,12,64,48,48)); wide=draw((batch,12,48,48,48))
    coarse=draw((batch,1,48,48,48)); residual=legacy.project(draw((batch,1,64,48,48)))
    # Registered context shifts, with corresponding physically aligned crops.
    shifts=[4 if i%2==0 else -4 for i in range(batch)]
    offset=torch.tensor([[s*models.WIDE_CELL,0.,0.] for s in shifts],device=device)
    aligned=torch.cat([legacy.replicate(coarse[i:i+1,:,16+s:32+s,18:30,18:30])
                       for i,s in enumerate(shifts)])
    regions=['wide'] if stage=='coarse' else (['left','right'] if name in ('D_fine','I_fine') else ['joint'])
    presentations=[]
    for region in regions:
        cond=models.FieldCondition(joint,wide,offset,region,
            aligned if stage=='fine' else None,coarse if stage=='fine' else None,
            'training_truth' if stage=='fine' else None)
        target=coarse if stage=='coarse' else models.parent_crop(residual,region)
        if name=='D_coarse': cond=wide
        elif name=='D_fine':
            local=models.parent_crop(joint,region)
            local=torch.cat((local,wide.mean((2,3,4),keepdim=True).expand_as(local)),1)
            cond=legacy.FineCondition(local,wide,cond.query_offset(),
                models.parent_crop(aligned,region),coarse,'training_truth')
        presentations.append((target,cond))
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    rng=torch.Generator(device=device if device!='meta' else 'cpu').manual_seed(cfg['objective_seed'])
    return model,optimizer,rng,presentations,kind


def update(case,cfg):
    model,opt,rng,presentations,kind=case; model.train(); opt.zero_grad(set_to_none=True)
    losses=[]
    for target,cond in presentations:
        value,_=(models.objective(model,target,cond,rng,kind,cfg['decoder_std'])
                 if isinstance(model,models.CoupledBackbone) else
                 legacy.loss(model,target,cond,rng,cfg['decoder_std']))
        if not torch.isfinite(value): raise FloatingPointError('nonfinite synthetic objective')
        (value/len(presentations)).backward(); losses.append(float(value.detach()))
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'],error_if_nonfinite=True)
    opt.step()
    return dict(loss=float(np.mean(losses)),gradient_norm=float(norm),parent_evaluations=len(presentations))


def cpu_state(value):
    if isinstance(value,torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value,dict): return {k:cpu_state(v) for k,v in value.items()}
    if isinstance(value,list): return [cpu_state(v) for v in value]
    if isinstance(value,tuple): return tuple(cpu_state(v) for v in value)
    return value


def pack(case,step,metrics):
    model,opt,rng,_,_=case
    return dict(binding=binding(),step=step,metrics=metrics,model=cpu_state(model.state_dict()),
        optimizer=cpu_state(opt.state_dict()),generator=rng.get_state().cpu(),
        torch_cpu=torch.get_rng_state(),torch_cuda=torch.cuda.get_rng_state().cpu(),
        python_rng=random.getstate(),numpy_rng=np.random.get_state())


def atomic_state(path,value):
    fd,temporary=tempfile.mkstemp(prefix='technical-state-',suffix='.tmp',dir=path.parent); os.close(fd)
    torch.save(value,temporary)
    with open(temporary,'rb') as stream: os.fsync(stream.fileno())
    os.link(temporary,path); os.unlink(temporary)
    return c.file_record(path,content_hash=True)


def restore(case,state):
    if state['binding']!=binding(): raise ValueError('technical checkpoint source drift')
    model,opt,rng,_,_=case
    model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer'])
    rng.set_state(state['generator']); torch.set_rng_state(state['torch_cpu'])
    torch.cuda.set_rng_state(state['torch_cuda']); random.setstate(state['python_rng'])
    np.random.set_state(state['numpy_rng'])


def inference_condition(cond,scalar=False):
    if isinstance(cond,torch.Tensor): return cond[:1] if scalar else cond
    data={field.name:getattr(cond,field.name) for field in fields(cond)}
    if scalar: data={k:v[:1] if isinstance(v,torch.Tensor) else v for k,v in data.items()}
    # A fixed zero synthetic coarse chart, NEVER the training coarse target.
    for key in ('coarse_local','coarse_joint','coarse_wide'):
        if key in data and data[key] is not None: data[key]=torch.zeros_like(data[key])
    if data.get('coarse_source') is not None: data['coarse_source']='fixed_mean'
    return type(cond)(**data)


def draw(model,cond,kind,nfe,seeds,cfg):
    if isinstance(model,models.CoupledBackbone):
        return models.sample(model,cond,kind,nfe if kind=='vdm' else nfe//2,seeds,cfg['vdm_noise_grid'])
    return legacy.coupled_sample(model,cond,nfe,seeds,noise_grid=cfg['vdm_noise_grid'])


def map_tensors(value, function):
    """Preserve factor/condition structure; repeated references copy separately."""
    if isinstance(value, torch.Tensor): return function(value)
    if is_dataclass(value):
        return type(value)(**{field.name: map_tensors(getattr(value, field.name), function)
                              for field in fields(value)})
    if isinstance(value, tuple): return tuple(map_tensors(v, function) for v in value)
    if isinstance(value, list): return [map_tensors(v, function) for v in value]
    if isinstance(value, dict): return {k: map_tensors(v, function) for k, v in value.items()}
    return value


def transfer_probe(presentations, batch_pairs):
    """Conservative synchronous/pageable host transfer, no overlap claim.

    This transfers the synthetic factor presentations including both independent
    parents. Shared observation tensors are intentionally copied per occurrence;
    it does not assume an unimplemented shared-tensor GPU cache. CPU conversion
    of the already-built fixture is outside the transfer measurement.
    """
    host = map_tensors(presentations, lambda v: v.detach().cpu().clone())
    sizes = []
    map_tensors(host, lambda v: sizes.append(v.numel() * v.element_size()) or v)
    elapsed = []
    for _ in range(3):
        torch.cuda.synchronize(); started = time.monotonic()
        copied = map_tensors(host, lambda v: v.to('cuda', non_blocking=False))
        torch.cuda.synchronize(); elapsed.append(time.monotonic() - started)
        del copied
    return dict(seconds_per_batch=elapsed, seconds_per_pair=float(np.median(elapsed)) / batch_pairs,
                tensor_bytes_per_batch=sum(sizes), repeats=3, pinned_memory=False,
                nonblocking=False, shared_references_deduplicated=False,
                asynchronous_overlap=False, synthetic_shapes_only=True)


def benchmark(name,folder):
    device=runtime(); cfg=config(); case=make_case(name,cfg,device)
    model,opt,rng,presentations,kind=case; records=[]
    transfer = transfer_probe(presentations, cfg['batch_pairs'])
    checkpoint_seconds = []
    def checkpoint(path, step, metrics):
        torch.cuda.synchronize(); started = time.monotonic()
        result = atomic_state(path, pack(case, step, metrics))
        checkpoint_seconds.append(time.monotonic() - started)
        return result
    for _ in range(cfg['warmup_updates']): metrics=update(case,cfg)
    initial=checkpoint(folder/'TECHNICAL_INITIAL.pt',cfg['warmup_updates'],metrics)
    metrics=update(case,cfg)
    expected=checkpoint(folder/'TECHNICAL_EXPECTED.pt',cfg['warmup_updates']+1,metrics)
    torch.cuda.reset_peak_memory_stats(); times=[]
    for _ in range(cfg['timed_updates']):
        torch.cuda.synchronize(); started=time.monotonic(); metrics=update(case,cfg); torch.cuda.synchronize()
        times.append(time.monotonic()-started)
    sampling_state=checkpoint(folder/'TECHNICAL_SAMPLING_STATE.pt',
        cfg['warmup_updates']+1+cfg['timed_updates'],metrics)
    saved_draws=[]
    for nfe in cfg['sample_nfe']:
        elapsed=0.; errors=[]; hashes=[]
        for side,(_,condition) in enumerate(presentations):
            cond=inference_condition(condition)
            # Independent parent streams; never crop a shared overlapping noise.
            seeds=[810000+side*1000+i for i in range(cfg['batch_pairs'])]
            torch.cuda.synchronize(); started=time.monotonic()
            z=draw(model,cond,kind,nfe,seeds,cfg); torch.cuda.synchronize()
            elapsed+=time.monotonic()-started
            if not torch.isfinite(z).all(): raise FloatingPointError('nonfinite technical draw')
            if name.endswith('fine') and float(legacy.block_mean(z).abs().max())>2e-6:
                raise ValueError('technical sampler left the fine subspace')
            if nfe==cfg['sample_nfe'][0]:
                scalar=draw(model,inference_condition(condition,True),kind,nfe,seeds[:1],cfg)
                relative=float((z[:1]-scalar).square().mean().sqrt()/scalar.square().mean().sqrt().clamp_min(1e-12))
                if relative>cfg['scalar_batch_relative_rms_tolerance']:
                    raise ValueError('scalar/batch path mismatch')
                errors.append(relative); del scalar
            hashes.append(hashlib.sha256(z.detach().cpu().contiguous().numpy().tobytes()).hexdigest())
            if nfe==cfg['sample_nfe'][-1]:
                item=atomic_state(folder/f'TECHNICAL_LATENT_NFE{nfe}_PART{side}.pt',z.detach().cpu())
                loaded=torch.load(item['path'],map_location='cpu',weights_only=True)
                if not torch.equal(loaded,z.detach().cpu()): raise ValueError('technical draw serialization mismatch')
                saved_draws.append(item); del loaded
            del z
        records.append(dict(nfe=nfe,network_evaluations_per_pair=nfe*len(presentations),
            batch_pair_seconds=elapsed,seconds_per_pair=elapsed/cfg['batch_pairs'],
            scalar_batch_relative_rms=errors,technical_draw_sha256=hashes))
    result=dict(**coord.provenance(),case=name,binding=binding(),outputs=[initial,expected,sampling_state,*saved_draws],
        device=torch.cuda.get_device_name(),torch_version=str(torch.__version__),cuda_version=torch.version.cuda,
        device_total_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
        cudnn_version=torch.backends.cudnn.version(),dtype='float32',tf32=False,attention_backend='math',
        parameters=sum(p.numel() for p in model.parameters()),batch_pairs=cfg['batch_pairs'],
        parent_evaluations_per_pair=len(presentations),update_seconds=times,
        median_update_seconds=float(np.median(times)),seconds_per_training_pair=float(np.median(times))/cfg['batch_pairs'],
        host_to_device=transfer,checkpoint_pack_write_fsync_hash_seconds=checkpoint_seconds,
        checkpoint_measurement_includes=['GPU_to_CPU_model_optimizer_RNG','serialization','fsync','SHA256'],
        unique_updates=cfg['warmup_updates']+1+cfg['timed_updates'],sampling=records,
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
        peak_host_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        checkpoint_replay_step=cfg['warmup_updates']+1,
        sampling_checkpoint_step=cfg['warmup_updates']+1+cfg['timed_updates'],
        scientific_fit=False,synthetic_only=True,**{'pass':True})
    c.atomic_json(folder/'TIMINGS.json',result)


def replay(name,folder):
    device=runtime(); cfg=config(); receipt=coord.verify_receipt(folder/'TIMINGS.json')
    if receipt['binding']!=binding() or receipt['case']!=name: raise ValueError('replay source/case mismatch')
    state=torch.load(receipt['outputs'][0]['path'],map_location='cpu',weights_only=False)
    expected=torch.load(receipt['outputs'][1]['path'],map_location='cpu',weights_only=False)
    case=make_case(name,cfg,device); restore(case,state)
    metrics=update(case,cfg); actual=pack(case,state['step']+1,metrics)
    if not equal_state(actual,expected): raise ValueError('fresh-process optimizer/model/RNG replay differs')
    c.atomic_json(folder/'REPLAY.json',dict(**coord.provenance(),case=name,binding=binding(),
        exact_fresh_process_replay=True,optimizer_updates=1,scientific_fit=False,
        sources=[c.file_record(folder/'TIMINGS.json',content_hash=True)],outputs=[],**{'pass':True}))


def run():
    runtime(); cfg=config(); usage=accounting()
    own=[row for row in usage['allocations'] if row['job_id']==os.environ['SLURM_JOB_ID']]
    if len(own)!=1 or own[0]['kind']!='gpu' or own[0]['gpus']!=1:
        raise PermissionError('register the approved single-GPU allocation before benchmarking')
    if usage['gpu_hours']>=c.config()['approval']['gpu_hours']:
        raise RuntimeError('technical GPU allowance exhausted')
    if used_bytes(c.ROOT)+cfg['new_artifact_reserve_bytes']>c.config()['approval']['scratch_bytes']:
        raise RuntimeError('insufficient approved artifact reserve')
    root=coord.ROOT/'technical_gpu'/c.digest(binding())[:16]; deadline=time.monotonic()+cfg['max_run_seconds']
    with c.single_writer(root):
        receipts=[]
        for name in CASES:
            complete=root/f'{name}_COMPLETE.json'
            if complete.exists():
                record=coord.verify_receipt(complete)
                if record['binding']!=binding(): raise ValueError('completed technical case source drift')
                for item in record['outputs']: coord.verify_receipt(item['path'])
                receipts.append(c.file_record(complete,content_hash=True)); continue
            if time.monotonic()>deadline-300: return 75
            folder=root/name/f'attempt_{time.time_ns()}'; folder.mkdir(parents=True,exist_ok=False)
            for worker in ('benchmark','replay'):
                command=[sys.executable,'-u','-m','workflows.sbi.e2e_coupled_gpu_benchmark',
                         '--worker',worker,'--case',name,'--folder',str(folder)]
                with (folder/f'{worker}.log').open('x') as log:
                    subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,
                                   timeout=max(1,min(900,deadline-time.monotonic()-60)))
            timing=coord.verify_receipt(folder/'TIMINGS.json'); replay_record=coord.verify_receipt(folder/'REPLAY.json')
            if not replay_record['exact_fresh_process_replay']: raise ValueError('replay did not qualify')
            result=dict(**coord.provenance(),case=name,binding=binding(),timing=timing,
                outputs=[c.file_record(folder/file,content_hash=True) for file in ('TIMINGS.json','REPLAY.json')],
                synthetic_only=True,scientific_fit=False,**{'pass':True})
            c.atomic_json(complete,result); receipts.append(c.file_record(complete,content_hash=True))
            print(json.dumps(dict(case=name,update_seconds=timing['median_update_seconds'],
                                  replay=True,sampling=timing['sampling'])),flush=True)
        final=root/'GPU_BENCHMARK_COMPLETE.json'
        if final.exists():
            previous=coord.verify_receipt(final)
            if previous['binding']!=binding() or previous['outputs']!=receipts:
                raise ValueError('full benchmark completion drift')
            return 0
        c.atomic_json(final,dict(**coord.provenance(),binding=binding(),
            cases=list(CASES),outputs=receipts,synthetic_only=True,scientific_fit=False,
            full_preparation_complete=False,**{'pass':True}))
    return 0


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',choices=['benchmark','replay']); parser.add_argument('--case',choices=CASES)
    parser.add_argument('--folder',type=Path)
    args=parser.parse_args()
    if args.worker:
        folder=c.guarded(args.folder,output=True)
        if coord.ROOT/'technical_gpu' not in folder.parents: raise PermissionError('technical worker path required')
        (benchmark if args.worker=='benchmark' else replay)(args.case,folder)
    else: raise SystemExit(run())
