"""Controlled diversity x global-normalization matrix, training phases only."""
import argparse
import copy
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import time
import h5py
import numpy as np
import torch
from torch import nn
from workflows.sbi import e2e_wide_pipeline as p
from workflows.sbi.e2e_wide_data import COARSE_CHANNELS, LOCAL_CHANNELS, IDENTITY
from workflows.sbi.e2e_wide_research_canary import preflight, NORMALIZATION
from workflows.sbi.e2e_wide_continue import checked_binding
from workflows.sbi.e2e_wide_denoising_audit import Bands, TRAIN384
from workflows.sbi.e2e_multinoise_models import coefficients, loss_for
from workflows.sbi.e2e_multinoise_test import exposure
from workflows.sbi.e2e_skip_path_test import make_skip

CONFIG = p.REPO/'configs/e2e_diversity_norm_20260915.json'
OLD = Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v2/wide_pipeline_v1/skip_path_20260915_58368502')


def overlap(a, b):
    return a['phase'] == b['phase'] and all(any(min(x[1], y[1]) > max(x[0], y[0]) for x in aa for y in bb)
        for aa, bb in zip(a['source_box_footprint'], b['source_box_footprint']))


def panels(rows):
    """Conservative 128-cell source-footprint exclusion, not just different IDs."""
    train = []; transfer = []
    for phase in ('ph000', 'ph002', 'ph003'):
        ev = sorted([r for r in rows if r['phase'] == phase and r['cap'] == 'SGC'
                     and r['anchor_id'].endswith('boundary_00')], key=lambda r: r['anchor_id'])
        candidates = [r for r in rows if r['phase'] == phase and r['cap'] == 'NGC' and not any(overlap(r, e) for e in ev)]
        tiny = next(r for r in candidates if r['anchor_id'].endswith('s0_boundary_00'))
        others = [r for r in candidates if r != tiny and not overlap(r, tiny)]
        valid = [combo for combo in itertools.combinations(others, 4)
                 if not any(overlap(a, b) for a, b in itertools.combinations(combo, 2))]
        if not valid or len(ev) != 4:
            raise ValueError('cannot construct registered disjoint panel')
        # Metadata-only shell coverage first, then separation; no target-based selection.
        def score(combo):
            rr = (tiny, *combo)
            distances = [sum((x-y)**2 for x,y in zip(a['center'],b['center'])) for a,b in itertools.combinations(rr,2)]
            return (-len({r['shell'] for r in rr}), -min(distances), tuple(sorted(r['anchor_id'] for r in combo)))
        pool = list(min(valid, key=score)); ordered = [tiny]
        while pool:
            nxt = min(pool, key=lambda r: (-min(sum((x-y)**2 for x,y in zip(r['center'],s['center'])) for s in ordered), r['anchor_id']))
            ordered.append(nxt); pool.remove(nxt)
        train.append(ordered); transfer.extend(ev)
    train = [train[phase][i] for i in range(5) for phase in range(3)]
    if any(overlap(a,b) for a,b in itertools.combinations(train,2)) or any(overlap(a,b) for a in train for b in transfer):
        raise ValueError('training/transfer source footprint overlap')
    keys = ('anchor_id','phase','cap','center','redshift','shell','support_stratum','science_core_support_fraction','source_box_footprint')
    return {'train': [{k:r[k] for k in keys} for r in train], 'transfer': [{k:r[k] for k in keys} for r in transfer],
            'transfer_overlap_pairs': [[a['anchor_id'],b['anchor_id']] for a,b in itertools.combinations(transfer,2) if overlap(a,b)],
            'source_mapping_caveat': 'uses registered source_box_footprint mapping; coarse conditioning domains may overlap'}


def identity_norm():
    return dict(target_mean=0., target_std=1., fine_mean=[0.]*13, fine_std=[1.]*13, wide_mean=[0.]*12, wide_std=[1.]*12)


def fit_norm(items):
    """Global train-only pooled moments, represented in original normalized units."""
    def moments(key):
        means = np.stack([np.mean(v[key], axis=(-3,-2,-1), dtype=np.float64) for v in items])
        squares = np.stack([np.mean(np.square(v[key], dtype=np.float64), axis=(-3,-2,-1)) for v in items])
        mean = means.mean(0); std = np.sqrt(np.maximum(squares.mean(0)-mean**2, 1e-12))
        return mean, std
    tm, ts = moments('target'); cm, cs = moments('coarse'); fm, fs = moments('condition'); wm, ws = moments('wide')
    for i, name in enumerate(LOCAL_CHANNELS):
        if name in IDENTITY: fm[i], fs[i] = 0., 1.
    for i, name in enumerate(COARSE_CHANNELS):
        if name in IDENTITY: wm[i], ws[i] = 0., 1.
    fm[-1], fs[-1] = cm[0], cs[0]  # same coarse chart, not a separate interpolation normalizer
    return dict(target_mean=float(tm[0]), target_std=float(ts[0]), fine_mean=fm.tolist(), fine_std=fs.tolist(),
                wide_mean=wm.tolist(), wide_std=ws.tolist())


def inverse_norm(norm):
    n = copy.deepcopy(norm)
    n['target_mean'] = -norm['target_mean']/norm['target_std']; n['target_std'] = 1/norm['target_std']
    for level in ('fine', 'wide'):
        n[level+'_mean'] = (-np.array(norm[level+'_mean'])/norm[level+'_std']).tolist()
        n[level+'_std'] = (1/np.array(norm[level+'_std'])).tolist()
    return n


class AffineChart(nn.Module):
    """Physical-noise/physical-loss-preserving VP chart, finite at both endpoints.

    New clean coordinate y'=(y-m)/s. q²=s²*a²+b²; x'=(x-a*m)/q,
    a'=s*a/q, b'=b/q. Return the new model's v in original coordinates:
    v=a*b*(1-s²)*x/q²-b*m/q²+(s/q)*v'. No division by a or b.
    An inverse inner chart is an exact frozen-function reparameterization control;
    swapping the chart alone is a sensitivity intervention, NOT an equivalent model.
    """
    def __init__(self, model, norm):
        super().__init__(); self.model = model; self.tau = model.tau; self.identity = norm == identity_norm()
        self.register_buffer('mean', torch.tensor(norm['target_mean'], dtype=torch.float32))
        self.register_buffer('scale', torch.tensor(norm['target_std'], dtype=torch.float32))
        if norm['target_std'] <= 0 or not np.isfinite([norm['target_mean'],norm['target_std']]).all():
            raise ValueError('finite positive target scale required')
        for level in ('fine', 'wide'):
            for key in ('mean','std'):
                v = torch.tensor(norm[level+'_'+key], dtype=torch.float32)[None,:,None,None,None]
                if not torch.isfinite(v).all() or (key == 'std' and (v <= 0).any()):
                    raise ValueError('invalid condition normalization')
                self.register_buffer(level+'_'+key, v)

    def forward(self, state, time, condition, wide_condition=None, context_present=None):
        if self.identity:
            return self.model(state, time, condition, wide_condition, context_present)
        a,b,_ = coefficients(time, self.tau); s,m = self.scale,self.mean
        q = (s*s*a*a+b*b).sqrt(); state_new = (state-a*m)/q
        t_new = (2/math.pi)*torch.atan2(b, s*a).flatten()
        c = (condition-self.fine_mean)/self.fine_std; w = (wide_condition-self.wide_mean)/self.wide_std
        v = self.model(state_new, t_new, c, w, context_present)
        return a*b*(1-s*s)*state/q.square()-b*m/q.square()+s*v/q


def source_hashes():
    names = ['e2e_diversity_norm.py','e2e_skip_path_test.py','e2e_multinoise_models.py','e2e_multinoise_test.py',
             'e2e_fine_learning_test.py','e2e_wide_denoising_audit.py','e2e_wide_data.py']
    paths = [p.REPO/'workflows/sbi'/name for name in names]+[CONFIG]
    return {str(x.relative_to(p.REPO)):p.sha256(x) for x in paths}


def prepare(out):
    device=p.runtime(); c,_,_,_ = preflight(); ds = p.dataset_for(c,NORMALIZATION); selection = panels(ds.rows)
    out = p.output_path(c,out); out.mkdir(parents=True,exist_ok=False); items = []; metadata = []
    with h5py.File(out/'cache.h5','x') as f:
        for row in selection['train']+selection['transfer']:
            item = ds[next(i for i,r in enumerate(ds.rows) if r['anchor_id']==row['anchor_id'])]
            arrays = dict(target=item['fine_target'],coarse=item['coarse_target'],condition=item['fine_condition'],wide=item['coarse_condition'])
            g = f.create_group(row['anchor_id'])
            for key,value in arrays.items():g.create_dataset(key,data=value)
            y = arrays['target'][0].astype(np.float64); mu=y.mean(); std=y.std(); z=(y-mu)/max(std,1e-10)
            physical = ds.inverse_target(y,'fine')
            # Metadata and target summaries are descriptive only, never inference-time normalizers.
            stats = dict(mean=float(physical.mean()),std=float(physical.std()),skew=float(np.mean(z**3)),
                         excess_kurtosis=float(np.mean(z**4)-3),q01=float(np.quantile(physical,.01)),
                         q99=float(np.quantile(physical,.99)),redshift=row['redshift'],shell=row['shell'],
                         support_fraction=row['science_core_support_fraction'],
                         observed_fraction=float(item['masks']['observed_parent'].mean()),
                         support_mean=float(arrays['condition'][1].mean()),
                         angular_response_mean=float(arrays['condition'][2].mean()))
            coarse_stats=ds.normalization['targets']['coarse']
            density=physical+arrays['condition'][-1]*coarse_stats['std']+coarse_stats['mean']
            counts_stats=ds.normalization['fine']['counts']
            counts=np.expm1(arrays['condition'][0]*counts_stats['std']+counts_stats['mean'])
            stats.update(density_mean=float(density.mean()),density_std=float(density.std()),
                         density_q01=float(np.quantile(density,.01)),density_q99=float(np.quantile(density,.99)),
                         density_under_mean_fraction=float(np.mean(density<0)),mean_galaxy_count=float(counts.mean()))
            metadata.append(dict(anchor_id=row['anchor_id'],phase=row['phase'],cap=row['cap'],stats=stats))
            if row in selection['train']:items.append(arrays)
            print('CACHE',row['anchor_id'],flush=True)
    strict = fit_norm(items); tiny = fit_norm(items[:3])
    cfg=json.loads(CONFIG.read_text());cfg.update(cfg['replicates'][0]);smoke=[]
    item={k:p.tensor(v,device) for k,v in items[0].items()};del items
    for name,norm in [('current',identity_norm()),('strict_global_train',strict)]:
        model=AffineChart(make_skip('unet_film',c,cfg,device),norm).to(device)
        torch.cuda.reset_peak_memory_stats();stamp=time.monotonic()
        for ratio in (.05,20.):
            noise=torch.randn(item['target'].shape,device=device,generator=torch.Generator(device=device).manual_seed(91533))
            model.zero_grad(set_to_none=True)
            loss,_,_=loss_for(model,item['target'],noise,2*math.atan(ratio)/math.pi,item['condition'],item['wide'])
            if not torch.isfinite(loss):raise FloatingPointError('smoke objective')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        torch.cuda.synchronize();smoke.append(dict(normalization=name,seconds=time.monotonic()-stamp,
            peak_bytes=torch.cuda.max_memory_allocated(),finite=True));del model
    p.write_json(out/'SMOKE.json',smoke)
    old = json.loads((OLD/'SKIP_COMPLETE.json').read_text())
    binding = checked_binding(TRAIN384,p.provenance(c,ds))
    if p.digest(binding)!=old['registration']['parent_binding_sha256']:raise ValueError('parent provenance drift')
    receipt = dict(selection=selection,normalizations={'current':identity_norm(),'strict_global_train':strict,'tiny_train':tiny},
                   metadata=metadata,original_normalization=ds.normalization,base_binding=binding,source_sha256=source_hashes(),
                   cache_sha256=p.sha256(out/'cache.h5'),prior_registration=old['registration'],
                   prior_checkpoint_sha256=p.sha256(OLD/'unet_film/update_003072.pt'),heldout_payloads_read=False,
                   config=json.loads(CONFIG.read_text()),job_id=os.environ['SLURM_JOB_ID'])
    p.write_json(out/'PREPARED.json',receipt); print('PREPARED',json.dumps(strict),flush=True)


def load_items(root,device):
    receipt=json.loads((root/'PREPARED.json').read_text())
    if p.sha256(root/'cache.h5')!=receipt['cache_sha256']:raise ValueError('cache drift')
    if receipt['source_sha256']!=source_hashes():raise ValueError('source drift')
    items={}
    with h5py.File(root/'cache.h5','r') as f:
        for row in receipt['selection']['train']+receipt['selection']['transfer']:
            g=f[row['anchor_id']];y=g['target'][:]
            items[row['anchor_id']]=dict(target=p.tensor(y,device),target_np=y[0],condition=p.tensor(g['condition'][:],device),
                wide=p.tensor(g['wide'][:],device),group='transfer' if row['cap']=='SGC' else 'train_pool',phase=row['phase'])
    return receipt,items


@torch.no_grad()
def evaluate(model,items,cfg,scale,bands,clean_ratios=None):
    model.eval(); rows=[]
    for anchor,item in items.items():
        y=item['target']; truth=item['target_np']*scale
        for ratio in (cfg['clean_ratios'] if clean_ratios is None else clean_ratios):
            t=2*math.atan(ratio)/math.pi; a=1/math.sqrt(1+ratio*ratio); b=ratio*a
            x=a*y; pred=model(x,y.new_tensor([t]),item['condition'],wide_condition=item['wide']); clean=a*x-b*pred
            error=(clean-y)[0,0].cpu().numpy()*scale
            metrics=bands.compare(clean[0,0].cpu().numpy()*scale,truth)
            if ratio==0 and float(np.max(np.abs(error)))>1e-6:raise ValueError('endpoint identity failed')
            rows.append(dict(anchor_id=anchor,group=item['group'],phase=item['phase'],kind='clean',ratio=ratio,
                rms=float(np.sqrt(np.mean(error.astype(np.float64)**2))),bias=float(error.mean()),
                max_abs=float(np.max(np.abs(error))),rms_over_injected=(float(np.sqrt(np.mean(error.astype(np.float64)**2)))/(scale*ratio) if ratio else None),metrics=metrics))
        for rep in range(cfg['evaluation_replicates']):
            seed=p.seed_for(cfg['evaluation_seed'],anchor,f'evaluation-{rep}','fine')
            noise=torch.randn(y.shape,device=y.device,generator=torch.Generator(device=y.device).manual_seed(seed))
            for ratio in cfg['noisy_ratios']:
                t=2*math.atan(ratio)/math.pi; a=1/math.sqrt(1+ratio*ratio); b=ratio*a
                x=a*y+b*noise; v=model(x,y.new_tensor([t]),item['condition'],wide_condition=item['wide']); clean=a*x-b*v
                metrics=bands.compare(clean[0,0].cpu().numpy()*scale,truth,noise[0,0].cpu().numpy()*scale)
                metrics['noise_amplitude']=(np.array(metrics['noise_leakage_coefficient'])/b).tolist()
                rows.append(dict(anchor_id=anchor,group=item['group'],phase=item['phase'],kind='noisy',ratio=ratio,rep=rep,seed=seed,
                                 velocity_mse=float((v-(a*noise-b*y)).square().mean()),metrics=metrics))
    return rows


def field_for(update,n,train):
    block=update//3; index=(block//6+block%6) % (n//3)
    return train[3*index+update%3]


def train(root,replica,out):
    device=p.runtime(); c,_,_,_=preflight(); receipt,items=load_items(root,device); cfg=receipt['config']; cfg.update(cfg['replicates'][replica])
    out=p.output_path(c,out);out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    scale=receipt['original_normalization']['targets']['fine']['std'];bands=Bands(96,3.383,[0,.08,.16,.32,np.inf])
    train_ids=[r['anchor_id'] for r in receipt['selection']['train']]; results=[]; hashes={}
    registration=dict(replica=replica,config=cfg,prepared_sha256=p.sha256(root/'PREPARED.json'),source_sha256=source_hashes(),
                      git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=p.REPO,text=True).strip(),job_id=os.environ['SLURM_JOB_ID'])
    p.write_json(out/'STARTED.json',registration)
    for n in cfg['field_counts']:
        for norm_name in cfg['normalizations']:
            stamp=time.monotonic(); label=f'n{n:02d}_{norm_name}';branch=out/label;branch.mkdir()
            raw=make_skip('unet_film',c,cfg,device);model=AffineChart(raw,receipt['normalizations'][norm_name]).to(device)
            optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
            binding={**receipt['base_binding'],'diversity_norm':dict(registration=registration,n=n,normalization=norm_name)}
            curve=[];history=[]
            for update in range(cfg['updates']+1):
                if time.monotonic()-start>cfg['maximum_work_seconds_per_replica']:raise TimeoutError('matrix cap reached')
                if update in cfg['evaluate_at']:
                    rows=evaluate(model,items,cfg,scale,bands)
                    point=dict(update=update,rows=rows);curve.append(point);p.write_json(branch/f'probe_{update:04d}.json',point)
                    print('PROBE',replica,label,update,flush=True)
                if update==cfg['updates']:break
                anchor=field_for(update,n,train_ids);item=items[anchor]
                # Same Gaussian tensor at a given update for every diversity/norm cell.
                seed=p.seed_for(cfg['train_seed'],train_ids[update%3],update,'noise')
                generator=torch.Generator(device=device).manual_seed(seed);noise=torch.randn(item['target'].shape,device=device,generator=generator)
                t,ratio,_=exposure(update,cfg);model.train();optimizer.zero_grad(set_to_none=True)
                loss,_,_=loss_for(model,item['target'],noise,t,item['condition'],item['wide'])
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite physical objective')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['clip'],error_if_nonfinite=True);optimizer.step()
                history.append(dict(update=update+1,anchor_id=anchor,noise_seed=seed,time=t,ratio=ratio,loss=float(loss.detach()),gradient_norm=float(norm)))
                if (update+1)%512==0:print('TRAIN',replica,label,update+1,float(np.mean([r['loss'] for r in history[-512:]])),flush=True)
                if update+1 in cfg['evaluate_at']:
                    path=branch/f'update_{update+1:06d}.pt';p.save_checkpoint(path,model=model,optimizer=optimizer,generator=generator,binding=binding,
                        stage='fine',method='diffusion',step=update+1,history=history);hashes[str(path.relative_to(out))]=p.sha256(path)
            replay=None
            if n==3 and norm_name=='current' and replica==0:
                expected={**receipt['base_binding'],'skip_path':dict(registration=receipt['prior_registration'],arm='unet_film')}
                prior=p.load_checkpoint(OLD/'unet_film/update_003072.pt',expected,'fine','diffusion')['model']
                diffs=[float((v.cpu()-prior[k]).abs().max()) for k,v in raw.state_dict().items()]
                for k,v in raw.state_dict().items():torch.testing.assert_close(v.cpu(),prior[k],atol=2e-6,rtol=2e-5)
                replay=dict(max_parameter_difference=max(diffs),exact=max(diffs)==0)
            result=dict(n=n,normalization=norm_name,replica=replica,curve=curve,history=history,replay=replay,elapsed_seconds=time.monotonic()-stamp)
            results.append(result);p.write_json(branch/'COMPLETE.json',result);print('CELL COMPLETE',replica,label,result['elapsed_seconds'],flush=True)
            del model,raw,optimizer
    if source_hashes()!=registration['source_sha256']:raise ValueError('source drift')
    p.write_json(out/'MATRIX_COMPLETE.json',dict(registration=registration,results=results,checkpoints=hashes,complete=True,
        elapsed_seconds=time.monotonic()-start,heldout_payloads_read=False,training_ready=False));print('MATRIX COMPLETE',replica,flush=True)


def frozen(root,out):
    device=p.runtime();c,_,_,_=preflight();receipt,items=load_items(root,device);cfg=receipt['config'];cfg.update(cfg['replicates'][0])
    expected={**receipt['base_binding'],'skip_path':dict(registration=receipt['prior_registration'],arm='unet_film')}
    checkpoint=p.load_checkpoint(OLD/'unet_film/update_003072.pt',expected,'fine','diffusion')
    raw=make_skip('unet_film',c,cfg,device);raw.load_state_dict(checkpoint['model']);raw.eval()
    scale=receipt['original_normalization']['targets']['fine']['std'];bands=Bands(96,3.383,[0,.08,.16,.32,np.inf])
    strict=receipt['normalizations']['strict_global_train'];schemes=copy.deepcopy(receipt['normalizations'])
    target=identity_norm();target.update(target_mean=strict['target_mean'],target_std=strict['target_std']);schemes['strict_target_only']=target
    context=copy.deepcopy(strict);context.update(target_mean=0.,target_std=1.);schemes['strict_conditions_only']=context
    results={}
    for name,norm in schemes.items():
        model=AffineChart(raw,norm).to(device);results[name]=evaluate(model,items,cfg,scale,bands);print('FROZEN',name,flush=True)
    roundtrip=AffineChart(AffineChart(raw,inverse_norm(strict)),strict).to(device)
    parity=[]
    with torch.no_grad():
        for anchor,item in items.items():
            for ratio in (0.,.001,.05,.2,20.):
                t=item['target'].new_tensor([2*math.atan(ratio)/math.pi]);a,b,_=coefficients(t)
                x=a*item['target'];v0=raw(x,t,item['condition'],item['wide']);v1=roundtrip(x,t,item['condition'],item['wide'])
                d0=a*x-b*v0;d1=a*x-b*v1;torch.testing.assert_close(d0,d1,atol=2e-5,rtol=3e-5)
                parity.append(dict(anchor_id=anchor,ratio=ratio,max_abs=float((d0-d1).abs().max())))
    parent_path=TRAIN384/'diffusion_fine/step_000384.pt'
    parent=p.build_model(c,'fine',device).eval()
    parent.load_state_dict(p.load_checkpoint(parent_path,receipt['base_binding'],'fine','diffusion')['model'])
    parent_rows=evaluate(parent,items,cfg,scale,bands)
    p.write_json(out,dict(complete=True,results=results,roundtrip=parity,parent384=parent_rows,parent_checkpoint_sha256=p.sha256(parent_path),prepared_sha256=p.sha256(root/'PREPARED.json'),
        source_sha256=source_hashes(),checkpoint_sha256=receipt['prior_checkpoint_sha256'],training_updates=0,
        interpretation='Direct scaler swaps are sensitivity interventions; conjugate chart roundtrip must preserve the same frozen physical function.',heldout_payloads_read=False))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=['prepare','train','frozen']);ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--output',type=Path);ap.add_argument('--replica',type=int,choices=[0,1],default=0);args=ap.parse_args()
    if args.command=='prepare':prepare(args.root)
    elif args.command=='train':train(args.root,args.replica,args.output)
    else:frozen(args.root,args.output)
