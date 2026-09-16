"""Bounded preservation-objective experiment; not an E2E or posterior release."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import numpy as np
import torch
from workflows.sbi import e2e_clean_limit as base
from workflows.sbi import e2e_diversity_norm as data
from workflows.sbi import e2e_diversity_norm_report as metrics
from workflows.sbi import e2e_durable as durable
from workflows.sbi import e2e_wide_pipeline as p
from workflows.sbi.e2e_clean_limit_launch import snapshot_paths, SCRATCH
from workflows.sbi.e2e_multinoise_models import loss_for
from workflows.sbi.e2e_preservation_loss import objective, auxiliary_ratio
from workflows.sbi.e2e_wide_continue import equal_state
from workflows.sbi.e2e_wide_denoising_audit import Bands

CONFIG = p.REPO / 'configs/e2e_preservation_objective_v1.json'
ARMS = ('control', 'identity_weak', 'identity_strong', 'identity_response')


def spec():
    value = json.loads(CONFIG.read_text())
    if (value['schema'] != 'e2e-preservation-objective-v1' or tuple(value['arms']) != ARMS or
            value['additional_updates'] != 6144 or value['parent_update'] != 24576 or
            value['replicas'] != [0, 1] or any(value[k] for k in
            ('heldout_access','full_e2e_training','automatic_extension','training_ready'))):
        raise ValueError('outside registered experiment')
    return value


def stage(root):
    """Commit first; stage source/configuration and receipts, not data payloads."""
    root = root.resolve(); s = spec(); old = Path(s['parent_root'])
    if root.parent != SCRATCH or not root.name.startswith('preservation_'):
        raise ValueError('new preservation_ child of the pipeline Scratch root required')
    if subprocess.check_output(['git','status','--porcelain'],cwd=p.REPO,text=True).strip():
        raise ValueError('commit source before freezing')
    for name,key in [('MANIFEST.json','parent_manifest_sha256'),('analysis/SUMMARY.json','parent_summary_sha256')]:
        if p.sha256(old/name) != s[key]: raise ValueError('parent evidence drift')
    revision = subprocess.check_output(['git','rev-parse','HEAD'],cwd=p.REPO,text=True).strip()
    names = snapshot_paths(subprocess.check_output(['git','ls-files','-z'],cwd=p.REPO).decode().split('\0'))
    names.append('docs/e2e_preservation_objective_v1.md')
    root.mkdir(exist_ok=False); source=root/'source';source.mkdir();(root/'logs').mkdir()
    archive=subprocess.Popen(['git','archive',revision,'--',*names],cwd=p.REPO,stdout=subprocess.PIPE)
    result=subprocess.run(['tar','-xf','-','-C',str(source)],stdin=archive.stdout)
    archive.stdout.close()
    if archive.wait() or result.returncode: raise RuntimeError('partial archive preserved')
    parents={}
    for replica in s['replicas']:
        folder=old/f'replica_{replica}'/s['parent_arm']
        complete=json.loads((folder/'COMPLETE.json').read_text())
        if not complete['complete'] or complete['update'] != s['parent_update']: raise ValueError('parent incomplete')
        parents[str(replica)]=dict(folder=str(folder),complete_sha256=p.sha256(folder/'COMPLETE.json'),
                                  checkpoint=complete['checkpoint'],binding=complete['binding'])
    durable.publish_json(root/'MANIFEST.json',dict(spec=s,parents=parents,git_revision=revision,source=str(source),
        source_sha256={name:p.sha256(source/name) for name in names if (source/name).is_file()},
        training_ready=False,full_e2e_training=False))
    print('STAGED',root,revision,flush=True)


def verify(root):
    m=json.loads((root/'MANIFEST.json').read_text())
    if m['spec'] != spec() or Path(m['source']).resolve()!=p.REPO.resolve():
        raise ValueError('launch from matching frozen snapshot')
    for rel,sha in m['source_sha256'].items():
        if p.sha256(p.REPO/rel)!=sha: raise ValueError('source drift: '+rel)
    return m


def inputs(root,replica,device):
    m=verify(root);s=m['spec'];old=Path(s['parent_root'])
    if p.sha256(old/'MANIFEST.json')!=s['parent_manifest_sha256'] or p.sha256(old/'analysis/SUMMARY.json')!=s['parent_summary_sha256']:
        raise ValueError('parent experiment drift')
    previous=json.loads((old/'MANIFEST.json').read_text())['spec']
    raw=Path(previous['parent_root'])
    if p.sha256(raw/'PREPARED.json')!=previous['prepared_sha256']: raise ValueError('data receipt drift')
    prepared,items=data.load_items(raw,device)
    parent=m['parents'][str(replica)];folder=Path(parent['folder'])
    if p.sha256(folder/'COMPLETE.json')!=parent['complete_sha256']: raise ValueError('parent completion drift')
    state,pointer=durable.load(folder,parent['binding'])
    if pointer!=parent['checkpoint'] or state['step']!=s['parent_update']: raise ValueError('parent checkpoint drift')
    return m,previous,prepared,items,state


def update(model,optimizer,generator,items,ids,cfg,s,previous,arm,index):
    anchor=data.field_for(index,15,ids);item=items[anchor]
    seed=p.seed_for(cfg['train_seed'],ids[index%3],index,'noise')
    generator.manual_seed(seed)
    noise=torch.randn(item['target'].shape,device=item['target'].device,generator=generator)
    aux_seed=p.seed_for(cfg['train_seed'],anchor,index,'preservation-auxiliary')
    aux=torch.randn(item['target'].shape,device=item['target'].device,
                    generator=torch.Generator(device=item['target'].device).manual_seed(aux_seed))
    t,ratio,_=base.schedule(index,cfg,previous,'near_zero')
    sigma=auxiliary_ratio(index,cfg['train_seed'],s)
    model.train();optimizer.zero_grad(set_to_none=True)
    loss,terms=objective(model,item['target'],noise,t,item['condition'],item['wide'],aux,
                         sigma,s['perturbation_fraction'],**s['arms'][arm])
    if not torch.isfinite(loss): raise FloatingPointError('nonfinite objective')
    loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['clip'],error_if_nonfinite=True)
    optimizer.step()
    return dict(update=index+1,anchor_id=anchor,noise_seed=seed,auxiliary_seed=aux_seed,time=t,ratio=ratio,
                auxiliary_ratio=sigma,loss=float(loss.detach()),gradient_norm=float(norm),
                terms={k:float(v) for k,v in terms.items()})


@torch.no_grad()
def evaluate(model,items,cfg,previous,prepared):
    """Legacy spectral gates plus direct physical noisy MSE and sub-noise response."""
    cfg=dict(cfg,clean_ratios=previous['clean_ratios'],noisy_ratios=previous['noisy_ratios'])
    scale=prepared['original_normalization']['targets']['fine']['std']
    bands=Bands(96,3.383,[0,.08,.16,.32,np.inf])
    # Original evaluation remains byte-for-byte callable; extra probes are separate.
    rows=data.evaluate(model,items,cfg,scale,bands)
    indexed={(r['anchor_id'],r['ratio'],r.get('rep')):r for r in rows if r['kind']=='noisy'}
    response=[];model.eval()
    for anchor,item in items.items():
        y=item['target']
        for rep in range(cfg['evaluation_replicates']):
            seed=p.seed_for(cfg['evaluation_seed'],anchor,f'evaluation-{rep}','fine')
            eps=torch.randn(y.shape,device=y.device,generator=torch.Generator(device=y.device).manual_seed(seed))
            for ratio in cfg['noisy_ratios']:
                a=1/math.sqrt(1+ratio*ratio);b=ratio*a;t=y.new_tensor([2*math.atan(ratio)/math.pi])
                x=a*y+b*eps;pred=a*x-b*model(x,t,item['condition'],wide_condition=item['wide'])
                indexed[anchor,ratio,rep]['physical_mse']=float((pred-y).double().square().mean())*scale**2
            # Distinct held-back fractions, not the .25 fraction used in the loss.
            for ratio in (.001,.01,.05):
                a=1/math.sqrt(1+ratio*ratio);b=ratio*a;t=y.new_tensor([2*math.atan(ratio)/math.pi])
                x=a*y;zero=a*x-b*model(x,t,item['condition'],wide_condition=item['wide'])
                for fraction in (.1,.5):
                    xp=x+b*fraction*eps;xm=x-b*fraction*eps
                    plus=a*xp-b*model(xp,t,item['condition'],wide_condition=item['wide'])
                    minus=a*xm-b*model(xm,t,item['condition'],wide_condition=item['wide'])
                    even=(plus+minus)/2;odd=(plus-minus)/2
                    mse=lambda z:float(z.double().square().mean())*scale**2
                    response.append(dict(anchor_id=anchor,phase=item['phase'],ratio=ratio,rep=rep,fraction=fraction,
                        identity_mse=mse(zero-y),even_drift_mse=mse(even-zero),odd_remaining_mse=mse(odd),
                        noisy_mse=.5*(mse(plus-y)+mse(minus-y)),
                        input_noise_mse=mse(b*fraction*eps)))
    if not metrics.finite_tree([rows,response]): raise ValueError('nonfinite evaluation')
    return dict(rows=rows,response=response)


def train(root,replica,arm):
    device=p.runtime();m,previous,prepared,items,parent=inputs(root,replica,device);s=m['spec']
    cfg,model,optimizer,generator=base.make(prepared,replica,device)
    ids=[r['anchor_id'] for r in prepared['selection']['train']]
    binding={**prepared['base_binding'],'preservation':dict(manifest_sha256=p.sha256(root/'MANIFEST.json'),
             replica=replica,arm=arm,parent_sha256=m['parents'][str(replica)]['checkpoint']['sha256'])}
    folder=root/f'replica_{replica}'/arm;end=s['parent_update']+s['additional_updates']
    probes=[s['parent_update']+u for u in s['evaluate_after']]
    with durable.single_writer(folder):
        state=parent
        if (folder/'LATEST.json').exists():state,pointer=durable.load(folder,binding)
        if (folder/'COMPLETE.json').exists():
            done=json.loads((folder/'COMPLETE.json').read_text())
            if state['step']!=end or done['checkpoint']!=pointer or done['binding']!=binding: raise ValueError('completion drift')
            print('ALREADY COMPLETE',replica,arm,flush=True);return
        base.restore(state,model,optimizer,generator);history=copy.deepcopy(state['history']);index=state['step']
        if len(history)!=index or not s['parent_update']<=index<=end:raise ValueError('invalid resume')
        def save():
            return durable.save(folder,model=model,optimizer=optimizer,generator=generator,binding=binding,
                                stage='fine',method='diffusion',step=index,history=history)
        if not (folder/'LATEST.json').exists():pointer=save()
        stopped=[]
        def stop(signum,frame):stopped.append(signum)
        signal.signal(signal.SIGUSR1,stop);signal.signal(signal.SIGTERM,stop)
        started=time.monotonic()
        while True:
            if stopped:
                if pointer['step']!=index:pointer=save()
                print('STOPPED SAFELY',index,flush=True);raise SystemExit(75)
            if index in probes:
                dest=folder/f'probe_{index:06d}.json'
                if dest.exists():
                    if json.loads(dest.read_text())['checkpoint']!=pointer:raise ValueError('probe drift')
                else:
                    result=evaluate(model,items,cfg,previous,prepared)
                    durable.publish_json(dest,dict(update=index,checkpoint=pointer,**result))
                print('PROBE',replica,arm,index,flush=True)
            if index==end:break
            history.append(update(model,optimizer,generator,items,ids,cfg,s,previous,arm,index));index+=1
            if index%s['checkpoint_every']==0 or index in probes or stopped:
                pointer=save();print('CHECKPOINT',replica,arm,index,history[-1]['terms'],flush=True)
        verify(root)
        durable.publish_json(folder/'COMPLETE.json',dict(complete=True,binding=binding,checkpoint=pointer,
            update=index,probes={str(u):p.sha256(folder/f'probe_{u:06d}.json') for u in probes},
            seconds=time.monotonic()-started,training_ready=False,heldout_payloads_read=False))


def gate(control, candidate, anchors, gate_spec, parent):
    """Paired fields, separate constraints; never trade a failed gate for a score."""
    idx=lambda rows:{(r['anchor_id'],r['kind'],r['ratio'],r.get('rep')):r for r in rows if r['anchor_id'] in anchors}
    before=idx(control);after=idx(candidate)
    if before.keys()!=after.keys():raise ValueError('unpaired evaluation rows')
    result={};checks=[]
    for ratio in sorted({r['ratio'] for r in candidate}):
        clean=[];noisy=[];phase={}
        for anchor in anchors:
            key=(anchor,'clean',ratio,None)
            if key in before:
                x=before[key]['rms'];y=after[key]['rms']
                clean.append(y/max(x,1e-30))
                if ratio==0:checks.append(after[key]['max_abs']<=1e-6)
            vals=[]
            for key in before:
                if key[:3]==(anchor,'noisy',ratio):
                    vals.append(after[key]['physical_mse']/max(before[key]['physical_mse'],1e-30))
                    phase.setdefault(after[key]['phase'],[]).append(vals[-1])
            if vals:noisy.append(float(np.median(vals)))
        record={}
        if clean and ratio>0:
            bound=gate_spec['maximum_median_clean_ratio'] if ratio in gate_spec['preservation_ratios'] else gate_spec['maximum_other_clean_ratio']
            record.update(median_clean_ratio=float(np.median(clean)),improved_fraction=float(np.mean(np.array(clean)<1)))
            checks.append(record['median_clean_ratio']<=bound)
            if ratio in gate_spec['preservation_ratios']:checks.append(record['improved_fraction']>=gate_spec['minimum_improved_fraction'])
        if noisy:
            record.update(median_noisy_mse_ratio=float(np.median(noisy)),
                          phase_noisy_mse_ratio={k:float(np.median(v)) for k,v in phase.items()})
            checks.append(record['median_noisy_mse_ratio']<=gate_spec['maximum_median_noisy_mse_ratio'])
            checks.extend(v<=gate_spec['maximum_phase_noisy_mse_ratio'] for v in record['phase_noisy_mse_ratio'].values())
        result[str(ratio)]=record
    legacy=metrics.summarize([r for r in candidate if r['ratio'] in (.05,.2)],anchors,parent)
    checks.extend(legacy[str(q)]['passed']==len(anchors) for q in (.05,.2))
    return dict(passed=bool(all(checks)),ratios=result,legacy=legacy)


def response_gate(control, candidate, anchors, gate_spec):
    key=lambda r:(r['anchor_id'],r['ratio'],r['fraction'],r['rep'])
    a={key(r):r for r in control if r['anchor_id'] in anchors}
    b={key(r):r for r in candidate if r['anchor_id'] in anchors}
    if not a or a.keys()!=b.keys():raise ValueError('unpaired near-clean probes')
    records=[]
    for ratio,fraction in sorted({(k[1],k[2]) for k in a}):
        fields=[];phases={}
        for anchor in anchors:
            keys=[k for k in a if k[:3]==(anchor,ratio,fraction)]
            if len(keys)!=2:raise ValueError('near-clean evaluation replicas missing')
            ratios={metric:float(np.median([b[k][metric]/max(a[k][metric],1e-30) for k in keys]))
                    for metric in ('noisy_mse','even_drift_mse','odd_remaining_mse')}
            fields.append(dict(anchor_id=anchor,**ratios))
            phases.setdefault(a[keys[0]]['phase'],[]).append(ratios['noisy_mse'])
        median=float(np.median([f['noisy_mse'] for f in fields]))
        worst=max(float(np.median(v)) for v in phases.values())
        records.append(dict(ratio=ratio,fraction=fraction,fields=fields,median_noisy_mse_ratio=median,
            worst_phase_ratio=worst,passed=median<=gate_spec['maximum_median_noisy_mse_ratio'] and
            worst<=gate_spec['maximum_phase_noisy_mse_ratio']))
    return dict(passed=all(r['passed'] for r in records),records=records)


def report(root):
    p.require_compute();m=verify(root);s=m['spec']
    old_spec=json.loads((Path(s['parent_root'])/'MANIFEST.json').read_text())['spec']
    if p.sha256(Path(s['parent_root'])/'MANIFEST.json')!=s['parent_manifest_sha256']:
        raise ValueError('parent manifest drift')
    raw=Path(old_spec['parent_root']);prepared=json.loads((raw/'PREPARED.json').read_text())
    if p.sha256(raw/'PREPARED.json')!=old_spec['prepared_sha256']:raise ValueError('prepared drift')
    frozen=json.loads((raw/'FROZEN.json').read_text())
    parent_manifest=json.loads((Path(s['parent_root'])/'MANIFEST.json').read_text())
    if p.sha256(raw/'FROZEN.json')!=parent_manifest['frozen_sha256']:raise ValueError('reference probe drift')
    reference={(r['anchor_id'],r['ratio'],r.get('rep')):r for r in frozen['parent384'] if r['kind']=='noisy'}
    groups=metrics.groups(prepared,15);loaded={};hashes={}
    for seed in s['replicas']:
        for arm in ARMS:
            folder=root/f'replica_{seed}'/arm;complete=json.loads((folder/'COMPLETE.json').read_text())
            if not complete['complete'] or complete['update']!=s['parent_update']+s['additional_updates']:raise ValueError('unfinished arm')
            checkpoint,pointer=durable.load(folder,complete['binding'])
            if pointer!=complete['checkpoint']:raise ValueError('completion/checkpoint drift')
            expected_binding={**prepared['base_binding'],'preservation':dict(
                manifest_sha256=p.sha256(root/'MANIFEST.json'),replica=seed,arm=arm,
                parent_sha256=m['parents'][str(seed)]['checkpoint']['sha256'])}
            if complete['binding']!=expected_binding:raise ValueError('wrong experiment/arm binding')
            expected_probes={str(s['parent_update']+u) for u in s['evaluate_after']}
            if set(complete['probes'])!=expected_probes:raise ValueError('missing registered checkpoint probes')
            cfg=copy.deepcopy(prepared['config']);cfg.update(cfg['replicates'][seed])
            ids=[r['anchor_id'] for r in prepared['selection']['train']]
            history=checkpoint['history'][s['parent_update']:]
            if len(history)!=s['additional_updates']:raise ValueError('wrong history length')
            for index,row in enumerate(history,s['parent_update']):
                t,q,_=base.schedule(index,cfg,old_spec,'near_zero')
                anchor=data.field_for(index,15,ids)
                if (row['update']!=index+1 or row['anchor_id']!=anchor or row['time']!=t or row['ratio']!=q or
                    row['noise_seed']!=p.seed_for(cfg['train_seed'],ids[index%3],index,'noise') or
                    row['auxiliary_seed']!=p.seed_for(cfg['train_seed'],anchor,index,'preservation-auxiliary') or
                    row['auxiliary_ratio']!=auxiliary_ratio(index,cfg['train_seed'],s) or not metrics.finite_tree(row)):
                    raise ValueError('training schedule drift')
            for index,digest in complete['probes'].items():
                path=folder/f'probe_{int(index):06d}.json'
                if p.sha256(path)!=digest:raise ValueError('probe checksum drift')
                probe=json.loads(path.read_text())
                if len(probe['rows'])!=810 or len(probe['response'])!=324 or not metrics.finite_tree(probe):
                    raise ValueError('incomplete/nonfinite probe panel')
                row_keys={(r['anchor_id'],r['kind'],r['ratio'],r.get('rep')) for r in probe['rows']}
                anchors=[r['anchor_id'] for r in prepared['selection']['train']+prepared['selection']['transfer']]
                expected={(anchor,'clean',q,None) for anchor in anchors for q in old_spec['clean_ratios']}
                expected|={(anchor,'noisy',q,rep) for anchor in anchors for q in old_spec['noisy_ratios'] for rep in range(2)}
                if row_keys!=expected:raise ValueError('missing or duplicate probe rows')
                response_keys={(r['anchor_id'],r['ratio'],r['fraction'],r['rep']) for r in probe['response']}
                if response_keys!={(anchor,q,f,rep) for anchor in anchors for q in (.001,.01,.05) for f in (.1,.5) for rep in range(2)}:
                    raise ValueError('missing or duplicate response probes')
                loaded[seed,arm,int(index)]=probe;hashes[str(path.relative_to(root))]=digest
    results=[]
    for arm in ARMS[1:]:
        tests=[]
        for seed in s['replicas']:
            for offset in s['gate']['last_two_evaluations']:
                index=s['parent_update']+offset
                for name in ('exposed','transfer'):
                    result=gate(loaded[seed,'control',index]['rows'],loaded[seed,arm,index]['rows'],groups[name],s['gate'],reference)
                    response=response_gate(loaded[seed,'control',index]['response'],loaded[seed,arm,index]['response'],groups[name],s['gate'])
                    result['passed']=result['passed'] and response['passed']
                    tests.append(dict(replica=seed,update=index,group=name,response=response,**result))
        results.append(dict(arm=arm,reproducible_local_gate=all(t['passed'] for t in tests),tests=tests))
    folder=root/'analysis';folder.mkdir(exist_ok=True)
    with durable.single_writer(folder):
        summary=dict(complete=True,results=results,inputs=hashes,
            manifest_sha256=p.sha256(root/'MANIFEST.json'),training_ready=False,full_e2e_restart_authorized=False,
            interpretation='Local gate only; no calibrated posterior or CFM claim. All raw response probes retained.')
        destination=folder/'SUMMARY.json'
        if destination.exists():
            if json.loads(destination.read_text())!=summary:raise ValueError('report conflict')
        else:durable.publish_json(destination,summary)
    print('REPORT COMPLETE',folder/'SUMMARY.json',flush=True)


def smoke(root):
    device=p.runtime();m,previous,prepared,items,parent=inputs(root,0,device);s=m['spec']
    ids=[r['anchor_id'] for r in prepared['selection']['train']];results=[]
    for arm in ARMS:
        cfg,model,opt,gen=base.make(prepared,0,device);base.restore(parent,model,opt,gen)
        folder=root/'smoke'/arm;folder.mkdir(parents=True,exist_ok=False)
        binding=dict(smoke_manifest=p.sha256(root/'MANIFEST.json'),arm=arm)
        history=copy.deepcopy(parent['history']);start=parent['step'];stamp=time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        endpoint_checks=[];item=items[ids[0]]
        eps=torch.randn(item['target'].shape,device=device,generator=torch.Generator(device=device).manual_seed(916171))
        eta=torch.randn(item['target'].shape,device=device,generator=torch.Generator(device=device).manual_seed(916172))
        for primary_time in (0.,1.):
            for ratio in (1e-5,.001,.05):
                opt.zero_grad(set_to_none=True)
                loss,terms=objective(model,item['target'],eps,primary_time,item['condition'],item['wide'],eta,
                                     ratio,s['perturbation_fraction'],**s['arms'][arm])
                if not torch.isfinite(loss):raise ValueError('nonfinite endpoint loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['clip'],error_if_nonfinite=True)
                endpoint_checks.append(dict(primary_time=primary_time,auxiliary_ratio=ratio,
                                            loss=float(loss.detach()),gradient_norm=float(norm)))
        base.restore(parent,model,opt,gen)
        for index in range(start,start+4):history.append(update(model,opt,gen,items,ids,cfg,s,previous,arm,index))
        durable.save(folder,model=model,optimizer=opt,generator=gen,binding=binding,stage='fine',method='diffusion',step=start+4,history=history)
        for index in range(start+4,start+8):update(model,opt,gen,items,ids,cfg,s,previous,arm,index)
        expected=copy.deepcopy(dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=p.rng_state(gen)))
        saved,_=durable.load(folder,binding);base.restore(saved,model,opt,gen)
        for index in range(start+4,start+8):update(model,opt,gen,items,ids,cfg,s,previous,arm,index)
        if not equal_state(expected,dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=p.rng_state(gen))):
            raise ValueError('GPU resume mismatch')
        parity=None
        if arm=='control':
            base.restore(parent,model,opt,gen)
            for index in range(start,start+8):base.step(model,opt,items,ids,cfg,previous,'near_zero',index,gen)
            parity=equal_state(expected,dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=p.rng_state(gen)))
            if not parity:raise ValueError('zero-weight control does not replay prior objective')
        results.append(dict(arm=arm,exact_resume=True,control_parity=parity,seconds=time.monotonic()-stamp,
                            peak_bytes=torch.cuda.max_memory_allocated(),last_terms=history[-1]['terms'],endpoints=endpoint_checks))
        del model,opt,expected,saved
    cfg,model,opt,gen=base.make(prepared,0,device);base.restore(parent,model,opt,gen)
    subset={a:items[a] for a in (ids[0],prepared['selection']['transfer'][0]['anchor_id'])}
    probe=evaluate(model,subset,cfg,previous,prepared)
    frozen=json.loads((Path(previous['parent_root'])/'FROZEN.json').read_text())
    reference={(r['anchor_id'],r['ratio'],r.get('rep')):r for r in frozen['parent384'] if r['kind']=='noisy'}
    unchanged=gate(probe['rows'],probe['rows'],list(subset),s['gate'],reference)
    response_check=response_gate(probe['response'],probe['response'],list(subset),s['gate'])
    if unchanged['passed'] or not response_check['passed']:
        raise ValueError('gate smoke: unchanged predictor cannot meet improvement gate')
    durable.publish_json(root/'SMOKE_PROBE.json',probe)
    verify(root)
    durable.publish_json(root/'SMOKE.json',dict(passed=True,manifest_sha256=p.sha256(root/'MANIFEST.json'),results=results,
        evaluation_rows=len(probe['rows']),response_rows=len(probe['response']),node=socket.gethostname(),
        gate_smoke_passed=True,job_id=os.environ['SLURM_JOB_ID'],scientific_fit_complete=False))
    print('SMOKE PASSED',json.dumps(results),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=['stage','smoke','train','report'])
    ap.add_argument('--root',type=Path,required=True);ap.add_argument('--replica',type=int,choices=[0,1],default=0)
    ap.add_argument('--arm',choices=ARMS,default='control');args=ap.parse_args()
    if args.command=='stage':stage(args.root)
    elif args.command=='smoke':smoke(args.root)
    elif args.command=='train':train(args.root,args.replica,args.arm)
    else:report(args.root)


if __name__=='__main__':main()
