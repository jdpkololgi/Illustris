"""Frozen synthetic oracle and read-only checkpoint-gradient experiment."""
import argparse
import itertools
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import time
import numpy as np
import torch
from workflows.sbi import e2e_wide_pipeline as p
from workflows.sbi import e2e_durable as durable
from workflows.sbi import e2e_clean_limit as base
from workflows.sbi import e2e_diversity_norm as data
from workflows.sbi.e2e_clean_limit_launch import snapshot_paths, SCRATCH
from workflows.sbi.e2e_analytic_fields import oracle_report, min_snr_v_weight
from workflows.sbi.e2e_loss_conflict import components, gradient_vectors, comparison, remove_conflicting_component

CONFIG=p.REPO/'configs/e2e_oracle_conflict_v1.json'


def spec():
    s=json.loads(CONFIG.read_text())
    if s['schema']!='e2e-oracle-conflict-v1' or any(s[k] for k in
            ('full_e2e_training','heldout_access','automatic_repair_training')):
        raise ValueError('outside diagnostic scope')
    return s


def stage(root):
    s=spec();root=root.resolve();old=Path(s['parent_root'])
    if root.parent!=SCRATCH or not root.name.startswith('oracle_conflict_'):
        raise ValueError('new registered Scratch child required')
    if subprocess.check_output(['git','status','--porcelain'],cwd=p.REPO,text=True).strip():
        raise ValueError('commit before staging')
    for rel,key in [('MANIFEST.json','parent_manifest_sha256'),('analysis/SUMMARY.json','parent_summary_sha256')]:
        if p.sha256(old/rel)!=s[key]:raise ValueError('parent evidence drift')
    summary=json.loads((old/'analysis/SUMMARY.json').read_text())
    for rel,digest in summary['inputs'].items():
        if p.sha256(old/rel)!=digest:raise ValueError('parent probe drift')
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=p.REPO,text=True).strip()
    names=snapshot_paths(subprocess.check_output(['git','ls-files','-z'],cwd=p.REPO).decode().split('\0'))
    names.append('docs/e2e_oracle_conflict_v1.md')
    root.mkdir(exist_ok=False);source=root/'source';source.mkdir();(root/'logs').mkdir()
    archive=subprocess.Popen(['git','archive',revision,'--',*names],cwd=p.REPO,stdout=subprocess.PIPE)
    result=subprocess.run(['tar','-xf','-','-C',str(source)],stdin=archive.stdout)
    archive.stdout.close()
    if archive.wait() or result.returncode:raise RuntimeError('partial archive preserved')
    parents={}
    for seed,arm in itertools.product(s['replicas'],s['arms']):
        path=old/f'replica_{seed}'/arm/'COMPLETE.json';c=json.loads(path.read_text())
        if not c['complete'] or c['update']!=30720:raise ValueError('wrong parent checkpoint')
        parents[f'{seed}/{arm}']=dict(complete_sha256=p.sha256(path),checkpoint=c['checkpoint'],binding=c['binding'])
    durable.publish_json(root/'MANIFEST.json',dict(spec=s,git_revision=revision,source=str(source),
        source_sha256={name:p.sha256(source/name) for name in names if (source/name).is_file()},parents=parents))
    print('STAGED',root,revision,flush=True)


def verify(root):
    m=json.loads((root/'MANIFEST.json').read_text())
    if m['spec']!=spec() or Path(m['source']).resolve()!=p.REPO.resolve():raise ValueError('wrong snapshot')
    for rel,digest in m['source_sha256'].items():
        if p.sha256(p.REPO/rel)!=digest:raise ValueError('source drift: '+rel)
    old=Path(m['spec']['parent_root'])
    for rel,key in [('MANIFEST.json','parent_manifest_sha256'),('analysis/SUMMARY.json','parent_summary_sha256')]:
        if p.sha256(old/rel)!=m['spec'][key]:raise ValueError('parent drift')
    return m


def gradients(root):
    device=p.runtime();m=verify(root);s=m['spec'];old=Path(s['parent_root'])
    previous=json.loads((old/'MANIFEST.json').read_text())['spec']
    clean_root=Path(previous['parent_root']);clean=json.loads((clean_root/'MANIFEST.json').read_text())['spec']
    if p.sha256(clean_root/'MANIFEST.json')!=previous['parent_manifest_sha256']:raise ValueError('old data contract drift')
    raw=Path(clean['parent_root'])
    if p.sha256(raw/'PREPARED.json')!=clean['prepared_sha256']:raise ValueError('prepared drift')
    prepared,items=data.load_items(raw,device)
    # Metadata-only first registered fitting field per phase; transfer never used
    # for gradient decisions. All three phases, two addressed noise draws.
    selected=prepared['selection']['train'][:s['gradient_fitting_fields']]
    if len({x['phase'] for x in selected})!=3:raise ValueError('three fitting phases required')
    folder=root/'gradients';folder.mkdir(exist_ok=True)
    all_records=[];cross=[];parents={}
    for seed,arm in itertools.product(s['replicas'],s['arms']):
        destination=folder/f'replica_{seed}_{arm}.json'
        if destination.exists():
            saved=json.loads(destination.read_text())
            if saved['manifest_sha256']!=p.sha256(root/'MANIFEST.json'):raise ValueError('receipt drift')
            all_records.extend(saved['records']);cross.extend(saved['cross_sigma']);parents[f'{seed}/{arm}']=saved['parent_checkpoint']
            continue
        branch=old/f'replica_{seed}'/arm;parent=m['parents'][f'{seed}/{arm}']
        if p.sha256(branch/'COMPLETE.json')!=parent['complete_sha256']:raise ValueError('parent receipt drift')
        state,pointer=durable.load(branch,parent['binding'])
        if pointer!=parent['checkpoint']:raise ValueError('parent pointer drift')
        cfg,model,optimizer,generator=base.make(prepared,seed,device);base.restore(state,model,optimizer,generator)
        model.eval();parameters=list(model.parameters());records=[];cross_records=[]
        for row in selected:
            anchor=row['anchor_id'];item=items[anchor]
            for rep in range(s['gradient_noise_replicates']):
                gen=torch.Generator(device=device).manual_seed(p.seed_for(s['gradient_seed'],anchor,rep,'gradient'))
                noise=torch.randn(item['target'].shape,device=device,generator=gen)
                aux=torch.randn(item['target'].shape,device=device,generator=gen)
                primary={}
                for sigma in s['gradient_sigmas']:
                    losses=components(model,item,noise,aux,sigma)
                    values={k:float(v.detach()) for k,v in losses.items()}
                    vectors=gradient_vectors(losses,parameters)
                    den=vectors['denoising'];identity=vectors['identity'];response=vectors['even_consistency']+vectors['odd_noise_response']
                    pairs={k:comparison(den,v) for k,v in vectors.items() if k!='denoising'}
                    pairs['response_sum']=comparison(den,response)
                    for name,addition in [('identity_weak',.1*identity),('identity_strong',identity),('identity_response',.1*(identity+response))]:
                        projected=remove_conflicting_component(den,addition)
                        pairs[name]=dict(**comparison(den,addition),
                            relative_aux_norm=float(addition.norm()/den.norm().clamp_min(1e-30)),
                            projected_dot=float((den*projected).sum()))
                    primary[sigma]=den.cpu()
                    records.append(dict(replica=seed,arm=arm,anchor_id=anchor,phase=row['phase'],rep=rep,sigma=sigma,
                        auxiliary_in_training_range=sigma<=.05,losses=values,pairs=pairs,
                        min_snr_v_weight=float(min_snr_v_weight(torch.tensor(sigma,dtype=torch.float64),s['min_snr_gamma']))))
                    del vectors,losses,den,identity,response
                for qa,qb in itertools.combinations(s['gradient_sigmas'],2):
                    cross_records.append(dict(replica=seed,arm=arm,anchor_id=anchor,rep=rep,sigma_a=qa,sigma_b=qb,
                                              **comparison(primary[qa],primary[qb])))
        # Read-only means exact parameter and Adam equality, not merely no save.
        from workflows.sbi.e2e_wide_continue import equal_state
        if not equal_state(model.state_dict(),state['model']) or not equal_state(optimizer.state_dict(),state['optimizer']):
            raise ValueError('diagnostic mutated checkpoint state')
        saved=dict(manifest_sha256=p.sha256(root/'MANIFEST.json'),parent_checkpoint=pointer,
                   unchanged_model_optimizer=True,records=records,cross_sigma=cross_records)
        durable.publish_json(destination,saved);all_records.extend(records);cross.extend(cross_records);parents[f'{seed}/{arm}']=pointer
        print('GRADIENT COMPLETE',seed,arm,len(records),flush=True)
        del state,model,optimizer,generator,primary
    summary=[]
    for seed,arm,q in itertools.product(s['replicas'],s['arms'],s['gradient_sigmas']):
        rows=[r for r in all_records if (r['replica'],r['arm'],r['sigma'])==(seed,arm,q)]
        result={}
        for key in rows[0]['pairs']:
            vals=[r['pairs'][key] for r in rows];cs=[v['cosine'] for v in vals if v['cosine'] is not None]
            result[key]=dict(median_cosine=float(np.median(cs)) if cs else None,
                negative_fraction=float(np.mean([c<0 for c in cs])) if cs else None,
                median_norm_ratio=float(np.median([v['norm_b']/max(v['norm_a'],1e-30) for v in vals])))
        summary.append(dict(replica=seed,arm=arm,sigma=q,n=len(rows),pairs=result))
    return dict(rows=len(all_records),cross_sigma_rows=len(cross),summary=summary,
        checkpoint_receipts=parents,raw_files={str(x.relative_to(root)):p.sha256(x) for x in folder.glob('*.json')},
        caveat='Raw Euclidean gradients, fixed checkpoints and six examples per sigma; projection is a local first-order diagnostic, not an Adam or generalization guarantee.')


def run(root):
    p.require_compute();m=verify(root);stamp=time.monotonic()
    with durable.single_writer(root):
        oracle_path=root/'ORACLES.json'
        if not oracle_path.exists():
            result=oracle_report(p.runtime(),m['spec']['oracle_grid'],m['spec']['oracle_draws'])
            durable.publish_json(oracle_path,dict(manifest_sha256=p.sha256(root/'MANIFEST.json'),**result))
        oracle=json.loads(oracle_path.read_text())
        if oracle['manifest_sha256']!=p.sha256(root/'MANIFEST.json'):raise ValueError('oracle binding drift')
        print('ORACLES',oracle['passed'],flush=True)
        # Preserve oracle negative results, but still collect independent gradients.
        gradient=gradients(root);verify(root)
        summary=dict(complete=True,oracle_passed=oracle['passed'],oracle_sha256=p.sha256(oracle_path),
            gradients=gradient,manifest_sha256=p.sha256(root/'MANIFEST.json'),
            job_id=os.environ.get('SLURM_JOB_ID'),node=socket.gethostname(),seconds=time.monotonic()-stamp,
            diffusers_reference='not installed; analytic identities and existing sampler tested, no library parity claim',
            full_e2e_training=False,repair_training_performed=False)
        durable.publish_json(root/'SUMMARY.json',summary)
        print('COMPLETE',root,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['stage','run'])
    parser.add_argument('--root',type=Path,required=True);args=parser.parse_args()
    (stage if args.action=='stage' else run)(args.root)

if __name__=='__main__':main()
