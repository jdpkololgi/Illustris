"""Validate/aggregate target and resolution controls without selecting seeds."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from workflows.sbi.e2e_conditional_reference import atomic_json
from workflows.sbi.e2e_conditional_reference_continue import digest
from workflows.sbi.e2e_conditional_reference_math import problem, probes


def group(rows, key):
    rr=[r for r in rows if r['nfe']==256]
    groups={}
    for r in rr:
        label=f'{r[key]}_'+('amortised' if r['fixed'] is None else 'fixed')
        groups.setdefault(label,[]).append(r)
    result={}
    for label,subset in groups.items():
        result[label]=dict(cells=len(subset), seeds=sorted(set(r['seed'] for r in subset)),
            means={k:float(np.mean([r[k] for r in subset])) for k in ['mean_rms','covariance_relative','variance_ratio','octant_coverage']},
            high_k_range=[float(min(r['power_ratio'][-1] for r in subset)),float(max(r['power_ratio'][-1] for r in subset))],
            power_ratio_mean=np.mean([r['power_ratio'] for r in subset],axis=0).tolist(),
            original_pass=sum(r['passed'] for r in subset), joint_pass=sum(r['passed'] and r['power_pass'] for r in subset),
            by_seed={str(s):dict(mean_rms=float(np.mean([r['mean_rms'] for r in subset if r['seed']==s])),
                                joint_pass=sum(r['passed'] and r['power_pass'] for r in subset if r['seed']==s)) for s in [17,29]})
        if 'risk' in subset[0]:
            unique={r['name']:r['risk']['exact_target_mse'] for r in subset}
            result[label]['mean_frozen_exact_target_mse']=float(np.mean(list(unique.values())))
        if 'common_observable_metrics' in subset[0]:
            result[label]['common_means']={k:float(np.mean([r['common_observable_metrics'][k] for r in subset])) for k in ['mean_rms','covariance_relative','variance_ratio','octant_coverage']}
            result[label]['common_power']=np.mean([r['common_observable_metrics']['power_ratio'] for r in subset],axis=0).tolist()
            def common_pass(c):
                return (c['mean_rms']<=.1 and c['covariance_relative']<=.15
                        and .9<=c['variance_ratio']<=1.1 and .85<=c['octant_coverage']<=.95
                        and all(.9<=v<=1.1 for v in c['power_ratio']))
            result[label]['common_joint_pass']=sum(common_pass(r['common_observable_metrics']) for r in subset)
    return result


def sampler_differences(rows):
    index={(r['name'],r['case'],r['nfe']):r for r in rows}
    output=[]
    for (name,case,nfe),a in index.items():
        if nfe!=128: continue
        b=index[name,case,256]
        output.append(dict(name=name,case=case,mean_rms_difference=b['mean_rms']-a['mean_rms'],
                           max_absolute_power_difference=float(np.max(np.abs(np.array(b['power_ratio'])-a['power_ratio'])))))
    return output


def affine_moments(root,update=16384):
    _,_,_,_,cases,_,radius=problem(8,4)
    q=probes(8);shell=np.floor(radius).astype(int)
    out=[]
    manifest=json.loads((root/'manifest.json').read_text())
    for item in manifest['items']:
        if not item['branch'].startswith('affine'):continue
        saved=torch.load(root/'results'/item['name']/f'checkpoint_{update}.pt',map_location='cpu',weights_only=False)
        if saved['sources']!=manifest['sources'] or saved['item']!=item or saved['update']!=update:
            raise ValueError('affine checkpoint provenance mismatch')
        w={k:v.numpy() for k,v in saved['model'].items()}
        for j in (range(4) if item['fixed'] is None else [item['fixed']]):
            c=cases[j];which=j%2
            mu=w['mean_map'][which]@c['y'] if item['fixed'] is None else w['mean']
            lam=np.exp(w['log_values'][which]).astype(float);vec=w['vectors'][which].astype(float)
            sigma=(vec*lam)@vec.T
            def spectrum(factor):
                return np.abs(np.fft.fftn(factor.T.reshape(-1,8,8,8),axes=(1,2,3),norm='ortho'))**2
            p=spectrum(vec*np.sqrt(lam)).sum(0);target=spectrum(c['chol']).sum(0)
            out.append(item|dict(case=j,ideal_continuous_time=True,
                mean_rms=float(np.linalg.norm(mu-c['mu'])/np.sqrt(np.trace(c['sigma']))),
                full_covariance_relative=float(np.linalg.norm(sigma-c['sigma'])/np.linalg.norm(c['sigma'])),
                probe_covariance_relative=float(np.linalg.norm(q.T@(sigma-c['sigma'])@q)/np.linalg.norm(q.T@c['sigma']@q)),
                power_ratio=[float(p[shell==i].mean()/target[shell==i].mean()) for i in np.unique(shell)]))
    return out


def main(args):
    root=Path(args.target);dest=Path(args.output);dest.mkdir(parents=True,exist_ok=True)
    target=json.loads((root/'COMPLETE.json').read_text())
    if len(target['rows'])!=120: raise ValueError('expected 120 final target ensembles')
    summary=dict(target=group(target['rows'],'branch'),sampler=sampler_differences(target['rows']),
                 affine_ideal=affine_moments(root))
    roots={'target':root}
    if args.affine:
        ar=Path(args.affine);adaptive=json.loads((ar/'COMPLETE.json').read_text())
        if len(adaptive['rows'])!=16:raise ValueError('expected16adaptive affine ensembles')
        summary['adaptive_affine']=group(adaptive['rows'],'branch')
        summary['adaptive_affine_ideal']=affine_moments(ar,65536)
        roots['adaptive_affine']=ar
    if args.resolution:
        rr=Path(args.resolution);resolution=json.loads((rr/'COMPLETE.json').read_text())
        if len(resolution['rows'])!=40:raise ValueError('expected40resolution ensembles')
        summary['resolution']=group(resolution['rows'],'n')
        summary['resolution_sampler']=sampler_differences(resolution['rows'])
        roots['resolution']=rr
    hashes={}
    for kind,run in roots.items():
        files=[run/'manifest.json',run/'COMPLETE.json']
        files+=list(run.glob('worker_*_COMPLETE.json'))
        files+=list((run/'results').glob('*/precision/*.json'))
        files+=list((run/'results').glob('*/evaluation_*.json'))
        files+=list((run/'results').glob('*/learning.jsonl'))
        for p in files:
            relative=p.relative_to(run);d=dest/kind/relative;d.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,d)
            hashes[str(Path(kind)/relative)]=digest(p)
        hashes[kind+'_generated_artifacts']={str(p.relative_to(run)):digest(p) for p in (run/'results').rglob('*') if p.is_file()}
    atomic_json(dest/'artifact_hashes.json',hashes)
    atomic_json(dest/'summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k in ['target','resolution','adaptive_affine']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--target',required=True);p.add_argument('--resolution');p.add_argument('--affine');p.add_argument('--output',required=True)
    main(p.parse_args())
