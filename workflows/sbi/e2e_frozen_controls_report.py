"""Descriptive paired summaries; no retrospective model-promotion gate."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
from workflows.sbi import e2e_wide_pipeline as p
from workflows.sbi import e2e_durable as durable


def optimizer_summary(records):
    keys=('replica','anchor_id','gradient_sigma','momentum','clipping')
    controls={tuple(r[k] for k in keys):r for r in records if r['variant']=='denoising_only'}
    output=[]
    for seed,momentum,clipping,variant in itertools.product((0,1),('retained','zero_first'),(True,False),
            ('denoising_only','identity_strong','projected_identity')):
        selected=[r for r in records if (r['replica'],r['momentum'],r['clipping'],r['variant'])==(seed,momentum,clipping,variant)]
        pairs=[]
        for r in selected:
            control=controls[tuple(r[k] for k in keys)]
            for after,reference,before in zip(r['after'],control['after'],r['before']):
                if after['sigma']!=reference['sigma'] or after['sigma']!=before['sigma']:raise ValueError('mispaired noise')
                pairs.append(dict(sigma=after['sigma'],clean_ratio=after['clean_mse']/reference['clean_mse'],
                    noisy_ratio=after['noisy_mse']/reference['noisy_mse'],
                    clean_vs_before=after['clean_mse']/before['clean_mse'],noisy_vs_before=after['noisy_mse']/before['noisy_mse']))
        if len(selected)!=9 or len(pairs)!=27:raise ValueError('incomplete optimizer factorial')
        output.append(dict(replica=seed,momentum=momentum,clipping=clipping,variant=variant,
            interventions=len(selected),evaluations=len(pairs),
            clipped=sum(r['clipping_scale']<1 for r in selected),
            median_clip_scale=float(np.median([r['clipping_scale'] for r in selected])),
            raw_negative=sum(r['raw_primary_dot_aux'] < -1e-12 for r in selected),
            incremental_adverse=sum(r['incremental_primary_dot_vs_control']>0 for r in selected),
            actual_primary_ascent=sum(r['actual_primary_dot_displacement']>0 for r in selected),
            joint_nonworse=sum(r['clean_ratio']<=1 and r['noisy_ratio']<=1 for r in pairs),
            **{f'median_{k}':float(np.median([r[k] for r in pairs])) for k in ('clean_ratio','noisy_ratio','clean_vs_before','noisy_vs_before')},
            by_sigma=[dict(sigma=q,**{f'median_{k}':float(np.median([r[k] for r in pairs if r['sigma']==q]))
                for k in ('clean_ratio','noisy_ratio')}) for q in (.005,.01,.05)]))
    return output


def sampler_summary(records):
    output=[]
    for seed,group in itertools.product((0,1),('train','transfer')):
        selected=[r for r in records if (r['replica'],r['group'])==(seed,group)]
        if len(selected)!=6:raise ValueError('incomplete sampler panel')
        summary={}
        for method in selected[0]['metrics']:
            vals=[r['metrics'][method] for r in selected]
            summary[method]=dict(median_relative_rms_to_reference=float(np.median([v['relative_rms_to_reference'] for v in vals])),
                median_density_below_minus_one=float(np.median([v['density_below_minus_one'] for v in vals])),
                median_density_std=float(np.median([v['density_std'] for v in vals])),
                median_residual_power_ratio=np.median([v['residual_spectrum']['power_ratio'] for v in vals],axis=0).tolist(),
                median_residual_gain=np.median([v['residual_spectrum']['gain'] for v in vals],axis=0).tolist())
        output.append(dict(replica=seed,group=group,n=len(selected),
            reference_resolved=sum(r['reference_resolved'] for r in selected),
            maximum_reference_refinement=max(r['reference_refinement_relative_rms'] for r in selected),methods=summary,
            heun_better_at_nfe={str(n):sum(r['metrics'][f'heun_{n}']['relative_rms_to_reference']<r['metrics'][f'ddim_{n}']['relative_rms_to_reference'] for r in selected) for n in (32,128)}))
    return output


def plot_panel(root,manifest,records):
    """Fixed first seed/phase/draw, no visual cherry-picking; physical total density."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import h5py
    parent=json.loads((Path(manifest['spec']['parent_root'])/'MANIFEST.json').read_text())['spec']
    raw=Path(json.loads((Path(parent['parent_root'])/'MANIFEST.json').read_text())['spec']['parent_root'])
    prepared=json.loads((raw/'PREPARED.json').read_text())
    if p.sha256(raw/'cache.h5')!=prepared['cache_sha256']:raise ValueError('plot cache drift')
    fine=prepared['original_normalization']['targets']['fine']
    coarse=prepared['original_normalization']['targets']['coarse']
    fig,axes=plt.subplots(2,4,figsize=(12,6),constrained_layout=True)
    for i,group in enumerate(('train','transfer')):
        row=next(r for r in records if r['replica']==0 and r['phase']=='ph000' and r['group']==group and r['draw']==0)
        with h5py.File(raw/'cache.h5','r') as f:
            target=f[row['anchor_id']]['target'][0]
            local=f[row['anchor_id']]['condition'][-1]*coarse['std']+coarse['mean']
        with np.load(root/'neural_sampler'/row['draw_file']) as z:
            fields=[target]+[z[k] for k in ('ddim_128','heun_128','heun_512')]
        for j,(field,title) in enumerate(zip(fields,('Truth','DDIM,128 NFE','Heun,128 NFE','Heun,512 NFE'))):
            density=field*fine['std']+fine['mean']+local
            im=axes[i,j].imshow(np.arcsinh(density[:,:,48]),origin='lower',cmap='magma',vmin=-1,vmax=3)
            axes[i,j].set_title(f'{group}: {title}',fontsize=10);axes[i,j].set_xticks([]);axes[i,j].set_yticks([])
    fig.colorbar(im,ax=axes,label='asinh(total density contrast); common display scale',shrink=.8)
    fig.suptitle('Frozen fine U-Net with TRUE coarse conditioning; seed0,ph000,draw0,z-cell48\nFixed illustrative slices, not posterior calibration',fontsize=11)
    path=root/'FROZEN_FIELDS.png';fig.savefig(path,dpi=150);plt.close(fig)
    return dict(path=str(path),sha256=p.sha256(path))


def run(root):
    p.require_compute()
    manifest=json.loads((root/'MANIFEST.json').read_text());mh=p.sha256(root/'MANIFEST.json')
    for rel,digest in manifest['source_sha256'].items():
        if p.sha256(Path(manifest['source'])/rel)!=digest:raise ValueError('frozen source drift')
    inputs={};results={}
    for name,fn in [('OPTIMIZER',optimizer_summary),('SAMPLER',sampler_summary)]:
        path=root/f'{name}.json';payload=json.loads(path.read_text())
        if not payload['complete'] or payload['manifest_sha256']!=mh:raise ValueError('incomplete or misbound result')
        inputs[path.name]=p.sha256(path);results[name.lower()]=fn(payload['records'])
        if name=='SAMPLER':
            for row in payload['records']:
                if p.sha256(root/'neural_sampler'/row['draw_file'])!=row['draw_sha256']:raise ValueError('draw drift')
    old=Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v2/wide_pipeline_v1/oracle_conflict_followup_20260916_58439522/ADAM_STEPS.json')
    if p.sha256(old)!='2c27c2801d78ee07da96636ec0e5ada241b213fba9ffe71930b006871e5849e8':raise ValueError('original step probe drift')
    previous=json.loads(old.read_text())['records']
    current=json.loads((root/'OPTIMIZER.json').read_text())['records']
    key=lambda x:(x['replica'],x['anchor_id'],x['gradient_sigma'],x['variant'])
    control={key(x):x for x in current if x['momentum']=='retained' and x['clipping']}
    fields=('before','after','raw_primary_dot_aux','actual_primary_dot_displacement','incremental_primary_dot_vs_control')
    parity=sum(all(x[k]==control[key(x)][k] for k in fields) for x in previous)
    if parity!=54 or len(control)!=54:raise ValueError('original control parity failed')
    figure=plot_panel(root,manifest,payload['records'])
    result=dict(complete=True,manifest_sha256=mh,git_revision=manifest['git_revision'],inputs=inputs,figure=figure,
        previous_control_exact_parity=parity,previous_probe_sha256=p.sha256(old),
        report_source_sha256=p.sha256(__file__),**results,full_e2e_training=False,
        note='Descriptive bounded panel; overlapping phases/fields and paired draws are not independent simulations. No calibrated-posterior or convergence claim from one-step probes.')
    durable.publish_json(root/'RESULTS.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    run(parser.parse_args().root)
