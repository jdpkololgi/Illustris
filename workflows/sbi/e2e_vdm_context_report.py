"""Frozen distributional comparison and H1/H2 assessment; no adaptive fitting."""
import argparse
from collections import defaultdict
from pathlib import Path
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from workflows.sbi import e2e_durable as durable,e2e_wide_pipeline as existing
from workflows.sbi.e2e_field_build_products import require_compute
from workflows.sbi.e2e_vdm_context_data import read_json,output_root,spec,ROLES
from workflows.sbi.e2e_vdm_context_dataset import Products,observation_summary
from workflows.sbi.e2e_vdm_context_analysis import (
    CORE,WIDE_EDGES,ensemble,parent_ensemble,density_spectra,core_features,closure_tensor,calibration_summary)
from workflows.sbi.e2e_vdm_context_physics import composite_tensor
from workflows.sbi.e2e_vdm_context_metrics import fair_energy,variogram_score,calibration
from workflows.sbi.e2e_vdm_context_train import verify_launch


def summarize_case(root,task,data):
    folder=root/'analysis/cases'
    folder.mkdir(parents=True,exist_ok=True)
    output=folder/(task['task_id']+'.json')
    if output.exists():
        receipt=read_json(output)
        if receipt['manifest_sha256']!=existing.sha256(root/'MANIFEST.json') or receipt['task']!=task:
            raise ValueError('case report drift')
        for path,digest in receipt['draw_receipts'].items():
            if existing.sha256(root/path)!=digest:
                raise ValueError('case input receipt drift')
        ensemble(root,task)  # Rehash actual arrays, not only unchanged receipts.
        if task['arm']=='D' and task['coarse_mode']!='oracle_diagnostic':
            parent_ensemble(root,task)
        return receipt
    started=time.monotonic()
    fields,inputs=ensemble(root,task)
    target=data.raw_targets(task['anchor'])
    truth=target['rho']-1
    true_features=core_features(truth[CORE],target['tensor'])
    raw=data.raw_observations(task['anchor'])
    support=raw['support'][CORE].reshape(-1)>=.5
    masks=dict(all=np.ones(len(support),bool),observed=support,unobserved=~support)
    scales=np.asarray(data.chart['physical_probe_scales']['std'])
    spectra,_=density_spectra(fields,truth)
    coarse=None
    if task['arm']=='D':
        coarse=(np.repeat(target['coarse'][None],len(fields),axis=0)
                if task['coarse_mode']=='oracle_diagnostic' else parent_ensemble(root,task))
    closures=['periodic','zero','reflect']+(['composite'] if coarse is not None else [])
    diagnostics={}
    summaries=None
    for closure in closures:
        if closure=='composite':
            tensors=(composite_tensor(x,coarse[i]-1,data.rows[task['anchor']].get('core_offset_raw',[0,0,0]))[CORE]
                     for i,x in enumerate(fields))
        else:
            tensors=(closure_tensor(x,closure)[CORE] for x in fields)
        features=np.stack([core_features(x[CORE],tensor) for x,tensor in zip(fields,tensors)])
        matched=core_features(truth[CORE],closure_tensor(truth,closure if closure!='composite' else 'periodic')[CORE])
        diagnostics[closure]=dict(physical={key:calibration_summary(features,true_features,scales,mask)
                                           for key,mask in masks.items()})
        if closure!='composite':
            diagnostics[closure]['matched_patch']={key:calibration_summary(features,matched,scales,mask)
                                                 for key,mask in masks.items()}
        if closure==('composite' if task['arm']=='D' else 'periodic'):
            summaries=features.mean(1)
    physical=diagnostics['composite' if task['arm']=='D' else 'periodic']['physical']['all']
    density=diagnostics['periodic']['physical']['all']
    onepoint=np.array([[x.mean(),x.std(),*np.quantile(x,[.001,.01,.1,.5,.9,.99,.999])] for x in fields])
    regional=lambda x:(x+1).reshape(2,24,2,24,2,24).mean((1,3,5)).ravel()
    regional_draws=np.stack([regional(x) for x in fields])
    regional_truth=regional(truth)
    rc=calibration(regional_draws,regional_truth)
    wide=None
    if coarse is not None and task['coarse_mode']=='sampled':
        wide,_=density_spectra(coarse-1,target['coarse']-1,cell=27.064,edges=WIDE_EDGES)
        # Marginal wide diagnostics are deliberately distinguished from a joint-field claim.
        wc=calibration((coarse-1).reshape(len(coarse),-1),(target['coarse']-1).ravel())
        wide.update(mean_crps=float(wc['crps'].mean()),coverage90=float(wc['covered_0.9'].mean()),
                    attainable90=wc['attainable']['0.9'],mean_bias=float(wc['bias'].mean()))
    result=dict(task=task,manifest_sha256=existing.sha256(root/'MANIFEST.json'),draw_receipts=inputs,
        phase=data.rows[task['anchor']]['phase'],cap=data.rows[task['anchor']]['cap'],
        source_center_mpc_h=data.rows[task['anchor']].get('source_center_mpc_h'),
        spectra=spectra,wide=wide,closures=diagnostics,observations=observation_summary(raw),
        onepoint_domain='parent325',onepoint_draws=onepoint.tolist(),onepoint_truth=[float(truth.mean()),float(truth.std()),*np.quantile(truth,[.001,.01,.1,.5,.9,.99,.999]).tolist()],
        core_onepoint_draws=[[float(x[CORE].mean()),float(x[CORE].std()),*np.quantile(x[CORE],[.001,.01,.1,.5,.9,.99,.999]).tolist()] for x in fields],
        core_onepoint_truth=[float(truth[CORE].mean()),float(truth[CORE].std()),*np.quantile(truth[CORE],[.001,.01,.1,.5,.9,.99,.999]).tolist()],
        regional=dict(mean_crps=float(rc['crps'].mean()),bias=rc['bias'].tolist(),spread=rc['std'].tolist(),
            coverage90=rc['covered_0.9'].tolist(),attainable90=rc['attainable']['0.9']),
        core_summary_draws=summaries.tolist(),core_summary_truth=true_features.mean(0).tolist(),
        primary=dict(density_crps=density['standardized_crps'][0],tidal_energy=physical['tidal_joint_energy'],
            density_rmse=density['rmse_mean'][0],density_coverage90=density['coverage']['0.9'][0],
            tidal_coverage90=np.mean(physical['coverage']['0.9'][1:4]),
            tidal_coverage90_components=physical['coverage']['0.9'][1:4],
            attainable90=density['attainable_coverage']['0.9']),
        elapsed_seconds=time.monotonic()-started,wide_unavailable_reason=None if wide else 'no sampled wide matter field in this case')
    durable.publish_json(output,result)
    print('CASE_REPORTED',task['task_id'],result['elapsed_seconds'],flush=True)
    return result


def pair_report(first,second,scales):
    x=np.concatenate([first['core_summary_draws'],second['core_summary_draws']],axis=1)/np.tile(scales,2)
    y=np.concatenate([first['core_summary_truth'],second['core_summary_truth']])/np.tile(scales,2)
    difference=x[:,:6]-x[:,6:]
    target=y[:6]-y[6:]
    c=calibration(difference,target)
    return dict(energy=float(fair_energy(x,y)),variogram=variogram_score(x,y),
        posterior_cross_covariance=np.cov(x,rowvar=False,ddof=1)[:6,6:].tolist(),
        difference_crps=c['crps'].tolist(),difference_coverage90=c['covered_0.9'].tolist(),
        attainable90=c['attainable']['0.9'],phase=first['phase'],cap=first['cap'],
        caveat='two caps per evaluation phase; covariance is descriptive, joint scores use realized truths')


def aggregate(records):
    means=lambda key:float(np.mean([r['primary'][key] for r in records]))
    sample=np.mean([r['spectra']['mean_sample_power'] for r in records],axis=0)
    truth=np.mean([r['spectra']['truth_power'] for r in records],axis=0)
    return dict(density_crps=means('density_crps'),tidal_energy=means('tidal_energy'),
        density_rmse=float(np.sqrt(np.mean([r['primary']['density_rmse']**2 for r in records]))),
        density_coverage90=means('density_coverage90'),tidal_coverage90=means('tidal_coverage90'),
        tidal_coverage90_components=np.mean([r['primary']['tidal_coverage90_components'] for r in records],axis=0).tolist(),
        attainable90=means('attainable90'),
        aggregate_sample_power_ratio=(sample/np.maximum(truth,1e-30)).tolist(),
        power_discrepancy=float(np.mean(np.abs(np.log(np.maximum(sample,1e-30)/np.maximum(truth,1e-30))))),
        anchors=len(records),independent_phases=len({r['phase'] for r in records}))


def block_interval(first,second,key,block_size,seed=741,draws=1024):
    if [r['task']['anchor'] for r in first]!=[r['task']['anchor'] for r in second]:
        raise ValueError('bootstrap requires paired anchors')
    groups=defaultdict(list)
    for i,row in enumerate(first):
        block=tuple(np.floor(np.asarray(row['source_center_mpc_h'])/block_size).astype(int))
        groups[(row['phase'],*block)].append(i)
    if len(groups)<3:
        return dict(blocks=len(groups),interval=None,reason='fewer than three occupied source blocks')
    groups=list(groups.values())
    difference=np.array([b['primary'][key]-a['primary'][key] for a,b in zip(first,second)])
    rng=np.random.default_rng(seed)
    values=[]
    for _ in range(draws):
        ids=np.concatenate([groups[j] for j in rng.integers(len(groups),size=len(groups))])
        values.append(difference[ids].mean())
    return dict(blocks=len(groups),interval=np.quantile(values,[.025,.975]).tolist(),
        point=float(difference.mean()),block_size_mpc_h=block_size,
        caveat='descriptive fixed-phase spatial bootstrap; not cosmological population uncertainty')


def contrast(first,second,key):
    a,b=aggregate(first),aggregate(second)
    score_gain=(a[key]-b[key])/max(abs(a[key]),1e-12)
    coverage='tidal_coverage90_components' if key=='tidal_energy' else 'density_coverage90'
    before=float(np.mean(np.abs(np.asarray(a[coverage])-a['attainable90'])))
    after=float(np.mean(np.abs(np.asarray(b[coverage])-b['attainable90'])))
    checks=dict(primary_direction=score_gain>0,coverage=before-after>=.05 or after<=.05,
                density_mean_nonregression=b['density_rmse']<=a['density_rmse']*1.05,
                sample_power_nonregression=b['power_discrepancy']<=a['power_discrepancy']+.05)
    return dict(first=a,second=b,relative_primary_gain=score_gain,coverage_gap_before=before,
        coverage_gap_after=after,checks=checks,passed=all(checks.values()),
        spatial_uncertainty={str(size):block_interval(first,second,key,size) for size in (500,1000)})


def contrast_decision(rows,key):
    """Equal seed/phase weights; pooled >=10%, positive direction in every cell."""
    if len(rows)!=4 or {(r['replica'],r['phase']) for r in rows}!={(s,p) for s in (0,1) for p in ('ph004','ph005')}:
        raise ValueError('exact two-seed two-phase contrast required')
    first=float(np.mean([r['first'][key] for r in rows]))
    second=float(np.mean([r['second'][key] for r in rows]))
    gain=(first-second)/max(abs(first),1e-12)
    return dict(equal_weight_first=first,equal_weight_second=second,relative_primary_gain=gain,
                passed=bool(gain>=.1 and all(r['passed'] for r in rows)),
                policy='pooled gain >=10%; all four seed/phase directions and nonregression checks pass')


_REPORT_DATA=None
_REPORT_ROOT=None


def initialize_case_worker(root):
    global _REPORT_DATA,_REPORT_ROOT
    _REPORT_ROOT=Path(root)
    # Parent already verifies all data payloads and the full matrix release.
    # Each worker still checks role/geometry/receipt bindings without rereading
    # the entire multi-GB product set for every one of the688 tasks.
    _REPORT_DATA=Products(_REPORT_ROOT,list(ROLES),targets=True,
        confirmation_receipt=_REPORT_ROOT/'MODELS_FROZEN.json',verify=False)


def report_case_worker(task):
    return summarize_case(_REPORT_ROOT,task,_REPORT_DATA)


def render_report(root,result,records,data):
    """Human-readable evidence and figures, with no checkpoint/model selection."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=root/'analysis'
    colors=dict(A='tab:blue',B='tab:orange',C='tab:green',D='tab:red')
    fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    panels=[('density_crps','Density fair CRPS'),('tidal_energy','Tidal joint energy'),
            ('density_coverage90','Density central coverage'),('tidal_coverage90','Mean eigenvalue central coverage')]
    for ax,(key,label) in zip(axes.flat,panels):
        for arm in 'ABCD':
            for seed in (0,1):
                for phase in ('ph004','ph005'):
                    rows=sorted((r for r in result['progression'] if r['arm']==arm and r['replica']==seed and r['phase']==phase),key=lambda r:r['checkpoint'])
                    ax.plot([r['checkpoint'] for r in rows],[r[key] for r in rows],
                        color=colors[arm],linestyle='-' if seed==0 else '--',marker='o' if phase=='ph004' else 's',
                        label=arm if seed==0 and phase=='ph004' else None)
        if 'coverage' in key:
            targets=[r['attainable90'] for r in result['progression']]
            ax.axhspan(min(targets),max(targets),color='grey',alpha=.2,label='finite-M target')
            ax.set_ylim(0,1)
        ax.set(title=label,xlabel='Updates per factor')
        ax.legend(fontsize=8)
    fig.suptitle('Fixed checkpoint progression; solid/dashed: seeds0/1; circles/squares: phases004/005')
    with (folder/'checkpoint_progression.png').open('xb') as stream:
        fig.savefig(stream,format='png',dpi=140)
    plt.close(fig)
    final=[r for r in records if r['task']['purpose']=='main' and r['task']['steps']==250 and r['task']['checkpoint']==20480]
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for arm in 'ABCD':
        rows=[r for r in final if r['task']['arm']==arm]
        truth=np.mean([r['spectra']['truth_power'] for r in rows],axis=0)
        for key,style,label in [('mean_sample_power','-',arm+' draws'),('posterior_mean_power','--',arm+' mean')]:
            value=np.mean([r['spectra'][key] for r in rows],axis=0)/np.maximum(truth,1e-30)
            axes[0].plot(range(len(value)),value,style,color=colors[arm],label=label)
        corr=np.mean([r['spectra']['correlation_posterior_mean'] for r in rows],axis=0)
        axes[1].plot(range(len(corr)),corr,'o-',color=colors[arm],label=arm)
    axes[0].axhline(1,color='grey',linewidth=.7)
    axes[0].set(title='Final sample power vs posterior-mean power',ylabel='Power / truth power',xlabel='Registered k band')
    axes[1].set(title='Posterior-mean cross-correlation',ylabel='r',xlabel='Registered k band')
    for ax in axes:
        ax.legend(fontsize=7)
    with (folder/'field_statistics.png').open('xb') as stream:
        fig.savefig(stream,format='png',dpi=140)
    plt.close(fig)
    anchor=read_json(root/'DRAW_LEDGER.json')['panels']['refinement'][0]
    truth=data.raw_targets(anchor)['rho']-1
    fig,axes=plt.subplots(3,5,figsize=(14,8),constrained_layout=True)
    lo,hi=np.quantile(truth,[.01,.99])
    axes[0,0].imshow(truth[24],origin='lower',vmin=lo,vmax=hi,cmap='magma')
    axes[0,0].set_title('Truth '+anchor,fontsize=7)
    axes[1,0].axis('off');axes[2,0].axis('off')
    for column,arm in enumerate('ABCD',1):
        row=next(r for r in final if r['task']['anchor']==anchor and r['task']['arm']==arm and r['task']['replica']==0)
        fields,_=ensemble(root,row['task'])
        for index,(label,value) in enumerate([('draw0',fields[0]),('mean',fields.mean(0)),('spread',fields.std(0,ddof=1))]):
            ax=axes[index,column]
            ax.imshow(value[24],origin='lower',cmap='magma',vmin=0 if index==2 else lo,vmax=None if index==2 else hi)
            ax.set_title(f'{arm} {label}; range[{value.min():.2g},{value.max():.2g}]',fontsize=8)
    for ax in axes.flat:
        ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle('Fixed development anchor, seed0, final checkpoint; density colors use truth1--99% range')
    with (folder/'posterior_fields.png').open('xb') as stream:
        fig.savefig(stream,format='png',dpi=140)
    plt.close(fig)
    lines=['# Controlled VDM field-posterior experiment','',
        'All registered checkpoints and draws are assessed; no validation-selected checkpoint.',
        'A:32 fields, summary context; B:384 fields, same context; C:spatial wide context; D:shared stochastic coarse/fine.',
        '',f"Saved central draws: {result['central_draws']}; shared coarse draws: {result['coarse_draws']}.",'',
        '| Contrast | Pooled primary-score gain | Joint registered gate |',
        '| --- | ---: | --- |']
    for label,row in result['contrast_decisions'].items():
        lines.append(f"| {label} | {100*row['relative_primary_gain']:.2f}% | {'PASS' if row['passed'] else 'NOT ESTABLISHED'} |")
    lines+=['','Individual seed/phase cells, coverage, power, paired dependence and spatial uncertainty are in RESULTS.json.',
        'A failed contrast is inconclusive at this budget, not proof that its physical hypothesis or VDM is false.',
        '', '![Checkpoint progression](checkpoint_progression.png)','',
        '![Field statistics](field_statistics.png)','', '![Fixed posterior fields](posterior_fields.png)','',
        '## Claim boundaries','']+['- '+item for item in result['limitations']]
    lines+=['','No real-DESI production release, full-field SBC claim, or automatic architecture/optimizer change.',
        'Final allocated GPU/CPU time and Scratch footprint are recorded in EXPERIMENT_COMPLETE.json after Slurm accounting closes.','']
    with (folder/'REPORT.md').open('x') as stream:
        stream.write('\n'.join(lines))
    durable.publish_json(folder/'FIGURES.json',dict(manifest_sha256=existing.sha256(root/'MANIFEST.json'),
        inputs_sha256=existing.sha256(folder/'RESULTS.json'),files={name:existing.sha256(folder/name) for name in
            ('REPORT.md','checkpoint_progression.png','field_statistics.png','posterior_fields.png')}))


def report(root,workers=1):
    require_compute()
    root=output_root(root)
    verify_launch(root)
    from workflows.sbi.e2e_vdm_context_control import verify_models_frozen
    verify_models_frozen(root)  # Before any confirmation target read, including cached reports.
    sampler=read_json(root/'analysis/SAMPLER_GATE.json')
    if not sampler['passed']:
        raise PermissionError('sampler gate failed')
    ledger=read_json(root/'DRAW_LEDGER.json')
    data=Products(root,list(ROLES),targets=True,confirmation_receipt=root/'MODELS_FROZEN.json')
    for arm in spec()['arms']:
        for replica in spec()['replicas']:
            done=read_json(root/'sampling'/f'{arm}_seed{replica}_ALL_COMPLETE.json')
            if done['manifest_sha256']!=existing.sha256(root/'MANIFEST.json'):
                raise ValueError('sampling branch incomplete/drifted')
    if not 1<=workers<=8:
        raise ValueError('bounded CPU reporting requires one to eight workers')
    if workers==1:
        records=[summarize_case(root,t,data) for t in ledger['tasks']]
    else:
        with ProcessPoolExecutor(max_workers=workers,initializer=initialize_case_worker,initargs=(str(root),)) as pool:
            records=list(pool.map(report_case_worker,ledger['tasks']))
    cases={r['task']['task_id']:r for r in records}
    grouped=defaultdict(list)
    for r in records:
        t=r['task']
        if t['purpose']=='main' and t['steps']==250:
            grouped[t['arm'],t['replica'],t['checkpoint'],r['phase']].append(r)
    progression=[dict(arm=a,replica=s,checkpoint=k,phase=p,**aggregate(v)) for (a,s,k,p),v in sorted(grouped.items())]
    pairs=[]
    scales=np.asarray(data.chart['physical_probe_scales']['std'])
    for arm in spec()['arms']:
        for replica in spec()['replicas']:
            for pair in ledger['panels']['pairs']:
                for purpose in (['joint','joint_fixed_mean','joint_oracle_diagnostic'] if arm=='D' else ['joint']):
                    selected=[next(r for r in records if r['task']['arm']==arm and r['task']['replica']==replica
                                   and r['task']['purpose']==purpose and r['task']['anchor']==anchor) for anchor in pair['cores']]
                    pairs.append(dict(arm=arm,replica=replica,purpose=purpose,domain=pair['domain'],
                                      **pair_report(*selected,scales)))
    contrasts=[]
    for a,b,key,label in [('A','B','density_crps','H1_diversity'),('B','C','density_crps','H2_observed_context'),('C','D','tidal_energy','H2_multiscale_package')]:
        for replica in spec()['replicas']:
            for phase in ('ph004','ph005'):
                first=sorted(grouped[a,replica,20480,phase],key=lambda r:r['task']['anchor'])
                second=sorted(grouped[b,replica,20480,phase],key=lambda r:r['task']['anchor'])
                result=contrast(first,second,key)
                if b=='D':
                    pa=[p for p in pairs if p['arm']==a and p['replica']==replica and p['phase']==phase and p['purpose']=='joint']
                    pb=[p for p in pairs if p['arm']==b and p['replica']==replica and p['phase']==phase and p['purpose']=='joint']
                    result['dependence']={score:dict(first=float(np.mean([p[score] for p in pa])),second=float(np.mean([p[score] for p in pb])))
                                          for score in ('energy','variogram')}
                    result['checks']['dependence_scores']=all(v['second']<v['first'] for v in result['dependence'].values())
                    result['passed']=all(result['checks'].values())
                contrasts.append(dict(label=label,first_arm=a,second_arm=b,replica=replica,phase=phase,**result))
    conditioned=[]
    for (arm,replica,checkpoint,phase),values in sorted(grouped.items()):
        if checkpoint!=20480:
            continue
        for key,edges in data.chart['conditioning_bin_edges'].items():
            for index in range(3):
                selected=[r for r in values if np.searchsorted(edges,r['observations'][key],side='right')==index]
                if selected:
                    conditioned.append(dict(arm=arm,replica=replica,phase=phase,variable=key,bin=index,
                        train_only_edges=edges,**aggregate(selected)))
    decision_details={label:contrast_decision([r for r in contrasts if r['label']==label],key)
        for label,key in [('H1_diversity','density_crps'),('H2_observed_context','density_crps'),('H2_multiscale_package','tidal_energy')]}
    decisions={label:value['passed'] for label,value in decision_details.items()}
    result=dict(manifest_sha256=existing.sha256(root/'MANIFEST.json'),central_draws=ledger['central_draws'],
        coarse_draws=ledger['coarse_draws'],case_receipts={t['task_id']:existing.sha256(root/'analysis/cases'/(t['task_id']+'.json')) for t in ledger['tasks']},
        progression=progression,contrasts=contrasts,pairs=pairs,conditioning=conditioned,
        contrast_decisions=decision_details,positive_contributions=decisions,
        no_automatic_model_change=True,production_ready=False,full_field_sbc=False,
        interpretation='A failed contrast is partial/inconclusive at this budget, not proof the hypothesis or VDM objective is false',
        limitations=['two programme-exposed evaluation phases','D adds a coarse model and compute',
                     'no audited fibre/redshift-success response','shared coarse does not prove full fine dependence',
                     'matter beyond the wide domain is absent','grid probes are correlated',
                     'coarse factor sees wide observations only; sufficiency for all local observations is unproven',
                     'fine cores are conditionally independent given the shared coarse field'])
    durable.publish_json(root/'analysis/RESULTS.json',result)
    render_report(root,result,records,data)
    print('REPORT_COMPLETE',decisions,flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    report(p.parse_args().root)


if __name__=='__main__':
    main()
