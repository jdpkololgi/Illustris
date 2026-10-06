"""Independent score arithmetic and source-backed static Stage A figures."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import e2e_stage_a_calibration as a
import numpy as np
b=a.b


def independent_joint(x,y):
    """Centered matrix products and Cholesky, not the production solve helper."""
    z=np.vstack((np.asarray(y).ravel(),np.asarray(x).reshape(len(x),-1)))
    scores=[]
    for i in range(len(z)):
        rest=np.concatenate((z[:i],z[i+1:]))
        centered=rest-rest.mean(0)
        cov=centered.T@centered/(len(rest)-1)
        scale=max(np.trace(cov)/len(cov),1e-12)
        cov=.8*cov+.2*np.diag(np.diag(cov))+np.eye(len(cov))*scale*1e-8
        d=np.linalg.solve(np.linalg.cholesky(cov),z[i]-rest.mean(0))
        scores.append(d@d)
    scores=np.asarray(scores);u=np.random.default_rng(119).uniform()
    return float(((scores<scores[0]).sum()+u*(scores==scores[0]).sum())/len(z))


def run(include_a2=False,figures=None):
    b.c.require_compute();root=a.ROOT
    result=json.loads((root/'a1/COMPLETE.json').read_text());assert result['passed'] and result['panel_complete']
    checks=0;maxrank=0.;records=[];supplement=[];cached_sources=[]
    for phase in a.PHASES:
        a.guard(phase)
        prior=json.loads((a.SAVED/'results'/f'COMPLETE_{phase}.json').read_text())
        cached_path=b.ROOT/'interim_scores'/phase/'SCORES.json'
        cached=json.loads(cached_path.read_text())
        assert cached['panel_complete'] and cached['phase']==phase
        assert cached['freeze']['sha256']==b.c.sha256(b.FREEZE)
        for receipt in cached['inputs']:
            assert b.c.sha256(receipt['path'])==receipt['sha256']
        cached_sources.append(b.c.file_record(cached_path,content_hash=True))
        for original in prior['cases']:
            pid=original['pair_id'];row=next(v for v in result['cases'] if v['pair_id']==pid)
            coarse=np.asarray([v['coarse_tensor_mean'] for v in original['arms']['canvas_raw']['scale']])[...,[0,3,5]].sum(-1)
            coarse_truth=np.asarray(original['truth_scale']['coarse_eigenvalues']).sum(-1)
            cached_row=next(v for v in cached['cases'] if v['pair_id']==pid)
            extra={k:row[k] for k in ('phase','cap','shell','support_stratum','pair_id')}
            extra['metrics']={'coarse_trace':a.scalar_scores(coarse,coarse_truth),
                'coarse_joint_trace':a.vector_scores(coarse,coarse_truth)}
            for name in ('density','core_mass'):
                q=cached_row['metrics'][name]
                extra['metrics']['cached_fullcap/'+name]=dict(kind='scalar',
                    coverage=q['coverage'],reference=q['attainable'],crps=q['crps'],
                    bias=q['bias'],mse=q['mse'],variance=q['variance'],
                    mse_minus_draw_noise=q['mse']-q['variance']/32,
                    rank_histogram=q['rank_histogram'],width90=q['width90'],
                    caveat='Cached score: interval width is available only at 90%; not regenerated.')
            supplement.append(extra)
            for arm in a.ARMS:
                for factor in ('coarse','fine'):
                    if factor=='coarse' and arm.startswith('oracle'):continue
                    x=np.array([v[f'{factor}_shear_rms'] for v in original['arms'][arm]['scale']])
                    truth=original['truth_scale'][f'{factor}_shear_rms'];ordered=np.sort(x)
                    score=row['metrics'][f'{arm}/{factor}_shear_rms']
                    assert float(ordered[1]<=truth<=ordered[30])==score['coverage']['0.9']
                    crps=abs(x-truth).mean()-abs(x[:,None]-x[None,:]).sum()/(2*32*31)
                    np.testing.assert_allclose(crps,score['crps'],atol=1e-14)
                    checks+=2
            with np.load(root/'a1'/(pid+'_SUMMARIES.npz')) as f:
                for arm,factor in [('canvas_raw','coarse'),('oracle_coarse_local_fine','fine')]:
                    rank=independent_joint(f[f'{arm}_{factor}_shear'],f[f'truth_{factor}_shear'])
                    delta=abs(rank-row['metrics'][f'{arm}/{factor}_joint_shear']['rank'])
                    maxrank=max(maxrank,delta);assert delta<1e-12;checks+=1
                rank=independent_joint(f['fullcap_eigenvalues'],f['fullcap_physical_truth'])
                delta=abs(rank-row['metrics']['fullcap/physical/joint6']['rank'])
                maxrank=max(maxrank,delta);assert delta<1e-12;checks+=1
    selected=('canvas_raw/coarse_shear_rms','canvas_raw/coarse_joint_shear',
        'canvas_raw/fine_shear_rms','canvas_coarse_local_fine/fine_shear_rms','oracle_coarse_local_fine/fine_shear_rms',
        'oracle_coarse_local_fine/fine_joint_shear','oracle_coarse_local_fine/matched/joint4gap',
        'fullcap/physical/joint4gap','fullcap/physical/joint6')
    for group in result['groups']:
        if group['group']=='all':
            records.append(dict(phase=group['phase'],metrics={k:group['metrics'][k] for k in selected}))
    report=dict(passed=True,checks=checks,independent_joint_rank_maxabs=maxrank,
        a1=b.c.file_record(root/'a1/COMPLETE.json',content_hash=True),headline=records,
        source=b.c.file_record(__file__,content_hash=True),posterior_qualified=False)
    report['supplement']=dict(cases=supplement,groups=a.aggregate(supplement),sources=cached_sources,
        caveat='Coarse trace rescored from saved tensor means; composed density and mass scores reused after freeze and source-receipt verification. Density interval widths at 50/68/95 were not present in the cached scores and remain unreported.')
    if include_a2:
        revised=json.loads((root/'a2/COMPLETE.json').read_text());assert revised['panel_complete'] and revised['passed']
        parity=0.;crps_error=0.;rows=[]
        for row in revised['cases']:
            path=root/'a2'/(row['pair_id']+'_SUMMARIES.npz')
            assert b.c.sha256(path)==row['summary']['sha256']
            with np.load(path) as f,np.load(root/'a1'/(row['pair_id']+'_SUMMARIES.npz')) as old:
                delta=float(abs(f['original']-old['fullcap_eigenvalues']).max());parity=max(parity,delta)
                assert delta<2e-7
                for method in ('original','corrected'):
                    x=f[method][:,:,2]-f[method][:,:,0]
                    y=f['physical_truth'][:,2]-f['physical_truth'][:,0]
                    direct=abs(x-y).mean()-abs(x[:,None]-x[None,:]).sum()/(2*32*31*2)
                    crps_error=max(crps_error,abs(direct-row['metrics'][f'{method}/physical/width']['crps']))
                    assert crps_error<1e-14
                    for th in (0.,.2):np.testing.assert_allclose(f[f'{method}_fractions_{th}'].sum(-1),1,atol=1e-14)
                for method in ('original','corrected'):
                    delta=abs(independent_joint(f[method],f['physical_truth'])-row['metrics'][f'{method}/physical/joint6']['rank'])
                    assert delta<1e-12
        for phase in a.PHASES:
            group=next(v for v in revised['groups'] if v['phase']==phase and v['group']=='all')
            cases=[v for v in revised['cases'] if v['phase']==phase]
            entry=dict(phase=phase,methods={})
            for method in ('original','corrected'):
                metrics=group['metrics'];width=metrics[f'{method}/physical/width']
                entry['methods'][method]=dict(width_deficit=-width['bias']/width['truth'],width_sd=width['sd'],
                    width_crps=width['crps'],width_coverage90=width['coverage']['0.9'],
                    joint6_coverage90=metrics[f'{method}/physical/joint6']['coverage']['0.9'],
                    joint4gap_coverage90=metrics[f'{method}/physical/joint4gap']['coverage']['0.9'],
                    joint6_energy=metrics[f'{method}/physical/joint6']['energy'],
                    cell_eigenvalue_crps=[metrics[f'{method}/physical/cell_lambda{i}']['crps'] for i in (1,2,3)],
                    cell_eigenvalue_coverage90=[metrics[f'{method}/physical/cell_lambda{i}']['coverage']['0.9'] for i in (1,2,3)],
                    class_fraction_offset={str(th):np.mean([np.array(v['classes'][f'{method}/{th}']['mean_fraction'])-
                        v['classes'][f'{method}/{th}']['truth_fraction'] for v in cases],axis=0).tolist() for th in (0.,.2)},
                    brier={str(th):float(np.mean([v['classes'][f'{method}/{th}']['brier'] for v in cases])) for th in (0.,.2)},
                    fair_brier={str(th):float(np.mean([v['classes'][f'{method}/{th}']['fair_brier'] for v in cases])) for th in (0.,.2)})
            rows.append(entry)
        report['a2']=b.c.file_record(root/'a2/COMPLETE.json',content_hash=True)
        report['reconstruction']=rows;report['original_regional_replay_maxabs']=parity
        report['width_crps_independent_maxabs']=crps_error
    b.c.atomic_json(root/('REVIEW.json' if include_a2 else 'A1_REVIEW.json'),report)
    if figures:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        figures.mkdir(parents=True,exist_ok=True)
        fig,axes=plt.subplots(1,2,figsize=(11,4.4),sharex=True,sharey=True)
        for ax,row in zip(axes,records):
            for key,label,color,marker in [('canvas_raw/coarse_shear_rms','Coarse','#2563a6','o'),
                ('canvas_coarse_local_fine/fine_shear_rms','Fine given sampled coarse','#d07a26','s'),
                ('oracle_coarse_local_fine/fine_shear_rms','Fine given true coarse','#597d46','^')]:
                q=row['metrics'][key];xs=[q['reference'][str(p)] for p in a.LEVELS]
                ax.plot(xs,[q['coverage'][str(p)] for p in a.LEVELS],label=label,color=color,marker=marker)
            ax.plot([.25,1],[.25,1],':',color='#333333');ax.set(title=row['phase'],xlabel='Attainable central coverage',xlim=(.25,1),ylim=(0,1.03));ax.grid(alpha=.15)
        axes[0].set_ylabel('Observed coverage');axes[0].legend(fontsize=8,loc='upper left')
        fig.suptitle('Shear RMS coverage by model factor')
        fig.text(.5,.01,'16 dependent pairs per phase; 32 draws. Two reused phases, no population confidence bands.\nCommon local operator; fine sampler held fixed. True coarse conditioning is privileged.',ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.08,1,.94));fig.savefig(figures/'factor_coverage.png',dpi=160);plt.close(fig)
        if include_a2:
            fig,axes=plt.subplots(1,3,figsize=(12,4.5))
            for ax,key,title in zip(axes,['width_coverage90','joint4gap_coverage90','joint6_coverage90'],
                                  ['Regional width','Joint four gaps','Joint six eigenvalues']):
                for row,color,marker in zip(report['reconstruction'],['#2563a6','#d07a26'],['o','s']):
                    ax.plot([0,1],[row['methods'][m][key] for m in ('original','corrected')],color=color,marker=marker,label=row['phase'])
                ax.axhline(29/33,color='#333333',ls=':',label='Attainable reference')
                ax.set(xticks=[0,1],xticklabels=['Original','Corrected'],ylim=(0,1.03),title=title);ax.grid(alpha=.15)
            axes[0].set_ylabel('Observed nominal-90% coverage');axes[0].legend(fontsize=8)
            fig.suptitle('Density-only operator correction on all 64 development cores')
            fig.text(.5,.015,'Unchanged 32 field draws and physical truth; local correction retains original distant tide.\nDependent cores in two phases; not full-cap convergence or posterior qualification.',ha='center',fontsize=9)
            fig.tight_layout(rect=(0,.1,1,.94));fig.savefig(figures/'operator_coverage.png',dpi=160);plt.close(fig)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--a2',action='store_true');p.add_argument('--figures',type=Path)
    args=p.parse_args();run(args.a2,args.figures)
