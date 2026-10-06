"""Independent weighted arithmetic review and static Stage B figures."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import e2e_stage_b_prepare as p
import numpy as np
c=p.c


def review(figures=None):
    c.require_compute()
    path=p.ROOT/'score/COMPLETE.json';scored=json.loads(path.read_text())
    assert scored['passed'] and scored['panel_complete']
    prepared=json.loads((p.ROOT/'prepare/COMPLETE.json').read_text())
    replay=json.loads((p.ROOT/'replay/COMPLETE.json').read_text())
    assert c.sha256(p.ROOT/'prepare/COMPLETE.json')==scored['binding']['prepared']['sha256']
    assert c.sha256(p.ROOT/'replay/COMPLETE.json')==scored['binding']['replay']['sha256']
    checks=0;error=0.
    for row in scored['cases']:
        pid=row['pair_id'];p.guard(row['phase'])
        a=next(q for q in prepared['cases'] if q['pair_id']==pid)
        b=next(q for q in replay['cases'] if q['pair_id']==pid)
        for q in (a,b):assert c.sha256(q['output']['path'])==q['output']['sha256']
        with np.load(a['output']['path']) as f:counts=f['cell_counts'];truth=f['truth_cell_labels'];flat=f['flat_index']
        with np.load(b['output']['path']) as f:labels=f['labels']
        np.testing.assert_array_equal(counts.ravel(),np.bincount(flat,minlength=8192));checks+=1
        for mi,method in enumerate(('original','corrected')):
            for ti,t in enumerate(p.THRESHOLDS):
                pp=np.stack([(labels[mi,ti]==k).sum(0)/32 for k in range(4)],axis=-1)
                yy=np.eye(4)[truth[ti]]
                raw=((pp-yy)**2*counts[...,None]).sum((0,1))
                fair=raw-(pp*(1-pp)*counts[...,None]).sum((0,1))/31
                qq=row['reliability'][f'{method}/{t}']
                err=float(abs(raw-qq['raw_brier_sum']).max());error=max(error,err)
                np.testing.assert_allclose(raw,qq['raw_brier_sum'],atol=1e-8,rtol=0)
                np.testing.assert_allclose(fair,qq['fair_brier_sum'],atol=1e-8,rtol=0)
                confusion=np.zeros((4,4),np.int64);hard=pp.argmax(-1)
                for actual in range(4):
                    for predicted in range(4):confusion[actual,predicted]=counts[(truth[ti]==actual)&(hard==predicted)].sum()
                np.testing.assert_array_equal(confusion,qq['confusion']);checks+=3
                for weighting,w in [('volume',np.ones_like(counts)),('galaxy',counts)]:
                    values=np.stack([np.column_stack([(labels[mi,ti,:,side]==k).astype(float)@w[side]
                        for side in range(2)]) for k in range(4)],axis=-1)/w.sum(-1)[None,:,None]
                    actual=np.stack([((truth[ti]==k)*w).sum(-1) for k in range(4)],axis=-1)/w.sum(-1)[:,None]
                    np.testing.assert_allclose(values.sum(-1),1,atol=1e-14)
                    direct=abs(values-actual).mean(0)-abs(values[:,None]-values[None,:]).sum((0,1))/(2*32*31)
                    recorded=row['regional'][f'{method}/{t}/{weighting}']['marginal']
                    np.testing.assert_allclose(direct,recorded['crps'],atol=1e-14)
                    # Independent all-pairs norm arithmetic on fixed simplex coordinates.
                    x=values[...,:3].reshape(32,6);y=actual[:,:3].ravel()
                    energy=np.linalg.norm(x-y,axis=-1).mean()-np.linalg.norm(x[:,None]-x[None,:],axis=-1).sum()/(2*32*31)
                    np.testing.assert_allclose(energy,row['regional'][f'{method}/{t}/{weighting}']['joint_pair']['energy'],atol=1e-14)
                    checks+=3
    headlines=[]
    for phase in p.a.PHASES:
        g=next(q for q in scored['groups'] if q['phase']==phase and q['group']=='all')
        selected=[q for q in prepared['cases'] if q['phase']==phase]
        entry=dict(phase=phase,galaxies=sum(q['galaxies'] for q in selected),cores=2*len(selected),
            zero_support_excluded=sum(q['zero_support_excluded'] for q in selected),methods={})
        for method in ('baseline','original','corrected'):
            q=g['reliability'][f'{method}/0.2']
            bins=[v for rows in q['bins'] for v in rows]
            entry['methods'][method]=dict(brier=q['brier'],fair_brier=q['fair_brier'],
                classwise_fair_brier=q['classwise_fair_brier'],
                reliable_bins=sum(v['screen']=='pass' for v in bins),failed_bins=sum(v['screen']=='fail' for v in bins),
                sparse_bins=sum(v['screen']=='sparse' for v in bins),empty_bins=sum(v['screen']=='empty' for v in bins),
                max_abs_residual_nonsparse=max([abs(v['residual']) for v in bins if v['screen'] in ('pass','fail')],default=None))
            if method!='baseline':
                entry['methods'][method]['regional']={w:g['regional'][f'{method}/0.2/{w}'] for w in ('volume','galaxy')}
        headlines.append(entry)
    summary=dict(passed=True,checks=checks,max_brier_sum_discrepancy=error,
        source=c.file_record(__file__,content_hash=True),score=c.file_record(path,content_hash=True),
        headline=headlines,posterior_qualified=False)
    c.atomic_json(p.ROOT/'REVIEW.json',summary)
    if figures:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        figures.mkdir(parents=True,exist_ok=True)
        names=('Void','Wall','Filament','Knot')
        style=(('original','#2563a6','o'),('corrected','#d07a26','s'))
        fig,axes=plt.subplots(2,4,figsize=(13,7),sharex=True,sharey=True)
        for i,phase in enumerate(p.a.PHASES):
            g=next(q for q in scored['groups'] if q['phase']==phase and q['group']=='all')
            for k,name in enumerate(names):
                ax=axes[i,k];ax.plot([0,1],[0,1],':',color='#333333')
                for method,color,marker in style:
                    q=g['reliability'][f'{method}/0.2']['bins'][k]
                    valid=[v for v in q if v['galaxies']]
                    ax.plot([v['mean_probability'] for v in valid],[v['truth_frequency'] for v in valid],color=color,alpha=.65,lw=1)
                    for v in valid:
                        ax.scatter(v['mean_probability'],v['truth_frequency'],marker=marker,s=32,
                            facecolors='none' if v['screen']=='sparse' else color,edgecolors=color)
                ax.set(title=f'{phase} · {name}',xlim=(-.02,1.02),ylim=(-.02,1.02));ax.grid(alpha=.15)
                if i==1:ax.set_xlabel('Mean predicted probability')
                if k==0:ax.set_ylabel('Observed truth frequency')
        from matplotlib.lines import Line2D
        handles=[Line2D([0],[0],color=color,marker=marker,label=method.capitalize()) for method,color,marker in style]
        fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.955),ncol=2)
        fig.suptitle('Galaxy-position class reliability at threshold 0.2')
        fig.text(.5,.012,'Observed-coordinate cell lookup; matched galaxies and physical R7 truth. Hollow points: fewer than four cores.\nTwo reused phases; dependent galaxies/cores, no iid confidence bands.',ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.07,1,.90));fig.savefig(figures/'galaxy_reliability.png',dpi=160);plt.close(fig)
        fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True,sharey=True)
        for i,weight in enumerate(('volume','galaxy')):
            for j,phase in enumerate(p.a.PHASES):
                g=next(q for q in scored['groups'] if q['phase']==phase and q['group']=='all');ax=axes[i,j]
                for offset,(method,color,marker) in zip((-.1,.1),style):
                    q=g['regional'][f'{method}/0.2/{weight}']
                    ax.scatter(np.arange(4)+offset,q['coverage']['0.9'],color=color,marker=marker,label=method.capitalize())
                ax.axhline(.9,ls=':',color='#333333');ax.set(title=f'{phase} · {weight}-weighted fractions',
                    xticks=np.arange(4),xticklabels=names,ylim=(0,1.04));ax.grid(axis='y',alpha=.15)
                if j==0:ax.set_ylabel('Randomization-averaged rank coverage')
        axes[0,0].legend(fontsize=8,loc='lower left')
        fig.suptitle('Regional class-fraction coverage at threshold 0.2')
        fig.text(.5,.015,'Nominal 90% reference is exact for this tie-aware rank diagnostic, not an inclusive interval convention.\n32 cores per phase; equal-core summaries, dependent regions, no population qualification.',ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.085,1,.95));fig.savefig(figures/'fraction_coverage.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(passed=True,checks=checks,headlines=headlines),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--figures',type=Path);args=parser.parse_args();review(args.figures)
