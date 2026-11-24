"""Matched, per-phase descriptive plots from completed blind scores. No voxel-level error bars."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lowz_blind_readout import summarize


def run(directory,out):
    report=summarize(directory)
    r={(v['phase'],v['shell'],v['method']):v for v in report['rows']
       if v['cap']=='all' and v['support']=='all' and v['weighting']=='cells'}
    out.mkdir(parents=True,exist_ok=False)
    fields=['Density',r'$\lambda_1$',r'$\lambda_2$',r'$\lambda_3$']
    methods=('cfm','baseline');colors=('#146c80','#aa4e24')
    shells=['.01–.15','.15–.25','.25–.35','.35–.45','.45–.55']
    fig,axes=plt.subplots(3,4,figsize=(13,8.5),layout='constrained')
    for j,label in enumerate(fields):
        axes[0,j].set_title(label)
        for i,metric in enumerate(['fair_crps','coverage90','width90']):
            ax=axes[i,j]
            for ph,style in [('ph018','-'),('ph019','--')]:
                for method,color in zip(methods,colors):
                    y=[r.get((ph,s,method),{}).get(metric,[np.nan]*4)[j] for s in range(5)]
                    ax.plot(range(5),y,style,marker='.',color=color,label=f'{method} {ph}')
            if i==1:ax.axhline(.9,color='0.4',lw=.9)
            ax.set_xticks(range(5),shells if i==2 else ['']*5,rotation=35)
            if j==0:ax.set_ylabel(['Fair CRPS','5–95% coverage','5–95% width'][i])
    axes[0,0].legend(fontsize=7)
    fig.suptitle('Blind mock confirmation · supported cells · both caps · two independent phases')
    fig.savefig(out/'marginals_by_shell.pdf');plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(13,6),layout='constrained')
    for j,label in enumerate(fields):
        axes[0,j].set_title(label)
        for ph,style in [('ph018','-'),('ph019','--')]:
            for method,color in zip(methods,colors):
                v=r[ph,'all',method]
                rank=np.array(v['ranks'][j]);axes[0,j].plot(rank/rank.sum(),style,color=color)
                rel=v['class_reliability'];w=np.array(rel['weight'][j]);sel=w>0
                axes[1,j].plot(np.array(rel['sum_probability'][j])[sel]/w[sel],
                               np.array(rel['sum_truth'][j])[sel]/w[sel],style,marker='.',color=color,
                               label=f'{method} {ph}')
        axes[0,j].axhline(1/33,color='0.5',lw=.8);axes[0,j].set_xlabel('Truth rank among 32 draws')
        axes[1,j].plot([0,1],[0,1],':',color='0.5');axes[1,j].set_title(report['class_order'][j])
        axes[1,j].set_xlabel('Predicted probability')
    axes[1,0].legend(fontsize=7);axes[0,0].set_ylabel('Cell fraction');axes[1,0].set_ylabel('Observed frequency')
    fig.suptitle('Single-cell ranks and T-web reliability · blind mocks, not DESI truth')
    fig.savefig(out/'ranks_and_web_reliability.pdf');plt.close(fig)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path);ap.add_argument('out',type=Path)
    args=ap.parse_args();run(args.directory,args.out)
