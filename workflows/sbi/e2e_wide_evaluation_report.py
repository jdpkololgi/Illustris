"""Summarize completed training-panel diagnostics; plotting remains on compute."""
import argparse
import json
from pathlib import Path
import numpy as np


def summarize(report, root):
    losses=report['fixed_loss_probes']
    optimization={}
    for method in ('cfm','diffusion'):
        for stage in ('coarse','fine'):
            rows=[r for r in losses if r['method']==method and r['stage']==stage]
            x=np.array([[r['losses'][str(s)] for s in (96,144,192)] for r in rows])
            means=x.mean(axis=0)
            optimization[f'{method}/{stage}']={'means_96_144_192':means.tolist(),
                'late_relative_improvement':float(1-means[2]/means[1]),
                'fraction_anchors_improving_late':float(np.mean(x[:,2]<x[:,1])),
                'by_phase':{ph:np.mean([[r['losses'][str(s)] for s in (96,144,192)] for r in rows if r['phase']==ph],axis=0).tolist()
                            for ph in ('ph000','ph002','ph003')}}
    refinement={}
    for method in ('cfm','diffusion'):
        values=[r for r in report['sampler_refinement'] if r['method']==method]
        refinements=[]
        for r in values:
            ensemble=json.loads((root/f'{r["anchor_id"]}_{method}.json').read_text())['ensemble']['observed']
            rms=np.array(r['masks']['observed']['eigen_rmse'])
            spread=np.array(ensemble['mean_pointwise_draw_std'])
            refinements.append(rms/np.maximum(spread,1e-30))
        refinement[method]={'median_eigen_rms_change_over_draw_std':np.median(refinements,axis=0).tolist(),
            'max_eigen_rms_change_over_draw_std':np.max(refinements,axis=0).tolist()}
    return {'optimization':optimization,'sampler_refinement':refinement,
            'science':report['summary'],'by_phase':report['by_phase'],
            'by_shell_support':report['by_shell_support'],
            'scope':'training panel; no held-out calibration; four draws per anchor'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    report=json.loads((args.root/'EVALUATION_COMPLETE.json').read_text())
    from workflows.sbi import e2e_wide_pipeline as p
    p.require_compute()
    result=summarize(report,args.root)
    p.write_json(args.root/'INTERPRETATION_SUMMARY.json',result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for ax,(name,r) in zip(axes.flat,result['optimization'].items()):
        for phase,values in r['by_phase'].items():
            ax.plot([96,144,192],values,'o-',label=phase,alpha=.65)
        ax.plot([96,144,192],r['means_96_144_192'],'ko-',label='All training phases',lw=2)
        ax.set(title=name,xlabel='Training update',ylabel='Fixed-noise objective loss',xticks=[96,144,192])
        ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Optimization probes on training data — not held-out calibration')
    fig.savefig(args.root/'fixed_noise_convergence.png',dpi=160)
    plt.close(fig)
    print(json.dumps(result['optimization'],indent=2))
    print(json.dumps(result['sampler_refinement'],indent=2))


if __name__=='__main__':
    main()
