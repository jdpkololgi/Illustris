"""Archive compact learning-test evidence without promoting diagnostic checkpoints."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from workflows.sbi import e2e_wide_pipeline as p


def summarize(data):
    if not data['complete'] or len(data['results']) != 6 or len(data['fields']) != 48:
        raise ValueError('incomplete learning test')
    result = {k: data[k] for k in ('registration','parents','checkpoints','elapsed_seconds','training_ready','calibration_pass')}
    result['branches'] = {}
    result['fields'] = {}
    result['field_sha256'] = {f'{r["method"]}/{r["label"]}/{r["anchor_id"]}': r['sample_sha256'] for r in data['fields']}
    result['counts'] = dict(updates=sum(len(r['training']) for r in data['results']),
                           probes=sum(len(c['probes']) for r in data['results'] for c in r['curve']),
                           draws=len(data['fields']),checkpoints=len(data['checkpoints']))
    for branch in data['results']:
        key = f'{branch["method"]}/{branch["arm"]}'
        curve = {str(c['update']): c['summary'] for c in branch['curve']}
        phase_curves = {}
        for point in branch['curve']:
            for group in ('fit','transfer'):
                for phase in ('ph000','ph002','ph003'):
                    for ratio in (.05,.2):
                        rows = [r for r in point['probes'] if r['group']==group and r['phase']==phase and r['ratio']==ratio]
                        phase_curves[f'{point["update"]}/{group}/{phase}/{ratio}'] = {
                            k: np.median([r['metrics'][k] for r in rows],axis=0).tolist()
                            for k in ('noise_amplitude','error_power','gain')}
        gradients=np.array([r['gradient_norm_before_clip'] for r in branch['training']])
        result['branches'][key] = dict(curve=curve,phase_curves=phase_curves,gate=branch['gate'],
                                       clipping_fraction=float(np.mean(gradients > 1.)),
                                       median_unclipped_gradient_norm=float(np.median(gradients)))
    for method in ('cfm','diffusion'):
        for label in ('parent','uniform_time','balanced_noise','near_clean'):
            for group in ('fit','transfer'):
                rows=[r for r in data['fields'] if r['method']==method and r['label']==label and r['group']==group]
                result['fields'][f'{method}/{label}/{group}'] = {
                    'power_ratio': np.median([r['metrics']['power_ratio'] for r in rows],axis=0).tolist(),
                    'density_below_minus_one_fraction': float(np.median([r['density_below_minus_one_fraction'] for r in rows]))}
    return result


def plot(data, out):
    colors={'uniform_time':'#777777','balanced_noise':'#3575a5','near_clean':'#b54e43'}
    fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
    for i,method in enumerate(('cfm','diffusion')):
        for j,ratio in enumerate((.05,.2)):
            ax=axes[i,j]
            for arm,color in colors.items():
                for group,style in [('fit','-'),('transfer','--')]:
                    curve=data['branches'][f'{method}/{arm}']['curve']
                    steps=sorted(map(int,curve))
                    noise=[curve[str(step)][f'{group}/{ratio}']['noise_amplitude'][-1] for step in steps]
                    ax.plot(steps,noise,marker='o',linestyle=style,color=color,
                            label=arm.replace('_',' ')+(f' ({group})' if i==0 and j==0 else ''))
            ax.axhline(.2,color='0.2',linestyle=':',label='registered amplitude threshold')
            ax.set(title=f'{method.upper()}, noise/signal = {ratio}',xlabel='Additional diagnostic updates',
                   ylabel='High-k residual noise / input noise',ylim=(-.1,1.1))
    axes[0,0].legend(fontsize=7,ncol=2)
    fig.suptitle('Fine-stage learnability: same network and parent weights\n'
                 'Solid: three fitted anchors. Dashed: three training-only transfer anchors.',fontsize=12)
    fig.savefig(out/'learning_curves.png',dpi=160);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args()
    data=json.loads(args.input.read_text());summary=summarize(data)
    summary.update(raw_path=str(args.input.resolve()),raw_sha256=p.sha256(args.input),report_sha256=p.sha256(__file__))
    p.write_json(args.output/'SUMMARY.json',summary)
    plot(summary,args.output)
    print(json.dumps({'counts':summary['counts'],'gates':{k:v['gate'] for k,v in summary['branches'].items()},
                      'fields':summary['fields']},indent=2))


if __name__=='__main__':
    main()
