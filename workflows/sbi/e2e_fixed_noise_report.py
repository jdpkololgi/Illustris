"""Archive fixed-noise capability curves, controls and paired phase gates."""
import argparse
import copy
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from workflows.sbi import e2e_wide_pipeline as p
from workflows.sbi.e2e_fine_learning_test import aggregate, gates


def noise_alignment(probes):
    """Descriptive existing-error decomposition, not a new success criterion."""
    result={}
    for group in ('fit','transfer'):
        for ratio in (.05,.2,1.,5.,20.):
            rows=[r for r in probes if r['group']==group and r['ratio']==ratio]
            fractions=[np.array(r['metrics']['noise_correlated_error_power'])/
                       np.maximum(r['metrics']['error_power'],1e-30) for r in rows]
            result[f'{group}/{ratio}']=np.median(fractions,axis=0).tolist()
    return result


def summarize(data):
    counts=dict(updates=sum(len(b['training']) for b in data['results']),
                probes=len(data['baseline'])+sum(len(r['probes']) for r in data['references'])+
                       sum(len(c['probes']) for b in data['results'] for c in b['curve']),
                reconstructions=len(data['fields']),checkpoints=len(data['checkpoints']))
    if not data['complete'] or counts!=dict(updates=4096,probes=2100,reconstructions=72,checkpoints=24):
        raise ValueError('incomplete registered fixed-noise experiment')
    summary={k:data[k] for k in ('registration','checkpoints','fields','elapsed_seconds','training_ready','calibration_pass')}
    summary.update(counts=counts,baseline=aggregate(data['baseline']),branches={},references={})
    for ref in data['references']:
        summary['references'][ref['label']]={k:ref[k] for k in ('summary','gate','noiseless')}
    for branch in data['results']:
        name=f'{branch["arm"]}/{branch["ratio"]}';grad=np.array([r['gradient_norm_before_clip'] for r in branch['training']])
        panel=copy.deepcopy(data['registration']['panel']);panel['gate']['near_clean_ratios']=[branch['ratio']]
        phase_curves={}
        for point in branch['curve']:
            for group in ('fit','transfer'):
                for phase in ('ph000','ph002','ph003'):
                    rows=[r for r in point['probes'] if r['group']==group and r['phase']==phase and r['ratio']==branch['ratio']]
                    if len(rows)!=2:
                        raise ValueError('missing paired phase measurements')
                    phase_curves[f'{point["update"]}/{group}/{phase}']={
                        k:np.median([r['metrics'][k] for r in rows],axis=0).tolist()
                        for k in ('noise_amplitude','error_power','gain')}
        summary['branches'][name]=dict(arm=branch['arm'],ratio=branch['ratio'],parameters=branch['parameters'],
            elapsed_seconds=branch['elapsed_seconds'],gate=branch['gate'],phase_curves=phase_curves,
            gate_curve={str(c['update']):gates(data['baseline'],c['probes'],panel) for c in branch['curve']},
            noise_aligned_error_fraction={str(c['update']):noise_alignment(c['probes']) for c in branch['curve']},
            curve={str(c['update']):c['summary'] for c in branch['curve']},
            clipped_fraction=float(np.mean(grad>data['registration']['config']['clip'])),
            gradient_norm_quantiles=np.quantile(grad,[0,.5,.9,1]).tolist(),
            loss_medians_by_128_updates=[float(np.median([r['loss'] for r in branch['training'][i:i+128]]))
                                       for i in range(0,512,128)])
    return summary


def verify(data,root):
    for path,expected in data['registration']['source_sha256'].items():
        if p.sha256(p.REPO/path)!=expected:
            raise ValueError('source mismatch: '+path)
    for path,expected in data['checkpoints'].items():
        if p.sha256(root/path)!=expected:
            raise ValueError('checkpoint mismatch: '+path)
    for record in data['fields']:
        if p.sha256(root/record['path'])!=record['sha256']:
            raise ValueError('reconstruction mismatch')
    common=None
    for branch in data['results']:
        seeds=[(r['anchor_id'],r['noise_seed']) for r in branch['training']]
        if common is not None and seeds!=common:
            raise ValueError('unpaired training noises')
        common=seeds
        evaluation={r['seed'] for c in branch['curve'] for r in c['probes']}
        if evaluation & {s for _,s in seeds}:
            raise ValueError('noise seed overlap')
    return dict(source_hashes=True,checkpoint_hashes=True,reconstruction_hashes=True,
                paired_training_noise=True,separate_evaluation_noise=True)


def plot(summary,out):
    fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
    colors=['#444444','#888888','#287cb0','#25905f']
    for i,group in enumerate(('fit','transfer')):
        for j,ratio in enumerate((.05,.2)):
            ax=axes[i,j];key=f'{group}/{ratio}'
            for arm,color in zip(summary['registration']['config']['arms'],colors):
                curve=summary['branches'][f'{arm}/{ratio}']['curve'];steps=sorted(map(int,curve))
                ax.plot(steps,[curve[str(s)][key]['noise_amplitude'][-1] for s in steps],
                        marker='o',color=color,label=arm)
            ax.axhline(summary['references']['lowpass']['summary'][key]['noise_amplitude'][-1],
                       color='#be8024',linestyle='--',label='nonperiodic linear reference')
            ax.axhline(.2,color='0.3',linestyle=':',label='amplitude threshold')
            ax.set(title=f'{group}: fixed noise ratio {ratio}',xlabel='Diagnostic updates',
                   ylabel='Residual high-k noise / injected VP noise')
    axes[0,0].legend(fontsize=7)
    fig.suptitle('Fixed-noise capability: separate specialist fit at each ratio\n'
                 'A pass also requires each-phase error reduction and signal preservation.')
    fig.savefig(out/'learning_curves.png',dpi=160);plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('input',type=Path);ap.add_argument('output',type=Path)
    args=ap.parse_args();data=json.loads(args.input.read_text());summary=summarize(data)
    summary.update(verification=verify(data,args.input.parent),raw_path=str(args.input.resolve()),
                   raw_sha256=p.sha256(args.input),report_sha256=p.sha256(__file__))
    p.write_json(args.output/'SUMMARY.json',summary);plot(summary,args.output)
    print(json.dumps(dict(counts=summary['counts'],gates={k:v['gate'] for k,v in summary['branches'].items()}),indent=2))


if __name__=='__main__':
    main()
