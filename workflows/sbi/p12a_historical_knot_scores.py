"""Score frozen historical replays on identical ph006 rows and natural weights."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve,average_precision_score
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from workflows.sbi.p12a_historical_knot_comparison import sample,digest,CANDIDATE

def measures(y,score,w,pred):
    p,r,t=precision_recall_curve(y,score,sample_weight=w)
    tp=w[y & pred].sum();den=w[pred].sum()
    return {'average_precision':float(average_precision_score(y,score,sample_weight=w)),
            'hard_recall':float(tp/w[y].sum()),'hard_precision':float(tp/den) if den else None,
            'recall_at_precision':{str(v):float(r[p>=v].max()) for v in [.5,.6,.7,.8,.9]},
            'curve_precision':p.tolist(),'curve_recall':r.tolist()}

def main(a):
    a.output.mkdir(parents=True,exist_ok=True);f=sample();n=len(f['parent_node_id'])
    audit=CANDIDATE/'posterior/calibration_audit'
    ck=torch.load(CANDIDATE/'posterior/fmpe_estimator.pt',weights_only=False,map_location='cpu')
    draws=np.load(audit/'evaluation_samples_scaled.npy',mmap_mode='r');probs=np.empty((n,4))
    for lo in range(0,n,1000):
        th=np.asarray(draws[lo:lo+1000],dtype=np.float64)*np.asarray(ck['theta_std'])+np.asarray(ck['theta_mean'])
        eig=np.empty_like(th);eig[:,:,0]=th[:,:,0];eig[:,:,1]=eig[:,:,0]+np.logaddexp(0,th[:,:,1]);eig[:,:,2]=eig[:,:,1]+np.logaddexp(0,th[:,:,2])
        cl=(eig>.2).sum(axis=2);probs[lo:lo+len(cl)]=np.stack([(cl==j).mean(axis=1) for j in range(4)],axis=1)
    score={'Current posterior':probs[:,3],'Current encoder':f['base_prediction_eigenvalues'].min(axis=1)}
    hard={'Current posterior':probs.argmax(axis=1)==3,'Current encoder':score['Current encoder']>.2}
    source={}
    for directory in a.replays:
        for p in sorted(directory.glob('*_r0.npz')):
            meta=json.loads(p.with_suffix('.json').read_text());assert meta['complete'],str(p)
            d=np.load(p);assert d['done'].all() and np.array_equal(d['parent_node_id'],f['parent_node_id'])
            name=('Extended ' if 'extension' in meta['binding']['checkpoint'] else 'July ')+('U-PATCH' if p.stem.startswith('unet') else 'G-PATCH')
            assert name not in score
            score[name]=d['prediction'].min(axis=1);hard[name]=score[name]>.2
            source[name]={'manifest':meta,'prediction_sha256':digest(p),'ordering_violation_rate':float(np.any(np.diff(d['prediction'],axis=1)<0,axis=1).mean())}
    y=(f['truth_eigenvalues']>.2).sum(axis=1)==3;w=f['natural_weight'];z=f['context'][:,3]
    subsets={'Full range':np.ones(n,bool),'0.20–0.30':(z>=.2)&(z<.3)}
    for lo,hi in [(.15,.25),(.25,.35),(.35,.45),(.45,.55)]:subsets[f'{lo:.2f}–{hi:.2f}']=(z>=lo)&(z<hi)
    reports={};fig,axs=plt.subplots(2,3,figsize=(16,9),layout='constrained')
    for ax,(label,k) in zip(axs.flat,subsets.items()):
        result={name:measures(y[k],s[k],w[k],hard[name][k]) for name,s in score.items()}
        matched_precision=result['Current posterior']['hard_precision']
        for m in result.values():
            m['recall_at_current_hard_precision']=float(np.asarray(m['curve_recall'])[np.asarray(m['curve_precision'])>=matched_precision].max())
        prevalence=float(np.average(y[k],weights=w[k]));reports[label]={'rows':int(k.sum()),'true_knots':int(y[k].sum()),'weighted_prevalence':prevalence,'models':result}
        for j,(name,m) in enumerate(result.items()):
            line,=ax.plot(m['curve_recall'],m['curve_precision'],label=f"{name} (AP {m['average_precision']:.2f})",lw=1.5,ls=['-','--',':','-.','-','--'][j])
            if m['hard_precision'] is not None:ax.scatter(m['hard_recall'],m['hard_precision'],s=18,color=line.get_color())
        ax.axhline(prevalence,color='gray',ls=':',lw=1);ax.set(xlim=(0,1),ylim=(0,1),xlabel='True-knot recall',ylabel='Knot precision',title=f'{label}: {int(k.sum()):,} galaxies');ax.legend(fontsize=7)
    fig.suptitle('Frozen models on the same exposed ph006 galaxies\nDots: original hard decisions; curves: knot probability or smallest-eigenvalue ranking')
    fig.savefig(a.output/'precision_recall.png',dpi=180);fig.savefig(a.output/'precision_recall.pdf');plt.close(fig)
    # Paired spatial bootstrap for ranking discrimination, not per-galaxy independence.
    _,blocks=np.unique(np.column_stack([f['cap'],f['superblock_id']]),axis=0,return_inverse=True)
    nb=blocks.max()+1;rng=np.random.default_rng(59024562);bootstrap={}
    for label in ['Full range','0.20–0.30']:
        k=subsets[label];rep={name:[] for name in score}
        for _ in range(64):
            mult=np.bincount(rng.integers(0,nb,nb),minlength=nb);bw=w[k]*mult[blocks[k]]
            for name,s in score.items():rep[name].append(average_precision_score(y[k],s[k],sample_weight=bw))
        bootstrap[label]={name:{'ap_interval16_84':np.quantile(v,[.16,.84]).tolist(),'ap_minus_current_interval16_84':np.quantile(np.array(v)-rep['Current posterior'],[.16,.84]).tolist()} for name,v in rep.items()}
    result={'scope':'Identical exposed ph006 evaluation rows; no blind phase, fitting, or VAC mutation',
            'threshold_caveat':'Recall at precision is an empirical evaluation frontier, not a validated deployment threshold. Point scores are not probabilities.',
            'sample_sha256':digest(CANDIDATE/'dataset/ph006_selection_sample.npz'),
            'index_sha256':digest(audit/'evaluation_index.npy'),'draws_sha256':digest(audit/'evaluation_samples_scaled.npy'),
            'posterior_checkpoint_sha256':digest(CANDIDATE/'posterior/fmpe_estimator.pt'),
            'script_sha256':digest(__file__),'source':source,'subsets':reports,'paired_spatial_bootstrap':bootstrap}
    (a.output/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez(a.output/'matched_scores.npz',parent_node_id=f['parent_node_id'],truth_knot=y,weight=w,z=z,**{f'score_{i}':v for i,v in enumerate(score.values())})
    (a.output/'SCORE_ORDER.json').write_text(json.dumps(list(score),indent=2)+'\n')
    print({label:{name:{k:v for k,v in m.items() if not k.startswith('curve')} for name,m in r['models'].items()} for label,r in reports.items()},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--replays',nargs='+',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
