"""Boundary reliability of saved P12-A draws on exposed ph006; no model fitting."""
import argparse,json,sys,os
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from workflows.sbi.p12a_historical_knot_comparison import sample,digest,CANDIDATE

THRESHOLD=.2
LABELS=['Filament ↔ knot','Wall ↔ filament','Void ↔ wall']

def decode(t):
    e=np.empty_like(t);e[...,0]=t[...,0]
    e[...,1]=e[...,0]+np.logaddexp(0,t[...,1])
    e[...,2]=e[...,1]+np.logaddexp(0,t[...,2])
    return e

class Summaries:
    def __init__(self,w,blocks):
        self.w=w;_,self.block=np.unique(blocks,axis=0,return_inverse=True)
        self.nb=self.block.max()+1;rng=np.random.default_rng(20260929)
        self.mult=np.stack([np.bincount(rng.integers(0,self.nb,self.nb),minlength=self.nb) for _ in range(128)])
    def mean(self,value,k):
        w=self.w[k];v=np.asarray(value)[k]
        if not len(w):return None
        den=np.bincount(self.block[k],weights=w,minlength=self.nb)
        num=np.bincount(self.block[k],weights=w*v,minlength=self.nb)
        bd=self.mult@den;good=bd>0;b=(self.mult@num)[good]/bd[good]
        return {'value':float(np.sum(w*v)/w.sum()),'interval16_84':np.quantile(b,[.16,.84]).tolist(),'bootstrap_valid':int(good.sum())}
    def count(self,k):
        w=self.w[k]
        return {'rows':int(k.sum()),'effective_rows':float(w.sum()**2/(w@w)) if len(w) else 0.}

def reliability(q,y,k,stats):
    out=[]
    for low in np.arange(0,1,.1):
        take=k&(q>=low)&(q<(low+.1 if low<.89 else 1.00001))
        if take.any():out.append({'lower':float(low),**stats.count(take),'predicted':stats.mean(q,take),'observed':stats.mean(y,take),'observed_minus_predicted':stats.mean(y-q,take)})
    return out

def run(a):
    a.output.mkdir(parents=True,exist_ok=False)
    f=sample();n=len(f['parent_node_id']);w=f['natural_weight'].astype(float)
    assert n==50000 and np.isfinite(w).all() and np.all(w>0)
    assert np.allclose(decode(f['theta_softplus']),f['truth_eigenvalues'],atol=2e-5)
    cp=CANDIDATE/'posterior/fmpe_estimator.pt';ck=torch.load(cp,map_location='cpu',weights_only=False)
    marker=CANDIDATE/'dataset/P12A_DATASET_READY.json'
    assert ck['dataset_marker_sha256']==digest(marker)
    assert not json.loads(marker.read_text()).get('sealed_phase_opened')
    path=CANDIDATE/'posterior/calibration_audit/evaluation_samples_scaled.npy'
    draws=np.load(path,mmap_mode='r');assert draws.shape==(n,512,3)
    q=np.empty((n,3));p=np.empty((n,4));quant=np.empty((5,n,3));pit=np.empty((n,3))
    truth=f['truth_eigenvalues'];true_class=(truth>THRESHOLD).sum(axis=1)
    for lo in range(0,n,1000):
        e=decode(np.asarray(draws[lo:lo+1000],dtype=float)*np.asarray(ck['theta_std'])+np.asarray(ck['theta_mean']))
        q[lo:lo+len(e)]=(e>THRESHOLD).mean(axis=1)
        cl=(e>THRESHOLD).sum(axis=2)
        p[lo:lo+len(e)]=np.stack([(cl==j).mean(axis=1) for j in range(4)],axis=1)
        quant[:,lo:lo+len(e)]=np.quantile(e,[.05,.16,.5,.84,.95],axis=1)
        pit[lo:lo+len(e)]=(e<=truth[lo:lo+len(e),None,:]).mean(axis=1)
    assert np.allclose(q[:,0],p[:,3]) and np.allclose(q[:,1],p[:,2:].sum(axis=1)) and np.allclose(q[:,2],1-p[:,0])
    cov68=(truth>=quant[1])&(truth<=quant[3]);cov90=(truth>=quant[0])&(truth<=quant[4])
    z=f['context'][:,3];stats=Summaries(w,np.column_stack([f['cap'],f['superblock_id']]))
    subsets={'Full range':np.ones(n,bool),'0.20–0.30':(z>=.2)&(z<.3)}
    for lo,hi in [(.15,.25),(.25,.35),(.35,.45),(.45,.55)]:subsets[f'{lo:.2f}–{hi:.2f}']=(z>=lo)&(z<hi)
    result={};hard=p.argmax(axis=1)
    edges=np.array([-np.inf,-.2,-.1,-.05,-.025,0,.025,.05,.1,.2,np.inf])
    for label,k in subsets.items():
        result[label]=[]
        for j in range(3):
            lo_class=2-j;hi_class=3-j
            pair=np.isin(true_class,[lo_class,hi_class]);d=truth[:,j]-THRESHOLD;y=truth[:,j]>THRESHOLD
            ambiguous=(q[:,j]>=.16)&(q[:,j]<=.84)
            confident=(q[:,j]<=.1)|(q[:,j]>=.9)
            wrong=(q[:,j]>=.5)!=y
            metrics={'coverage68':cov68[:,j],'coverage90':cov90[:,j],
                     'interval68_crosses_threshold':(quant[1,:,j]<=THRESHOLD)&(quant[3,:,j]>=THRESHOLD),
                     'posterior_ambiguous':ambiguous,'confident_wrong':confident&wrong,
                     'binary_error':wrong,'hard_class_error':hard!=true_class,
                     'true_class_probability':p[np.arange(n),true_class],
                     'adjacent_pair_probability':p[:,lo_class]+p[:,hi_class],
                     'mean_probability_upper_side':q[:,j],'true_upper_side_fraction':y,
                     'median_error':quant[2,:,j]-truth[:,j],
                     'width68':quant[3,:,j]-quant[1,:,j]}
            def summarize(mask):
                return {**stats.count(mask),**{name:stats.mean(v,mask) for name,v in metrics.items()}}
            cohorts={'all':k,'posterior_transition_candidates':k&ambiguous,
                     'posterior_median_within_0.05':k&(abs(quant[2,:,j]-THRESHOLD)<=.05),
                     'confident_binary':k&confident}
            for radius in [.025,.05,.1]:
                near=pair&(abs(d)<=radius)
                cohorts[f'true_pair_within_{radius}']=k&near
                others=np.delete(abs(truth-THRESHOLD),j,axis=1)
                cohorts[f'true_pair_clean_within_{radius}']=k&near&np.all(others>radius,axis=1)
            cohorts['true_pair_farther_0.1']=k&pair&(abs(d)>.1)
            bins=[]
            for low,high in zip(edges[:-1],edges[1:]):
                take=k&pair&(d>=low)&(d<high)
                if take.any():bins.append({'low':None if not np.isfinite(low) else float(low),'high':None if not np.isfinite(high) else float(high),**summarize(take)})
            result[label].append({'boundary':LABELS[j],'eigenvalue_index':j+1,'cohorts':{name:summarize(mask) for name,mask in cohorts.items()},'true_distance_bins':bins,'reliability_all_rows':reliability(q[:,j],y,k,stats),'confident_expected_error':stats.mean(np.minimum(q[:,j],1-q[:,j]),k&confident)})
        print('Scored',label,flush=True)
    report={'threshold':THRESHOLD,'ordering':'ascending','scope':'Saved exposed ph006 50k; no ph001, no refitting or VAC modification',
            'bootstrap':'128 paired cap/superblock resamples; 16–84% intervals; not independent-phase uncertainty',
            'warning':'Truth-selected narrow-band interval coverage need not equal nominal even for a correct Bayesian posterior. Use posterior-selected cohorts and probability reliability for deployment calibration diagnostics.',
            'job':os.environ.get('SLURM_JOB_ID'),'source_hashes':{str(x):digest(x) for x in [cp,marker,path,CANDIDATE/'dataset/ph006_selection_sample.npz',CANDIDATE/'posterior/calibration_audit/evaluation_index.npy',Path(__file__)]},
            'subsets':result}
    (a.output/'RESULTS.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez(a.output/'transition_rows.npz',parent_node_id=f['parent_node_id'],truth=truth,probability_exceed=q,class_probability=p,quantiles=quant,pit=pit,weight=w,z=z,block=stats.block)
    figures=[]
    fig,axs=plt.subplots(1,3,figsize=(15,4.7),layout='constrained')
    for j,ax in enumerate(axs):
        for label,color in [('Full range','#2864a5'),('0.20–0.30','#bb5529')]:
            bins=[b for b in result[label][j]['reliability_all_rows'] if b['effective_rows']>=30]
            x=np.array([b['predicted']['value'] for b in bins]);y=np.array([b['observed']['value'] for b in bins]);ci=np.array([b['observed']['interval16_84'] for b in bins]).T
            ax.plot(x,y,'o-',color=color,label=label,ms=4);ax.fill_between(x,ci[0],ci[1],color=color,alpha=.15)
        ax.plot([0,1],[0,1],'k:',lw=1);ax.set(xlim=(0,1),ylim=(0,1),xlabel=f'Posterior P(λ{j+1} > 0.2)',ylabel='Observed fraction above threshold',title=LABELS[j]);ax.legend(fontsize=9)
    fig.suptitle('Are boundary-crossing probabilities reliable?\nAll galaxies in each redshift selection; shaded 16–84% spatial-bootstrap intervals')
    figures.append(('01_probability_reliability',fig))
    fig,axs=plt.subplots(2,3,figsize=(15,8.5),layout='constrained')
    for j in range(3):
        bins=result['Full range'][j]['true_distance_bins'];bins=[b for b in bins if b['low'] is not None and b['high'] is not None]
        x=np.array([.5*(b['low']+b['high']) for b in bins])
        for metric,label,style in [('mean_probability_upper_side','Mean probability of upper class side','-'),('true_upper_side_fraction','True side','--')]:
            yy=[b[metric]['value'] for b in bins]
            if metric=='true_upper_side_fraction':
                axs[0,j].step(x,yy,where='mid',ls=style,label=label)
            else:
                line,=axs[0,j].plot(x,yy,style,marker='o',ms=3,label=label)
                ci=np.array([b[metric]['interval16_84'] for b in bins]).T
                axs[0,j].fill_between(x,ci[0],ci[1],color=line.get_color(),alpha=.15)
        for metric,label in [('interval68_crosses_threshold','68% interval spans boundary'),('confident_wrong','Confidently wrong side')]:
            line,=axs[1,j].plot(x,[b[metric]['value'] for b in bins],marker='o',ms=3,label=label)
            ci=np.array([b[metric]['interval16_84'] for b in bins]).T
            axs[1,j].fill_between(x,ci[0],ci[1],color=line.get_color(),alpha=.15)
        for ax in axs[:,j]:ax.axvline(0,color='gray',ls=':');ax.set(ylim=(0,1),xlabel=f'True λ{j+1} − 0.2',ylabel='Weighted fraction / probability');ax.legend(fontsize=8)
        axs[0,j].set_title(LABELS[j])
    fig.suptitle('Behaviour through true class transitions\nAdjacent true classes only; truth-selected cohorts are descriptive, not nominal-coverage tests')
    figures.append(('02_transition_behaviour',fig))
    fig,axs=plt.subplots(1,3,figsize=(15,4.8),layout='constrained')
    groups=['all','posterior_transition_candidates','posterior_median_within_0.05']
    for j,ax in enumerate(axs):
        for level,offset,color in [('coverage68',-.1,'#2864a5'),('coverage90',.1,'#bb5529')]:
            rr=[result['Full range'][j]['cohorts'][g][level] for g in groups];y=np.array([v['value'] for v in rr]);ci=np.array([v['interval16_84'] for v in rr]).T;x=np.arange(3)+offset
            ax.scatter(x,y,color=color,label=level.replace('coverage','Nominal ')+'%');ax.vlines(x,ci[0],ci[1],color=color)
        ax.axhline(.68,color='#2864a5',ls=':');ax.axhline(.90,color='#bb5529',ls=':')
        ax.set(ylim=(.4,1),xticks=np.arange(3),xticklabels=['All','0.16 ≤ q ≤ 0.84','Median within\n±0.05'],ylabel='Empirical interval coverage',title=LABELS[j]);ax.legend(fontsize=8)
    fig.suptitle('Continuous-eigenvalue coverage in observable transition selections\nFull range; 16–84% spatial-bootstrap intervals')
    figures.append(('03_transition_coverage',fig))
    with PdfPages(a.output/'transition_audit.pdf') as pdf:
        for name,fig in figures:fig.savefig(a.output/f'{name}.png',dpi=180);pdf.savefig(fig);plt.close(fig)
    (a.output/'ARTIFACT_SHA256.json').write_text(json.dumps({p.name:digest(p) for p in a.output.iterdir()},indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a)
