"""Independent moment/score arithmetic and source-backed Stage C figures."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import e2e_stage_c_mechanism as d
import numpy as np
c=d.c


def independent_moments(x,y,q):
    n,draws,dim=x.shape;errors=x.mean(1)-y
    cov=sum(np.cov(v,rowvar=False,ddof=1) for v in x)/n
    second=sum(np.outer(v,v) for v in errors)/n
    bias=errors.mean(0);adjusted=second-cov/draws
    centered=second-np.outer(bias,bias)-cov/draws
    for value,key in [(cov,'predictive_covariance'),(second,'error_second_moment'),
            (adjusted,'adjusted_error_second_moment'),(centered,'centered_adjusted_error_second_moment'),(bias,'bias')]:
        np.testing.assert_allclose(value,q[key],rtol=1e-11,atol=1e-13)
    np.testing.assert_allclose(np.diag(adjusted),q['adjusted_mse'],atol=1e-13)
    if np.trace(cov)>1e-15:
        np.testing.assert_allclose(np.trace(centered)/np.trace(cov),q['trace_centered_adjusted_mse_over_variance'],atol=1e-12)
    if n>=4:
        # Generalized eigenvalues through a Cholesky solve, independent of the
        # symmetric eigendecomposition whitening used by the production code.
        eig,vec=np.linalg.eigh(cov);regular=(vec*np.maximum(eig,q['covariance_floor']))@vec.T
        lower=np.linalg.cholesky(regular)
        for matrix,key in [(adjusted,'generalized_eigenvalues'),(centered,'centered_generalized_eigenvalues')]:
            white=np.linalg.solve(lower,matrix)
            white=np.linalg.solve(lower,white.T).T
            # Redundant simplex directions require a floor and are numerically
            # much less informative than the six-coordinate paired summaries.
            np.testing.assert_allclose(np.linalg.eigvalsh(white),q[key],atol=2e-6,rtol=2e-6)
    return 9


def energy(x,y):
    x=x.reshape(len(x),-1);y=y.ravel();m=len(x)
    return float(np.linalg.norm(x-y,axis=1).mean()-np.linalg.norm(x[:,None]-x[None,:],axis=-1).sum()/(2*m*(m-1)))


def review(figures=None):
    c.require_compute();root=d.ROOT/'full';path=root/'COMPLETE.json'
    result=json.loads(path.read_text());assert result['passed'] and result['panel_complete']
    for src in result['binding']['source']+[result['binding']['config'],result['panel']]:
        assert c.sha256(src['path'])==src['sha256']
    with np.load(result['panel']['path']) as f:arrays={key:f[key] for key in f.files}
    rows=result['cases'];xall=arrays['fractions'];yall=arrays['truth'];checks=0
    old=json.loads(Path(result['binding']['b_score']['path']).read_text())
    assert c.sha256(result['binding']['b_score']['path'])==result['binding']['b_score']['sha256']
    headlines=[]
    for r in result['fraction_results']:
        phase=np.array([i for i,v in enumerate(rows) if v['phase']==r['phase']])
        meta=[rows[i] for i in phase]
        subset=next(index for name,index in d.selections(meta) if name==r['group']);index=phase[subset]
        mi=d.METHODS.index(r['operator']);ti=d.p.THRESHOLDS.index(r['threshold']);wi=d.WEIGHTS.index(r['weighting'])
        x=xall[index,mi,ti,wi];y=yall[index,ti,wi]
        checks+=independent_moments(x.transpose(0,2,1,3).reshape(-1,32,4),y.reshape(-1,4),r['classwise'])
        checks+=independent_moments(x[...,:3].reshape(len(x),32,6),y[...,:3].reshape(len(y),6),r['joint'])
        for key,xx,yy in [('left',x[:,:,0],y[:,0]),('right',x[:,:,1],y[:,1]),
                ('mean',x.mean(2),y.mean(1)),('difference',x[:,:,1]-x[:,:,0],y[:,1]-y[:,0])]:
            checks+=independent_moments(xx,yy,r['cross_core'][key])
        if r['group']!='all':continue
        opposite=np.array([i for i,v in enumerate(rows) if v['phase']!=r['phase']])
        bias=(xall[opposite,mi,ti,wi].mean(1)-yall[opposite,ti,wi]).mean((0,1))
        np.testing.assert_allclose(bias,r['transfer_bias'],atol=1e-14);checks+=1
        original=next(q for q in old['groups'] if q['phase']==r['phase'] and q['group']=='all')['regional'][f"{r['operator']}/{r['threshold']}/{r['weighting']}"]
        unshifted=r['sensitivity']['no_bias/scale1.0']['score']
        for key in ('joint_pair','joint_difference'):
            for level in d.b.LEVELS:
                np.testing.assert_allclose(unshifted[key]['coverage'][str(level)],original[key]['coverage'][str(level)],atol=1e-14);checks+=1
        for name,probe in r['sensitivity'].items():
            scale=float(name.split('scale')[-1]);offset=bias if name.startswith('other_phase') else np.zeros(4)
            z=x+(scale-1)*(x-x.mean(1,keepdims=True))-offset
            invalid=np.any((z < -1e-12)|(z > 1+1e-12),axis=-1)
            assert int(invalid.sum())==probe['support']['out_of_simplex_vectors'];checks+=1
            for key,xx,yy in [('joint_pair',z[...,:3],y[...,:3]),
                    ('joint_difference',z[:,:,1,:3]-z[:,:,0,:3],y[:,1,:3]-y[:,0,:3])]:
                np.testing.assert_allclose(np.mean([energy(a,b) for a,b in zip(xx,yy)]),probe['score'][key]['energy'],atol=1e-13);checks+=1
            direct=abs(z-y[:,None]).mean((0,1,2))-abs(z[:,:,None]-z[:,None,:]).sum((0,1,2,3))/(len(z)*2*2*32*31)
            np.testing.assert_allclose(direct,probe['score']['crps'],atol=1e-13);checks+=1
        if r['threshold']==.2 and r['weighting']=='galaxy':headlines.append(r)
    for r in result['factor_results']:
        phase=np.array([i for i,v in enumerate(rows) if v['phase']==r['phase']]);meta=[rows[i] for i in phase]
        subset=next(index for name,index in d.selections(meta) if name==r['group']);index=phase[subset]
        key=r['factor'].replace('/','_');x=arrays[key+'_draws'][index];y=arrays[key+'_truth'][index]
        checks+=independent_moments(x.transpose(0,2,1,3).reshape(-1,32,5),y.reshape(-1,5),r['projection'])
        checks+=independent_moments(x.reshape(len(x),32,10),y.reshape(len(y),10),r['joint'])
    for r in result['cyclic_case_results']:
        index=next(i for i,v in enumerate(rows) if v['pair_id']==r['pair_id'])
        mi=d.METHODS.index(r['operator']);wi=d.WEIGHTS.index(r['weighting'])
        x=xall[index,mi,0,wi];y=yall[index,0,wi];cross=[]
        for offset in range(32):
            z=np.stack((x[:,0],x[(np.arange(32)-offset)%32,1]),axis=1)
            cov=np.cov(z.reshape(32,8),rowvar=False);cross.append(cov[:4,4:])
            for key,xx,yy in [('joint_pair',z[...,:3],y[:,:3]),('joint_difference',z[:,1,:3]-z[:,0,:3],y[1,:3]-y[0,:3])]:
                np.testing.assert_allclose(energy(xx,yy),r['result'][key]['energy'][offset],atol=1e-13);checks+=1
        np.testing.assert_allclose(np.mean(cross,0),0,atol=1e-14);checks+=1
    out=dict(passed=True,checks=checks,source=c.file_record(__file__,content_hash=True),
        result=c.file_record(path,content_hash=True),tests=c.file_record(Path(__file__).resolve().parents[2]/'tests/phase4/test_e2e_stage_c_mechanism.py',content_hash=True),
        headline=headlines,factor_headline=[r for r in result['factor_results'] if r['group']=='all'],
        posterior_qualified=False,causal_mechanism_identified=False)
    c.atomic_json(root/'REVIEW.json',out)
    if figures:plot(out,figures)
    print(json.dumps(dict(passed=True,checks=checks,review=str(root/'REVIEW.json'))))


def plot(review,folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(10,4.8),sharey=True)
    styles=[('original','#2563a6','o',-.08),('corrected','#d07a26','s',.08)]
    for ax,phase in zip(axes,d.p.a.PHASES):
        for method,color,marker,offset in styles:
            r=next(q for q in review['headline'] if q['phase']==phase and q['operator']==method)
            ax.scatter(np.arange(4)+offset,r['classwise']['adjusted_mse_over_variance'],color=color,marker=marker,label=method.capitalize())
        ax.axhline(1,color='#333333',ls=':');ax.set(title=phase,xticks=np.arange(4),xticklabels=['Void','Wall','Filament','Knot'],ylim=(0,2.6))
        ax.grid(axis='y',alpha=.15)
    axes[0].set_ylabel('Adjusted mean squared error / posterior variance');axes[0].legend()
    fig.suptitle('Galaxy-weighted fraction error and variance by class')
    fig.text(.5,.015,'Threshold 0.2; 32 equally weighted cores per phase; draw-mean Monte Carlo term removed.\nReference 1 is a conditional-calibration moment identity, not a finite-panel acceptance band.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.09,1,.94));fig.savefig(folder/'class_error_variance.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4.8),sharex=True,sharey=True)
    levels=np.array(d.b.LEVELS)
    styles=[('no_bias/scale1.0','Original','#2563a6','o'),
        ('other_phase_bias/scale1.0','Other-phase bias shift','#d07a26','s'),
        ('no_bias/scale1.5','Spread x 1.5','#758046','^'),('shuffle','Cyclic pairing','#a85f87','D')]
    for ax,phase in zip(axes,d.p.a.PHASES):
        r=next(q for q in review['headline'] if q['phase']==phase and q['operator']=='original')
        ax.plot([0,1],[0,1],':',color='#333333')
        for key,label,color,marker in styles:
            q=r['cyclic_pairing']['joint_pair'] if key=='shuffle' else r['sensitivity'][key]['score']['joint_pair']
            ax.plot(levels,[q['coverage'][str(v)] for v in levels],label=label,color=color,marker=marker)
        ax.set(title=phase,xlabel='Nominal rank level',xlim=(.45,1),ylim=(0,1.04));ax.grid(alpha=.15)
    axes[0].set_ylabel('Joint rank inclusion');axes[0].legend(fontsize=8,loc='upper left')
    fig.suptitle('Summary-space sensitivity of paired galaxy fractions')
    fig.text(.5,.015,'Original operator, threshold 0.2; 16 pairs per phase. Shift/scale can leave the simplex.\nSensitivity probes are not physical posterior repairs; neither phase is fresh validation.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.09,1,.94));fig.savefig(folder/'joint_sensitivity.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--figures',type=Path);args=parser.parse_args();review(args.figures)
