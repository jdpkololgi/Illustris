"""CPU-only, development-only paired contrast replay; run in production import environment."""
import hashlib
import json
from pathlib import Path

import numpy as np


def contrast(a, mask, dx=6.766):
    """Masked, demeaned local pseudo-power in h/Mpc bins; not deconvolved P(k)."""
    v = a[:,mask]
    if not mask.any() or not np.isfinite(v).all():
        raise ValueError("Invalid supported field")
    win = mask.astype(float)
    for j,n in enumerate(mask.shape):
        shape=[1,1,1];shape[j]=n
        win *= np.hanning(n).reshape(shape)
    result=dict(sd=v.std(1).tolist(), q95_q05=np.diff(np.quantile(v,[.05,.95],axis=1),axis=0)[0].tolist())
    if win.sum()==0:
        return dict(result,power=None)
    a=np.where(mask,a,0.)
    mu=np.sum(a*win,axis=(1,2,3))/win.sum()
    spec=np.abs(np.fft.fftn((a-mu[:,None,None,None])*win,axes=(1,2,3)))**2*dx**3/(win**2).sum()
    ks=np.meshgrid(*[2*np.pi*np.fft.fftfreq(n,dx) for n in mask.shape],indexing="ij")
    k=np.sqrt(sum(g*g for g in ks))
    edges=np.linspace(2*np.pi/(min(mask.shape)*dx),np.pi/dx,5)
    bins=[(k>=lo)&(k<hi) for lo,hi in zip(edges,edges[1:])]
    return dict(result,k_edges=edges.tolist(),modes=[int(b.sum()) for b in bins],
                power=[spec[:,b].mean(1).tolist() for b in bins])


def run():
    import lowz_ensemble_scores as replay
    from matplotlib import pyplot as plt
    ref=replay.root/'lowz_calibration_61647276/CALIBRATION.json'
    digest=hashlib.sha256(ref.read_bytes()).hexdigest()
    assert digest=='72a63078c8728b35f5564c95c3af7871c33171333038cd5d13cbb10c5987bbae'
    records=[]; rng=np.random.default_rng(617)
    def reduce(out,label,x,y,counts,mask,z):
        if not np.isfinite(counts[mask]).all() or np.min(counts[mask])<0:
            raise ValueError("Invalid observed counts")
        rho=np.concatenate([y[None,...,0],x[...,0]])
        case=dict(**label,contrast=contrast(rho,mask),shells=[])
        for lo,hi in zip([.01,.15,.25,.35,.45],[.15,.25,.35,.45,.55]):
            m=mask&(z>=lo)&(z<hi)
            if not m.any():continue
            def fractions(v):return [(v==i).mean(-1).tolist() for i in range(4)]
            shell=dict(z=[lo,hi],n=int(m.sum()),fractions=fractions((x[:,m,1:]>.2).sum(-1)),
                       truth_fractions=fractions((y[m,1:]>.2).sum(-1)),by_count=[])
            for l,h in [(0,1),(1,2),(2,4),(4,np.inf)]:
                s=m&(counts>=l)&(counts<h)
                if s.any():
                    xx,yy=x[:,s],y[s]
                    q=np.quantile(xx,[.05,.95],axis=0)
                    shell['by_count'].append(dict(lower=l,**replay.metrics(xx,yy,np.ones(s.sum()),rng),
                        percentile90_width=(q[1]-q[0]).mean(0).tolist(),
                        percentile90_coverage=((yy>=q[0])&(yy<=q[1])).mean(0).tolist()))
            assert sum(b['cells'] for b in shell['by_count'])==int(m.sum())
            case['shells'].append(shell)
        records.append(case)
        imgs=[*rho[:4,:,:,8],x[:,:,:,8,0].mean(0),x[:,:,:,8,0].std(0,ddof=1)]
        fig,axes=plt.subplots(1,6,figsize=(14,3),layout='constrained')
        for ax,v,title in zip(axes,imgs,['Truth','Draw 0','Draw 1','Draw 2','Draw mean','Draw SD']):
            im=ax.imshow(np.ma.array(v.T,mask=~mask[:,:,8].T),origin='lower',interpolation='nearest',
                         vmin=0,vmax=2,cmap='viridis');ax.set_title(title);ax.axis('off')
        fig.suptitle(label['pair']+' | supported core slice');fig.colorbar(im,ax=list(axes),shrink=.65)
        fig.savefig(out/(label['phase']+'_'+label['pair']+'.png'));plt.close(fig)
    out=replay.run(reduce,'lowz_finish_dev_')
    if replay.read(ref)!=replay.read(out/'CALIBRATION.json'):
        raise ValueError("Historical score replay mismatch")
    assert len(records)==48
    replay.c.atomic_json(out/'FINISH_COMPLETE.json',dict(source_sha256=digest,passed=True,
        contrast_axis='truth, then draws 0..31',fractions_axes='class, draw; registered supported cells only',
        count_bins='gridded observed counts [0,1),[1,2),[2,4),[4,infinity)',records=records))


if __name__=='__main__':run()
