"""Paired development scorecard and historical regional statistics from saved draws."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.distance import pdist
import lowz_ensemble_scores as e
from workflows.sbi.e2e_wholefield_fullcap_extract import summary


def run():
    import matplotlib.pyplot as plt
    base=e.root/'lowz_count_reference_61848915'
    ref=e.root/'lowz_calibration_61647276/CALIBRATION.json'
    saved=e.read(ref)
    bench=e.read(base/'REFERENCE_COMPLETE.json')
    assert bench['passed'] and bench['archive']==e.c.sha256(ref)
    getkey=lambda r:(r['pair'],r['weighting'],tuple(r['z']))
    a,b=[{getkey(r):r for r in rr} for rr in (saved['rows'],bench['rows'])]
    assert a.keys()==b.keys() and len(a)==len(saved['rows'])==len(bench['rows'])
    reg=[]; rng=np.random.default_rng(4971)
    def collect(out,label,x,y,n,mask,z):
        # Exact historical six-component geometry; includes modeled, unsupported
        # cells inside each core. Do not identify with supported-voxel calibration.
        x=x.astype('f8').reshape(32,2,4096,4); y=y.astype('f8').reshape(2,4096,4)
        xx=summary(x[...,0],x[...,1:]); yy=summary(y[...,0],y[...,1:])
        sd=xx.std(0,ddof=1); assert (sd>0).all()
        dif=xx.mean(0)-yy; q=np.quantile(xx,[.05,.95],axis=0)
        reg.append(dict(**label,draws=xx.tolist(),truth=yy.tolist(),err=dif.tolist(),sd=sd.tolist(),
                    percentile90=((q[0]<=yy)&(yy<=q[1])).tolist(),width=(q[1]-q[0]).tolist(),
                    ranks=((xx<yy).sum(0)+rng.random(6)*((xx==yy).sum(0)+1)).tolist(),
                    fair_energy=np.linalg.norm(xx-yy,axis=1).mean()-.5*pdist(xx).mean()))
    out=e.run(collect,'lowz_finish_scorecard_')
    assert len(reg)==48 and e.read(out/'CALIBRATION.json')==saved
    card=[]
    for weighting in ('cells','galaxies'):
        subsets=[('all','all',lambda r:True)]
        for axis in ('phase','cap','stratum','z'):
            for v in sorted({str(r[axis]) for r in a.values()}):
                subsets.append((axis,v,lambda r,i=axis,k=v:str(r[i])==k))
        for axis,v,keep in subsets:
            keys=[k for k,r in a.items() if r['weighting']==weighting and keep(r)]
            w=[a[k]['weight'] for k in keys]
            rr=dict(weighting=weighting,axis=axis,value=v,weight=sum(w),cells=sum(a[k]['cells'] for k in keys))
            for name,src in [('cfm',a),('counts',b)]:
                avg=lambda fun:np.average([fun(src[k]) for k in keys],weights=w,axis=0).tolist()
                rr[name]=dict(crps=avg(lambda s:s['fair_crps']),bias=avg(lambda s:s['bias']),
                    fair_brier=avg(lambda s:[i['fair_brier'] for i in s['classes']]),
                    intervals={j:{m:avg(lambda s:s['intervals'][j][m]) for m in ('coverage','width')}
                               for j in ('1','3','8')})
            rr['crps_skill']=(1-np.array(rr['cfm']['crps'])/rr['counts']['crps']).tolist()
            card.append(rr)
    e.c.atomic_json(out/'SCORECARD.json',dict(rows=card,regions=reg,comparison=e.c.sha256(base/'REFERENCE_COMPLETE.json'),
        archive=e.c.sha256(ref),script=e.c.sha256(__file__),passed=True,
        region_scope='exact historical two complete 16-cubed cores, density inner 12-cubes; modeled cells not restricted to survey support'))
    fig,ax=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    fields=['Density','λ1','λ2','λ3']; phases=e.phases
    for i,p in enumerate(phases):
        r=next(v for v in card if (v['weighting'],v['axis'],v['value'])==('cells','phase',p))
        ax[0].plot(fields,r['crps_skill'],'o-',label=p)
        ax[1].plot(fields,r['cfm']['bias'],'o-',label=p)
    for aa in ax:aa.axhline(0,color='grey',linewidth=0.8)
    ax[0].set_ylabel('Fair CRPS skill vs count regression');ax[0].legend()
    ax[1].set_ylabel('Mean prediction minus full-box truth')
    fig.savefig(out/'MATCHED.pdf');plt.close(fig)


if __name__=='__main__':run()
