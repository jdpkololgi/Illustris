"""Single-cell probabilistic benchmark, not a spatial posterior sampler.

Run with the frozen production imports used for lowz_operator_closure. Only
local count, random support and observed radius enter the predictor. A simple
ridge regression and phase-held-out residuals are fitted on training mocks.
"""
import os
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter
import lowz_ph006_cap_draw as p
from lowz_ensemble_scores import metrics

c=p.c
EDGES=np.array([.01,.15,.25,.35,.45,.55])
read=lambda f:json.loads(f.read_text())


def observations(raw,row):
    shape=np.array([32,16,16]); pad=5; dx=6.766
    start=np.array(row['center'])-16
    def extract(name):
        a=p.products.padded_extract(raw[name],start-2*pad,2*(shape+2*pad))
        return a.reshape(shape[0]+2*pad,2,shape[1]+2*pad,2,shape[2]+2*pad,2).mean((1,3,5),dtype='f8')
    n=8*extract('counts'); support=extract('support_random')
    crop=(slice(pad,-pad),)*3
    smooth=lambda v:gaussian_filter(v,7/dx,mode='constant',truncate=4.)[crop]
    sm=smooth(support)
    cnt=np.log1p(smooth(n)/np.maximum(sm,.02))
    xyz=np.meshgrid(*[(np.arange(a)+.5)*dx+b*dx/2+o for a,b,o in
        zip(shape,start,row['grid']['origin_mpc_h'])],indexing='ij')
    z=p.coord.redshift(np.sqrt(sum(a*a for a in xyz))); zz=z*2
    x=np.stack([np.ones_like(z),cnt,cnt**2,zz,zz**2,zz**3,sm,sm**2,cnt*zz,cnt*zz**2,cnt*sm,sm*zz],-1)
    bins=np.searchsorted(EDGES,z,side='right')-1
    return x,bins,support[crop],sm,n[crop]


def truth(phase,row):
    pair=row['pair_id']
    rec=read(p.coord.ROOT/'targets'/phase/(pair+'.json'))
    assert rec['pass'] and (rec['phase'],rec['pair_id'])==(phase,pair)
    t=p.read_target(p.checked(rec),phase,pair)
    return np.concatenate([p.owned_core_arrays(t['rho_joint']).reshape(32,16,16,1),
            p.tidal.eigensystem(t['fullbox_tensor_cores']).reshape(32,16,16,3)],-1)


def encode(y):
    rho,a,b,d=y.T
    assert (rho>0).all() and (b>a).all() and (d>b).all()
    return np.stack([np.log(rho),b,np.log(b-a),np.log(d-b)],-1)


def decode(x):
    q,b,u,v=x.transpose(2,0,1)
    return np.stack([np.exp(q),b-np.exp(u),b,b+np.exp(v)],-1)


def groups(phase):
    assert phase in (*c.TRAIN,*c.DEVELOPMENT)
    g=p.coord.verify_receipt(p.coord.ROOT/'geometry'/phase/'GEOMETRY_COMPLETE.json',payload=False)
    # One replicate in every cap/shell/boundary stratum, chosen by name only.
    rows=[r for r in g['pairs'] if r['pair_id'].endswith('_00')]
    assert len(rows)==16 and len({r['pair_id'] for r in rows})==16
    for cap in ('NGC','SGC'):
        rec=read(p.coord.ROOT/'observations'/phase/(cap+'_COMPLETE.json'))
        assert rec['pass'] and rec['phase']==phase
        with h5py.File(p.checked(rec)) as raw:
            for row in rows:
                if row['cap']==cap:yield row,observations(raw,row)


def fit(out):
    assert len(c.TRAIN)==20 and not set(c.TRAIN)&set(c.DEVELOPMENT)
    xs=[];ys=[];bs=[];pairs=[]
    for ph in c.TRAIN:
        xx=[];yy=[];bb=[]
        for row,(x,b,s,sm,n) in groups(ph):
            m=(b>=0)&(b<5)&(s>0)
            # Deterministic equal-probability subsample of supported cells per domain.
            gen=np.random.default_rng(int(c.digest(row['pair_id'])[:15],16))
            ix=np.flatnonzero(m);ix=gen.choice(ix,min(2048,len(ix)),replace=False)
            xx.append(x.reshape(-1,x.shape[-1])[ix]);yy.append(encode(truth(ph,row).reshape(-1,4)[ix]))
            bb.append((2*b+(sm>=.8)).ravel()[ix]);pairs.append(row['pair_id'])
        xs.append(np.concatenate(xx));ys.append(np.concatenate(yy));bs.append(np.concatenate(bb))
        print('baseline training inputs',ph,flush=True)
    def coef(ii):
        g=sum(xs[i].T@xs[i] for i in ii);h=sum(xs[i].T@ys[i] for i in ii)
        return np.linalg.solve(g+np.eye(len(g))*.0001*sum(len(xs[i]) for i in ii),h)
    ids=list(range(len(xs))); cf=coef(ids)
    res=np.concatenate([ys[i]-xs[i]@coef([j for j in ids if j!=i]) for i in ids])
    bins=np.concatenate(bs);assert np.isfinite(res).all()
    file=out/'TRAIN_ONLY.npz'
    with file.open('xb') as f:np.savez_compressed(f,coef=cf,residual=res,bin=bins)
    c.atomic_json(out/'FIT.json',dict(phases=list(c.TRAIN),pairs=pairs,source=c.sha256(__file__),
        library=c.file_record(file,content_hash=True),ridge=.0001,radius_mpc_h=7.,
        residual_policy='joint pointwise residual vectors, leave-one-training-phase-out; shell x smoothed support 0.8; fallback within shell or globally for <32 rows'))


def run():
    c.require_compute()
    out=c.ROOT.parent/('lowz_count_reference_'+os.environ['SLURM_JOB_ID']);out.mkdir(exist_ok=False)
    fit(out)
    fitrec=read(out/'FIT.json');assert fitrec['library']['sha256']==c.sha256(out/'TRAIN_ONLY.npz')
    with np.load(out/'TRAIN_ONLY.npz') as f:cf,res,bins=[f[k] for k in ('coef','residual','bin')]
    ref=c.ROOT.parent/'lowz_calibration_61647276/CALIBRATION.json'
    assert c.sha256(ref)=='72a63078c8728b35f5564c95c3af7871c33171333038cd5d13cbb10c5987bbae'
    archive=read(ref); cmp={(r['pair'],tuple(r['z']),r['weighting']):r for r in archive['rows']}
    output=[]; rng=np.random.default_rng(61703)
    for ph in c.DEVELOPMENT:
        for row,(x,b,s,sm,n) in groups(ph):
            pair=row['pair_id'];cap=row['cap']
            path=Path(archive['paths'][str((ph,cap,0))])/'density/COMPLETE.json'
            record=read(path);assert record['passed'] and (record['phase'],record['cap'])==(ph,cap)
            with h5py.File(record['field']['path']) as f:
                sup=f['support'][p.cuts(row,f.attrs['origin_mpc_h'])]>0
            np.testing.assert_array_equal(s>0,sup)
            for i in range(5):
                k=(b==i)&sup
                if not k.any():continue
                mu=x[k]@cf; draw=np.empty((32,len(mu),4))
                for high in (0,1):
                    cells=(sm[k]>=.8)==high
                    if not cells.any():continue
                    pool=np.flatnonzero(bins==2*i+high)
                    if len(pool)<32:pool=np.flatnonzero(bins//2==i)
                    if len(pool)<32:pool=np.arange(len(res))
                    draw[:,cells]=mu[cells]+res[rng.choice(pool,(32,int(cells.sum())))]
                draw=decode(draw)
                assert np.isfinite(draw).all()
                # Only after all predictions at these cells are made, read full-box truth for scoring.
                y=truth(ph,row)[k]; z=EDGES[i:i+2].tolist()
                for name,w in [('cells',np.ones(len(y))),('galaxies',n[k])]:
                    if w.sum()==0:continue
                    old=cmp[(pair,tuple(z),name)];assert old['cells']==len(w)
                    np.testing.assert_allclose(old['weight'],w.sum(),rtol=2e-7)
                    scored=metrics(draw,y,w,rng)
                    output.append(dict(phase=ph,cap=cap,pair=pair,z=z,stratum=row['support_stratum'],
                        weighting=name,**scored))
    assert len(output)==len(cmp)
    c.atomic_json(out/'REFERENCE_COMPLETE.json',dict(rows=output,fit=c.sha256(out/'FIT.json'),
        archive=c.sha256(ref),scope='Single-cell regression benchmark; not a coherent baseline density-to-tides ensemble. Fraction ranks across cells not interpreted.',passed=True))


if __name__=='__main__':run()
