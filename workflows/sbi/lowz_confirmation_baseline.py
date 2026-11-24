"""Freeze matched pointwise reference predictions; no truth or fitting stage."""
import os, sys, socket
from pathlib import Path
import lowz_confirmation_inputs as b


def run():
    assert __debug__ and socket.gethostname().startswith('nid') and os.environ['SLURM_JOB_ID']
    assert Path.cwd()==b.DEPLOY and os.environ['GRAPHWEB_CFM_DATA_CONFIG']==b.CONFIG
    _,a=b.release('ph018')
    index=b.REPO/'docs/evidence/lowz_blind_cfm_prediction_index.json'
    assert b.sha(index)=='e9c78744e3c45b6679dbe42288f95f37d30f1821efc9ace6ddd9ab4f40f52397'
    geom=b.OUT/'GEOMETRY_FROZEN.json'
    assert b.sha(geom)=='eb20683dbbf052f19ec651b8711c403eab3f75a967d235faf12011f316b2e737'
    geometry=b.read(geom); assert geometry['release']==b.sha(b.RELEASE)
    for rel,digest in a['source_files'].items(): assert b.sha(b.DEPLOY/rel)==digest
    ref=a['methods']['lowz_count_reference.py']
    lib=a['baseline']; fit=a['baseline_recipe']
    parity=next(x for x in a['development_results'] if x['path'].endswith('/MATCHED_PERCENTILE90.json'))
    for x in (ref,lib,fit,parity,*a['sampler_helpers'].values()): assert b.sha(x['path'])==x['sha256']
    sys.path[:0]=[str(b.DEPLOY),str(Path(a['sampler_helpers']['lowz_ph006_cap_draw.py']['path']).parent)]
    import lowz_count_reference as r
    import numpy as np, h5py
    assert b.sha(r.__file__)==ref['sha256'] and b.read(fit['path'])['phases']==list(r.c.TRAIN)
    with np.load(lib['path'],allow_pickle=False) as f: cf,res,bins=(f[k] for k in ('coef','residual','bin'))

    def draws(x,sm,k,i,rng):
        mu=x[k]@cf; v=np.empty((32,len(mu),4))
        for high in (0,1):
            cells=(sm[k]>=.8)==high
            if not cells.any(): continue
            pool=np.flatnonzero(bins==2*i+high)
            if len(pool)<32: pool=np.flatnonzero(bins//2==i)
            if len(pool)<32: pool=np.arange(len(res))
            v[:,cells]=mu[cells]+res[rng.choice(pool,(32,int(cells.sum())))]
        v=r.decode(v)
        assert np.isfinite(v).all() and np.all(v[...,0]>0) and np.all(np.diff(v[...,1:],axis=-1)>0)
        return v

    dev=iter(r.groups('ph006'))
    row,(x,z,s,sm,_)=next(dev); dev.close()
    old=b.read(parity['path'])['rows'][0]; k=(z==0)&(s>0)
    assert old['pair']==row['pair_id'] and old['n']==int(k.sum()) and old['z']==[.01,.15]
    v=draws(x,sm,k,0,np.random.default_rng(61703))
    lo,hi=np.quantile(v,[.05,.95],axis=0)
    np.testing.assert_allclose((hi-lo).mean(0),old['count_width'],rtol=1e-10,atol=0)
    print('DEVELOPMENT_PARITY_PASS',row['pair_id'],flush=True)
    indexrows={(ph,cap,i):(inp,d,t) for ph,cap,i,inp,d,t in b.read(index)['rows']}
    out=b.OUT/'matched_reference_v1'
    out.mkdir(exist_ok=False)
    products=[]
    rng=np.random.default_rng(61703)
    for phase in ('ph018','ph019'):
        b.release(phase)
        panel=geometry['phases'][phase]
        for cap in ('NGC','SGC'):
            base=b.OUT/phase/cap
            receipt=base/'RESPONSE_COMPLETE.json'
            assert b.sha(receipt)==panel['response_receipts'][cap]
            native=b.verify_record(b.read(receipt)['outputs'][0],base)
            assert native==base/'NATIVE.h5'
            density=base/'predictions/draw_000/density'
            assert b.sha(density/'COMPLETE.json')==indexrows[phase,cap,0][1]
            field=b.verify_record(b.read(density/'COMPLETE.json')['field'],density)
            path=out/f'{phase}_{cap}.h5'
            rows=[row for row in panel['pairs'] if row['cap']==cap]
            assert len(rows)==8
            cells=0
            with h5py.File(native,'r') as raw,h5py.File(field,'r') as f,h5py.File(path,'x') as dest:
                dest.attrs.update(phase=phase,cap=cap,channels='rho,lambda1,lambda2,lambda3',flat_index_order='C:32,16,16')
                for row in rows:
                    x,z,s,sm,n=r.observations(raw,row)
                    support=f['support'][r.p.cuts(row,f.attrs['origin_mpc_h'])]>0
                    np.testing.assert_array_equal(support,s>0)
                    for i in range(5):
                        k=(z==i)&support
                        if not k.any(): continue
                        g=dest.create_group(row['pair_id']+'/'+str(i))
                        g['flat_cell']=np.flatnonzero(k); g['galaxy_weight']=n[k]
                        g.create_dataset('draws',data=draws(x,sm,k,i,rng),compression='lzf')
                        cells+=int(k.sum())
            assert cells>0
            products.append(dict(phase=phase,cap=cap,cells=cells,**r.c.file_record(path,content_hash=True)))
            print('BASELINE_CAP',phase,cap,cells,flush=True)
    r.c.atomic_json(out/'BASELINE_FROZEN.json',dict(passed=True,outputs=products,draws=32,
        cfm_index=b.sha(index),geometry=b.sha(geom),release=b.sha(b.RELEASE),baseline=lib['sha256'],
        fit=fit['sha256'],runner=b.sha(__file__),seed=61703,development_parity=old['pair'],
        truth_read=False,refit=False,job=os.environ['SLURM_JOB_ID']))


if __name__=='__main__': run()
