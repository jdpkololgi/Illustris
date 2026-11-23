"""Historical empirical 5/95 intervals, same 32 reference draws, no refitting."""
import numpy as np
import lowz_count_reference as r


def run():
    r.c.require_compute()
    base=r.c.ROOT.parent/'lowz_count_reference_61848915'
    fit=r.read(base/'FIT.json')
    assert fit['source']==r.c.sha256(r.__file__) and fit['library']['sha256']==r.c.sha256(base/'TRAIN_ONLY.npz')
    with np.load(base/'TRAIN_ONLY.npz') as f:cf,res,bins=[f[k] for k in ('coef','residual','bin')]
    ref=r.read(base/'REFERENCE_COMPLETE.json')
    bench={(s['pair'],s['z'][0],s['weighting']):s for s in ref['rows']}
    saved=r.read(r.c.ROOT.parent/'lowz_finish_dev_61811977/FINISH_COMPLETE.json')
    assert saved['source_sha256']==ref['archive']
    cfms={(a['pair'],s['z'][0]):s for a in saved['records'] for s in a['shells']}
    rng=np.random.default_rng(61703); rows=[]
    for ph in r.c.DEVELOPMENT:
        for row,(x,b,s,sm,n) in r.groups(ph):
            pair=row['pair_id'];yfull=r.truth(ph,row)
            for i in range(5):
                m=(b==i)&(s>0)
                if not m.any():continue
                mu=x[m]@cf; draws=np.empty((32,len(mu),4))
                for high in (0,1):
                    k=(sm[m]>=.8)==high
                    if not k.any():continue
                    pool=np.flatnonzero(bins==2*i+high)
                    if len(pool)<32:pool=np.flatnonzero(bins//2==i)
                    if len(pool)<32:pool=np.arange(len(res))
                    draws[:,k]=mu[k]+res[rng.choice(pool,(32,int(k.sum())))]
                draws=r.decode(draws);y=yfull[m];lo,hi=np.quantile(draws,[.05,.95],axis=0)
                old=cfms[pair,r.EDGES[i]]
                assert old['n']==len(y)
                def matched(key):
                    return np.average([s[key] for s in old['by_count']],axis=0,weights=[s['cells'] for s in old['by_count']]).tolist()
                for name,w in [('cells',np.ones(len(y))),('galaxies',n[m])]:
                    if w.sum():
                        scores=r.metrics(draws,y,w,rng)
                        np.testing.assert_allclose(scores['fair_crps'],bench[pair,r.EDGES[i],name]['fair_crps'],rtol=1e-10)
                rows.append(dict(phase=ph,cap=row['cap'],pair=pair,stratum=row['support_stratum'],z=old['z'],n=len(y),
                    cfm_coverage=matched('percentile90_coverage'),cfm_width=matched('percentile90_width'),
                    count_coverage=((lo<=y)&(y<=hi)).mean(0).tolist(),count_width=(hi-lo).mean(0).tolist()))
    assert len(rows)==len(cfms)
    r.c.atomic_json(base/'MATCHED_PERCENTILE90.json',dict(rows=rows,passed=True,script=r.c.sha256(__file__),
        scope='linear-interpolated empirical 5th–95th interval from 32 draws. Finite-sample coverage under exchangeability need not be 90%.'))


if __name__=='__main__':run()
