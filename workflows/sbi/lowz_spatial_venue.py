"""Development-only spatial readout. Donors are whole residual regions from training."""
from pathlib import Path
import json, os
import numpy as np
import h5py
from scipy.spatial.distance import cdist,pdist
from scipy.ndimage import minimum_filter
import lowz_count_reference as b

p=b.p
root=p.c.ROOT.parent
out=root/('spatial_venue_'+os.environ['SLURM_JOB_ID'])
out.mkdir(exist_ok=False)
rng=np.random.default_rng(711)
cf=np.load(root/'lowz_count_reference_61848915/TRAIN_ONLY.npz')['coef']
archive=b.read(root/'lowz_calibration_61647276/CALIBRATION.json')
bank={}
for ph in p.c.TRAIN:
    for row,(x,*_) in b.groups(ph):
        y=b.truth(ph,row)
        key=row['pair_id'].split('_',1)[1]
        bank.setdefault(key,[]).append((b.encode(y.reshape(-1,4)).reshape(y.shape)-x@cf).astype('f4'))
    print('training donor',ph,flush=True)

def stat(f,m,d):
    # Connected two-point indicator covariance along the three grid axes.
    num=np.zeros(len(f)); left=num.copy(); right=num.copy(); n=0
    for axis in range(3):
        s=[slice(None)]*3;t=s.copy();s[axis]=slice(None,-d);t[axis]=slice(d,None)
        s=tuple(s);t=tuple(t);v=m[s]&m[t];a=f[(slice(None),*s)][:,v];h=f[(slice(None),*t)][:,v]
        num+=(a&h).sum(1);left+=a.sum(1);right+=h.sum(1);n+=v.sum()
    return num/n-left*right/n**2

records=[]
for ph in p.c.DEVELOPMENT:
    for row,(x,z,s,sm,n) in b.groups(ph):
        cap=row['cap'];m=(z>=0)&(z<5)&(s>0); pair=row['pair_id']
        eligible=minimum_filter(m.astype('u1'),size=5,mode='constant')>0
        center=np.array(np.unravel_index(np.argmax(np.where(eligible,x[...,1],-np.inf)),m.shape))
        hood=tuple(slice(v-2,v+3) for v in center)
        rho=[];fil=[]
        for i in range(32):
            path=Path(archive['paths'][str((ph,cap,i))])
            with h5py.File(path/'density/FIELD.h5') as f:
                sl=p.cuts(row,f.attrs['origin_mpc_h']);rho.append(f['rho'][(0,*sl)])
            with h5py.File(path/'tides/COMPLETED_FIELD.h5') as f:
                fil.append(((f['eigenvalues'][(0,*sl)]>.2).sum(-1)==2))
        f=np.stack(fil); shuffled=np.take_along_axis(f,np.argsort(rng.random(f.shape),axis=0),axis=0)
        pool=bank[pair.split('_',1)[1]];assert len(pool)==20
        baseline=[]
        for j in rng.integers(20,size=32):
            _,mid,u,v=(x@cf+pool[j]).T
            count=(mid-np.exp(u)>.2).astype(int)+(mid>.2)+(mid+np.exp(v)>.2)
            baseline.append((count==2).T)
        y=b.truth(ph,row);true=(y[...,1:]>.2).sum(-1)==2
        rec=dict(phase=ph,pair=pair,stats={})
        for d in (1,2,4):
            rec['stats'][d]=[stat(a,m,d).tolist() for a in (true[None],f,shuffled,np.stack(baseline))]
        if eligible.any():
            a=np.stack([y[...,0],*rho])[(slice(None),*hood)].reshape(33,-1)
            pos=np.stack(np.unravel_index(a.argmax(1),(5,5,5)),axis=1)*6.766
            rec['peak']=[float(cdist(pos[:1],pos[1:]).mean()),float(pdist(pos[1:]).mean()),float(a[0].max()),float(np.median(a[1:].max(1)))]
        records.append(rec)
    (out/(ph+'.json')).write_text(json.dumps([r for r in records if r['phase']==ph],allow_nan=False))
    print(ph,'done',flush=True)
