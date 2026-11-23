"""Registered supported-core readout for the November coherent cap ensemble."""
import json
import os
import subprocess
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist
import lowz_ph006_cap_draw as pilot
from workflows.sbi.e2e_cfm_calibration_figures import tarp_curves

c = pilot.c
root = c.ROOT.parent
phases = ('ph006','ph012','ph013')
read = lambda p: json.loads(p.read_text())


def panel():
    # Read only the explicitly submitted jobs. Raw Slurm ID names the Scratch output.
    jobmap = {'61646793':'NGC','61646799':'SGC','61646846':'NGC','61646847':'SGC','61646642':'SGC'}
    jobs = subprocess.check_output(['sacct','-X','-nP','-j',','.join(jobmap),
                                    '-o','JobID,JobIDRaw,State'],text=True)
    result = {('ph006',cap,0): root/f'lowz_cap_draw_{job}'/cap
              for cap,job in [('NGC',61610899),('SGC',61582948)]}
    for line in jobs.splitlines():
        task, raw, state = line.strip().split('|')[:3]
        parent, ix = task.split('_'); ix = int(ix)
        assert state == 'COMPLETED' and raw.isdigit(), line
        cap = jobmap[parent]
        if parent=='61646642' and ix==0: continue
        key = phases[ix//32],cap,ix%32
        assert key not in result
        result[key] = root/f'lowz_cap_draw_{raw}'/cap
    assert len(result)==192 and len(set(result.values()))==192
    for (ph,cap,ix),d in result.items():
        for stage,key in [('density','field'),('tides','output')]:
            r=read(d/stage/'COMPLETE.json')
            assert (r['passed'],r['phase'],r['cap'],r['first'],r['draws'])==(True,ph,cap,ix,1)
            assert c.sha256(r[key]['path'])==r[key]['sha256']
        assert (d/'SCORES.json').is_file()
    return result


def metrics(x,y,w,rng):
    # x: draw, supported location, physical field. No independence assumed across cells.
    m=len(x); avg=lambda v: np.average(v,axis=0,weights=w/w.sum())
    xs=np.sort(x,axis=0)
    spread=np.einsum('i,ijk->jk',2*np.arange(m)-m+1,xs)/(m*(m-1))
    ranks=(x<y).sum(0)+np.floor(rng.random(y.shape)*((x==y).sum(0)+1)).astype(int)
    out=dict(weight=float(w.sum()),cells=len(w),bias=avg(x.mean(0)-y).tolist(),
             fair_crps=avg(abs(x-y).mean(0)-spread).tolist(),
             rank_counts=[np.bincount(v,weights=w,minlength=m+1).tolist() for v in ranks.T])
    out['intervals']={}
    for k in (1,3,8):
        lo,hi=xs[k-1],xs[-k]
        out['intervals'][str(k)]=dict(nominal=1-2*k/(m+1),coverage=avg((y>=lo)&(y<=hi)).tolist(),
                                      width=avg(hi-lo).tolist())
    cls=(x[:,:,1:]>.2).sum(-1); truth=(y[:,1:]>.2).sum(-1)
    out['classes']=[]
    for k in range(4):
        p=(cls==k).mean(0); t=(truth==k)
        bins=np.minimum((p*10).astype(int),9)
        frac=np.average(cls==k,axis=1,weights=w); tf=np.average(t,weights=w)
        out['classes'].append(dict(fair_brier=float(avg((p-t)**2-p*(1-p)/(m-1))),
            fraction_rank=float(((frac<tf).sum()+rng.random()*((frac==tf).sum()+1))/(m+1)),
            reliability={name:np.bincount(bins,weights=val,minlength=10).tolist()
                         for name,val in [('weight',w),('sum_probability',w*p),('sum_truth',w*t)]}))
    return out


def run(region_callback=None, prefix='lowz_calibration_'):
    c.require_compute(); paths=panel(); out=root/(prefix+os.environ['SLURM_JOB_ID'])
    out.mkdir(exist_ok=False); rng=np.random.default_rng(1751); rows=[]; joint=[]
    for ph in phases:
        pilot.phase=ph
        for cap in ('NGC','SGC'):
            dirs=[paths[ph,cap,i] for i in range(32)]
            rawpath=pilot.checked(read(pilot.coord.ROOT/'observations'/ph/(cap+'_COMPLETE.json')))
            fig,axes=plt.subplots(8,4,figsize=(10,15),layout='constrained')
            for ax,row in zip(axes,pilot.rows(cap),strict=True):
                pair=row['pair_id']; x=[]
                for i,d in enumerate(dirs):
                    with h5py.File(d/'density/FIELD.h5') as f,h5py.File(d/'tides/COMPLETED_FIELD.h5') as t:
                        org=f.attrs['origin_mpc_h']; sl=pilot.cuts(row,org); sup=f['support'][sl]>0
                        if i==0: origin=org; support=sup
                        np.testing.assert_array_equal(org,origin);np.testing.assert_array_equal(sup,support)
                        x.append(np.concatenate((f['rho'][(0,*sl)][...,None],t['eigenvalues'][(0,*sl)]),axis=-1))
                x=np.asarray(x); shape=x.shape[1:-1]
                tr=pilot.read_target(pilot.checked(read(pilot.coord.ROOT/'targets'/ph/(pair+'.json'))),ph,pair)
                y=np.concatenate((pilot.owned_core_arrays(tr['rho_joint']).reshape(*shape,1),
                                  pilot.tidal.eigensystem(tr['fullbox_tensor_cores']).reshape(*shape,3)),axis=-1)
                pos=np.meshgrid(*[(np.arange(s.start,s.stop)+.5)*6.766+o for s,o in zip(sl,origin)],indexing='ij')
                z=pilot.coord.redshift(np.sqrt(sum(v*v for v in pos)))
                mask=support&(z>=.01)&(z<.55)
                with h5py.File(rawpath) as f:
                    counts=pilot.products.padded_extract(f['counts'],np.asarray(row['center'])-16,np.array(shape)*2)
                    weights=counts.reshape(32,2,16,2,16,2).sum((1,3,5),dtype='f8')
                assert np.isfinite(x[:,mask]).all() and np.isfinite(y[mask]).all()
                label=dict(phase=ph,cap=cap,pair=pair,stratum=row['support_stratum'])
                if region_callback is not None:
                    region_callback(out, label, x, y, weights, mask, z)
                for lo,hi in zip([.01,.15,.25,.35,.45],[.15,.25,.35,.45,.55]):
                    v=mask&(z>=lo)&(z<hi)
                    if not v.any():continue
                    for name,w in [('cells',np.ones(v.sum())),('galaxies',weights[v])]:
                        if w.sum():rows.append(dict(**label,z=[lo,hi],weighting=name,**metrics(x[:,v],y[v],w,rng)))
                # Two neighboring core means in physical overdensity units, supported locations only.
                for name,parts in [('two_cores',[(slice(a,a+16),slice(None),slice(None)) for a in (0,16)]),
                                   ('eight_subregions',[(slice(a,a+16),slice(b,b+8),slice(k,k+8))
                                     for a in (0,16) for b in (0,8) for k in (0,8)])]:
                    if not all(mask[s].any() for s in parts):continue
                    xx=np.stack([x[(slice(None),*s,0)][:,mask[s]].mean(1)-1 for s in parts],axis=1)
                    yy=np.array([y[(*s,0)][mask[s]].mean()-1 for s in parts])
                    curves={}
                    for width in (.25,1.):
                        alpha,curve=tarp_curves(xx[None],yy[None],seed=int(rng.integers(2**31)),width=width)
                        curves[str(width)]=curve[0].tolist()
                    joint.append(dict(**label,region=name,alpha=alpha.tolist(),curves=curves,
                        fair_energy=float(np.linalg.norm(xx-yy,axis=1).mean()-.5*pdist(xx).mean())))
                slz=8; cl=(x[:,:,:,slz,1:]>.2).sum(-1); true=(y[:,:,slz,1:]>.2).sum(-1)
                imgs=[y[:,:,slz,0],x[:,:,:,slz,0].mean(0),true,(cl==true).mean(0)]
                for a,img,cm,mx in zip(ax,imgs,['viridis','viridis','plasma','magma'],[2,2,3,1]):
                    a.imshow(np.ma.array(img.T,mask=~mask[:,:,slz].T),origin='lower',cmap=cm,vmin=0,vmax=mx)
                    a.set_xticks([]);a.set_yticks([])
                ax[0].set_ylabel(pair.split(cap+'_')[-1],fontsize=7)
            for a,title in zip(axes[0],['True density','Posterior mean','True T-web','P(true class)']):a.set_title(title)
            fig.savefig(out/f'{ph}_{cap}.png');plt.close(fig)
    c.atomic_json(out/'CALIBRATION.json',dict(fields=['rho','lambda1','lambda2','lambda3'],draws=32,
        paths={str(k):str(v) for k,v in paths.items()},rows=rows,joint=joint,independent_units=list(phases)))
    return out


if __name__=='__main__':run()
