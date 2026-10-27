"""Controlled TNG density x smoothing experiment; stages are independently saved."""
import argparse
import gc
import hashlib
import json
import struct
import time
import zipfile
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
CFG=ROOT/'configs/tng_density_control_20261015.json'
CAT=Path('/pscratch/sd/d/dkololgi/TNG300-1/groups_099')
DENS=Path('/pscratch/sd/d/dkololgi/IllustrisTNG300_densities/snap_099_total_dens_0.npz')
OLD_EIG=Path('/pscratch/sd/d/dkololgi/Cosmic_env_TNG_cache/snap_099_eig_vals_0.npz')
GRAPH=Path('/pscratch/sd/d/dkololgi/tng_illustris/cache/from_repo_root/processed_gcn_data_alpha_mc1e+09.pt')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def save_json(path,value):
    tmp=Path(str(path)+'.partial')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def mmap_member(path,name):
    with zipfile.ZipFile(path) as z:
        info=z.getinfo(name+'.npy');assert info.compress_type==zipfile.ZIP_STORED
    with path.open('rb') as f:
        f.seek(info.header_offset);header=f.read(30)
        fn,extra=struct.unpack_from('<HH',header,26);f.seek(fn+extra,1)
        version=np.lib.format.read_magic(f)
        reader=np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0
        shape,order,dtype=reader(f);assert not order
        return np.memmap(path,mode='r',offset=f.tell(),dtype=dtype,shape=shape)

def catalogue(c,out):
    import h5py
    import torch
    torch.set_num_threads(1)
    columns={k:[] for k in ['subhalo_id','pos_hmpc','mass_msun','sfr','g_r','flag']}
    sources=[];offset=0
    for part in range(600):
        p=CAT/f'fof_subhalo_tab_099.{part}.hdf5'
        with h5py.File(p,'r') as f:
            a=f['Header'].attrs
            assert int(a['NumFiles'])==600 and np.isclose(a['HubbleParam'],c['h'])
            n=int(a['Nsubgroups_ThisFile'])
            if not n:continue
            group=f['Subhalo'];rawmass=group['SubhaloMassType'][:,4]
            # Same float32 selection expression as the archived GAT loader.
            keep=rawmass>=c['stellar_mass_min_msun']*c['h']/1e10
            ids=np.flatnonzero(keep)
            columns['subhalo_id'].append(offset+ids)
            columns['pos_hmpc'].append(group['SubhaloPos'][...][keep].astype(np.float64)/1000.)
            columns['mass_msun'].append(rawmass[keep].astype(np.float64)*1e10/c['h'])
            columns['sfr'].append(group['SubhaloSFR'][...][keep].astype(np.float64))
            phot=group['SubhaloStellarPhotometrics'][...][keep]
            columns['g_r'].append((phot[:,4]-phot[:,5]).astype(np.float64))
            columns['flag'].append(group['SubhaloFlag'][...][keep].astype(bool))
            # Hash only the extracted relevant arrays, not unrelated large fields.
            hh=hashlib.sha256()
            for k in columns:hh.update(np.asarray(columns[k][-1]).tobytes())
            sources.append(dict(path=str(p),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns,selected_rows=len(ids),selected_fields_sha256=hh.hexdigest()))
            offset+=n
        if part%100==0:print('catalogue chunk',part,flush=True)
    d={k:np.concatenate(v) for k,v in columns.items()}
    graph,_,_=torch.load(GRAPH,map_location='cpu',weights_only=False)
    gp=graph.pos.numpy().astype(np.float64)*c['h']
    assert gp.shape==d['pos_hmpc'].shape
    delta=np.abs(gp-d['pos_hmpc']);maxdelta=float(delta.max())
    assert maxdelta<5e-5,maxdelta
    del graph,gp
    d['valid_query']=d['flag']&np.isfinite(d['mass_msun'])&np.isfinite(d['sfr'])&(d['sfr']>=0)&np.isfinite(d['g_r'])
    assert np.unique(d['subhalo_id']).size==len(d['subhalo_id'])
    np.savez_compressed(out/'catalogue.npz',**d)
    save_json(out/'catalogue_receipt.json',dict(rows=len(d['subhalo_id']),source_subhaloes=offset,valid_queries=int(d['valid_query'].sum()),excluded_queries=int((~d['valid_query']).sum()),zero_sfr=int((d['sfr']==0).sum()),position_parity_max_hmpc=maxdelta,graph=str(GRAPH),graph_sha256=sha(GRAPH),sources=sources))
    print('catalogue complete',len(d['subhalo_id']),maxdelta,flush=True)

def sampled_tensor(phi,pos,box,interpolate=True):
    """Hessian of phi using two centred first derivatives, as in Cactus."""
    n=phi.shape[0];dx=box/n
    coords=pos/dx-.5 if interpolate else pos/dx
    base=np.floor(coords).astype(np.int64)
    frac=coords-base
    tensor=np.zeros((len(pos),3,3),dtype=np.float64)
    corners=[(i,j,k) for i in [0,1] for j in [0,1] for k in [0,1]] if interpolate else [(0,0,0)]
    for corner in corners:
        idx=base+np.array(corner)
        weight=np.prod(np.where(np.array(corner),frac,1-frac),axis=1) if interpolate else 1.
        def at(shift):
            v=(idx+np.asarray(shift))%n
            return np.asarray(phi[v[:,0],v[:,1],v[:,2]],dtype=np.float64)
        centre=at([0,0,0])
        for i in range(3):
            plus=np.zeros(3,dtype=int);plus[i]=2
            tensor[:,i,i]+=weight*(at(plus)-2*centre+at(-plus))/(4*dx*dx)
            for j in range(i+1,3):
                pp=np.zeros(3,dtype=int);pm=pp.copy();pp[i]=1;pp[j]=1;pm[i]=1;pm[j]=-1
                v=weight*(at(pp)-at(pm)-at(-pm)+at(-pp))/(4*dx*dx)
                tensor[:,i,j]+=v;tensor[:,j,i]+=v
    return tensor

def potential(delta_k,box,radius):
    from scipy.fft import irfftn,fftfreq,rfftfreq
    n=delta_k.shape[0]
    k=2*np.pi*fftfreq(n,d=box/n);kr=2*np.pi*rfftfreq(n,d=box/n)
    k2=(k[:,None,None]**2+k[None,:,None]**2+kr[None,None,:]**2)
    k2[0,0,0]=1.
    kernel=-np.exp(-.5*radius*radius*k2)/k2;kernel[0,0,0]=0.
    work=delta_k*kernel
    del kernel,k2
    phi=irfftn(work,s=(n,n,n),workers=1,overwrite_x=True)
    return phi

def operator_checks():
    from scipy.fft import rfftn
    n=32;box=32.;dx=box/n;r=2.;m=np.array([1,2,1]);k=2*np.pi*m/box
    axes=(np.arange(n)+.5)*dx
    phase=k[0]*axes[:,None,None]+k[1]*axes[None,:,None]+k[2]*axes[None,None,:]
    delta=.2*np.cos(phase)
    phi=potential(rfftn(delta,workers=1),box,r)
    pix=np.array([[3,4,5],[0,31,17],[31,0,0],[14,8,2]])
    pos=(pix+.5)*dx
    actual=sampled_tensor(phi,pos,box)
    v=np.sin(k*dx)/dx
    amp=.2*np.exp(-.5*r*r*np.dot(k,k))*np.cos(np.sum(k*pos,axis=1))/np.dot(k,k)
    expected=amp[:,None,None]*v[None,:,None]*v[None,None,:]
    error=float(np.max(np.abs(actual-expected)));assert error<1e-12,error
    # Constant potential is zero tide, including across the periodic boundary.
    assert np.max(abs(sampled_tensor(np.ones((8,8,8)),np.array([[0.,0.,0.],[7.99,0.,4.]]),8.)))==0
    return dict(plane_wave_hessian_max_error=error,periodic_constant_test=True)

def truth(c,out):
    from scipy.fft import rfftn
    from scipy.stats import spearmanr
    checks=operator_checks()
    d=np.load(out/'catalogue.npz');pos=d['pos_hmpc'];box=c['box_hmpc']
    native=mmap_member(DENS,'dens')
    assert native.shape==(512,512,512)
    results={};receipt={'operator_checks':checks,'grids':[]}
    for n in [c['truth_ngrid'],c['check_ngrid']]:
        print('truth grid',n,flush=True)
        if n==512:delta=np.array(native,dtype=np.float64)
        else:
            factor=512//n;delta=np.empty((n,n,n),dtype=np.float64)
            for i in range(n):delta[i]=np.asarray(native[i*factor:(i+1)*factor]).reshape(factor,n,factor,n,factor).mean((0,2,4))
        density_mean=float(delta.mean());delta/=density_mean;delta-=1
        dk=rfftn(delta,workers=1,overwrite_x=True);del delta;gc.collect()
        for r in c['truth_radii_hmpc']:
            phi=potential(dk,box,r)
            eig=np.linalg.eigvalsh(sampled_tensor(phi,pos,box,interpolate=True))
            results[f'eig_R{int(r)}_n{n}']=eig.astype(np.float32)
            info=dict(ngrid=n,radius=r,density_mean=density_mean,eigenvalue_std=eig.std(0).tolist(),class_counts=np.bincount((eig>=c['class_threshold']).sum(1),minlength=4).tolist())
            if n==512 and r==2:
                q=np.random.default_rng(20261015).choice(len(pos),8192,replace=False)
                pix=(pos[q]/box*512).astype(int)
                old=mmap_member(OLD_EIG,'eig_vals')
                saved=np.array(old[:,pix[:,0],pix[:,1],pix[:,2]]).T
                replay=np.linalg.eigvalsh(sampled_tensor(phi,pos[q],box,interpolate=False))
                info['native_nearest_replay_max_abs_error']=float(np.max(abs(replay-saved)))
                info['native_nearest_class_agreement']=float(np.mean((replay>=.2).sum(1)==(saved>=.2).sum(1)))
                assert info['native_nearest_replay_max_abs_error']<1e-5,info
            receipt['grids'].append(info)
            del phi,eig;gc.collect();print('truth complete',n,r,info,flush=True)
        del dk;gc.collect()
    receipt['coarse_comparison']={}
    for r in c['truth_radii_hmpc']:
        a=results[f'eig_R{int(r)}_n512'];b=results[f'eig_R{int(r)}_n256']
        receipt['coarse_comparison'][str(r)]=dict(mean_abs_difference_over_fine_std=(np.mean(abs(a-b),axis=0)/a.std(0)).tolist(),class_agreement=float(np.mean((a>=.2).sum(1)==(b>=.2).sum(1))),trace_spearman=float(spearmanr(a.sum(1),b.sum(1)).statistic))
    np.savez_compressed(out/'truth.npz',**results)
    receipt.update(density_path=str(DENS),density_file_bytes=DENS.stat().st_size,density_sha256=sha(DENS),old_eigenvalues_path=str(OLD_EIG),old_eigenvalues_sha256=sha(OLD_EIG),config_sha256=sha(CFG),script_sha256=sha(Path(__file__)))
    save_json(out/'truth_receipt.json',receipt)

def features(c,out):
    from scipy.spatial import cKDTree
    d=np.load(out/'catalogue.npz');pos=d['pos_hmpc'];box=c['box_hmpc'];n=len(pos)
    n_bgs=round(c['target_density_h3_mpc3']*box**3)
    result={};receipt=dict(full_rows=n,bgs_rows=n_bgs,full_density=n/box**3,bgs_density=n_bgs/box**3,thinning_fraction=n_bgs/n,checks=[])
    ranks=np.asarray(c['feature_neighbour_ranks'])
    subsets={'dense':np.arange(n)}
    subsets.update({f'thin_{seed}':np.sort(np.random.default_rng(seed).choice(n,n_bgs,replace=False)) for seed in c['seeds']})
    for name,ids in subsets.items():
        tree=cKDTree(pos[ids],boxsize=box);member=np.zeros(n,dtype=bool);member[ids]=True
        counts=np.stack([tree.query_ball_point(pos,r,return_length=True,workers=1)-member for r in c['feature_apertures_hmpc']],axis=1)
        dist,_=tree.query(pos,k=int(ranks.max()+1),workers=1)
        nearest=dist[np.arange(n)[:,None],ranks[None,:]-1+member[:,None]]
        x=np.column_stack([np.log1p(counts),np.log(np.maximum(nearest,1e-6))]).astype(np.float32)
        assert np.isfinite(x).all() and (counts>=0).all()
        for q in [int(ids[0]),int(ids[-1]),int(np.flatnonzero(~member)[0]) if (~member).any() else n//2]:
            diff=np.abs(pos[ids]-pos[q]);dd=np.sqrt((np.minimum(diff,box-diff)**2).sum(1));dd[ids==q]=np.inf
            brute=np.array([(dd<=r).sum() for r in c['feature_apertures_hmpc']])
            assert np.array_equal(brute,counts[q])
            assert np.allclose(np.sort(dd)[ranks-1],nearest[q],atol=1e-10,rtol=0)
        result[name]=x;result[name+'_tracer_ids']=ids
        receipt['checks'].append(dict(condition=name,periodic_brute_queries=3,input_sha256=hashlib.sha256(x.tobytes()).hexdigest()))
        print('features complete',name,flush=True)
    for key,(lo,hi) in c['split_x_box_fraction'].items():
        result[key+'_ids']=np.flatnonzero(d['valid_query']&(pos[:,0]>=lo*box)&(pos[:,0]<hi*box))
    assert not np.intersect1d(result['train_ids'],result['test_ids']).size
    receipt['split_rows']={key:len(result[key+'_ids']) for key in c['split_x_box_fraction']}
    receipt['split_buffer_hmpc']=.1*box
    receipt['config_sha256']=sha(CFG)
    np.savez_compressed(out/'features.npz',**result);save_json(out/'features_receipt.json',receipt)

def eigen_to_z(e):
    gaps=np.maximum(np.diff(e,axis=1),1e-7)
    return np.column_stack([e[:,0],gaps+np.log(-np.expm1(-gaps))])

def z_to_eigen(z):
    return np.column_stack([z[:,0],z[:,0,None]+np.cumsum(np.logaddexp(0,z[:,1:]),axis=1)])

def train(c,out):
    import copy
    import torch
    from torch import nn
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    f=np.load(out/'features.npz');truths=np.load(out/'truth.npz')
    ti=f['train_ids'];vi=f['validation_ids']
    curves={};predictions={}
    for r in c['truth_radii_hmpc']:
        target=truths[f'eig_R{int(r)}_n512'].astype(np.float64)
        z=eigen_to_z(target);assert np.max(abs(z_to_eigen(z)-target))<1e-6
        ym=z[ti].mean(0);ys=z[ti].std(0)
        y=torch.from_numpy(((z-ym)/ys).astype(np.float32))
        for seed in c['seeds']:
            for condition in ['dense',f'thin_{seed}']:
                name=f'R{int(r)}_{condition}_seed{seed}'
                x=f[condition];xm=x[ti].mean(0);xs=np.maximum(x[ti].std(0),1e-6)
                xt=torch.from_numpy(((x-xm)/xs).astype(np.float32))
                torch.manual_seed(seed)
                model=nn.Sequential(nn.Linear(x.shape[1],64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,3))
                opt=torch.optim.Adam(model.parameters(),lr=c['learning_rate'])
                rng=np.random.default_rng(seed);best=float('inf');best_state=None;history=[];best_step=0
                for step in range(1,c['training_steps']+1):
                    ix=rng.choice(ti,c['batch_size'],replace=True)
                    model.train();opt.zero_grad(set_to_none=True)
                    loss=((model(xt[ix])-y[ix])**2).mean();loss.backward();opt.step()
                    if step%c['validation_every']==0:
                        model.eval()
                        with torch.no_grad():vl=float(((model(xt[vi])-y[vi])**2).mean())
                        history.append(dict(step=step,train_loss=float(loss.detach()),validation_loss=vl))
                        if vl<best:best=vl;best_step=step;best_state=copy.deepcopy(model.state_dict())
                model.load_state_dict(best_state);model.eval()
                with torch.no_grad():
                    zp=np.concatenate([model(batch).numpy() for batch in xt.split(16384)])*ys+ym
                ep=z_to_eigen(zp).astype(np.float32)
                assert np.isfinite(ep).all() and (np.diff(ep,axis=1)>=0).all()
                predictions[name]=ep
                if condition=='dense':
                    thin=torch.from_numpy(((f[f'thin_{seed}']-xm)/xs).astype(np.float32))
                    with torch.no_grad():frozen=np.concatenate([model(batch).numpy() for batch in thin.split(16384)])*ys+ym
                    predictions[f'R{int(r)}_frozen_thin_seed{seed}']=z_to_eigen(frozen).astype(np.float32)
                curves[name]=dict(best_step=best_step,best_validation_loss=best,history=history)
                torch.save(dict(state=best_state,x_mean=xm,x_std=xs,y_mean=ym,y_std=ys,config=c),out/(name+'.pt'))
                print('fit complete',name,'best step',best_step,'validation',best,flush=True)
                # Durable partial results permit recovery without repeating completed fits.
                np.savez_compressed(out/'predictions.npz',**predictions)
                save_json(out/'training_receipt.json',curves)

def tail_membership(values):
    """Fractional membership gives equal-sized quartiles, sharing boundary ties."""
    _,inv,counts=np.unique(values,return_inverse=True,return_counts=True)
    ends=np.cumsum(counts);starts=ends-counts;n=len(values)
    low=np.maximum(0,np.minimum(ends,n/4)-starts)/counts
    high=np.maximum(0,ends-np.maximum(starts,3*n/4))/counts
    return low[inv],high[inv]

def analyse(c,out):
    from scipy.stats import spearmanr
    d=np.load(out/'catalogue.npz');f=np.load(out/'features.npz');t=np.load(out/'truth.npz');p=np.load(out/'predictions.npz')
    ids=f['test_ids'];mass=np.log10(d['mass_msun'][ids]);pos=d['pos_hmpc'][ids]
    bins=np.floor((mass-c['mass_range_dex'][0])/c['mass_bin_width_dex']).astype(int)
    maxbin=round(np.diff(c['mass_range_dex'])[0]/c['mass_bin_width_dex'])
    eligible=[b for b in range(maxbin) if (bins==b).sum()>=c['min_mass_bin_rows_for_quartiles']]
    use=np.isin(bins,eligible);ids=ids[use];mass=mass[use];pos=pos[use];bins=bins[use]
    ssfr=d['sfr'][ids]/d['mass_msun'][ids]
    properties=np.column_stack([ssfr<1e-11,d['g_r'][ids],np.log10(np.maximum(ssfr,1e-14))]).astype(float)
    props=['quenched_fraction','g_r','log10_ssfr_capped']
    blockijk=np.minimum((pos/c['box_hmpc']*c['bootstrap_blocks_per_axis']).astype(int),c['bootstrap_blocks_per_axis']-1)
    blockraw=np.ravel_multi_index(blockijk.T,(c['bootstrap_blocks_per_axis'],)*3)
    blocklabels,block=np.unique(blockraw,return_inverse=True);nb=len(blocklabels)
    rng=np.random.default_rng(818265)
    bw=rng.multinomial(nb,np.ones(nb)/nb,size=c['bootstrap_draws'])
    def contrast(members,ys=properties):
        point=np.zeros(3);boot=np.zeros((len(bw),3));total=sum((bins==b).sum() for b in members)
        if not total:return None,None
        for b,(lo,hi) in members.items():
            ii=np.flatnonzero(bins==b);weight=len(ii)/total
            vals=[];bs=[]
            for member in [lo,hi]:
                counts=np.bincount(block[ii],weights=member,minlength=nb)
                sums=np.stack([np.bincount(block[ii],weights=member*ys[ii,k],minlength=nb) for k in range(3)],axis=1)
                vals.append(sums.sum(0)/counts.sum())
                denom=bw@counts
                bs.append(np.divide(bw@sums,denom[:,None],out=np.full((len(bw),3),np.nan),where=denom[:,None]>0))
            point+=weight*(vals[1]-vals[0]);boot+=weight*(bs[1]-bs[0])
        return point,boot
    def summarize(point,boot):
        return {pr:dict(estimate=float(point[j]),ci95=np.nanquantile(boot[:,j],[.025,.975]).tolist()) for j,pr in enumerate(props)}
    conditions={f'R{int(r)}_truth':t[f'eig_R{int(r)}_n512'][ids] for r in c['truth_radii_hmpc']}
    conditions.update({k:p[k][ids] for k in p.files})
    outputs={};bootstore={};points={};metrics={};classes={}
    for name,e in conditions.items():
        trace=e.sum(1);cl=(e>=c['class_threshold']).sum(1);classes[name]=cl
        memberships={b:tail_membership(trace[bins==b]) for b in eligible}
        point,boot=contrast(memberships);points[name]=point;bootstore[name]=boot;outputs[name]=summarize(point,boot)
        if 'truth' not in name:
            truth=conditions[name.split('_')[0]+'_truth'];truecl=(truth>=c['class_threshold']).sum(1)
            shear=lambda ev:np.sqrt(np.sum((ev-ev.mean(1)[:,None])**2,axis=1))
            metrics[name]=dict(eigenvalue_r2=(1-np.sum((e-truth)**2,axis=0)/np.sum((truth-truth.mean(0))**2,axis=0)).tolist(),trace_spearman=float(spearmanr(trace,truth.sum(1)).statistic),shear_spearman=float(spearmanr(shear(e),shear(truth)).statistic),class_accuracy=float(np.mean(cl==truecl)),class_balanced_accuracy=float(np.mean([np.mean(cl[truecl==k]==k) for k in range(4)])),class_counts=np.bincount(cl,minlength=4).tolist())
    aggregate={};aggregate_boot={};aggregate_point={};effects={}
    for r in c['truth_radii_hmpc']:
        prefix=f'R{int(r)}'
        for arm in ['dense','thin','frozen_thin']:
            names=[f'{prefix}_dense_seed{s}' if arm=='dense' else f'{prefix}_thin_{s}_seed{s}' if arm=='thin' else f'{prefix}_frozen_thin_seed{s}' for s in c['seeds']]
            point=np.mean([points[k] for k in names],axis=0);boot=np.mean([bootstore[k] for k in names],axis=0)
            name=f'{prefix}_{arm}';aggregate_point[name]=point;aggregate_boot[name]=boot
            aggregate[name]=summarize(point,boot)
            for j,pr in enumerate(props):aggregate[name][pr]['seed_range']=[float(min(points[k][j] for k in names)),float(max(points[k][j] for k in names))]
        dp=aggregate_point[prefix+'_dense'];tp=aggregate_point[prefix+'_thin'];db=aggregate_boot[prefix+'_dense'];tb=aggregate_boot[prefix+'_thin']
        effects[prefix]=dict(dense_minus_thin=summarize(dp-tp,db-tb),fraction_of_dense_contrast_lost=summarize(1-tp/dp,1-tb/db),fraction_lost_frozen=summarize(1-aggregate_point[prefix+'_frozen_thin']/dp,1-aggregate_boot[prefix+'_frozen_thin']/db),truth_minus_dense=summarize(points[prefix+'_truth']-dp,bootstore[prefix+'_truth']-db))
    effects['R2_truth_minus_R7_truth']=summarize(points['R2_truth']-points['R7_truth'],bootstore['R2_truth']-bootstore['R7_truth'])
    effects['fraction_truth_contrast_lost_R2_to_R7']=summarize(1-points['R7_truth']/points['R2_truth'],1-bootstore['R7_truth']/bootstore['R2_truth'])
    # A property permutation within the same mass bins checks the contrast pipeline.
    shuffled=properties.copy()
    for b in eligible:
        ii=np.flatnonzero(bins==b);shuffled[ii]=properties[rng.permutation(ii)]
    null={}
    for r in c['truth_radii_hmpc']:
        name=f'R{int(r)}_truth';trace=conditions[name].sum(1)
        point,boot=contrast({b:tail_membership(trace[bins==b]) for b in eligible},shuffled)
        null[name]=summarize(point,boot)
    # Repeated randomizations characterize the null distribution; a single
    # realized shuffle need not have a spatial-bootstrap interval covering zero.
    coefficients=[]
    for r in c['truth_radii_hmpc']:
        trace=conditions[f'R{int(r)}_truth'].sum(1);coef=np.zeros(len(ids))
        for b in eligible:
            ii=np.flatnonzero(bins==b);lo,hi=tail_membership(trace[ii])
            coef[ii]=len(ii)/len(ids)*(hi/hi.sum()-lo/lo.sum())
        coefficients.append(coef)
    permutation_rng=np.random.default_rng(199372);null_draws=[]
    for _ in range(200):
        permuted=properties.copy()
        for b in eligible:
            ii=np.flatnonzero(bins==b);permuted[ii]=properties[permutation_rng.permutation(ii)]
        null_draws.append(np.asarray(coefficients)@permuted)
    null_draws=np.asarray(null_draws)
    permutation_null={f'R{int(r)}':{pr:dict(mean=float(null_draws[:,i,j].mean()),permutation_interval95=np.quantile(null_draws[:,i,j],[.025,.975]).tolist()) for j,pr in enumerate(props)} for i,r in enumerate(c['truth_radii_hmpc'])}
    # Physical classes are a secondary contrast on common mass support per radius.
    class_contrasts={}
    for r in c['truth_radii_hmpc']:
        prefix=f'R{int(r)}';names=[k for k in conditions if k.startswith(prefix+'_') and 'frozen' not in k]
        supported=[b for b in eligible if all(np.sum((bins==b)&(classes[k]==cl))>=c['min_class_bin_rows'] for k in names for cl in [0,3])]
        entry=dict(common_mass_bins=supported,rows=int(np.isin(bins,supported).sum()),contrasts={})
        for name in names:
            if supported:
                point,boot=contrast({b:((classes[name][bins==b]==0).astype(float),(classes[name][bins==b]==3).astype(float)) for b in supported})
                entry['contrasts'][name]=summarize(point,boot)
        class_contrasts[prefix]=entry
    class_cross_bins=sorted(set.intersection(*(set(v['common_mass_bins']) for v in class_contrasts.values())))
    class_cross={};class_cross_points={};class_cross_boot={}
    for name in conditions:
        if 'frozen' in name:continue
        point,boot=contrast({b:((classes[name][bins==b]==0).astype(float),(classes[name][bins==b]==3).astype(float)) for b in class_cross_bins})
        if point is not None:
            class_cross[name]=summarize(point,boot);class_cross_points[name]=point;class_cross_boot[name]=boot
    class_cross_effects={}
    if class_cross_bins:
        class_cross_effects['truth_R2_minus_R7']=summarize(class_cross_points['R2_truth']-class_cross_points['R7_truth'],class_cross_boot['R2_truth']-class_cross_boot['R7_truth'])
        for r in c['truth_radii_hmpc']:
            dn=[f'R{int(r)}_dense_seed{s}' for s in c['seeds']];tn=[f'R{int(r)}_thin_{s}_seed{s}' for s in c['seeds']]
            dp=np.mean([class_cross_points[k] for k in dn],axis=0);tp=np.mean([class_cross_points[k] for k in tn],axis=0)
            db=np.mean([class_cross_boot[k] for k in dn],axis=0);tb=np.mean([class_cross_boot[k] for k in tn],axis=0)
            class_cross_effects[f'R{int(r)}_dense']=summarize(dp,db);class_cross_effects[f'R{int(r)}_thin']=summarize(tp,tb)
            class_cross_effects[f'R{int(r)}_fraction_lost']=summarize(1-tp/dp,1-tb/db)
    # Oracle smoothing contrasts on the coarser mesh expose numerical sensitivity.
    coarse={}
    for r in c['truth_radii_hmpc']:
        trace=t[f'eig_R{int(r)}_n256'][ids].sum(1)
        point,boot=contrast({b:tail_membership(trace[bins==b]) for b in eligible})
        coarse[f'R{int(r)}']=summarize(point,boot)
    aperture={}
    for r in c['truth_radii_hmpc']:
        col=c['feature_apertures_hmpc'].index(r);entry={};tp=[];tb=[]
        for name in ['dense']+[f'thin_{s}' for s in c['seeds']]:
            v=f[name][ids,col];point,boot=contrast({b:tail_membership(v[bins==b]) for b in eligible})
            entry[name]=summarize(point,boot)
            if name=='dense':dp,db=point,boot
            else:tp.append(point);tb.append(boot)
        entry['fraction_of_dense_contrast_lost']=summarize(1-np.mean(tp,axis=0)/dp,1-np.mean(tb,axis=0)/db)
        aperture[f'R{int(r)}']=entry
    result=dict(schema='tng-density-smoothing-results-v1',test_rows_before_mass_support=len(f['test_ids']),test_rows=len(ids),mass_bins=eligible,mass_bin_counts={str(b):int((bins==b).sum()) for b in eligible},occupied_spatial_blocks=nb,bootstrap_draws=len(bw),class_names=['void','wall','filament','cluster'],property_names=props,quartile_contrasts=outputs,seed_mean_contrasts=aggregate,paired_effects=effects,inference_metrics=metrics,property_shuffle_null=null,physical_cluster_minus_void=class_contrasts,coarse_oracle_contrasts=coarse,aperture_rank_contrasts=aperture,bootstrap_scope='Paired spatial block resampling within one held-out part of one TNG box; mean over three fixed paired fit/thinning seeds; not independent-volume or full model-training uncertainty',config_sha256=sha(CFG))
    result['permutation_null_200']=permutation_null
    result['physical_class_common_support_across_scales']=dict(mass_bins=class_cross_bins,rows=int(np.isin(bins,class_cross_bins).sum()),contrasts=class_cross,effects=class_cross_effects)
    save_json(out/'RESULTS.json',result)
    np.savez_compressed(out/'contrast_bootstraps.npz',**bootstore,**aggregate_boot)
    # Row-level held-out data make the scientific contrasts independently auditable.
    np.savez_compressed(out/'heldout_analysis.npz',ids=ids,mass_log10=mass,mass_bins=bins,block=block,properties=properties,**conditions)
    print(json.dumps(dict(rows=len(ids),aggregate=aggregate,effects=effects,classes=class_contrasts),indent=2),flush=True)

def verify(c,out):
    import torch
    from torch import nn
    torch.set_num_threads(1)
    a=np.load(out/'heldout_analysis.npz');f=np.load(out/'features.npz');p=np.load(out/'predictions.npz');b=np.load(out/'contrast_bootstraps.npz');r=json.loads((out/'RESULTS.json').read_text())
    checks=0;maxpoint=0.;maxboot=0.;maxreplay=0.
    ids=a['ids'];bins=a['mass_bins'];ys=a['properties'];block=a['block'];nb=int(block.max()+1)
    draws=np.random.default_rng(818265).multinomial(nb,np.ones(nb)/nb,size=c['bootstrap_draws'])
    for name,reported in r['quartile_contrasts'].items():
        score=a[name].sum(1);point=np.zeros(3);boots=np.zeros((2,3))
        for mb in r['mass_bins']:
            ii=np.flatnonzero(bins==mb);n=len(ii);order=np.argsort(score[ii],kind='stable');s=score[ii][order]
            edges=np.r_[0,np.flatnonzero(np.diff(s)!=0)+1,n]
            lo=np.zeros(n);hi=np.zeros(n)
            for start,end in zip(edges[:-1],edges[1:]):
                lo[order[start:end]]=max(0,min(end,n*.25)-start)/(end-start)
                hi[order[start:end]]=max(0,end-max(start,n*.75))/(end-start)
            point+=n/len(ids)*(np.average(ys[ii],axis=0,weights=hi)-np.average(ys[ii],axis=0,weights=lo))
            for k in range(2):
                w=draws[k,block[ii]]
                boots[k]+=n/len(ids)*(np.average(ys[ii],axis=0,weights=hi*w)-np.average(ys[ii],axis=0,weights=lo*w))
        expected=np.array([reported[pr]['estimate'] for pr in r['property_names']])
        err=float(np.max(abs(point-expected)));be=float(np.max(abs(boots-b[name][:2])))
        assert err<1e-10 and be<1e-10,(name,err,be)
        maxpoint=max(maxpoint,err);maxboot=max(maxboot,be);checks+=3
    for name in p.files:
        assert np.isfinite(p[name]).all() and np.all(np.diff(p[name],axis=1)>=0);checks+=2
        if 'frozen' in name:continue
        ck=torch.load(out/(name+'.pt'),map_location='cpu',weights_only=False)
        condition='dense' if '_dense_' in name else f'thin_{name.split("_seed")[-1]}'
        sample=ids[:128];x=(f[condition][sample]-ck['x_mean'])/ck['x_std']
        model=nn.Sequential(nn.Linear(x.shape[1],64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,3));model.load_state_dict(ck['state']);model.eval()
        with torch.no_grad():z=model(torch.from_numpy(x.astype(np.float32))).numpy()*ck['y_std']+ck['y_mean']
        ep=z_to_eigen(z);error=float(np.max(abs(ep-p[name][sample])))
        assert error<1e-4,error
        maxreplay=max(maxreplay,error);checks+=1
    class_report=r['physical_class_common_support_across_scales'];class_point_error=0.
    support=class_report['mass_bins'];support_rows=int(np.isin(bins,support).sum())
    for name,reported in class_report['contrasts'].items():
        cl=(a[name]>=c['class_threshold']).sum(1);point=np.zeros(3)
        for mb in support:
            mask=bins==mb
            point+=mask.sum()/support_rows*(ys[mask&(cl==3)].mean(0)-ys[mask&(cl==0)].mean(0))
        expected=np.array([reported[pr]['estimate'] for pr in r['property_names']])
        error=float(np.max(abs(point-expected)));assert error<1e-10
        class_point_error=max(class_point_error,error);checks+=1
    neighbour_extent={}
    for name in ['dense']+[f'thin_{s}' for s in c['seeds']]:
        dist=np.exp(f[name][:,-1]);neighbour_extent[name]=dict(max_40th_neighbour_hmpc=float(dist.max()),queries_beyond_split_buffer=int((dist>20.5).sum()))
    receipt=dict(passed=True,checks=checks,independent_point_contrast_max_error=maxpoint,independent_raw_row_bootstrap_max_error=maxboot,independent_class_contrast_max_error=class_point_error,checkpoint_prediction_replay_max_error=maxreplay,neighbour_extent=neighbour_extent,config_sha256=sha(CFG),script_sha256=sha(Path(__file__)))
    save_json(out/'VALIDATION.json',receipt);print(json.dumps(receipt,indent=2),flush=True)

def render(c,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    result=json.loads((out/'RESULTS.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(12,4.5),layout='constrained')
    specifications=[('quenched_fraction',100,'Quenched fraction contrast (percentage points)'),('g_r',1,'g − r contrast (mag)'),('log10_ssfr_capped',1,'Capped log sSFR contrast (dex)')]
    arms=[('truth','True matter field','#333333','o',-.20),('dense','Dense tracers','#2467a6','s',0),('thin','BGS-density tracers','#c66a20','^',.20)]
    for ax,(prop,scale,label) in zip(axes,specifications):
        for arm,title,color,marker,offset in arms:
            values=[];lower=[];upper=[]
            for r in [2,7]:
                value=result['quartile_contrasts'][f'R{r}_truth'][prop] if arm=='truth' else result['seed_mean_contrasts'][f'R{r}_{arm}'][prop]
                values.append(value['estimate']*scale);lower.append(value['ci95'][0]*scale);upper.append(value['ci95'][1]*scale)
            ax.errorbar(np.arange(2)+offset,values,yerr=[np.asarray(values)-lower,np.asarray(upper)-values],fmt=marker,color=color,linestyle='none',capsize=3,label=title,markersize=6)
        ax.axhline(0,color='.6',linewidth=.7);ax.set_xticks([0,1],['R = 2 h⁻¹ Mpc','R = 7 h⁻¹ Mpc']);ax.set_ylabel(label)
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.18)
    axes[0].legend(frameon=False,fontsize=9)
    fig.suptitle('Same TNG galaxies: property contrast across environment quartiles',fontsize=14)
    fig.supxlabel('Highest minus lowest trace quartile within stellar-mass bins; 95% paired spatial bootstrap intervals\nFixed held-out galaxies at z = 0; three paired fits; one TNG300 box',fontsize=10)
    fig.savefig(out/'tng_density_control.png',dpi=180);fig.savefig(out/'tng_density_control.pdf');plt.close(fig)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['catalogue','truth','features','train','analyse','verify','render']);args=parser.parse_args()
    c=json.loads(CFG.read_text());out=Path(c['out_dir']);out.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    globals()[args.stage](c,out)
    print(args.stage,'elapsed_seconds',time.monotonic()-start,flush=True)

if __name__=='__main__':main()
