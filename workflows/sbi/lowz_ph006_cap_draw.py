"""Single corrected ph006 draw for each cap. Independent stage processes."""
import argparse
import os
import time
import h5py
import numpy as np
import torch
from lowz_canvas_check import m, canvas, Geometry, coarse_run, NORM, read, galactic_cap, fitsio, block_mean
from workflows.sbi import e2e_coupled_coordinates as coord
from workflows.sbi import e2e_coupled_condition_products as products
from workflows.sbi import e2e_wholefield_pilot_data as wd
from workflows.sbi import e2e_footprint_tidal_operator as tidal
from workflows.sbi.e2e_wholefield_fullcap_extract import cuts
from workflows.sbi.e2e_coupled_product_audit import read_target
from workflows.sbi.e2e_coupled_target_products import owned_core_arrays
from workflows.sbi.e2e_exterior_cap_tides import run as run_tides
p=m.pilot
c=p.c
phase='ph006'
draw=0
prior=c.ROOT/'exterior_learned_pad32_v1'
def checked(rec):
    item=rec['outputs'][0]
    assert c.sha256(item['path'])==item['sha256']
    return item['path']
def rows(cap):
    g=coord.verify_receipt(coord.ROOT/'geometry'/phase/'GEOMETRY_COMPLETE.json',payload=False)
    r=[v for v in g['pairs'] if v['cap']==cap]
    assert len(r)==8
    return r
def freeze():
    f=read(prior/'FREEZE.json')
    assert f['prior_sha256']==c.sha256(prior/'PRIOR.json') and f['training_phases']==list(c.TRAIN)
    assert f['method']=='learned' and f['padding']==32
    return f
@torch.inference_mode()
def density(cap, dest):
    m.fine.t.setup(17)
    chart=p.views.load_chart(NORM)
    r=rows(cap); grid=r[0]['grid']; stride=np.array([64,32,32])
    cat=fitsio.read(checked(read(c.ROOT/'observations'/phase/'OBSERVED_COMPLETE.json')),columns=['RA','DEC','Z'])
    k=(cat['Z']>=.01)&(cat['Z']<.55)&(galactic_cap(cat['RA'],cat['DEC'])==(cap=='NGC'))
    xyz=coord.sky_mpc_h(cat['RA'][k],cat['DEC'][k],cat['Z'][k])
    keys=np.floor((xyz-grid['origin_mpc_h'])/grid['cell_mpc_h']/stride).astype('i4')
    extra=[(np.array(row['center'])+[side,0,0])//stride for row in r for side in (0,32)]
    mids=np.unique(np.vstack((keys,extra)),axis=0)*stride+stride//2
    assert len(mids)>0
    if phase=='ph006': assert len(mids)==dict(NGC=1584,SGC=814)[cap]
    del xyz,keys,cat
    cg=Geometry.make(mids,'coarse'); fg=canvas.geometry(mids)
    models=[p.CoupledBackbone('coarse','wide',base=24,levels=3),m.fine.t.FineModel('unet','x')]
    hashes=['b79083d1df2a4129dc9511d1aa7a9d29ededa61d6b139656f66086520f75e6dd','90a6f257c013160c330f75a7e2effbfa441ac39b025b241dd2f50907c96259c7']
    for model, name, sha in zip(models,['lowz_coarse_velocity_seed17_v1','lowz_unet_endpoint_20261118_seed17'],hashes):
        ck=c.ROOT.parent/name/'LATEST.pt'; assert c.sha256(ck)==sha
        model.load_state_dict(torch.load(ck,map_location='cpu',weights_only=False)['ema']);model.cuda().eval()
    dest.mkdir(parents=True,exist_ok=False)
    t=time.monotonic(); cc=[]; joint=[]; support=np.zeros(fg.shape,'f4')
    ten=lambda x:torch.as_tensor(x,device='cuda',dtype=torch.float32)[None]
    with h5py.File(checked(read(coord.ROOT/'observations'/phase/(cap+'_COMPLETE.json')))) as raw,h5py.File(checked(read(coord.ROOT/'conditions'/phase/(cap+'_FACTOR4.json')))) as wide:
        for i,mid in enumerate(mids):
            small=products.extract_pair(dict(center=mid-[16,0,0],grid=grid),raw,wide)
            obs=p.views.observation_view(small,phase,(0,0,0),chart)
            x=wd.build_lowz_fine_observations(grid,mid,raw)
            sl=fg.crop(i)[2:]; support[sl]=np.maximum(support[sl],x[products.LOCAL_CHANNELS.index('support_random')])
            full=ten(p.views.scale_channels(x,chart['joint']))
            core=full[(slice(None),slice(None),*wd.OLD_CROP)]
            if i==0: torch.testing.assert_close(core,ten(obs['joint']),rtol=0,atol=1e-5)
            joint.append(full); cc.append(p.FieldCondition(core,ten(obs['wide']),ten(obs['joint_center_from_wide_mpc_h']),'wide'))
            if i%100==0: print(cap,'prepared',i,flush=True)
    prep=time.monotonic()-t;t=time.monotonic()
    def noise(g, stage):
        seed=int(c.digest([phase+'_cap_draw_v1',cap,stage,draw])[:15],16)
        return torch.randn((1,1,*g.shape),device='cuda',generator=torch.Generator(device='cuda').manual_seed(seed))
    z,_=coarse_run(models[0],'coarse',cc,noise(cg,'coarse'),cg,64)
    fc=[m.fine_models.Condition(j,cond.wide,z[cg.crop(i)],'sampled') for i,(j,cond) in enumerate(zip(joint,cc))]
    u,w=canvas.integrate(models[1],fc,noise(fg,'fine'),fg,64)
    torch.cuda.synchronize(); solve=time.monotonic()-t
    offset=np.array(fg.origin)//4-cg.origin; size=np.array(fg.shape)//4
    cut=tuple(slice(a,a+b) for a,b in zip(offset,size))
    cs=chart['coarse_logrho']; mass=(z[(...,*cut)].double()*cs['std'][0]+cs['mean'][0]).exp()
    rho=m.decode_density(mass,u.double()*chart['fine_residual']['std'][0])
    err=((block_mean(rho)-mass)/mass).abs().max().item()
    assert err<1e-10 and rho.isfinite().all() and rho.min()>0
    path=dest/'FIELD.h5'; covered=w[0,0].cpu().numpy().astype('u2')
    values=rho[0,0].float().cpu().numpy();values[covered==0]=np.nan
    with h5py.File(path,'x') as f:
        f.create_dataset('rho',data=values[None],compression='lzf')
        f.create_dataset('coverage',data=covered,compression='lzf')
        f.create_dataset('support',data=support,compression='lzf')
        f.attrs.update(origin_mpc_h=np.array(grid['origin_mpc_h'])+np.array(fg.origin)*6.766,phase=phase,cap=cap,first_draw=draw,cell_mpc_h=6.766)
    c.atomic_json(dest/'COMPLETE.json',dict(passed=True,phase=phase,cap=cap,first=draw,draws=1,windows=len(mids),field=c.file_record(path,content_hash=True),prep_seconds=prep,solve_seconds=solve,mass_error=err,checkpoints=hashes,normalizer=NORM,nfe=64))
def score(cap,d):
    vals=[]
    with h5py.File(d/'tides/COMPLETED_FIELD.h5') as t,h5py.File(d/'density/FIELD.h5') as f:
        for row in rows(cap):
            a=row['pair_id']; tr=read_target(checked(read(coord.ROOT/'targets'/phase/(a+'.json'))),phase,a)
            sel=cuts(row,f.attrs['origin_mpc_h']); mask=f['support'][sel]>0
            pred=t['eigenvalues'][(0,*sel)]; target=tidal.eigensystem(tr['fullbox_tensor_cores']).reshape(pred.shape)
            rho=f['rho'][(0,*sel)]; ref=owned_core_arrays(tr['rho_joint']).reshape(rho.shape)
            pos=np.stack(np.broadcast_arrays(*[(np.arange(s.start,s.stop)+.5).reshape(tuple(-1 if i==k else 1 for i in range(3)))*6.766+f.attrs['origin_mpc_h'][k] for k,s in enumerate(sel)]),axis=-1)
            z=coord.redshift(np.linalg.norm(pos,axis=-1))
            for l,h in zip([.01,.15,.25,.35,.45],[.15,.25,.35,.45,.55]):
                v=mask&(z>=l)&(z<h)
                if not v.any():continue
                delta=np.column_stack([(rho-ref)[v],(pred-target)[v]])
                classes=(pred[v]>.2).sum(-1); true=(target[v]>.2).sum(-1)
                vals.append(dict(pair=a,stratum=row['support_stratum'],z=[l,h],n=int(v.sum()),bias=delta.mean(0).tolist(),mse=(delta**2).mean(0).tolist(),confusion=np.bincount(4*true+classes,minlength=16).reshape(4,4).tolist()))
    assert vals
    c.atomic_json(d/'SCORES.json',dict(phase=phase,cap=cap,rows=vals,operator='cap free-space cell-average, no subcell correction',scope='one draw, not posterior coverage'))
if __name__=='__main__':
    args=argparse.ArgumentParser();args.add_argument('stage',choices=['density','tides','score']);args.add_argument('cap',choices=['NGC','SGC'])
    args.add_argument('--phase',choices=('ph006','ph012','ph013'),default='ph006')
    args.add_argument('--draw',type=int,choices=range(32),default=0)
    a=args.parse_args();phase=a.phase;draw=a.draw
    assert c.DEVELOPMENT==('ph006','ph012','ph013')
    c.require_compute(); freeze(); d=c.ROOT.parent/('lowz_cap_draw_'+os.environ['SLURM_JOB_ID'])/a.cap
    if a.stage=='density': density(a.cap,d/'density')
    elif a.stage=='tides': run_tides(d/'density/COMPLETE.json',prior/'PRIOR.json',d/'tides',method='learned',padding=32)
    else: score(a.cap,d)
