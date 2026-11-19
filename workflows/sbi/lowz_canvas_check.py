"""Bounded trained-weight benchmark, not a full posterior generator."""
import json, os, socket, time
from pathlib import Path
import fitsio
import numpy as np
import torch
from workflows.sbi import lowz_coarse_fit as m
from workflows.sbi.e2e_global_canvas import Geometry, integrate as coarse_run
from workflows.sbi.e2e_vdm_context_models import block_mean
from workflows.abacus_tweb.p3br_build_random_response import galactic_cap
import lowz_endpoint_canvas as canvas

NORM='ce4e185228982341b52ab560c15090be2c3d1606c661d1add788402656dc46da'
DEV=('ph006','ph012','ph013')
def read(path): return json.loads(path.read_text())
def sync(): torch.cuda.synchronize(); return time.monotonic()
def plan(phase, cap, cat, rows):
    from workflows.sbi import e2e_coupled_coordinates as coord
    z=cat['Z']
    keep=(z>=.01)&(z<.55)&(galactic_cap(cat['RA'],cat['DEC'])==(cap=='NGC'))
    assert np.any(z[keep]<.15)
    xyz=coord.sky_mpc_h(cat['RA'][keep],cat['DEC'][keep],z[keep])
    rows=[r for r in rows if r['cap']==cap]
    grid=rows[0]['grid']; stride=np.array([64,32,32])
    keys=np.floor((xyz-grid['origin_mpc_h'])/grid['cell_mpc_h']/stride).astype('i4')
    extra=[(np.array(r['center'])+[side,0,0])//stride for r in rows for side in (0,32)]
    mids=np.unique(np.vstack([keys,extra]),axis=0)*stride+stride//2
    return dict(windows=len(mids),shape=list(map(int,canvas.geometry(mids).shape)),
                lowz_galaxies=int((z[keep]<.15).sum()))

@torch.inference_mode()
def main():
    m.fine.t.setup(17)
    p=m.pilot; root=p.c.ROOT.parent; data=m.fine.t.ROOT/'data'
    dest=root/('lowz_canvas_check_'+os.environ['SLURM_JOB_ID']); dest.mkdir(exist_ok=False)
    chart=p.views.load_chart(NORM)
    models=[p.CoupledBackbone('coarse','wide',base=24,levels=3),m.fine.t.FineModel('unet','x')]
    for model,name,sha in zip(models,('lowz_coarse_velocity_seed17_v1','lowz_unet_endpoint_20261118_seed17'),('b79083d1df2a4129dc9511d1aa7a9d29ededa61d6b139656f66086520f75e6dd','90a6f257c013160c330f75a7e2effbfa441ac39b025b241dd2f50907c96259c7')):
        ck=root/name/'LATEST.pt'; assert p.c.sha256(ck)==sha
        model.load_state_dict(torch.load(ck,map_location='cpu',weights_only=False)['ema'])
        model.cuda().eval()
    plans={}
    for phase in DEV:
        path=read(p.c.ROOT/'observations'/phase/'OBSERVED_COMPLETE.json')['outputs'][0]['path']
        cat=fitsio.read(path,columns=['RA','DEC','Z'])
        rows=read(p.coord.ROOT/'geometry'/phase/'GEOMETRY_COMPLETE.json')['pairs']
        for cap in ('NGC','SGC'): plans[phase+'/'+cap]=plan(phase,cap,cat,rows)
    p.c.atomic_json(dest/'PLANS.json',plans)
    phase=DEV[0]; results={}
    ten=lambda x: torch.as_tensor(x,device='cuda',dtype=torch.float32)[None]
    for cap in ('NGC','SGC'):
        rows=[r for r in read(data/phase/'COMPLETE.json')['pairs'] if r['cap']==cap and r['shell']==0]
        assert len(rows)==2
        mids=np.array([r['center'] for r in rows])+[16,0,0]
        cg=Geometry.make(mids,'coarse'); fg=canvas.geometry(mids)
        cond=[]; expanded=[]
        for r in rows:
            a=r['pair_id']; obs=p.views.load_observations(phase,a,NORM)
            cond.append(p.FieldCondition(ten(obs['joint']),ten(obs['wide']),ten(obs['joint_center_from_wide_mpc_h']),'wide'))
            with np.load(data/phase/a/'OBSERVATIONS.npz') as f:
                np.testing.assert_allclose(p.views.scale_channels(f['wide'],chart['wide']),obs['wide'],rtol=0,atol=1e-5)
                expanded.append(ten(p.views.scale_channels(f['joint'],chart['joint'])))
        n=8; noise=torch.randn((1,1,*cg.shape),device='cuda')
        # Reference coarse sampler generates noise using its addressed generator.
        one=Geometry.make(mids[:1],'coarse')
        n1=torch.randn((1,1,*one.shape),device='cuda',generator=torch.Generator(device='cuda').manual_seed(12345))
        z1,_=coarse_run(models[0],'coarse',cond[:1],n1,one,n)
        gold=m.coarse_models.sample(models[0],cond[0],'cfm',n//2,[12345])
        torch.testing.assert_close(z1,gold,rtol=1e-6,atol=1e-6)
        start=sync(); z,_=coarse_run(models[0],'coarse',cond,noise,cg,n); coarse_seconds=sync()-start
        fc=[m.fine_models.Condition(expanded[i],cond[i].wide,z[cg.crop(i)],'sampled') for i in range(2)]
        eps=torch.randn((1,1,*fg.shape),device='cuda')
        u1,_=canvas.integrate(models[1],fc[:1],eps[fg.crop(0)].contiguous(),canvas.geometry(mids[:1]),n)
        ref=m.fine_models.sample(models[1],fc[0],eps[fg.crop(0)].contiguous(),nfe=n)
        torch.testing.assert_close(u1,ref,rtol=1e-6,atol=1e-6)
        start=sync(); u,w=canvas.integrate(models[1],fc,eps,fg,n); fine_seconds=sync()-start
        lo=np.maximum(*map(np.asarray,fg.starts)); hi=np.minimum(*[np.array(s)+fg.window for s in fg.starts]); assert (hi>lo).all()
        pieces=[]; masserr=0
        for i in range(2):
            cs=chart['coarse_logrho']; mass=(z[cg.crop(i)].double()*cs['std'][0]+cs['mean'][0]).exp()[(...,*m.COARSE_CROP)]
            rho=m.decode_density(mass,u[fg.crop(i)].double()*chart['fine_residual']['std'][0])
            assert rho.isfinite().all() and rho.min()>0
            masserr=max(masserr,((block_mean(rho)-mass)/mass).abs().max().item())
            pieces.append(rho[(...,*tuple(slice(a,b) for a,b in zip(lo-fg.starts[i],hi-fg.starts[i])))])
        diff=(pieces[0]-pieces[1]).abs().max().item()
        assert diff<1e-10 and masserr<1e-10
        results[cap]=dict(midpoints=mids.tolist(),coarse_seconds=coarse_seconds,fine_seconds=fine_seconds,nfe=n,
          overlap_cells=int(np.prod(hi-lo)),density_overlap_error=diff,mass_error=masserr,
          coarse_parity=(z1-gold).abs().max().item(),fine_parity=(u1-ref).abs().max().item(),
          two_window_peak_gib=torch.cuda.max_memory_allocated()/2**30)
        p.c.atomic_json(dest/(cap+'.json'),results[cap])
        print(cap,results[cap],flush=True)
    p.c.atomic_json(dest/'COMPLETE.json',dict(node=socket.gethostname(),plans=plans,results=results))
if __name__=='__main__': main()
