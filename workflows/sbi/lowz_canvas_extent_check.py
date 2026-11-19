"""Sparse real-window benchmark on actual cap-sized canvases; not posterior output."""
from dataclasses import replace
from lowz_canvas_check import *
from workflows.sbi import e2e_coupled_coordinates as coord
@torch.inference_mode()
def main():
    m.fine.t.setup(17); p=m.pilot; phase='ph006'; base=p.c.ROOT.parent
    dest=base/('lowz_canvas_check_'+os.environ['SLURM_JOB_ID'])
    plans=read(dest/'PLANS.json'); chart=p.views.load_chart(NORM)
    models=[p.CoupledBackbone('coarse','wide',base=24,levels=3),m.fine.t.FineModel('unet','x')]
    for model,name,sha in zip(models,('lowz_coarse_velocity_seed17_v1','lowz_unet_endpoint_20261118_seed17'),('b79083d1df2a4129dc9511d1aa7a9d29ededa61d6b139656f66086520f75e6dd','90a6f257c013160c330f75a7e2effbfa441ac39b025b241dd2f50907c96259c7')):
        path=base/name/'LATEST.pt'; assert p.c.sha256(path)==sha
        model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['ema']); model.cuda().eval()
    data=m.fine.t.ROOT/'data'/phase
    cat=fitsio.read(read(p.c.ROOT/'observations'/phase/'OBSERVED_COMPLETE.json')['outputs'][0]['path'],columns=['RA','DEC','Z'])
    for cap in ('NGC','SGC'):
        rows=[r for r in read(data/'COMPLETE.json')['pairs'] if r['cap']==cap]
        grid=rows[0]['grid']; stride=np.array([64,32,32])
        keep=(cat['Z']>=.01)&(cat['Z']<.55)&(galactic_cap(cat['RA'],cat['DEC'])==(cap=='NGC'))
        xyz=coord.sky_mpc_h(cat['RA'][keep],cat['DEC'][keep],cat['Z'][keep])
        keys=np.floor((xyz-grid['origin_mpc_h'])/grid['cell_mpc_h']/stride).astype('i4')
        extra=[(np.array(r['center'])+[side,0,0])//stride for r in rows for side in (0,32)]
        mids=np.unique(np.vstack((keys,extra)),axis=0)*stride+stride//2
        assert len(mids)==plans[phase+'/'+cap]['windows']
        local=np.array([r['center'] for r in rows])+[16,0,0]
        # Include registered windows in the timing extent if a crop extends past the cap plan.
        def geom(stage,k):
            f=canvas.geometry if stage=='fine' else lambda x:Geometry.make(x,'coarse')
            g=f(np.unique(np.vstack((mids,local)),axis=0))
            starts=(local[:k]-np.array(g.window))//2 if stage=='fine' else (local[:k]-192)//8
            return replace(g,starts=tuple(map(tuple,starts-g.origin)))
        host=[]; ten=lambda x:torch.as_tensor(x,device='cuda',dtype=torch.float32)[None]
        for r in rows:
            a=r['pair_id']; obs=p.views.load_observations(phase,a,NORM)
            with np.load(data/a/'OBSERVATIONS.npz') as f:
                host.append((obs,p.views.scale_channels(f['joint'],chart['joint'])))
        report=[]
        for k in (2,8,8):
            cg=geom('coarse',k); fg=geom('fine',k)
            torch.cuda.reset_peak_memory_stats()
            start=sync()
            cc=[p.FieldCondition(ten(o['joint']),ten(o['wide']),ten(o['joint_center_from_wide_mpc_h']),'wide') for o,j in host[:k]]
            z,_=coarse_run(models[0],'coarse',cc,torch.randn((1,1,*cg.shape),device='cuda'),cg,4)
            ct=sync()-start; start=sync()
            fc=[m.fine_models.Condition(ten(host[i][1]),cc[i].wide,z[cg.crop(i)],'sampled') for i in range(k)]
            u,_=canvas.integrate(models[1],fc,torch.randn((1,1,*fg.shape),device='cuda'),fg,4)
            ft=sync()-start
            report.append(dict(k=k,nfe=4,coarse_seconds=ct,fine_seconds=ft,peak_gib=torch.cuda.max_memory_allocated()/2**30,shape=list(map(int,fg.shape))))
            del u,z,fc,cc,_
        p.c.atomic_json(dest/(cap+'_extent.json'),dict(trials=report));print(cap,report,flush=True)
if __name__=='__main__': main()
