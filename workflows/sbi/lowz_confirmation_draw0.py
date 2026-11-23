"""One authorized ph018 draw per cap. Prediction only; no truth/scoring entry point."""
import gc,json,os,sys,time,socket
from pathlib import Path
import lowz_confirmation_inputs as b

stage,cap=sys.argv[1:]
phase='ph018'
draw=0
assert stage in ('density','tides') and cap in ('NGC','SGC') and __debug__
assert socket.gethostname().startswith('nid') and os.environ['SLURM_JOB_ID']
assert Path.cwd()==b.DEPLOY and os.environ['GRAPHWEB_CFM_DATA_CONFIG']==b.CONFIG
spec,art=b.release(phase)
assert b.sha(b.RELEASE)=='ac947c30a5d3e0706f7de261a255ea51277813e7f4720b1915afc0e027179ac7'
freeze=b.OUT/'GEOMETRY_FROZEN.json'
assert b.sha(freeze)=='eb20683dbbf052f19ec651b8711c403eab3f75a967d235faf12011f316b2e737'
g=b.read(freeze)
assert not g['truth_read'] and g['release']==b.sha(b.RELEASE)
for rel,h in art['source_files'].items(): assert b.sha(b.DEPLOY/rel)==h
for v in [*art['sampler_helpers'].values(),*(art[k] for k in ('coarse','fine','normalizer','prior','prior_freeze'))]:
    assert b.sha(v['path'])==v['sha256']
sys.path[:0]=[str(b.DEPLOY),str(Path(art['sampler_helpers']['lowz_canvas_check.py']['path']).parent)]
import h5py,numpy as np,torch
from lowz_canvas_check import m,canvas,Geometry,coarse_run,NORM,galactic_cap,fitsio,block_mean
from workflows.sbi import e2e_coupled_coordinates as coord,e2e_coupled_condition_products as products,e2e_wholefield_pilot_data as wd
from workflows.sbi import e2e_spatial_exterior as spatial,e2e_footprint_tidal_operator as tidal,e2e_loa_field_sampling as sampling
from workflows.sbi.e2e_exterior_cap_tides import CapTidal
p=m.pilot
c=p.c
rows=[r for r in g['phases'][phase]['pairs'] if r['cap']==cap]
assert len(rows)==8
directory=b.OUT/phase/cap
rec=directory/'RESPONSE_COMPLETE.json'
assert b.sha(rec)==g['phases'][phase]['response_receipts'][cap]
r=b.read(rec)
assert r['passed'] and r['binding']['phase']==phase and r['binding']['cap']==cap
catalog=b.verify_record(b.read(b.OLD/phase/'OBSERVED_COMPLETE.json')['outputs'][0],b.OLD/phase)
native,wide_path=[b.verify_record(v,directory) for v in r['outputs']]
assert [native.name,wide_path.name]==['NATIVE.h5','FACTOR4.h5']
dest=directory/'predictions/draw_000'
prior_path=Path(art['prior']['path'])

@torch.inference_mode()
def density(cap, dest):
    m.fine.t.setup(17)
    chart=p.views.load_chart(NORM)
    r=rows; grid=r[0]['grid']; stride=np.array([64,32,32])
    cat=fitsio.read(catalog,columns=['RA','DEC','Z'])
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
    with h5py.File(native) as raw,h5py.File(wide_path) as wide:
        for i,mid in enumerate(mids):
            small=products.extract_pair(dict(center=mid-[16,0,0],grid=grid),raw,wide)
            products.validate_arrays(small)
            cfg=products.op.layout()
            start=np.asarray(cfg['wide_crop_base_start'])
            slices=tuple(slice(int(a),int(a+n)) for a,n in zip(start,cfg['wide_crop_shape']))
            obs=dict(joint=p.views.scale_channels(small['joint'],chart['joint']),
                     wide=p.views.scale_channels(small['wide_extended'][(slice(None),*slices)],chart['wide']),
                     joint_center_from_wide_mpc_h=np.zeros(3,'f4'))
            if i==0:
                # Compare the same in-memory arrays with the frozen evaluation transform.
                ref=p.views.observation_view(small,'ph006',(0,0,0),chart)
                assert all(np.array_equal(v,ref[k]) for k,v in obs.items())
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

def tides():
    field_receipt=dest/'density/COMPLETE.json'
    record=b.read(field_receipt)
    assert record['passed'] and record['phase']==phase and record['cap']==cap and record['first']==0
    assert b.verify_record(record['field'],dest/'density')==dest/'density/FIELD.h5'
    out=dest/'tides'
    method,padding,old_loa,limit='learned',32,False,None
    fit = json.loads(prior_path.read_text())
    if fit['training_phases'] != list(c.TRAIN):
        raise ValueError('Spatial prior is not training-only')
    sampling.setup(True)
    out.mkdir(parents=True, exist_ok=False)
    binding = dict(phase=phase, cap=cap, method=method, padding=padding,
                   input_receipt=c.file_record(field_receipt, content_hash=True),
                   prior=c.file_record(prior_path, content_hash=True),
                   sources={Path(v.__file__).name: c.sha256(v.__file__) for v in (spatial, tidal)},
                   runner_sha256=c.sha256(__file__), posterior_qualified=False,
                   cap_joint_law='independent capwise modular approximation',
                   opposite_cap_observations_condition_exterior=False,
                   freeze_sha256=None if freeze is None else c.sha256(freeze))
    c.atomic_json(out/'BINDING.json', binding)
    with h5py.File(record['field']['path']) as source:
        shape = source['rho'].shape[1:]
        support = source['support'][:]
        count = source['rho'].shape[0] if limit is None else min(limit, source['rho'].shape[0])
        origin = np.asarray(source.attrs['origin_mpc_h'])
        first = 0 if old_loa else int(source.attrs['first_draw'])
        start = time.monotonic()
        prior = spatial.CudaSpatialPrior(tuple(n+2*padding for n in shape), fit, workers=4)
        operator = CapTidal(shape, padding)
        setup_seconds = time.monotonic()-start
        diagnostics = []
        output = out/'COMPLETED_FIELD.h5'
        with h5py.File(output, 'x') as f:
            complete = f.create_dataset('rho_completed', shape=(count, *prior.shape), dtype='f4',
                                        chunks=(1, 16, 32, 32), compression='lzf')
            eigen = f.create_dataset('eigenvalues', shape=(count, *shape, 3), dtype='f4',
                                     chunks=(1, 8, 32, 32, 3), compression='lzf')
            f.create_dataset('support', data=support, compression='lzf')
            f.create_dataset('coverage', data=source['coverage'][:], compression='lzf')
            f.attrs.update(phase=phase, cap=cap, cell_mpc_h=6.766, origin_mpc_h=origin,
                           completed_origin_mpc_h=origin-padding*6.766, first_draw=first,
                           method=method, padding_cells=padding, posterior_qualified=False,
                           cap_joint_law=binding['cap_joint_law'], smoothing_mpc_h=7., smoothing_count=1)
            for i in range(count):
                tick = time.monotonic()
                rho = source['rho'][i]
                seed = int(c.digest(['spatial-exterior-cap-v1', phase, cap, first+i])[:15], 16)
                field, diagnostic = spatial.complete_density(rho, support, prior, np.random.default_rng(seed), method, padding)
                completion_seconds = time.monotonic()-tick
                complete[i] = field.astype('f4')
                # The measured full-cap smoke peak is below 15 GiB host RSS.
                # This 1–2 GiB array avoids unreclaimed HDF5 scratch chunks
                # inflating every completed product by a temporary tensor.
                temp = np.empty((*shape, 6), dtype=np.float64)
                for component, values in enumerate(operator.components(field)):
                    temp[..., component] = values
                    del values
                trace_error = 0.
                for lo in range(0, shape[0], 8):
                    hi = min(lo+8, shape[0]); tensor = temp[lo:hi]
                    local = field[padding+lo:padding+hi, padding:padding+shape[1], padding:padding+shape[2]]
                    trace_error = max(trace_error, float(np.max(abs(tensor[..., [0, 3, 5]].sum(-1)-(local-1)))))
                    ev=tidal.eigensystem(tensor).astype('f4')
                    assert np.isfinite(ev).all() and (np.diff(ev,axis=-1)>=0).all()
                    eigen[i, lo:hi]=ev
                if trace_error > 1e-9*max(1., float(field.max())):
                    raise ValueError('Full-cap free-space trace identity failed')
                del temp
                diagnostic.update(draw=first+i, seconds=time.monotonic()-tick,
                                  completion_seconds=completion_seconds, trace_maxabs=trace_error,
                                  minimum_density=float(field.min()))
                diagnostics.append(diagnostic); print(json.dumps(diagnostic), flush=True)
                del field, rho; gc.collect()
        final = dict(phase=phase, cap=cap, first=first, draws=count, method=method,
                     binding=c.sha256(out/'BINDING.json'), diagnostics=diagnostics,
                     setup_seconds=setup_seconds, seconds=time.monotonic()-start,
                     output=c.file_record(output, content_hash=True), passed=True,
                     posterior_qualified=False, cap_joint_law=binding['cap_joint_law'],
                     job=os.environ['SLURM_JOB_ID'])
        c.atomic_json(out/'COMPLETE.json', final)



if stage=='density':
    assert not dest.exists()
    dest.mkdir(parents=True)
    c.atomic_json(dest/'INPUTS.json',dict(phase=phase,cap=cap,draw=0,geometry=b.sha(freeze),release=b.sha(b.RELEASE),
        runner=b.sha(__file__),response=b.sha(rec),truth_read=False,job=os.environ['SLURM_JOB_ID']))
    density(cap,dest/'density')
else:
    receipt=b.read(dest/'INPUTS.json')
    assert receipt['job']==os.environ['SLURM_JOB_ID'] and receipt['runner']==b.sha(__file__)
    tides()
