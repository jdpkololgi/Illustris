"""Transform prepared Loa inputs into the native/wide CFM response; no inference."""
import sys,os,socket,json
from pathlib import Path
import lowz_confirmation_inputs as b
from lowz_confirmation_inputs import check,read,sha,native_values


def run(receipt,expected):
    assert __debug__ and socket.gethostname().startswith('nid') and os.environ['SLURM_JOB_ID']
    assert Path.cwd()==b.DEPLOY and os.environ['GRAPHWEB_CFM_DATA_CONFIG']==b.CONFIG
    assert receipt.name=='OBSERVATIONS_PREPARED.json' and receipt.parent.parent==b.BASE and sha(receipt)==expected
    art=read(b.FREEZE)['artifacts']
    assert sha(b.FREEZE)=='80436676ce3cc07e4364a1ee62d524452a0635e2e08fb12975d5167b8a437be1'
    for rel,h in art['source_files'].items():assert sha(b.DEPLOY/rel)==h
    sys.path.insert(0,str(b.DEPLOY))
    from workflows.sbi import cfm_lowz_selection_response as low,build_cfm_matched_response as matched
    from workflows.sbi import e2e_coupled_condition_products as products,e2e_coupled_coordinates as coord
    from workflows.abacus_tweb import p3br_build_random_response as random,p3a_build_canonical_fields as grid
    import h5py, fitsio, numpy as np, healpy as hp
    p=read(receipt);assert p['phase']=='Loa' and p['passed'] and not p['refit'] and p['method']==sha(b.FREEZE)
    obs,ang=[b.verify_record(v,receipt.parent) for v in p['outputs']]
    rows=fitsio.read(obs,columns=['RA','DEC','Z'])
    with np.load(ang,allow_pickle=False) as f: angular={k:f[k] for k in ('domain','support','angular_response')}
    plan,policy,_=low.load_plan();cfg=read(b.DEPLOY/b.CONFIG)
    selection_path=Path(cfg['observation']['fixed_selection_manifest']);sel=read(selection_path)
    assert sha(selection_path)=='17321b87097468a478e9d581ae7ff8c1b5b7fce6054c491f578fb86735eba8ee'
    assert sel['fit_phases']==policy['training'] and sel['omitted_phase'] is None
    b.dev_parity(plan,sel)
    out=b.BASE/('lowz_loa_response_'+os.environ['SLURM_JOB_ID']);out.mkdir(exist_ok=False)
    def hist(cat,common):
        pix=hp.ang2pix(256,cat['RA'],cat['DEC'],lonlat=True)
        caps=random.galactic_cap(cat['RA'],cat['DEC']);z=cat['Z']
        return [np.histogram(z[common[pix]&(caps==j)&(angular['domain'][pix]//2==j)&(z<.55)],p['shell_edges'])[0].tolist() for j in range(2)]
    controls={}
    for ph in ('ph006','ph012','ph013'):
        dev=read(coord.ROOT/'observations'/ph/'NGC_COMPLETE.json')
        src=dev['matched_source'];assert sha(src['path'])==src['sha256']
        inputs=read(src['path'])['binding']; oa,aa=[inputs[k]['payload'] for k in ('observed','angular')]
        assert sha(oa['path'])==oa['sha256'] and sha(aa['path'])==aa['sha256']
        with np.load(aa['path'],allow_pickle=False) as f:
            common=angular['support'].astype(bool)&f['support'].astype(bool)&(angular['domain']==f['domain'])
        controls[ph]=dict(loa=hist(rows,common),mock=hist(fitsio.read(oa['path'],columns=['RA','DEC','Z']),common),
            common_pixels=[int((common&(angular['domain']//2==j)).sum()) for j in range(2)])
    phase='Loa';workers=16
    for cap in ('NGC','SGC'):
        dest=out/cap;dest.mkdir()
        marker=dest/'RESPONSE_COMPLETE.json'
        bound=dict(observations=expected,method=sha(b.FREEZE),selection=sha(selection_path),phase=phase,cap=cap)
        native,wide = dest/'NATIVE.h5',dest/'FACTOR4.h5'
        check(not native.exists() and not wide.exists(), 'incomplete cap: stop for review')
        geometry = low.support_grid(angular,cap,coord.config()['raw_cell_mpc_h'])
        g = dict(origin_mpc_h=list(geometry.origin),shape=list(geometry.shape),cell_mpc_h=geometry.cell_mpc,
                 distance_unit='Mpc/h',padding_mpc_h=0.,coordinate_sha256=sha(coord.CONFIG))
        mask,audit = matched.source_point_mask(rows,angular,cap,plan)
        selected = rows[mask]
        points=coord.sky_mpc_h(selected['RA'],selected['DEC'],selected['Z'])
        deposited,cic = grid.cic_deposit(points,geometry)
        check(cic['lost_weight']==0 and np.isclose(deposited.sum(dtype=float),mask.sum(),rtol=2e-6),'CIC conservation')
        boundary = random.angular_boundary_distance(angular['support'] & (angular['domain']//2==low.CAPS.index(cap)))
        chunks=tuple(min(n,32) for n in geometry.shape)
        bounds=[tuple((s.start,s.stop) for s in sl) for sl in grid.iter_chunks(geometry.shape,chunks)]
        args=(angular,cap,plan,sel,geometry.origin,geometry.cell_mpc)
        with matched.projected_chunks(bounds,workers,args) as results, h5py.File(native,'x') as f:
            fields={}
            f.attrs.update(phase=phase,cap=cap,contains_matter=False,coordinate_sha256=g['coordinate_sha256'],
                distance_unit='Mpc/h',origin_mpc_h=geometry.origin,cell_mpc_h=geometry.cell_mpc)
            for bb,expected in results:
                sl=tuple(slice(a,b) for a,b in bb)
                vals=native_values(geometry,sl,angular,cap,plan,sel,boundary,deposited[sl],expected)
                for key,v in vals.items():
                    check(np.isfinite(v).all(),key)
                    if key not in fields:
                        fields[key]=f.create_dataset(key,shape=geometry.shape,dtype='u1' if key in ('selection_valid','support_random') else 'f4',
                                                     chunks=chunks,compression='lzf',shuffle=True)
                    fields[key][sl]=v
        conservation=products.coarsen_response(native,wide,g)
        result=dict(binding=bound,grid=g,audit=audit,conservation=conservation,development_parity=True,
                    outputs=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in (native,wide)],
                    low_z_rows=int(((selected['Z']>=.01)&(selected['Z']<.15)).sum()),
                    passed=True,job=os.environ['SLURM_JOB_ID'])
        with marker.open('x') as f: json.dump(result,f,sort_keys=True,allow_nan=False)
        print('RESPONSE_COMPLETE',phase,cap,flush=True)

    coord.c.atomic_json(out/'RESPONSE_PREPARED.json',dict(passed=True,obs_receipt=expected,observations_path=str(receipt),method=sha(b.FREEZE),
        source=sha(__file__),cap_order=['SGC','NGC'],shell_edges=p['shell_edges'],common_domain_counts=controls,
        caps={cap:sha(out/cap/'RESPONSE_COMPLETE.json') for cap in ('NGC','SGC')},
        scope='Matched cap response and transfer diagnostics; not proof of DESI calibration'))
if __name__=='__main__':run(Path(sys.argv[1]),sys.argv[2])
