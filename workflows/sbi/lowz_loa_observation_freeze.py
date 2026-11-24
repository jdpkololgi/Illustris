"""Paper 3 Loa target-free observation and angular inputs. No model fitting or inference."""
import os
import socket
import sys
from pathlib import Path
import lowz_confirmation_inputs as b


def run():
    assert __debug__ and socket.gethostname().startswith('nid') and os.environ['SLURM_JOB_ID']
    assert Path.cwd()==b.DEPLOY and os.environ['GRAPHWEB_CFM_DATA_CONFIG']==b.CONFIG
    reg=b.REPO/'configs/p10_response_sources_v1.json'
    assert b.sha(reg)=='11d7fd79b89e7dc065a27db4cfccd940de5e88be67b92f26fd6703271cf89d71'
    assert b.sha(b.FREEZE)=='80436676ce3cc07e4364a1ee62d524452a0635e2e08fb12975d5167b8a437be1'
    art=b.read(b.FREEZE)['artifacts']
    for rel,h in art['source_files'].items(): assert b.sha(b.DEPLOY/rel)==h
    sys.path.insert(0,str(b.DEPLOY))
    import fitsio
    import healpy as hp
    import numpy as np
    from workflows.abacus_tweb import p3br_build_random_response as response
    from workflows.sbi import e2e_coupled_contract as c,cfm_lowz_selection_response as low
    a=b.read(reg)['desi_candidate']
    lss=Path('/global/cfs/cdirs/desi/survey/catalogs/DA2/LSS/loa-v1/LSScats/v2.1')
    data=a['data']['full']; randoms=a['full_random']
    assert Path(data['path'])==lss/'BGS_BRIGHT_full_HPmapcut.dat.fits'
    assert [Path(r['path']) for r in randoms]==[lss/f'BGS_BRIGHT_{i}_full_HPmapcut.ran.fits' for i in range(18)]
    out=b.BASE/('lowz_loa_preparation_'+os.environ['SLURM_JOB_ID']);out.mkdir(exist_ok=False)
    counts=np.zeros((4,hp.nside2npix(256)),dtype=np.int64)
    for i,r in enumerate(randoms):
        b.verify_record(r,lss); audit=response.add_random_file(counts,r)
        assert not audit['maskbits_filter_applied']
        print('random complete',i,flush=True)
    angular=response.normalized_map(counts)
    ap=out/'ANGULAR.npz'
    with ap.open('xb') as f:np.savez_compressed(f,**{k:v for k,v in angular.items() if k!='metadata'})
    src=b.verify_record(data,lss)
    plan,_,_=low.load_plan()
    edges=[.01,.15,.25,.35,.45,.55]
    hist=np.zeros((2,5),dtype=np.int64); supported=hist.copy()
    path=out/'OBSERVED.fits'; written=0
    with fitsio.FITS(src) as f,fitsio.FITS(path,'rw') as dst:
        t=f[1]; assert t.get_nrows()==data['rows']
        for start in range(0,t.get_nrows(),250000):
            g=t.read_slice(start,min(t.get_nrows(),start+250000))
            z=g['Z_not4clus']
            k=low.radial_mask(z,plan)&(g['ZWARN']==0)&((g['BGS_TARGET']&2)!=0)
            k&=(np.char.strip(g['SPECTYPE'].astype('U'))=='GALAXY')&(g['DELTACHI2']>=25)
            if not k.any():continue
            g=g[k]; z=z[k]
            v=np.empty(len(z),dtype=[('TARGETID','i8'),('RA','f8'),('DEC','f8'),('Z','f8')])
            for key in ('TARGETID','RA','DEC'):v[key]=g[key]
            v['Z']=z
            assert np.isfinite(v['RA']).all() and np.isfinite(v['DEC']).all()
            if not written:dst.write(v,extname='OBSERVED')
            else:dst[-1].append(v)
            written+=len(v)
            caps=response.galactic_cap(v['RA'],v['DEC'])
            pix=hp.ang2pix(256,v['RA'],v['DEC'],lonlat=True)
            valid=angular['support'][pix]&(angular['domain'][pix]//2==caps)
            for cap in (0,1):
                zz=z[(caps==cap)&(z<edges[-1])]
                hist[cap]+=np.histogram(zz,edges)[0]
                supported[cap]+=np.histogram(z[(caps==cap)&valid&(z<edges[-1])],edges)[0]
    assert written and np.all(supported>0)
    c.atomic_json(out/'OBSERVATIONS_PREPARED.json',dict(scope='Loa observation inputs; not a transfer or inference pass',
        phase='Loa',release=a['release'],registry=b.sha(reg),method=b.sha(b.FREEZE),runner=b.sha(__file__),
        source=data['sha256'],randoms=[r['sha256'] for r in randoms],rows=written,cap_order=['SGC','NGC'],
        shell_edges=edges,counts=hist.tolist(),supported_counts=supported.tolist(),outputs=[
        c.file_record(p,content_hash=True) for p in (path,ap)],refit=False,passed=True))


if __name__=='__main__':run()
