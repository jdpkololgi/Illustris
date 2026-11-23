"""Read-only ph006/Loa input check, not DESI inference or full transfer validation."""
import json
import os
from pathlib import Path

import fitsio
import healpy as hp
import numpy as np

from workflows.abacus_tweb import p3br_build_random_response as response
from workflows.sbi import e2e_coupled_contract as c


def run():
    c.require_compute()
    cfg = c.config()['observation']
    if cfg['release'] != 'DA2/SecondGenMocks/AbacusSummitBGS_v2/altmtl/kibo-v1':
        raise ValueError('Unexpected mock input family')
    dev = c.paths('ph006')
    reg = json.loads((c.REPO/'configs/p10_response_sources_v1.json').read_text())['desi_candidate']
    files = {'mock': (dev['full'], dev['randoms'][0]),
             'loa': tuple(Path(p['path']) for p in (reg['data']['full'],reg['full_random'][0]))}
    stamps = lambda: {str(p):[p.stat().st_size,p.stat().st_mtime_ns] for pair in files.values() for p in pair}
    before=stamps()
    maps={}
    for name,(_,random) in files.items():
        arr=np.zeros((4,hp.nside2npix(256)),dtype='i8')
        response.add_random_file(arr,{'path':str(random)})
        maps[name]=arr
    common=(maps['mock']>0)&(maps['loa']>0)
    edges=[.01,.15,.25,.35,.45,.55]
    output={}
    for name,(data,_) in files.items():
        hist=np.zeros((2,4,5),dtype='i8')  # proxy and quality, domain, shell
        trace=[]
        with fitsio.FITS(data) as f:
            t=f[1]
            for start in range(0,t.get_nrows(),250000):
                b=t.read_slice(start,min(start+250000,t.get_nrows()))
                z=b['Z_not4clus']
                k=np.isfinite(z)&(z>=edges[0])&(z<edges[-1])&(b['ZWARN']==0)&((b['BGS_TARGET']&2)!=0)
                g=b[k]; z=z[k]
                dom=response.galactic_cap(g['RA'],g['DEC'])*2+response.photsys_code(g['PHOTSYS'])
                pix=hp.ang2pix(256,g['RA'],g['DEC'],lonlat=True)
                good=common[dom,pix]
                quality=(np.char.strip(g['SPECTYPE'].astype('U'))=='GALAXY')&(g['DELTACHI2']>=25) if name=='loa' else np.ones(len(g),bool)
                for i in range(4):
                    for j,sel in enumerate((good,good&quality)):
                        hist[j,i]+=np.histogram(z[sel&(dom==i)],edges)[0]
                for ix in np.flatnonzero(good&quality)[:max(0,10-len(trace))]:
                    trace.append([int(g['TARGETID'][ix]),float(z[ix]),int(g['ZWARN'][ix]),int(g['BGS_TARGET'][ix]),int(dom[ix])])
        assert len(trace)==10 and (hist[1]<=hist[0]).all()
        output[name]=dict(success_proxy=hist[0].tolist(),quality25=(hist[1].tolist() if name=='loa' else None),
                          ten_selected_rows=trace)
    assert before==stamps()
    report=dict(science_z_edges=edges,rows=output,source_stats=before,
        domain_order=['SGC/S','SGC/N','NGC/S','NGC/N'] if response.photsys_code(np.array(['S']))[0]==0 else ['SGC/N','SGC/S','NGC/N','NGC/S'],
        common_pixels=common.sum(1).tolist(), random_index=0, nside_ring=256,
        row_columns=['TARGETID','Z_not4clus','ZWARN','BGS_TARGET','cap_photsys_domain'],
        scope='Unweighted observed counts on shared random-occupied cap/PHOTSYS pixels. Single development phase and one random per release; not exact area, clustering, or equivalence.',
        transfer_qualified=False, script_sha256=c.sha256(__file__))
    out=c.ROOT.parent/('lowz_loa_input_check_'+os.environ['SLURM_JOB_ID'])
    out.mkdir(exist_ok=False)
    c.atomic_json(out/'INPUT_CHECK.json',report)
    print(out,flush=True)


if __name__=='__main__':run()
