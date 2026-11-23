"""Register blind regions from support only; replay the development panel first."""
import os
import socket
import sys
import json
from pathlib import Path
import lowz_confirmation_inputs as bind


def panel(phase, inputs):
    from workflows.sbi.cfm_geometry_validation_cache import cached_grid_validation
    with cached_grid_validation():
        return _panel(phase, inputs)


def _panel(phase, inputs):
    import numpy as np
    import h5py
    from workflows.sbi import e2e_coupled_geometry as g
    from workflows.sbi import prepare_cfm_lowz_geometry_candidates as low
    from workflows.sbi import prepare_cfm_lowz_campaign as campaign
    buckets = {}
    for cap in ('NGC','SGC'):
        grid, native, angular = inputs[cap]
        cfg = g.c.config()['geometry']
        rng = np.random.default_rng(g.address(cfg['seed'],phase,cap,'pair-geometry'))
        centers = np.column_stack([rng.integers(2,(n-m)//8+1,cfg['candidates_per_cap'])*8
                                   for n,m in zip(grid['shape'],(96,64,64))])
        with h5py.File(native,'r') as f:
            support = f['support_random'][:].astype(bool)
        bind.check(list(support.shape)==grid['shape'],'native dimensions')
        regular = {(s,k):[] for s in range(4) for k in ('interior','boundary')}
        for key in dict.fromkeys(map(tuple,centers.tolist())):
            center=np.array(key)
            xyz=np.asarray(grid['origin_mpc_h'])+(center+[16,0,0])*grid['cell_mpc_h']
            z=float(g.coord.redshift(np.linalg.norm(xyz)))
            shell=int(np.searchsorted((.15,.25,.35,.45,.55),z,side='right')-1)
            if not 0<=shell<4: continue
            frac=[float(support[tuple(slice(v-16,v+16) for v in center+[dx,0,0])].mean()) for dx in (0,32)]
            if min(frac)<.25: continue
            kind='interior' if min(frac)>=.95 else 'boundary'
            mid,cores=g.domain_positions(center,grid)
            regular[shell,kind].append(dict(phase=phase,role=g.c.ROLES[phase],cap=cap,
                center=list(key),grid=grid,shell=shell,redshift=z,support_stratum=kind,
                science_core_support_fractions=frac,source_midpoint_mpc_h=mid.tolist(),
                source_owned_core_centers_mpc_h=cores.tolist()))
        del support
        with np.load(angular,allow_pickle=False) as a:
            extra,_=low.candidates(phase,cap,{'path':str(native)},{k:a[k] for k in ('support','domain')})
        for kind in ('interior','boundary'):
            rows=(dict(r,role=g.c.ROLES[phase],shell=0) for r in extra if r['support_stratum']==kind)
            regular[0,kind]=sorted(rows,key=lambda r:(r['redshift'],tuple(r['center'])))+regular[0,kind]
        buckets.update({(cap,*k):v for k,v in regular.items()})
    rows=campaign.select_panel(phase,buckets)
    bind.check(len(rows)==16 and all(any(r['cap']==c and r['redshift']<.15 for r in rows)
                                     for c in ('NGC','SGC')),'panel incomplete')
    return rows


def run():
    bind.check(socket.gethostname().startswith('nid') and os.environ.get('SLURM_JOB_ID'),'compute required')
    bind.check(Path.cwd()==bind.DEPLOY and os.environ.get('GRAPHWEB_CFM_DATA_CONFIG')==bind.CONFIG,'deployment')
    sys.path.insert(0,str(bind.DEPLOY))
    _,art=bind.release('ph018')
    for rel,digest in art['source_files'].items():
        bind.check(bind.sha(bind.DEPLOY/rel)==digest,'deployed code drift')
    from workflows.sbi import e2e_coupled_coordinates as coord
    obs=coord.ROOT/'observations/ph006'
    dev={}
    for cap in ('NGC','SGC'):
        ref=bind.read(obs/(cap+'_COMPLETE.json'))
        angular=bind.read(ref['matched_source']['path'])['binding']['angular']['payload']['path']
        dev[cap]=(ref['grid'],ref['outputs'][0]['path'],angular)
    bind.check(panel('ph006',dev)==bind.read(coord.ROOT/'geometry/ph006/GEOMETRY_COMPLETE.json')['pairs'],
               'full development panel mismatch')
    result={}
    for phase in ('ph018','ph019'):
        spec,_=bind.release(phase)
        inputs={}; manifests={}
        ar=bind.OLD/phase/'angular'
        angular=bind.verify_record(bind.read(ar/'ANGULAR_COMPLETE.json')['outputs'][0],ar)
        for cap in ('NGC','SGC'):
            d=bind.OUT/phase/cap
            p=d/'RESPONSE_COMPLETE.json'
            r=bind.read(p); b=r['binding']
            bind.check(r['passed'] and r['angular_replay_equal'] and b['release']==bind.sha(bind.RELEASE)
                       and b['phase']==phase and b['cap']==cap and b['selection']==spec['selection_sha256']
                       and b['angular']==bind.sha(angular),'response binding')
            native=bind.verify_record(r['outputs'][0],d)
            bind.check(native==d/'NATIVE.h5','wrong native path')
            inputs[cap]=(r['grid'],native,angular); manifests[cap]=bind.sha(p)
        result[phase]=dict(response_receipts=manifests,pairs=panel(phase,inputs))
    record=dict(release=bind.sha(bind.RELEASE),adapter=bind.sha(__file__),development_parity='ph006',
                phases=result,truth_read=False,job=os.environ['SLURM_JOB_ID'])
    out=bind.OUT/'GEOMETRY_FROZEN.json'
    tmp=out.with_suffix('.'+record['job']+'.tmp')
    with tmp.open('x') as f: json.dump(record,f,sort_keys=True,allow_nan=False)
    os.link(tmp,out)
    tmp.unlink()
    print('GEOMETRY_FROZEN',bind.sha(out),flush=True)


if __name__=='__main__': run()
