"""Observation-only release adapter. Run from the checksum-bound deployment on a CPU node.

No training/development guard overrides. No target or other-campaign writes.
The output is native and factor-four response, not posterior predictions.
"""
import argparse
import hashlib
import json
import os
import socket
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BASE = Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v3')
OUT = BASE/'lowz_cfm_blind_confirmation_v1'
OLD = BASE/'coupled_20260918_v1/observations'
DEPLOY = BASE.parent/'coarse_source_61551798'
RELEASE = REPO/'configs/lowz_cfm_confirmation_release_v1.json'
FREEZE = BASE/'lowz_mock_dev_method_binding_v1/METHOD_FROZEN.json'
CONFIG = 'configs/e2e_coupled_data_lowz_20261029.json'


def check(ok, message):
    if not ok:
        raise ValueError(message)


def sha(p):
    digest = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def release(phase):
    spec = read(RELEASE)
    check(phase in ('ph018','ph019') and set(spec['receipts']) == {'ph018','ph019'}, 'phase not authorized')
    check(sha(FREEZE) == spec['method_sha256'] and sha(__file__) == spec['adapter_sha256'], 'release drift')
    art = read(FREEZE)['artifacts']
    check(sha(art['design']['path']) == art['design']['sha256'], 'design drift')
    design = read(art['design']['path'])
    check(design['confirmation_after_separate_freeze'] == ['ph018','ph019'] and
          design['draw_indices'] == [0,31] and design['caps'] == ['NGC','SGC'], 'wrong design')
    for name, expected in spec['receipts'][phase].items():
        check(sha(OLD/phase/name) == expected, 'receipt drift: '+name)
    return spec, art


def verify_record(rec, directory):
    path = Path(rec['path']).resolve()
    check(path.is_relative_to(directory.resolve()) and path.stat().st_size == rec['bytes']
          and sha(path) == rec['sha256'], 'payload identity: '+str(path))
    return path


def native_values(geometry, sl, angular, cap, plan, sel, boundary, counts, expected):
    import numpy as np
    from workflows.sbi import cfm_lowz_selection_response as low
    from workflows.sbi.build_cfm_matched_response import density_at
    vals,z=low.response_chunk(geometry,sl,angular,cap,plan,boundary=boundary)
    nbar,known=density_at(z,sel,cap)
    valid=known & vals['support_random'].astype(bool) & low.radial_mask(z,plan)
    vals.update(counts=counts,expected_counts_random=expected,ntilde_h3_mpc3=nbar,
        selection_valid=valid.astype('u1'),log_count_ratio_random=np.where(valid,np.log((counts+.001)/(expected+.001)),0.))
    for key in ('effective_volume','los_valid'): del vals[key]
    return vals


def dev_parity(plan,sel):
    """Exercise the adapter against fixed development windows BEFORE any holdout payload."""
    import h5py
    import numpy as np
    from workflows.sbi import cfm_lowz_selection_response as low, cfm_cic_quadrature as quad
    from workflows.sbi import e2e_coupled_coordinates as coord
    from workflows.abacus_tweb import p3br_build_random_response as random
    obs=coord.ROOT/'observations/ph006'
    pairs=read(coord.ROOT/'geometry/ph006/GEOMETRY_COMPLETE.json')['pairs']
    from workflows.sbi import build_cfm_matched_response as matched
    import fitsio
    from workflows.abacus_tweb import p3a_build_canonical_fields as grid
    for cap in ('NGC','SGC'):
        ref=read(obs/(cap+'_COMPLETE.json'))
        b=read(ref['matched_source']['path'])['binding']
        with np.load(b['angular']['payload']['path'],allow_pickle=False) as maps:
            a={k:maps[k] for k in ('domain','support','angular_response')}
        cat=fitsio.read(b['observed']['payload']['path'],columns=['RA','DEC','Z'])
        chosen,_=matched.source_point_mask(cat,a,cap,plan)
        xyz=coord.sky_mpc_h(*(cat[k][chosen] for k in ('RA','DEC','Z')))
        g=low.support_grid(a,cap,coord.config()['raw_cell_mpc_h'])
        dep,_=grid.cic_deposit(xyz,g)
        bd=random.angular_boundary_distance(a['support'].astype(bool)&(a['domain']//2==low.CAPS.index(cap)))
        with h5py.File(ref['outputs'][0]['path'],'r') as f:
            for r in (r for r in pairs if r['cap']==cap):
                start=np.asarray(r['center'])//32*32
                size=np.minimum(32,np.array(g.shape)-start)
                sl=tuple(slice(v,v+n) for v,n in zip(start,size))
                e=quad.project(np.array(g.origin)+start*g.cell_mpc,size,g.cell_mpc,a,cap,plan,sel,4)
                vals=native_values(g,sl,a,cap,plan,sel,bd,dep[sl],e)
                for k,v in vals.items():
                    check(np.array_equal(f[k][sl],v.astype(f[k].dtype)), 'development parity failed: '+cap+'/'+k)
        del dep,xyz,cat


def prepare(phase, caps, workers):
    spec, art = release(phase)
    check(socket.gethostname().startswith('nid') and os.environ.get('SLURM_JOB_ID'), 'compute node required')
    check(os.environ.get('GRAPHWEB_CFM_DATA_CONFIG') == CONFIG and Path.cwd() == DEPLOY, 'deployment environment')
    sys.path.insert(0, str(DEPLOY))
    for rel, digest in art['source_files'].items():
        check(sha(DEPLOY/rel) == digest, 'deployed source changed: '+rel)
    from workflows.sbi import cfm_lowz_selection_response as low
    from workflows.sbi import build_cfm_matched_response as matched
    from workflows.sbi import e2e_coupled_condition_products as products
    from workflows.sbi import e2e_coupled_coordinates as coord
    from workflows.abacus_tweb import p3br_build_random_response as random
    from workflows.abacus_tweb import p3a_build_canonical_fields as grid
    import fitsio
    import h5py
    import numpy as np
    plan, policy, _ = low.load_plan()
    selection_path = Path(read(DEPLOY/CONFIG)['observation']['fixed_selection_manifest'])
    sel = read(selection_path)
    check(sha(selection_path) == spec['selection_sha256'] and sel['fit_phases'] == policy['training']
          and sel['omitted_phase'] is None, 'not the frozen full-training selection')
    dev_parity(plan,sel)
    directory = OLD/phase
    observed = read(directory/'OBSERVED_COMPLETE.json')
    angular_dir = directory/'angular'
    ar = read(angular_dir/'ANGULAR_COMPLETE.json')
    check(all(r['pass'] and r['phase'] == phase and len(r['outputs']) == 1 for r in (observed,ar))
          and observed['target_columns_present'] is False and observed['source_code_sha256'] == spec['observed_builder']
          and ar['random_ids'] == list(range(18)) and len(ar['sources']) == 18, 'wrong lineage')
    counts = np.zeros((4,12*256**2), dtype=np.int64)
    mock = int(phase[2:])
    lss = Path('/global/cfs/cdirs/desi/survey/catalogs/DA2/mocks/SecondGenMocks/AbacusSummitBGS_v2')/f'altmtl{mock}/kibo-v1/mock{mock}/LSScats'
    for i, entry in enumerate(ar['sources']):
        path = angular_dir/f'random_{i:02d}.json'
        check(str(path) == entry['receipt'] and sha(path) == entry['sha256'], 'random receipt')
        r = read(path)
        source = lss/f'BGS_BRIGHT_{i}_full_HPmapcut.ran.fits'
        check(r['pass'] and r['phase'] == phase and r['source']['path'] == str(source) and
              r['source']['bytes'] == source.stat().st_size and not r['audit']['maskbits_filter_applied'], 'random source')
        with np.load(verify_record(r['outputs'][0],angular_dir), allow_pickle=False) as data:
            v = data['counts']
            check(v.dtype.kind in 'iu' and v.shape == counts.shape and (v>=0).all()
                  and v.sum() == r['audit']['accepted_rows'], 'invalid random counts')
            counts += v
    angular = random.normalized_map(counts)
    with np.load(verify_record(ar['outputs'][0],angular_dir), allow_pickle=False) as data:
        for key in ('raw_counts_by_domain','support','domain','angular_response'):
            check(np.array_equal(data[key],angular[key]), 'angular parity: '+key)
    src = lss/'BGS_BRIGHT_full_HPmapcut.dat.fits'
    check(str(src)==observed['source']['path'] and src.stat().st_size==observed['source']['bytes'], 'galaxy source')
    obs_path = verify_record(observed['outputs'][0],directory)
    rows = fitsio.read(obs_path,columns=['RA','DEC','Z'])
    check(len(rows)==observed['observed_rows'] and all(np.isfinite(rows[k]).all() for k in rows.dtype.names), 'galaxy rows')
    for cap in caps:
        dest = OUT/phase/cap
        dest.mkdir(parents=True,exist_ok=True)
        marker = dest/'RESPONSE_COMPLETE.json'
        bound = {'release':sha(RELEASE),'selection':sha(selection_path),'observed':observed['outputs'][0]['sha256'],
                 'angular':ar['outputs'][0]['sha256'],'phase':phase,'cap':cap}
        if marker.exists():
            rec=read(marker)
            check(rec['binding']==bound and rec['passed'], 'previous response identity')
            for v in rec['outputs']: verify_record(v,dest)
            continue
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
        result=dict(binding=bound,grid=g,audit=audit,conservation=conservation,angular_replay_equal=True,
                    outputs=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in (native,wide)],
                    low_z_rows=int(((selected['Z']>=.01)&(selected['Z']<.15)).sum()),
                    passed=True,job=os.environ['SLURM_JOB_ID'])
        with marker.open('x') as f: json.dump(result,f,sort_keys=True,allow_nan=False)
        print('RESPONSE_COMPLETE',phase,cap,flush=True)


if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('phase',choices=['ph018','ph019'])
    ap.add_argument('--caps',nargs='+',choices=['NGC','SGC'],default=['NGC','SGC'])
    ap.add_argument('--workers',type=int,choices=[1,2,4,8,16],default=16)
    ap.add_argument('--metadata-only',action='store_true')
    args=ap.parse_args()
    if args.metadata_only:
        release(args.phase)
        print('RELEASE_METADATA_OK',args.phase)
    else:
        prepare(args.phase,args.caps,args.workers)
