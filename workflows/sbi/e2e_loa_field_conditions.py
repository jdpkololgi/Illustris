"""Shared Loa count/coarse-response cache with nonperiodic local observations."""
import argparse
import json
from pathlib import Path
import time
import h5py
import numpy as np
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_coordinates as coord
from workflows.sbi import e2e_coupled_observations as obs
from workflows.sbi import e2e_coupled_condition_products as products
from workflows.sbi import e2e_coupled_conditions as conditions
from workflows.sbi import e2e_coupled_operators as op
from workflows.sbi.e2e_field_wide_coarse import block_sum
from workflows.sbi.e2e_cfm_field_gallery import LOA,compare_channels
from workflows.abacus_tweb import p3a_build_canonical_fields as grid_ops
from workflows.abacus_tweb import p3br_build_random_response as response_ops


def load_response(cap):
    with np.load(LOA/'random_angular.npz') as f:
        support=f['support'].astype(bool)&(f['domain']//2==(1 if cap=='NGC' else 0))
        response=f['angular_response'] if 'angular_response' in f.files else f['response']
    with np.load(LOA/'boundary_angles.npz') as f:boundary=f[cap]
    selection_path=Path(c.config()['observation']['fixed_selection_manifest'])
    selection=json.loads(selection_path.read_text())
    return support,response,boundary,selection


def cache_cap(root,cap):
    c.require_compute();plan=json.loads((root/'PLAN.json').read_text());grid=plan['grids'][cap]
    marker=root/(cap+'_CACHE.json')
    if marker.exists():return coord.verify_receipt(marker)
    with np.load(root/'catalogue.npz') as f:xyz=f['xyz'][f['cap']==(1 if cap=='NGC' else 0)]
    spec=grid_ops.GridSpec(tuple(grid['origin_mpc_h']),tuple(grid['shape']),3.383,grid['padding_mpc_h'])
    counts,audit=grid_ops.cic_deposit(xyz,spec);del xyz
    if audit['lost_weight']!=0:raise ValueError('global cap CIC lost observations')
    if abs(counts.sum(dtype=np.float64)-audit['input_points'])>2e-6*audit['input_points']:
        raise ValueError('global CIC count conservation failed')
    support,response,boundary,selection=load_response(cap)
    path=root/(cap+'_observation_cache.h5');coarse_shape=tuple((n+3)//4 for n in spec.shape)
    totals={name:[0.,0.] for name in products.SUMMED};start=time.monotonic()
    with h5py.File(path,'x') as f:
        f.attrs.update(distance_unit='Mpc/h',coordinate_sha256=c.sha256(coord.CONFIG),contains_matter=False)
        raw=f.create_dataset('counts_raw',data=counts,chunks=(32,64,64),compression='lzf',shuffle=True)
        wide=f.create_group('wide')
        for name in products.WIDE_CHANNELS[:8]:
            wide.create_dataset(name,shape=coarse_shape,dtype='f4',chunks=True,compression='lzf',shuffle=True)
        for i,sl in enumerate(grid_ops.iter_chunks(spec.shape,(32,128,128))):
            values,_=obs.response_chunk(spec,sl,support,response,boundary,selection,cap)
            values['counts']=counts[sl]
            out=tuple(slice(s.start//4,(s.stop+3)//4) for s in sl)
            for name in (*products.SUMMED,*products.AVERAGED):
                pooled=block_sum(values[name],4)/(64 if name in products.AVERAGED else 1)
                wide[name][out]=pooled
                if name in totals:
                    totals[name][0]+=float(values[name].sum(dtype=np.float64))
                    totals[name][1]+=float(pooled.astype('f4').sum(dtype=np.float64))
            wide['geometry_valid_fraction'][out]=block_sum(np.ones(values['counts'].shape,dtype='u1'),4)/64
            wide['log_count_ratio_random'][out]=np.log((wide['counts'][out]+.5)/(wide['expected_counts_random'][out]+.5))
            if i%100==0:print(json.dumps(dict(cap=cap,response_chunk=i,seconds=time.monotonic()-start)),flush=True)
        f.flush()
    for before,after in totals.values():
        if abs(before-after)>2e-7*max(1.,before):raise ValueError('coarsened count conservation failed')
    record=dict(cap=cap,grid=grid,cic=audit,conservation=totals,elapsed=time.monotonic()-start,
        source=c.file_record(__file__,content_hash=True),outputs=[c.file_record(path,content_hash=True)],**{'pass':True})
    c.atomic_json(marker,record);return record


def build(grid,midpoint,cap,counts,coarse,response_data):
    """Exactly the established transform-before-pool information recipe."""
    midpoint=np.asarray(midpoint,dtype=int);start=midpoint-[64,48,48];shape=(128,96,96)
    if np.any(midpoint%8):raise ValueError('unaligned field tile')
    spec=grid_ops.GridSpec(tuple(grid['origin_mpc_h']),tuple(grid['shape']),3.383,grid['padding_mpc_h'])
    raw_counts=products.padded_extract(counts,start,shape)
    support,response,boundary,selection=response_data
    joint=np.empty((12,64,48,48),dtype='f4')
    for i in range(0,128,16):
        local=(slice(i,i+16),slice(0,96),slice(0,96))
        sl=tuple(slice(int(a+s.start),int(a+s.stop)) for a,s in zip(start,local))
        values,z=obs.response_chunk(spec,sl,support,response,boundary,selection,cap)
        values['counts']=raw_counts[local]
        values['log_count_ratio_random']=grid_ops.log_count_ratio(values['counts'],values['expected_counts_random'],
                                                                  values['exposure_apodized_random'],1e-3,1e-4)
        axes=[np.arange(s.start,s.stop) for s in sl]
        valid=((axes[0][:,None,None]>=0)&(axes[0][:,None,None]<spec.shape[0]) &
               (axes[1][None,:,None]>=0)&(axes[1][None,:,None]<spec.shape[1]) &
               (axes[2][None,None,:]>=0)&(axes[2][None,None,:]<spec.shape[2]))
        for name in values:values[name]=np.where(valid,values[name],0)
        values['observer_redshift']=z
        for j,name in enumerate(products.LOCAL_CHANNELS):
            joint[j,i//2:(i+16)//2]=op.mean_pool(products.transformed(name,values[name]),2)
    wide_start=(midpoint-192)//4;extra=products.radial_coordinates(grid,wide_start,(96,96,96),stride=4)
    wide=np.stack([op.mean_pool(products.transformed(name,extra[name] if name in extra else
                    products.padded_extract(coarse[name],wide_start,(96,96,96))),2).astype('f4')
                    for name in products.WIDE_CHANNELS])
    if not np.isfinite(joint).all() or not np.isfinite(wide).all():raise ValueError('nonfinite condition')
    return dict(joint=joint,wide=wide,support=joint[1].copy())


def golden_checks(root):
    """Both-cap mock replay, then independent Loa direct-builder replay."""
    checks={};phase='ph016'
    geo=coord.verify_receipt(coord.ROOT/'geometry'/phase/'GEOMETRY_COMPLETE.json',payload=False)
    angular_record=c.verify_receipt(c.ROOT/'observations'/phase/'angular/ANGULAR_COMPLETE.json')
    with np.load(angular_record['outputs'][0]['path']) as f:angular={k:f[k] for k in f.files}
    selection=json.loads(Path(c.config()['observation']['fixed_selection_manifest']).read_text())
    for cap in ('NGC','SGC'):
        pair=f'{phase}_{cap}_s0_interior_00';row=next(r for r in geo['pairs'] if r['pair_id']==pair)
        raw=coord.verify_receipt(coord.ROOT/'observations'/phase/(cap+'_COMPLETE.json'),payload=False)
        coarse=coord.verify_receipt(coord.ROOT/'conditions'/phase/(cap+'_FACTOR4.json'),payload=False)
        # Large immutable source checksums were qualified upstream; verify pinned receipts and conditions here.
        support=angular['support'].astype(bool)&(angular['domain']//2==(1 if cap=='NGC' else 0))
        response_data=(support,angular['angular_response'],response_ops.angular_boundary_distance(support),selection)
        with h5py.File(raw['outputs'][0]['path'],'r') as r,h5py.File(coarse['outputs'][0]['path'],'r') as w:
            rebuilt=build(row['grid'],np.asarray(row['center'])+[16,0,0],cap,r['counts'],w,response_data)
        original=conditions.crop_context(conditions.load_pair(phase,pair),phase)
        checks[cap]=compare_channels(rebuilt,original)
    plan=json.loads((root/'PLAN.json').read_text())
    old=Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/cfm_gallery_20260930_v2')
    receipt=json.loads((old/'COMPLETE.json').read_text());mid=receipt['loa']['midpoint']
    with h5py.File(root/'NGC_observation_cache.h5','r') as f:
        rebuilt=build(plan['grids']['NGC'],mid,'NGC',f['counts_raw'],f['wide'],load_response('NGC'))
    with np.load(old/'loa_conditions.npz') as f:original={k:f[k] for k in f.files}
    checks['Loa_NGC']=compare_channels(rebuilt,original)
    record=dict(checks=checks,passed=all(r['passed'] for rows in checks.values() for r in rows),
                source=c.file_record(__file__,content_hash=True))
    c.atomic_json(root/'CONDITION_CHECKS.json',record);print(json.dumps(record),flush=True)
    if not record['passed']:raise ValueError('golden condition parity failed')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    c.require_compute()
    for cap in ('NGC','SGC'):cache_cap(a.root,cap)
    golden_checks(a.root)
