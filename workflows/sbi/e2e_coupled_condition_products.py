"""Target-free rectangular condition shards in the verified Cartesian frame.

No matter path is opened here. Counts/expectations are summed on the intermediate
wide grid, transformed, then averaged, matching the previous information recipe.
Observation padding is zero/missing, never periodic like the matter cube.
"""
import argparse
import json
import time

import h5py
import numpy as np

from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_coordinates as coord
from workflows.sbi import e2e_coupled_geometry as geometry
from workflows.sbi import e2e_coupled_operators as op
from workflows.sbi import e2e_coupled_observations as obs
from workflows.sbi.e2e_field_wide_coarse import block_sum

LOCAL_CHANNELS = ('counts', 'support_random', 'angular_response',
    'exposure_apodized_random', 'expected_counts_random', 'log_count_ratio_random',
    'distance_to_support_boundary', 'ntilde_h3_mpc3', 'los_x', 'los_y', 'los_z',
    'observer_redshift')
WIDE_CHANNELS = ('counts', 'expected_counts_random', 'support_random',
    'angular_response', 'exposure_apodized_random', 'ntilde_h3_mpc3',
    'geometry_valid_fraction', 'log_count_ratio_random', 'los_x', 'los_y', 'los_z',
    'observer_radius_mpc_h')
SUMMED = ('counts', 'expected_counts_random')
AVERAGED = ('support_random', 'angular_response', 'exposure_apodized_random',
            'ntilde_h3_mpc3')
SCHEMA = 'e2e-coupled-condition-v1'


def source_binding():
    from workflows.sbi import e2e_field_wide_coarse as legacy
    return dict(layout_sha256=c.sha256(op.LAYOUT), builder_sha256=c.sha256(__file__),
                geometry_sha256=c.sha256(geometry.__file__),
                operators_sha256=c.sha256(op.__file__),
                block_sum_source_sha256=c.sha256(legacy.__file__),
                coordinates_source_sha256=c.sha256(coord.__file__))


def padded_extract(dataset, start, shape):
    start, shape = np.asarray(start, dtype=int), np.asarray(shape, dtype=int)
    if start.shape != (3,) or shape.shape != (3,) or np.any(shape <= 0):
        raise ValueError('positive rectangular 3D extraction required')
    lo, hi = np.maximum(start, 0), np.minimum(start+shape, dataset.shape)
    out = np.zeros(tuple(shape), dtype=np.float32)
    if np.all(hi > lo):
        out[tuple(slice(a,b) for a,b in zip(lo-start, hi-start))] = dataset[
            tuple(slice(a,b) for a,b in zip(lo,hi))]
    return out


def transformed(name, values):
    x = np.asarray(values, dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError('nonfinite observation channel')
    if name in SUMMED or name == 'ntilde_h3_mpc3':
        if np.any(x < 0):
            raise ValueError('negative count/expectation/number density')
        if name in SUMMED:
            return np.log1p(x)
        # Same numerical floor in physical units as the legacy Mpc^-3 recipe.
        return np.log(np.maximum(x, 1e-10 / c.config()['target']['coordinate_h']**3))
    return x


def radial_coordinates(grid, start, shape, stride=1):
    coord.validate_grid(grid)
    axes = [(grid['origin_mpc_h'][a]+(start[a]+np.arange(shape[a])+.5)*
             stride*grid['cell_mpc_h']) for a in range(3)]
    xyz = np.meshgrid(*axes, indexing='ij', sparse=True)
    radius = np.sqrt(sum(v*v for v in xyz))
    extra = {name:np.broadcast_to(v/np.maximum(radius,1e-30), radius.shape)
             for name,v in zip(('los_x','los_y','los_z'),xyz)}
    extra['observer_radius_mpc_h'] = radius
    return extra


def coarsen_response(source_path, destination, grid):
    """Conservative factor-four cap response; independent of targets/geometry."""
    coord.validate_grid(grid)
    shape = tuple(grid['shape'])
    with h5py.File(source_path, 'r') as src, h5py.File(destination, 'x') as dst:
        if (src.attrs.get('coordinate_sha256') != c.sha256(coord.CONFIG)
                or src.attrs.get('distance_unit') != 'Mpc/h'):
            raise ValueError('unverified source coordinate frame')
        target = tuple((n+3)//4 for n in shape)
        dst.attrs.update(schema=SCHEMA, contains_matter=False, factor=4,
                         coordinate_sha256=c.sha256(coord.CONFIG), distance_unit='Mpc/h',
                         padding='zero/missing; nonperiodic')
        for name in WIDE_CHANNELS[:8]:
            dst.create_dataset(name, shape=target, dtype='f4', chunks=True, compression='lzf')
        conservation = {name:[0.,0.] for name in SUMMED}
        for i in range(0, shape[0], 16):
            stop = min(i+16, shape[0]); sl = slice(i,stop); out = slice(i//4,(stop+3)//4)
            for name in (*SUMMED,*AVERAGED):
                values = src[name][sl]
                if not np.isfinite(values).all() or np.any(values < 0):
                    raise ValueError('invalid raw response')
                pooled = block_sum(values,4)/(64 if name in AVERAGED else 1)
                dst[name][out] = pooled
                if name in SUMMED:
                    conservation[name][0] += float(values.sum(dtype=np.float64))
                    conservation[name][1] += float(pooled.astype('f4').sum(dtype=np.float64))
            dst['geometry_valid_fraction'][out] = block_sum(
                np.ones((stop-i,*shape[1:]),dtype=np.uint8),4)/64
            dst['log_count_ratio_random'][out] = np.log(
                (dst['counts'][out]+.5)/(dst['expected_counts_random'][out]+.5))
        for name,(before,after) in conservation.items():
            if abs(before-after) > 2e-7*max(1.,abs(before)):
                raise ValueError('coarsened count conservation failed')
            dst.attrs[name+'_sum_in'],dst.attrs[name+'_sum_out'] = before,after
        dst.flush()
    obs.durable(destination)
    return conservation


def extract_pair(row, raw, coarse):
    """Transforms precede the final factor-two average, as in the old recipe."""
    cfg = op.layout(); center = np.asarray(row['center'], dtype=int)
    joint_shape = tuple(cfg['joint_density_shape'])
    start = center+cfg['joint_raw_start_from_first_center']
    shape = tuple(2*n for n in joint_shape)
    local_z = coord.redshift(radial_coordinates(row['grid'],start,shape)['observer_radius_mpc_h'])
    local = []
    for name in LOCAL_CHANNELS:
        values = local_z if name == 'observer_redshift' else padded_extract(raw[name],start,shape)
        local.append(op.mean_pool(transformed(name,values),2).astype('f4'))
    support = op.mean_pool(padded_extract(raw['support_random'],start,shape),2).astype('f4')
    midpoint = center+cfg['joint_midpoint_from_first_center']
    wide_raw = midpoint+cfg['wide_raw_start_from_joint_midpoint']
    if np.any(wide_raw % 4):
        raise ValueError('wide context is not on the cap factor-four grid')
    wide_start = wide_raw//4
    wide_shape = tuple(2*n for n in cfg['wide_extended_shape'])
    extra = radial_coordinates(row['grid'],wide_start,wide_shape,stride=4)
    wide = []
    for name in WIDE_CHANNELS:
        values = extra[name] if name in extra else padded_extract(coarse[name],wide_start,wide_shape)
        wide.append(op.mean_pool(transformed(name,values),2).astype('f4'))
    result = dict(joint=np.stack(local),wide_extended=np.stack(wide),support=support)
    validate_arrays(result)
    return result


def validate_arrays(arrays):
    cfg=op.layout()
    shapes=dict(joint=(len(LOCAL_CHANNELS),*cfg['joint_density_shape']),
                wide_extended=(len(WIDE_CHANNELS),*cfg['wide_extended_shape']),
                support=tuple(cfg['joint_density_shape']))
    if set(arrays)!=set(shapes):
        raise ValueError('unexpected condition payload keys')
    for name,shape in shapes.items():
        if arrays[name].shape!=shape or not np.isfinite(arrays[name]).all():
            raise ValueError('invalid condition shape/finiteness: '+name)
    if np.min(arrays['support'])<0 or np.max(arrays['support'])>1:
        raise ValueError('invalid fractional observation support')


def build(phase, limit=None):
    c.require_compute(); c.phase_guard(phase); coord.bind(); coord.require_host_checks()
    source = coord.ROOT/'geometry'/phase/'GEOMETRY_COMPLETE.json'
    geo = coord.verify_receipt(source, payload=False)
    geometry.verify_rows(geo['pairs'],phase)
    directory=coord.ROOT/'conditions'/phase
    binding=dict(**source_binding(),geometry_receipt_sha256=c.sha256(source))
    with c.single_writer(directory):
        marker=directory/'CONDITIONS_COMPLETE.json'
        if marker.exists():
            record=coord.verify_receipt(marker)
            if record['binding']!=binding:
                raise ValueError('condition builder/binding drift')
            for item in record['pair_receipts']:
                if c.sha256(item['path'])!=item['sha256']:
                    raise ValueError('pair receipt drift')
                coord.verify_receipt(item['path'])
            return record
        records=[]; built=0
        for cap in ('NGC','SGC'):
            source_path=coord.ROOT/'observations'/phase/f'{cap}_COMPLETE.json'
            if c.sha256(source_path)!=geo['sources'][cap]['sha256']:
                raise ValueError('observation receipt differs from geometry source')
            raw=coord.verify_receipt(source_path)
            cap_binding=dict(**binding,response_receipt_sha256=c.sha256(source_path))
            cap_marker=directory/f'{cap}_FACTOR4.json'
            if cap_marker.exists():
                cap_record=coord.verify_receipt(cap_marker)
                if cap_record['binding']!=cap_binding:
                    raise ValueError('coarse condition source drift')
            else:
                output=directory/f'{cap}_factor4_generation_{time.time_ns()}.h5'
                qa=coarsen_response(raw['outputs'][0]['path'],output,raw['grid'])
                cap_record=dict(**coord.provenance(),phase=phase,cap=cap,binding=cap_binding,
                    conservation=qa,outputs=[c.file_record(output,content_hash=True)],**{'pass':True})
                c.atomic_json(cap_marker,cap_record)
            with h5py.File(raw['outputs'][0]['path'],'r') as fine, \
                 h5py.File(cap_record['outputs'][0]['path'],'r') as wide:
                for row in (r for r in geo['pairs'] if r['cap']==cap):
                    pair_marker=directory/f"{row['pair_id']}.json"
                    pair_binding=dict(**cap_binding,coarse_receipt_sha256=c.sha256(cap_marker),
                                      row_sha256=c.digest(row))
                    if pair_marker.exists():
                        record=coord.verify_receipt(pair_marker)
                        if record['binding']!=pair_binding:
                            raise ValueError('condition pair source drift')
                    else:
                        if limit is not None and built>=limit:
                            return dict(phase=phase,partial=True,new_pairs=built)
                        arrays=extract_pair(row,fine,wide)
                        output=directory/f"{row['pair_id']}_generation_{time.time_ns()}.h5"
                        with h5py.File(output,'x') as saved:
                            saved.attrs.update(schema=SCHEMA,phase=phase,role=c.ROLES[phase],
                                pair_id=row['pair_id'],contains_matter=False,
                                local_channels=json.dumps(LOCAL_CHANNELS),wide_channels=json.dumps(WIDE_CHANNELS),
                                distance_unit='Mpc/h',coordinate_sha256=c.sha256(coord.CONFIG))
                            for name,values in arrays.items():
                                saved.create_dataset(name,data=values,compression='lzf',shuffle=True)
                            saved.flush()
                        obs.durable(output)
                        record=dict(**coord.provenance(),phase=phase,pair_id=row['pair_id'],
                            binding=pair_binding,contains_matter=False,
                            outputs=[c.file_record(output,content_hash=True)],**{'pass':True})
                        c.atomic_json(pair_marker,record)
                        built+=1
                        print(json.dumps(dict(phase=phase,pair=row['pair_id'],new_pairs=built)),flush=True)
                    records.append(dict(path=str(pair_marker),sha256=c.sha256(pair_marker)))
        if len(records)!=len(geo['pairs']):
            raise ValueError('incomplete condition panel')
        result=dict(**coord.provenance(),phase=phase,role=c.ROLES[phase],binding=binding,
                    pair_receipts=records,contains_matter=False,outputs=[],**{'pass':True})
        c.atomic_json(marker,result)
        return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',required=True); p.add_argument('--limit',type=int)
    a=p.parse_args(); print(json.dumps(build(a.phase,a.limit)),flush=True)
