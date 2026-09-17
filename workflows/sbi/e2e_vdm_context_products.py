"""Exact block-averaged cubic R7 products for the controlled VDM matrix.

Coarse mass is computed by averaged interpolation operators, not by replacing
the target with a lowpass approximation. Observations and targets live in
separate files. Confirmation packaging performs no predictive scoring.
"""
from contextlib import ExitStack
import argparse
import gc
import os
from pathlib import Path
import socket
import time

import h5py
import numpy as np
from scipy import fft, sparse
from astropy.cosmology import Planck18

from workflows.sbi.e2e_vdm_context_data import (
    CONFIG, ROLES, guarded, read_json, spec, output_root, verify_geometry)
from workflows.sbi.e2e_field_build_products import require_compute, sha256, TENSOR_COMPONENTS
from workflows.sbi.e2e_field_error_budget import smoothed_spectrum, inverse_component
from workflows.sbi.e2e_field_wide_coarse import block_sum, padded_extract
from workflows.sbi.e2e_field_dataset import transformed
from workflows.sbi.e2e_wide_data import LOCAL_CHANNELS, COARSE_CHANNELS
from workflows.sbi.e2e_durable import publish_json, single_writer


def mean_pool(x, factor):
    x = np.asarray(x)
    d, h, w = x.shape
    if any(n % factor for n in x.shape):
        raise ValueError('block-aligned array required')
    return x.reshape(d//factor, factor, h//factor, factor, w//factor, factor).mean((1,3,5))


def averaged_cubic_operator(raw_starts, average, origin, c, ngrid):
    """Average exact cubic Lagrange point-samples before any downsampling.

    Each row averages `average` original 3.383 Mpc/h samples. Tensor products
    therefore reproduce the original fine chart, then its physical block means.
    Periodic matter sampling is distinct from nonperiodic observation padding.
    """
    points = np.asarray(raw_starts)[:,None]+np.arange(average)[None,:]+.5
    coordinates = ((origin+points*c['raw_cell_mpc'])*c['coordinate_h']+c['box_offset_mpc_h'])
    coordinates = np.mod(coordinates, c['box_mpc_h'])/(c['box_mpc_h']/ngrid)
    base = np.floor(coordinates).astype(np.int64)
    t = coordinates-base
    offsets = (-1,0,1,2)
    rr, cc, vv = [], [], []
    for offset in offsets:
        weight = np.ones_like(t)
        for other in offsets:
            if offset != other:
                weight *= (t-other)/(offset-other)
        rr.extend(np.repeat(np.arange(len(points)), average))
        cc.extend(((base+offset) % ngrid).ravel())
        vv.extend((weight/average).ravel())
    result = sparse.csr_matrix((vv,(rr,cc)), shape=(len(points),ngrid))
    result.sum_duplicates()
    if np.max(np.abs(np.asarray(result.sum(1)).ravel()-1)) > 1e-12:
        raise ValueError('interpolation operator fails DC preservation')
    return result


def sample_averaged(field, raw_starts, average, grid, c):
    operators = [averaged_cubic_operator(starts, average, grid['origin_mpc'][a], c, field.shape[0])
                 for a, starts in enumerate(raw_starts)]
    output = np.empty(tuple(op.shape[0] for op in operators), dtype=np.float64)
    x, y, z = operators
    for i in range(x.shape[0]):
        lo, hi = x.indptr[i:i+2]
        plane = np.zeros(field.shape[1:], dtype=np.float64)
        for index, weight in zip(x.indices[lo:hi], x.data[lo:hi]):
            plane += weight*field[index]
        output[i] = z.dot(y.dot(plane).T).T
    return output


def target_parent(field, row, c, side=48):
    starts = [centre-side+2*np.arange(side) for centre in row['center']]
    return sample_averaged(field, starts, 2, row['grid'], c)


def pair_rows(rows):
    result, pairs = [], []
    for phase in ('ph004', 'ph005'):
        for cap in ('NGC', 'SGC'):
            anchor = next(r for r in rows if r['phase']==phase and r['cap']==cap
                          and r['shell']==2 and r['support_stratum']=='interior')
            companion = dict(anchor, anchor_id=anchor['anchor_id']+'_adjacent_x', small_train=False,
                center=[anchor['center'][0]+32, *anchor['center'][1:]],
                parent_domain_anchor=anchor['anchor_id'], core_offset_raw=[32,0,0])
            result.append(companion)
            pairs.append(dict(domain=anchor['anchor_id'], cores=[anchor['anchor_id'], companion['anchor_id']],
                              phase=phase, cap=cap, independent_pair=False))
    return result, pairs


def verification_sources(source, c):
    index = read_json(c['spectral_source'])
    expected = {str(guarded(v['path'])):v['sha256'] for v in index['sources_verified']}
    # Original spectral sources retain the original verified observation inputs.
    paths = [source['response_file']['path']]+[v['path'] for v in source['virtual_sources']]
    verified = {}
    for name in sorted(set(paths)):
        path = guarded(name, source['phase'])
        checksum = expected.get(str(path))
        if checksum is None or sha256(path) != checksum:
            raise ValueError('observation/VDS source hash mismatch: '+str(path))
        verified[str(path)] = checksum
    return verified


def coarse_response(source, destination):
    """Raw factor4 cap grid; no periodic wrapping and explicit geometry validity."""
    factor = 4
    summed = ('counts', 'expected_counts_random')
    averaged = ('support_random','angular_response','exposure_apodized_random','ntilde_mpc3')
    shape = source['grid']['shape']
    with h5py.File(guarded(source['response_file']['path'], source['phase']), 'r') as src, \
         h5py.File(destination, 'x') as dst:
        target = tuple((n+factor-1)//factor for n in shape)
        for name in COARSE_CHANNELS[:8]:
            dst.create_dataset(name, shape=target, dtype='f4', chunks=True, compression='lzf')
        conservation = {name: [0.,0.] for name in summed}
        for i in range(0, shape[0], 16):
            stop = min(i+16, shape[0])
            sl, out = slice(i,stop), slice(i//4,(stop+3)//4)
            for name in (*summed,*averaged):
                x = src[name][sl]
                if not np.isfinite(x).all():
                    raise ValueError('nonfinite response')
                values = block_sum(x, 4)/(64 if name in averaged else 1)
                dst[name][out] = values
                if name in summed:
                    conservation[name][0] += float(x.sum(dtype=np.float64))
                    conservation[name][1] += float(values.astype('f4').sum(dtype=np.float64))
            valid = np.ones((stop-i,*shape[1:]), dtype=np.uint8)
            dst['geometry_valid_fraction'][out] = block_sum(valid,4)/64
            dst['log_count_ratio_random'][out] = np.log((dst['counts'][out]+.5)/(dst['expected_counts_random'][out]+.5))
        for name,(before,after) in conservation.items():
            if abs(before-after) > 2e-7*max(1.,abs(before)):
                raise ValueError('count conservation failure')
            dst.attrs[name+'_sum_in'] = before
            dst.attrs[name+'_sum_out'] = after


def observations(row, source, coarse_file, c):
    centre = np.asarray(row['center'])
    # Companion cores keep the designated common observation/coarse parent.
    context_centre = centre-np.asarray(row.get('core_offset_raw',[0,0,0]))
    start = centre-48
    xyz = [(row['grid']['origin_mpc'][a]+(start[a]+np.arange(96)+.5)*c['raw_cell_mpc']) for a in range(3)]
    radius = np.sqrt(xyz[0][:,None,None]**2+xyz[1][None,:,None]**2+xyz[2][None,None,:]**2)
    z_grid = np.linspace(0,.8,4001)
    r_grid = Planck18.comoving_distance(z_grid).value
    local = []
    with h5py.File(guarded(source['response_file']['path'], row['phase']), 'r') as raw:
        for name in LOCAL_CHANNELS:
            values = np.interp(radius, r_grid,z_grid) if name=='observer_redshift' else padded_extract(raw[name],start,96)
            local.append(mean_pool(transformed(name,values),2).astype('f4'))
        support = mean_pool(padded_extract(raw['support_random'], start,96),2)
    wide_start = context_centre//4-56
    positions = [(row['grid']['origin_mpc'][a]+(wide_start[a]+np.arange(112)+.5)*4*c['raw_cell_mpc']) for a in range(3)]
    axes = np.meshgrid(*positions,indexing='ij',sparse=True)
    radius = np.sqrt(sum(x*x for x in axes))
    extra = {name:np.broadcast_to(axis/np.maximum(radius,1e-30),radius.shape)
             for name,axis in zip(('los_x','los_y','los_z'),axes)}
    extra['observer_radius_mpc'] = radius
    wide = []
    with h5py.File(coarse_file,'r') as raw:
        for name in COARSE_CHANNELS:
            values = extra[name] if name in extra else padded_extract(raw[name],wide_start,112)
            wide.append(mean_pool(transformed(name,values),2).astype('f4'))
    return np.stack(local), np.stack(wide), support.astype('f4')


def build_phase(root, phase):
    require_compute()
    c = spec()
    if phase not in ROLES:
        raise PermissionError('phase excluded before opening any payload')
    root = output_root(root)
    geometry = read_json(root/'data/GEOMETRY.json')
    if not geometry['quotas_pass'] or geometry['config_sha256'] != sha256(CONFIG):
        raise ValueError('geometry gate missing or drifted')
    verify_geometry(geometry['rows'],c)
    companions,pairs = pair_rows(geometry['rows'])
    rows = [r for r in geometry['rows']+companions if r['phase']==phase]
    started = time.monotonic()
    folder = root/'data'/phase
    binding = dict(config_sha256=sha256(CONFIG), geometry_sha256=sha256(root/'data/GEOMETRY.json'),
                   builder_sha256=sha256(__file__))
    with single_writer(folder):
        done = folder/'COMPLETE.json'
        if done.exists():
            receipt = read_json(done)
            if receipt['binding'] != binding or any(sha256(f['path']) != f['sha256'] for f in receipt['files']):
                raise ValueError('completed phase is not an exact resume')
            print('VERIFIED_SKIP',phase,flush=True)
            return receipt
        if any(p.name!='WRITER.lock' for p in folder.iterdir()):
            raise FileExistsError('partial products preserved; explicit recovery required')
        screened = read_json(c['screened_source'])
        sources = [s for s in screened['sources'] if s['phase']==phase]
        native = next(s for s in read_json(c['native_source'])['phases'] if s['phase']==phase)
        density_path = guarded(native['density_path'],phase)
        if sha256(density_path) != native['density_sha256']:
            raise ValueError('native density source hash mismatch')
        verified = {str(density_path):native['density_sha256']}
        files = []
        for source in sources:
            verified.update(verification_sources(source,c))
            coarse_file = folder/f"{source['cap']}_factor4_response.h5"
            coarse_response(source,coarse_file)
            destination = folder/f"{source['cap']}_observations.h5"
            with h5py.File(destination,'x') as saved:
                saved.attrs.update(role=ROLES[phase],phase=phase,contains_matter=False)
                for row in [r for r in rows if r['cap']==source['cap']]:
                    local,wide,support = observations(row,source,coarse_file,c)
                    group = saved.create_group(row['anchor_id'])
                    for key,value in (('local',local),('wide_extended',wide),('support',support)):
                        group.create_dataset(key,data=value,compression='lzf')
            files.extend([coarse_file,destination])
            print('OBSERVATIONS_COMPLETE',phase,source['cap'],flush=True)
        target_file = folder/'targets.h5'
        print('NATIVE_R7_START',phase,flush=True)
        spectrum,mean = smoothed_spectrum(density_path,c['box_mpc_h'],c['smoothing_mpc_h'])
        with h5py.File(target_file,'x') as saved:
            saved.attrs.update(role=ROLES[phase],phase=phase,contains_observations=False)
            field = fft.irfftn(spectrum.copy(),s=(c['native_ngrid'],)*3,workers=32,overwrite_x=True)
            for source in sources:
                cap_rows = [r for r in rows if r['cap']==source['cap']]
                ordinary = [r for r in cap_rows if 'parent_domain_anchor' not in r]
                centres = np.array([r['center'] for r in ordinary])//8
                lo,hi = centres.min(0)-28,centres.max(0)+28
                starts = [np.arange(a,b)*8 for a,b in zip(lo,hi)]
                coarse = 1+sample_averaged(field,starts,8,source['grid'],c)
                if not np.isfinite(coarse).all() or np.min(coarse)<=0:
                    raise ValueError('coarse physical density not positive')
                for row in cap_rows:
                    group = saved.create_group(row['anchor_id'])
                    rho = 1+target_parent(field,row,c)
                    if not np.isfinite(rho).all() or np.min(rho)<=0:
                        raise ValueError('fine physical density not positive')
                    centre = np.asarray(row['center'])//8
                    start = centre-6-lo
                    from_coarse = coarse[tuple(slice(s,s+12) for s in start)]
                    error = float(np.max(np.abs(mean_pool(rho,4)-from_coarse)))
                    if error > 2e-6:
                        raise ValueError('exact coarse/fine mass target mismatch')
                    group.attrs['coarse_fine_mass_max_abs'] = error
                    group.create_dataset('rho_parent',data=rho,compression='lzf')
                    group.create_dataset('fullbox_tensor_core',shape=(16,16,16,6),dtype='f8')
                    if 'parent_domain_anchor' not in row:
                        begin = centre-28-lo
                        extended = coarse[tuple(slice(s,s+56) for s in begin)]
                        group.create_dataset('coarse_rho_extended',data=extended,compression='lzf')
                print('MASS_TARGETS_COMPLETE',phase,source['cap'],flush=True)
            del field
            gc.collect()
            for column,component in enumerate(TENSOR_COMPONENTS):
                field = inverse_component(spectrum,c['box_mpc_h'],component)
                for row in rows:
                    saved[row['anchor_id']]['fullbox_tensor_core'][...,column] = target_parent(field,row,c,side=16)
                del field
                gc.collect()
                print('TENSOR_COMPONENT_COMPLETE',phase,component,flush=True)
            del spectrum
            gc.collect()
            checks = []
            for row in rows:
                group = saved[row['anchor_id']]
                truth = group['rho_parent'][16:32,16:32,16:32]-1
                tensor = group['fullbox_tensor_core'][:]
                trace_error = float(np.max(np.abs(tensor[...,[0,3,5]].sum(-1)-truth)))
                if trace_error > 2e-6:
                    raise ValueError('fullbox tensor trace mismatch')
                checks.append(dict(anchor=row['anchor_id'], trace_max_abs=trace_error,
                                   mass_max_abs=float(group.attrs['coarse_fine_mass_max_abs'])))
        files.append(target_file)
        receipt = dict(binding=binding,phase=phase,role=ROLES[phase],rows=rows,
            files=[dict(path=str(f),sha256=sha256(f),bytes=f.stat().st_size) for f in files],
            verified_inputs=verified,checks=checks,pairs=[p for p in pairs if p['phase']==phase],
            science_scores_evaluated=False,representation_gate_pass=None,native_count_mean=mean,
            elapsed_seconds=time.monotonic()-started,job=os.environ['SLURM_JOB_ID'],node=socket.gethostname())
        publish_json(done,receipt)
        print('PHASE_COMPLETE',phase,receipt['elapsed_seconds'],flush=True)
        return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--phases',nargs='+',required=True,choices=list(ROLES))
    args = parser.parse_args()
    for phase in args.phases:
        build_phase(args.root,phase)


if __name__ == '__main__':
    main()
