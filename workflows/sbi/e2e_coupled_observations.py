"""Target-free P3b observations in the verified DESI/Abacus Mpc/h frame."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import time

import fitsio
import h5py
import healpy as hp
import numpy as np
from scipy.ndimage import gaussian_filter

from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_coordinates as coord
from workflows.abacus_tweb import p3a_build_canonical_fields as grid_ops
from workflows.abacus_tweb import p3br_build_random_response as response_ops


def angular_maps(phase, directory):
    """One immutable receipt per random file; never rerun finished streams."""
    directory.mkdir(parents=True,exist_ok=True)
    final = directory/'ANGULAR_COMPLETE.json'
    if final.exists():
        return c.verify_receipt(final)
    counts = np.zeros((4,hp.nside2npix(256)),dtype=np.int64)
    sources = []
    for index,path in enumerate(c.paths(phase)['randoms']):
        marker = directory/f'random_{index:02d}.json'
        if marker.exists():
            record = c.verify_receipt(marker)
            current = c.file_record(path)
            if any(current[k] != record['source'][k] for k in ('bytes','mtime_ns','path')):
                raise ValueError('random source drift')
            with np.load(record['outputs'][0]['path']) as data:
                delta = data['counts']
        else:
            before = c.file_record(path,content_hash=True)
            delta = np.zeros_like(counts)
            audit = response_ops.add_random_file(delta,before)
            if c.file_record(path) != {k:before[k] for k in ('path','bytes','mtime_ns')}:
                raise ValueError('random source mutated during build')
            output = publish_npz(directory, f'random_{index:02d}', counts=delta)
            record = dict(**c.provenance(),phase=phase,source=before,audit=audit,
                          outputs=[c.file_record(output,content_hash=True)],**{'pass':True})
            c.atomic_json(marker,record)
        counts += delta
        sources.append(dict(receipt=str(marker),sha256=c.sha256(marker)))
        print(json.dumps(dict(phase=phase,random_complete=index)),flush=True)
    result = response_ops.normalized_map(counts)
    output = publish_npz(directory, 'angular', **{k:v for k,v in result.items() if k != 'metadata'})
    record = dict(**c.provenance(),phase=phase,random_ids=list(range(18)),sources=sources,
                  metadata=result['metadata'],legacy_algorithm_sha256=c.sha256(response_ops.__file__),
                  outputs=[c.file_record(output,content_hash=True)],**{'pass':True})
    c.atomic_json(final,record)
    return record


def durable(path):
    with open(path, 'rb') as stream:
        os.fsync(stream.fileno())


def publish_npz(directory, prefix, **values):
    # Only a committed receipt makes a generation visible. A crash before
    # its receipt leaves an unreferenced generation, not a poisoned canonical
    # filename. Retain orphan generations for audit/storage accounting.
    output = directory/f'{prefix}_generation_{time.time_ns()}.npz'
    with output.open('xb') as stream:
        np.savez_compressed(stream, **values)
        stream.flush(); os.fsync(stream.fileno())
    return output


def observed_points(path):
    source = fitsio.read(path,columns=['RA','DEC','Z'])
    z = source['Z']
    mask = (z >= .10) & (z < .60) & ~((z >= .585) & (z < .595))
    source = source[mask]
    xyz = coord.sky_mpc_h(source['RA'], source['DEC'], source['Z'])
    cap = response_ops.galactic_cap(source['RA'],source['DEC'])
    return xyz,cap


def response_chunk(spec,slices,cap_support,response,boundary,selection,cap_name):
    halo = 5  # 4.0596 Mpc/h apodization / 3.383 Mpc/h cells, as before.
    gx,gy,gz = grid_ops.coordinate_block(spec,slices,halo=halo)
    shape = tuple(sl.stop-sl.start+2*halo for sl in slices)
    radius = np.sqrt(gx*gx+gy*gy+gz*gz)
    safe = np.maximum(radius,1e-12)
    pix = hp.vec2pix(256,np.broadcast_to(gx,shape)/safe,
                    np.broadcast_to(gy,shape)/safe,np.broadcast_to(gz,shape)/safe,nest=False)
    redshift = coord.redshift(radius)
    radial = (redshift >= .10) & (redshift < .60) & ~((redshift >= .585)&(redshift < .595))
    binary_ext = radial & cap_support[pix]
    apod_ext = gaussian_filter(binary_ext.astype(np.float32),sigma=1.2,
                               mode='constant',cval=0.,truncate=4.)
    trim = tuple(slice(halo,halo+sl.stop-sl.start) for sl in slices)
    support = binary_ext[trim]
    apod = np.clip(apod_ext[trim],0.,1.).astype(np.float32)
    radius,pix,redshift = radius[trim],pix[trim],redshift[trim]
    angular = response[pix].astype(np.float32)*support
    curve = selection['rotations']['0']['caps'][cap_name]
    nbar = np.zeros_like(redshift)
    valid = ((redshift>=selection['cosmology']['redshift_grid'][0]) &
             (redshift<=selection['cosmology']['redshift_grid'][-1]) & (apod>1e-4))
    nbar[valid] = (np.interp(redshift[valid],curve['grid_z'],curve['ntilde'])*
                  coord.selection_volume_jacobian(redshift[valid],selection))
    nbar = nbar.astype(np.float32)
    expected = (nbar.astype(np.float64)*spec.cell_mpc**3*angular.astype(np.float64)*apod.astype(np.float64)).astype(np.float32)
    distances = [np.abs(radius-coord.radius_mpc_h(z)) for z in (.10,.60,.585,.595)]
    boundary_distance = np.minimum(boundary[pix].astype(np.float64)*radius,np.minimum.reduce(distances))
    boundary_distance = boundary_distance.astype(np.float32)*support
    gx,gy,gz = grid_ops.coordinate_block(spec,slices)
    safe = np.maximum(radius,1e-12)
    return dict(support_random=support.astype(np.uint8),angular_response=angular,
                exposure_apodized_random=apod,expected_counts_random=expected,
                distance_to_support_boundary=boundary_distance,ntilde_h3_mpc3=nbar,
                los_x=np.broadcast_to(gx/safe,safe.shape).astype(np.float32),
                los_y=np.broadcast_to(gy/safe,safe.shape).astype(np.float32),
                los_z=np.broadcast_to(gz/safe,safe.shape).astype(np.float32)),redshift


def build_cap(phase,cap_name,xyz,angular,selection,selection_record,directory):
    marker = directory/f'{cap_name}_COMPLETE.json'
    if marker.exists():
        return coord.verify_receipt(marker)
    started = time.monotonic()
    spec = grid_ops.grid_from_xyz(xyz,coord.config()['raw_cell_mpc_h'],coord.config()['padding_mpc_h'])
    counts,audit = grid_ops.cic_deposit(xyz,spec)
    if audit['lost_weight'] != 0 or abs(float(counts.sum(dtype=np.float64))-len(xyz)) > 2e-6*len(xyz):
        raise ValueError('observed CIC count conservation failed')
    cap_id = 1 if cap_name == 'NGC' else 0
    support = angular['support'].astype(bool) & (angular['domain']//2 == cap_id)
    boundary = response_ops.angular_boundary_distance(support)
    shape = tuple(min(64,n) for n in spec.shape)
    output = directory/f'{cap_name}_response_generation_{time.time_ns()}.h5'
    totals = {str(i):dict(observed=0.,expected=0.,support_voxels=0) for i in range(4)}
    with h5py.File(output,'x') as out:
        out.attrs.update(schema='e2e-coupled-observations-v2',phase=phase,cap=cap_name,
                         origin_mpc_h=spec.origin,shape=spec.shape,cell_mpc_h=spec.cell_mpc,
                         coordinate_sha256=c.sha256(coord.CONFIG),distance_unit='Mpc/h',
                         density_unit='(Mpc/h)^-3',axis_order='ix,iy,iz')
        fields = {}
        for i,slices in enumerate(grid_ops.iter_chunks(spec.shape,shape)):
            values,z = response_chunk(spec,slices,support,angular['angular_response'],boundary,selection,cap_name)
            values['counts'] = counts[slices]
            values['log_count_ratio_random'] = grid_ops.log_count_ratio(
                values['counts'],values['expected_counts_random'],values['exposure_apodized_random'],1e-3,1e-4)
            if not fields:
                fields = {k:out.create_dataset(k,shape=spec.shape,dtype=v.dtype,chunks=shape,
                            compression='lzf',shuffle=True,fillvalue=0) for k,v in values.items()}
            if not all(np.isfinite(v).all() for v in values.values()):
                raise ValueError('nonfinite response')
            # LOS is defined everywhere, including outside survey support.
            for k,v in values.items():
                if np.any(v):
                    fields[k][slices] = v
            for j,(lo,hi) in enumerate(zip((.15,.25,.35,.45),(.25,.35,.45,.55))):
                mask = (z >= lo)&(z < hi)
                totals[str(j)]['observed'] += float(values['counts'][mask].sum(dtype=np.float64))
                totals[str(j)]['expected'] += float(values['expected_counts_random'][mask].sum(dtype=np.float64))
                totals[str(j)]['support_voxels'] += int(values['support_random'][mask].sum())
            if i%100 == 0:
                print(json.dumps(dict(phase=phase,cap=cap_name,response_chunk=i)),flush=True)
        for alias,key in (('exposure_binary','support_random'),('exposure_apodized','exposure_apodized_random'),
                          ('expected_counts','expected_counts_random'),('log_count_ratio','log_count_ratio_random')):
            out[alias] = fields[key]
        out.flush()
    durable(output)
    closure = {key:value['observed']/value['expected'] if value['expected']>0 else None
               for key,value in totals.items()}
    gates = dict(count_conservation=True,all_finite=True,
                 cap_shell_closure_within_25pct=all(v is not None and .75<=v<=1.25 for v in closure.values()))
    record = dict(**coord.provenance(),phase=phase,cap=cap_name,grid=coord.grid_record(spec),
                  source_code_sha256=c.sha256(__file__),cic=audit,selection=selection_record,
                  shell_totals=totals,observed_expected_ratio=closure,gates=gates,
                  outputs=[c.file_record(output,content_hash=True)],
                  elapsed_seconds=time.monotonic()-started,**{'pass':all(gates.values())})
    c.atomic_json(marker,record)
    if not record['pass']:
        raise ValueError(f'cap response qualification failed: {closure}')
    return record


def build(phase):
    c.require_compute(); c.phase_guard(phase); coord.bind(); coord.require_host_checks()
    directory = coord.ROOT/'observations'/phase
    with c.single_writer(directory):
        upstream = c.ROOT/'observations'/phase
        observed = c.verify_receipt(upstream/'OBSERVED_COMPLETE.json')
        with c.single_writer(upstream/'angular'):
            angular_record = angular_maps(phase,upstream/'angular')
        with np.load(angular_record['outputs'][0]['path']) as f:
            angular = {k:f[k] for k in f.files}
        selection_path = Path(c.config()['observation']['fixed_selection_manifest'])
        selection = json.loads(selection_path.read_text())
        if (not selection['pass'] or set(selection['fit_phases']) != {'ph000','ph002','ph003','ph004','ph005'}):
            raise ValueError('inherited selection fit provenance changed')
        selection_record = c.file_record(selection_path,content_hash=True)
        selection_record['fit_phases'] = selection['fit_phases']
        xyz,cap = observed_points(observed['outputs'][0]['path'])
        records = [build_cap(phase,name,xyz[cap==index],angular,selection,selection_record,directory)
                   for index,name in ((1,'NGC'),(0,'SGC'))]
        return dict(phase=phase,caps=records)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',required=True)
    args=p.parse_args()
    print(json.dumps(build(args.phase)),flush=True)
