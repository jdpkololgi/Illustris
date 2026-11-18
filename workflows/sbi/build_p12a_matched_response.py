"""Build P12 source-matched counts and a full/OOF reusable CIC response basis.

Separate immutable products; never changes the historical P3 raw fields. The
three basis arrays are label-free angular/radial geometry, not fitted densities.
Every full/OOF density is bound separately and evaluated linearly by the loader.
No identity, host-fold or training-readiness claim is made here.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
from pathlib import Path
import time

import numpy as np

from workflows.sbi import prepare_p12a_lowz_inputs as p12
from workflows.sbi import p12a_cic_response_basis as basis
from workflows.sbi.build_cfm_matched_response import source_point_mask
from workflows.sbi.prepare_cfm_lowz_inputs import require_compute,save_json,sha256


_WORK=None


def initialize(angular,cap,edges,valid,origin,cell):
    global _WORK
    _WORK=(angular,cap,edges,valid,np.asarray(origin),cell)


def chunk(bounds):
    angular,cap,edges,valid,origin,cell=_WORK
    start=np.array([a for a,b in bounds]);shape=np.array([b-a for a,b in bounds])
    weights,bins=basis.project_basis(origin+start*cell,shape,cell,angular,cap,edges,valid)
    return bounds,weights,bins


def selection_contract(root,policy):
    directory=root/'point_selection_candidates_v1'
    complete=p12.read(directory/'CANDIDATES_COMPLETE.json')
    if complete['pipeline']!='p12' or complete['policy']!=policy:
        raise ValueError('Point-selection campaign policy mismatch')
    fits={};edges=None;valid=None
    for omitted in [None,*policy['training']]:
        name='full' if omitted is None else 'oof_'+omitted
        path=directory/(name+'.json');value=p12.read(path)
        expected=[p for p in policy['training'] if p!=omitted]
        if (sha256(path)!=complete['fits'][name] or value['fit_phases']!=expected
                or value['omitted_phase']!=omitted or value['density_unit']!='Mpc^-3'):
            raise ValueError('Full/OOF selection authority failure')
        bins=np.asarray(value['bin_edges'],float);known=np.asarray(value['valid'],bool)
        if edges is None:edges=bins;valid=known
        if not np.array_equal(edges,bins) or not np.array_equal(valid,known):
            raise ValueError('Response basis requires identical full/OOF bin support')
        for phase,digest in value['statistics_sha256'].items():
            if phase not in expected or sha256(directory/(phase+'.json'))!=digest:
                raise ValueError('Selection statistics changed')
        if set(value['statistics_sha256'])!=set(expected):
            raise ValueError('Selection statistics phase membership differs')
        fits[name]=dict(path=str(path),sha256=sha256(path))
    return p12.read(directory/'full.json'),fits,edges,valid


def build(phase,caps=(0,1),workers=16):
    require_compute()
    plan,_,policy=p12.load_plan()
    if phase not in policy['training']+policy['development_and_calibration']:
        raise PermissionError('Phase outside visible preparation scope')
    if workers not in (1,2,4,8,16) or not caps or not set(caps)<={0,1}:
        raise ValueError('Invalid CPU/cap request')
    import fitsio
    import h5py
    from astropy.cosmology import Planck18
    from workflows.abacus_tweb.p3a_build_canonical_fields import GridSpec,cic_deposit,iter_chunks
    root=Path(plan['output_root']);phase_root=root/phase
    fields_path=phase_root/'p3_fields/field_manifest.json';fields=p12.read(fields_path)
    raw=p12.read(phase_root/'RAW_PREPARATION_COMPLETE.json')
    if not raw['pass_raw_preparation'] or sha256(fields_path)!=raw['fields_sha256']:
        raise ValueError('Unqualified P12 raw fields')
    p12.require_matching_domains(fields,plan)
    p1_path=phase_root/'p1_canonical/manifest.json';parent=p12.read(p1_path)
    if sha256(p1_path)!=fields['p1_manifest_sha256'] or sha256(parent['parent'])!=parent['parent_sha256']:
        raise ValueError('P12 parent source drift')
    angular_path=Path(fields['angular_support']['source'])
    if sha256(angular_path)!=fields['angular_support']['sha256']:
        raise ValueError('P12 random support changed')
    with np.load(angular_path,allow_pickle=False) as saved:
        angular={k:saved[k] for k in ('support','domain','angular_response')}
    selection,fits,edges,valid=selection_contract(root,policy)
    radii=Planck18.comoving_distance(edges).value
    rows=fitsio.read(parent['parent'],columns=['RA','DEC','Z'])
    source_marker=Path(parent['source_completion_marker'])
    if sha256(source_marker)!=parent['source_completion_marker_sha256']:
        raise ValueError('P1 catalogue completion binding changed')
    source_completion=p12.read(source_marker)
    if (not source_completion['pass'] or source_completion['phase']!=phase
            or source_completion['geometry']['units']!='Mpc'
            or source_completion['geometry']['cosmology']!='Astropy Planck18'
            or source_completion['artifacts']['points']!=parent['points']):
        raise ValueError('P1 coordinate authority mismatch')
    points=np.load(parent['points'],mmap_mode='r')
    if len(points)!=len(rows) or sha256(parent['points'])!=source_completion['artifacts']['points_sha256']:
        raise ValueError('P12 point identity changed')
    radial_plan=dict(context_redshift=[0.,.6],excluded_sentinel_redshift=[.585,.595])
    output=phase_root/'matched_cic_response_v1';output.mkdir(exist_ok=True)
    binding=dict(raw_fields_sha256=sha256(fields_path),p1_sha256=sha256(p1_path),
        angular_sha256=sha256(angular_path),selection_fits=fits,
        builder_sha256=sha256(__file__),basis_sha256=sha256(basis.__file__),
        source_point_mask_sha256=sha256(Path(__file__).with_name('build_cfm_matched_response.py')))
    for cap in caps:
        name=('SGC','NGC')[cap];marker=output/(name+'_COMPLETE.json')
        if marker.exists():
            old=p12.read(marker)
            if old['binding']!=binding or sha256(old['output']['path'])!=old['output']['sha256']:
                raise ValueError('Existing P12 response changed')
            continue
        path=output/(name+'_response.h5')
        if path.exists():raise FileExistsError('Partial response requires review: '+str(path))
        started=time.monotonic();source=fields['components'][name];grid=source['grid']
        origin=np.asarray(grid['origin_mpc'],float);shape=tuple(grid['shape']);cell=float(grid['cell_mpc'])
        basis.validate_geometry(radii,cell)
        if sha256(source['file'])!=source['file_sha256']:
            raise ValueError('P12 raw field changed')
        mask,count_qa=source_point_mask(rows,angular,name,radial_plan)
        counts,cic_qa=cic_deposit(np.asarray(points[mask,:3],float),GridSpec(tuple(origin),shape,cell,0.))
        if cic_qa['lost_weight']!=0 or not np.isclose(counts.sum(dtype=float),mask.sum(),rtol=2e-6):
            raise ValueError('P12 supported source count conservation failed')
        bounds=[tuple((s.start,s.stop) for s in sl) for sl in iter_chunks(shape,tuple(min(n,32) for n in shape))]
        arguments=(angular,cap,radii,valid[cap],origin,cell)
        pool=None
        if workers==1:
            initialize(*arguments);results=map(chunk,bounds)
        else:
            pool=ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
                                     initializer=initialize,initargs=arguments)
            results=pool.map(chunk,bounds,chunksize=1)
        sums=np.zeros(3);expected_sum=0.
        try:
            with h5py.File(source['file'],'r') as old,h5py.File(path,'x') as saved:
                saved.attrs.update(schema=basis.SCHEMA,phase=phase,cap=name,distance_unit='Mpc',
                    cosmology='Planck18',quadrature_factor=4,contains_matter=False,
                    cell_mpc=cell,origin_mpc=origin,model_inputs_ready=False)
                for key in ('exposure_binary','exposure_apodized','los_x','los_y','los_z'):
                    saved[key]=h5py.ExternalLink(str(Path(source['file']).resolve()),'/'+key)
                spec=dict(shape=shape,chunks=tuple(min(n,32) for n in shape),compression='lzf',shuffle=True)
                out={key:saved.create_dataset(key,dtype='i2' if key=='radial_bin' else 'f4',**spec)
                     for key in ('counts','basis_0','basis_1','basis_2','radial_bin')}
                for ordinal,(bb,weights,bins) in enumerate(results):
                    slices=tuple(slice(a,b) for a,b in bb)
                    out['counts'][slices]=counts[slices];out['radial_bin'][slices]=bins
                    for i in range(3):out['basis_'+str(i)][slices]=weights[i]
                    sums+=weights.sum(axis=(1,2,3))
                    expected_sum+=basis.expected_counts(weights,bins,selection['density'][cap],valid[cap]).sum()
                    if ordinal%100==0:print('P12_BASIS',phase,name,ordinal,len(bounds),flush=True)
                saved.flush()
        finally:
            if pool is not None:pool.shutdown(wait=True,cancel_futures=True)
        if sha256(__file__)!=binding['builder_sha256'] or sha256(basis.__file__)!=binding['basis_sha256']:
            raise ValueError('P12 response implementation changed')
        save_json(marker,dict(schema=basis.SCHEMA,phase=phase,cap=name,binding=binding,
            output=dict(path=str(path),sha256=sha256(path),bytes=path.stat().st_size),
            external_geometry=dict(path=source['file'],sha256=source['file_sha256']),
            radius_edges_mpc=radii.tolist(),valid_bins=valid[cap].tolist(),grid=grid,
            count_source_audit=count_qa,cic_audit=cic_qa,basis_volume_sums_mpc3=sums.tolist(),
            full_expected_count_sum=float(expected_sum),elapsed_seconds=time.monotonic()-started,
            job=os.environ['SLURM_JOB_ID'],geometry_basis_ready=True,
            model_inputs_ready=False,contains_matter=False))
        print('P12_BASIS_COMPLETE',phase,name,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--phase',required=True)
    parser.add_argument('--caps',nargs='+',type=int,choices=(0,1),default=[0,1])
    parser.add_argument('--workers',type=int,choices=(1,2,4,8,16),default=16)
    args=parser.parse_args();build(args.phase,args.caps,args.workers)
