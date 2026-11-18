"""Full-range observed-only response with point selection and matched CIC.

Counts and intensity both use the same source-point random-footprint selection.
Old geometry channels are immutable external links with a bound source hash.
Each cap commits separately. No target access, model fit, or readiness promotion.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
from pathlib import Path
import time

import numpy as np

from workflows.sbi import cfm_lowz_selection_response as lowz
from workflows.sbi import e2e_coupled_coordinates as coord
from workflows.sbi import cfm_cic_quadrature as quadrature
from workflows.sbi.prepare_cfm_lowz_local_condition_canary import density_at
from workflows.sbi.prepare_cfm_lowz_inputs import require_compute, save_json, sha256

_WORK = None


def initialize(angular, cap, plan, selection, origin, cell):
    global _WORK
    _WORK = (angular, cap, plan, selection, np.asarray(origin), cell)
    coord.table()


def project_chunk(bounds):
    angular, cap, plan, selection, origin, cell = _WORK
    start = np.array([x[0] for x in bounds])
    shape = np.array([x[1]-x[0] for x in bounds])
    result = quadrature.project(origin+start*cell, shape, cell, angular, cap, plan, selection, 4)
    return bounds, result


@contextmanager
def projected_chunks(bounds, workers, arguments):
    if workers == 1:
        initialize(*arguments)
        yield map(project_chunk, bounds)
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                initializer=initialize, initargs=arguments) as pool:
            yield pool.map(project_chunk, bounds, chunksize=1)


def source_point_mask(rows, angular, cap, plan):
    import healpy as hp
    from workflows.abacus_tweb.p3br_build_random_response import galactic_cap
    radial = lowz.radial_mask(rows['Z'], plan)
    in_cap = galactic_cap(rows['RA'], rows['DEC']) == lowz.CAPS.index(cap)
    pixels = hp.ang2pix(256, rows['RA'], rows['DEC'], lonlat=True)
    support = angular['support'].astype(bool) & (angular['domain']//2 == lowz.CAPS.index(cap))
    selected = radial & in_cap & support[pixels]
    return selected, dict(context_cap_rows=int((radial & in_cap).sum()),
                         supported_source_rows=int(selected.sum()),
                         unsupported_source_rows=int((radial & in_cap & ~support[pixels]).sum()))


def build(phase, caps, workers):
    require_compute()
    if workers not in (1, 2, 4, 8, 16) or not caps or not set(caps) <= set(lowz.CAPS):
        raise ValueError('Invalid cap/worker request')
    plan, policy, source = lowz.load_plan()
    lowz.phase_guard(phase, policy)
    import h5py
    import fitsio
    from workflows.abacus_tweb.p3a_build_canonical_fields import GridSpec, cic_deposit, coordinate_block, iter_chunks
    root = Path(plan['output_root'])
    source_receipt = root/'responses_full'/phase/'RESPONSE_PREPARED.json'
    receipt = lowz.read_json(source_receipt)
    if receipt['phase'] != phase or not receipt['label_free']:
        raise ValueError('Invalid source response')
    selection_path = root/'point_selection_candidates_v1/full.json'
    selection = lowz.read_json(selection_path)
    if selection['fit_phases'] != policy['training'] or selection['omitted_phase'] is not None:
        raise PermissionError('Full expected response requires all and only training phases')
    observed_path, observed_binding = lowz.source_payload(Path(source['root']), phase,
        'OBSERVED_COMPLETE.json', plan['source_campaign']['sha256'], True)
    angular_path, angular_binding = lowz.source_payload(Path(source['root']), phase,
        'angular/ANGULAR_COMPLETE.json', plan['source_campaign']['sha256'])
    with np.load(angular_path, allow_pickle=False) as data:
        angular = {key:data[key] for key in ('support','domain','angular_response')}
    rows = fitsio.read(observed_path, columns=['RA','DEC','Z'])
    if not all(np.isfinite(rows[k]).all() for k in ('RA','DEC','Z')):
        raise ValueError('Nonfinite observations')
    directory = root/'responses_matched_cic_v1'/phase
    directory.mkdir(parents=True, exist_ok=True)
    binding = dict(selection_sha256=sha256(selection_path), source_response_sha256=sha256(source_receipt),
                   builder_sha256=sha256(__file__), quadrature_sha256=sha256(quadrature.__file__),
                   point_selection_reader_sha256=sha256(Path(__file__).with_name('prepare_cfm_lowz_local_condition_canary.py')),
                   observed=observed_binding, angular=angular_binding)
    for cap in caps:
        started = time.monotonic()
        marker = directory/(cap+'_COMPLETE.json')
        if marker.exists():
            old = lowz.read_json(marker)
            if old['binding'] != binding or sha256(old['output']['path']) != old['output']['sha256']:
                raise ValueError('Completed cap binding/payload changed')
            continue
        output = directory/(cap+'_response.h5')
        if output.exists():
            raise FileExistsError('Incomplete output requires explicit review: '+str(output))
        raw_record = receipt['outputs'][lowz.CAPS.index(cap)]
        raw_path = Path(raw_record['path'])
        if sha256(raw_path) != raw_record['sha256']:
            raise ValueError('Original raw response payload changed')
        with h5py.File(raw_path, 'r') as raw:
            shape = raw['counts'].shape
            origin = np.asarray(raw.attrs['origin_mpc_h'])
            cell = float(raw.attrs['cell_mpc_h'])
            if raw.attrs['phase'] != phase or raw.attrs['cap'] != cap:
                raise ValueError('Cap identity mismatch')
        mask, count_audit = source_point_mask(rows, angular, cap, plan)
        spec = GridSpec(tuple(origin), tuple(shape), cell, 0.)
        points = coord.sky_mpc_h(rows['RA'][mask], rows['DEC'][mask], rows['Z'][mask])
        counts, cic_audit = cic_deposit(points, spec)
        if cic_audit['lost_weight'] != 0 or not np.isclose(counts.sum(dtype=float), mask.sum(), rtol=2e-6):
            raise ValueError('Supported-source CIC conservation failed')
        del points
        bounds = [tuple((s.start,s.stop) for s in sl) for sl in iter_chunks(shape, tuple(min(n,32) for n in shape))]
        print(phase, cap, 'start', dict(shape=shape, chunks=len(bounds), workers=workers, **count_audit), flush=True)
        sums = dict(expected=0., counts=float(counts.sum(dtype=float)))
        with projected_chunks(bounds, workers, (angular,cap,plan,selection,origin,cell)) as results:
            with h5py.File(raw_path,'r') as raw, h5py.File(output,'x') as saved:
                saved.attrs.update(schema='cfm-lowz-matched-cic-response-v1',phase=phase,cap=cap,
                    origin_mpc_h=origin,cell_mpc_h=cell,distance_unit='Mpc/h',contains_matter=False,
                    coordinate_sha256=sha256(coord.CONFIG),model_inputs_ready=False,
                    selection_sha256=binding['selection_sha256'],quadrature_factor=4,
                    source_point_policy='positive-z nonsentinel, cap and angular random support for both counts and expectation')
                replaced={'counts','expected_counts_random','ntilde_h3_mpc3','selection_valid','log_count_ratio_random'}
                for key in raw:
                    if key not in replaced:
                        saved[key] = h5py.ExternalLink(str(raw_path.resolve()), '/'+key)
                saved['counts_all_observed'] = h5py.ExternalLink(str(raw_path.resolve()), '/counts')
                datasets={key:saved.create_dataset(key,shape=shape,dtype='u1' if key=='selection_valid' else 'f4',
                    chunks=tuple(min(n,32) for n in shape),compression='lzf',shuffle=True) for key in replaced}
                for ordinal,(bb,expected) in enumerate(results):
                    slices=tuple(slice(a,b) for a,b in bb)
                    axes=coordinate_block(spec,slices)
                    z=coord.redshift(np.sqrt(sum(v*v for v in axes)))
                    nbar,known=density_at(z,selection,cap)
                    support=raw['support_random'][slices].astype(bool)
                    valid=known & support & lowz.radial_mask(z,plan)
                    values=dict(counts=counts[slices],expected_counts_random=expected,
                                ntilde_h3_mpc3=nbar,selection_valid=valid.astype('u1'),
                                log_count_ratio_random=np.where(valid,np.log((counts[slices]+.001)/(expected+.001)),0.))
                    if not all(np.isfinite(v).all() for v in values.values()) or np.any(expected<0):
                        raise ValueError('Nonfinite/negative matched response')
                    for key,value in values.items():datasets[key][slices]=value
                    sums['expected']+=float(expected.sum())
                    if ordinal%100==0:print(phase,cap,'chunk',ordinal,'of',len(bounds),'elapsed',round(time.monotonic()-started,1),flush=True)
                saved.flush()
        if (sha256(selection_path)!=binding['selection_sha256'] or sha256(__file__)!=binding['builder_sha256']
                or sha256(quadrature.__file__)!=binding['quadrature_sha256']):
            raise ValueError('Response implementation changed during build')
        save_json(marker,dict(schema='cfm-lowz-matched-cic-cap-v1',phase=phase,cap=cap,binding=binding,
            output=dict(path=str(output),sha256=sha256(output),bytes=output.stat().st_size),
            external_source=raw_record,grid=dict(origin_mpc_h=origin.tolist(),shape=list(shape),cell_mpc_h=cell),
            count_source_audit=count_audit,cic_audit=cic_audit,sums=sums,elapsed_seconds=time.monotonic()-started,
            job=os.environ['SLURM_JOB_ID'],model_inputs_ready=False,contains_matter=False,**{'pass':True}))
        print('MATCHED_CIC_CAP_COMPLETE',phase,cap,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',required=True)
    parser.add_argument('--caps',nargs='+',choices=lowz.CAPS,default=list(lowz.CAPS))
    parser.add_argument('--workers',type=int,choices=(1,2,4,8,16),default=8)
    args=parser.parse_args();build(args.phase,args.caps,args.workers)
