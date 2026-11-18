"""Recover low/high branch host namespaces against native cleaned halo tables.

Read-only scientific diagnostic, with an optional qualified host-map output.
Never deduplicate galaxies. The public generator filters resolved low-z halos
by M>=1e11 and observer distance chi(0.16), preserving table order. Test that
mapping against every selected mass and central position, not only a sample.
Unresolved pseudo-hosts remain a separate namespace and must not occur in the
high branch. No fitted tolerance or nearest-neighbour substitution is allowed.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import time

import numpy as np

from workflows.sbi import prepare_p12a_lowz_inputs as p12
from workflows.sbi import cfm_lowz_selection_response as lowz
from workflows.sbi.prepare_cfm_lowz_inputs import require_compute, save_json, sha256
from workflows.abacus_tweb.galaxy_position_mapping import reconstruct_native, periodic_delta, comoving_distance

NATIVE=Path('/global/cfs/cdirs/desi/public/cosmosim/AbacusSummit')
GENERATOR_COMMIT='a40a5d3be570a6ce9d5a0698268b04864d4d3af2'
POSITION_TOLERANCE=.005
MASS_TOLERANCE=1e-5


def native_filter_rows(positions, mass, radius):
    """Preserve the generator's native position dtype and strict radial cut."""
    shifted=np.array(positions,copy=True)
    if shifted.ndim!=2 or shifted.shape[1]!=3 or not np.issubdtype(shifted.dtype,np.floating):
        raise ValueError('Native floating point positions required')
    shifted+=1000.
    shifted[shifted>1000.]-=2000.
    shifted[shifted< -1000.]+=2000.
    distance=np.sum(shifted**2,axis=1)**.5
    return np.flatnonzero((np.asarray(mass)>=10.)&(distance<radius))


def check_mapping(rows, positions, mass, native_xyz, mapping, native_radius=None):
    indices=np.asarray(rows['HALO_INDEX'],dtype=np.int64)
    in_range=(indices>=0)&(indices<len(mapping))
    mapped=np.full(len(rows),-1,dtype=np.int64)
    mapped[in_range]=mapping[indices[in_range]]
    mass_ok=np.zeros(len(rows),bool)
    mass_ok[in_range]=np.abs(mass[mapped[in_range]]/rows['HALO_MASS'][in_range]-1)<MASS_TOLERANCE
    central=rows['CEN']==1
    position_ok=np.zeros(len(rows),bool);position_ok[~central]=True
    checked=central&in_range
    residual=np.linalg.norm(periodic_delta(positions[checked],native_xyz[mapped[checked]]),axis=1)
    position_ok[checked]=residual<POSITION_TOLERANCE
    satellites=(~central)&in_range
    satellite_bad=0
    if native_radius is not None:
        offset=np.linalg.norm(periodic_delta(positions[satellites],native_xyz[mapped[satellites]]),axis=1)
        within=offset<=np.asarray(native_radius)[mapped[satellites]]+POSITION_TOLERANCE
        position_ok[satellites]=within
        satellite_bad=int((~within).sum())
    qa=dict(rows=len(rows),out_of_range=int((~in_range).sum()),mass_mismatches=int((~mass_ok).sum()),
        centrals=int(central.sum()),central_position_mismatches=int((central&~position_ok).sum()),
        satellite_radius_mismatches=satellite_bad,satellite_radius_checked=native_radius is not None,
        central_position_max_mpc_h=float(residual.max(initial=0.)),
        passed=bool(np.all(in_range&mass_ok&position_ok)))
    return mapped,qa


def qualified_filter_radius(certificate):
    """Accept only an unfitted, source-bound generator-background calculation."""
    from workflows.sbi.prepare_p12a_generator_distance import EXPECTED_INI, COMMIT, CLASS_VERSION, CLONE_ADDITIONS, NBODYKIT_COMMIT, CLASSYLSS_TAG, CLASSYLSS_PATCHES
    path=Path(certificate)
    record=json.loads(path.read_text())
    if (record.get('schema')!='p12a-generator-distance-v1'
            or record.get('parameters')!=EXPECTED_INI
            or record.get('generator_commit')!=COMMIT
            or record.get('class_version')!=CLASS_VERSION
            or record.get('clone_additions')!=CLONE_ADDITIONS
            or record.get('nbodykit_commit')!=NBODYKIT_COMMIT
            or record.get('classylss_tag')!=CLASSYLSS_TAG
            or record.get('classylss_patches')!=list(CLASSYLSS_PATCHES)
            or record.get('catalogue_data_used') is not False
            or record.get('fitted_radius') is not False):
        raise ValueError('Unqualified generator distance certificate')
    measured=record['measurement']
    radius=float(measured['distance_mpc_h'])
    if (measured['redshift']!=.16 or measured['h']!=.6736
            or abs(measured['omega_ncdm']-.0006442)>1e-12
            or measured['Omega_Lambda']!=0. or not .6<measured['Omega_fld']<.8
            or not np.isfinite(radius) or not 460.<radius<462.
            or abs(radius-measured['conformal_difference_mpc_h'])>1e-8):
        raise ValueError('Wrong generator background')
    for item in [*record['sources'],record['normalized_ini'],record['helper'],record['binary'],record['preparer']]:
        if sha256(item['path'])!=item['sha256']:
            raise ValueError('Generator distance source changed: '+item['path'])
    return radius,dict(path=str(path),sha256=sha256(path),measurement=measured,
        solver_selection=record['solver_selection'],fitted_radius=False)


def run(phase, distance_certificate=None):
    require_compute();plan,_,policy=p12.load_plan();lowz.phase_guard(phase,policy)
    import fitsio
    from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog
    root=Path(plan['output_root'])/phase
    destination=root/('native_host_mapping_class263_v2' if distance_certificate else 'native_host_mapping_v1')
    if distance_certificate:
        radius,radius_source=qualified_filter_radius(distance_certificate)
    else:
        radius=float(comoving_distance(np.array([.16]))[0])
        radius_source=dict(method='historical Astropy approximation; failed ph001 diagnostic')
    if destination.exists():raise FileExistsError('Existing diagnostic requires explicit review: '+str(destination))
    destination.mkdir()
    started=time.monotonic();p1path=root/'p1_canonical/manifest.json';p1=p12.read(p1path)
    for path,digest in ((p1['parent'],p1['parent_sha256']),(p1['index'],p1['index_sha256'])):
        if sha256(path)!=digest:raise ValueError('P1 identity source changed')
    table=fitsio.read(p1['parent'],columns=['TARGETID','SOURCE_ROW','RA','DEC','Z_COSMO','FILE_NUM',
                                           'BOX_INDEX','HALO_INDEX','RES','CEN','HALO_MASS'])
    with np.load(p1['index'],allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['targetid'],table['TARGETID'])
        active=saved['active'].astype(bool)&saved['valid_target'].astype(bool)
    ids=np.flatnonzero(active);rows=table[ids]
    xyz=reconstruct_native(rows['RA'],rows['DEC'],rows['Z_COSMO'])
    low=rows['BOX_INDEX']<0;resolved=rows['RES']==1
    # Pseudo-hosts cannot cross into a purely resolved cubic branch.
    namespace_qa=dict(high_unresolved=int((~low&~resolved).sum()),
        low_resolved_below_mass_cut=int((low&resolved&(rows['HALO_MASS']<10.)).sum()),
        low_unresolved_above_mass_cut=int((low&~resolved&(rows['HALO_MASS']>10.)).sum()))
    native_rows=np.full(len(rows),-1,dtype=np.int64)
    diagnostics=[];sources=[]
    for slab in np.unique(rows['FILE_NUM'][resolved]):
        path=NATIVE/f'AbacusSummit_base_c000_{phase}'/'halos/z0.200/halo_info'/f'halo_info_{int(slab):03d}.asdf'
        before=path.stat()
        cat=CompaSOHaloCatalog(path,fields=['N','x_L2com'],subsamples=False,cleaned=True,convert_units=True,verbose=False)
        if cat.header['SimName']!=f'AbacusSummit_base_c000_{phase}':raise ValueError('Wrong native simulation')
        native_positions=np.asarray(cat.halos['x_L2com'])
        native_xyz=np.asarray(native_positions,float)
        mass=np.asarray(cat.halos['N'],float)*float(cat.header['ParticleMassHMsun'])/1e10
        # The documented generator draws its NFW satellite radius in [0,R_SO].
        # Convert its mean-density definition to comoving Mpc/h; this checks
        # proposed fixed-index hosts, never chooses a nearest-neighbour host.
        mean_density=2.77536627e11*float(cat.header['Omega_M'])
        native_radius=(3.*mass*1e10/(4*np.pi*float(cat.header['SODensityL1'])*mean_density))**(1./3.)
        keep=native_filter_rows(native_positions,mass,radius)
        for branch,mapping in ((False,np.arange(len(mass),dtype=np.int64)),(True,keep)):
            chosen=np.flatnonzero(resolved&(rows['FILE_NUM']==slab)&(low==branch))
            mapped,qa=check_mapping(rows[chosen],xyz[chosen],mass,native_xyz,mapping,native_radius)
            native_rows[chosen]=mapped
            diagnostics.append(dict(file_num=int(slab),low_branch=branch,**qa))
            print('NATIVE_HOST_LOOKUP',phase,int(slab),'low' if branch else 'high',qa,flush=True)
        after=path.stat()
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('Native file changed during read')
        sources.append(dict(path=str(path),bytes=before.st_size,mtime_ns=before.st_mtime_ns,
            cleaned=True,rows=len(mass),low_filtered_rows=len(keep),particle_mass=float(cat.header['ParticleMassHMsun']),
            selection_position_dtype=str(native_positions.dtype)))
        del cat,mass,native_positions,native_xyz,native_radius,keep;gc.collect()
    passed=all(d['passed'] for d in diagnostics) and not any(namespace_qa.values())
    report=dict(phase=phase,job=os.environ['SLURM_JOB_ID'],p1_sha256=sha256(p1path),
        source_code_sha256=sha256(__file__),generator_commit=GENERATOR_COMMIT,
        generator_url='https://github.com/amjsmith/shared_code/tree/'+GENERATOR_COMMIT+'/abacus_mocks',
        mapping='resolved high: cleaned native row; resolved low: order-preserving mass>=1e11 and chi(0.16) filter',
        filter_radius_mpc_h=radius,position_tolerance_mpc_h=POSITION_TOLERANCE,mass_relative_tolerance=MASS_TOLERANCE,
        filter_radius_source=radius_source,galaxy_position_label_method_changed=False,
        namespace_qa=namespace_qa,diagnostics=diagnostics,native_sources=sources,
        native_raw_payloads_rehashed=False,all_selected_masses_and_centrals_checked=True,
        cross_branch_host_mapping_pass=passed,fold_policy_qualified=False,model_inputs_ready=False,
        elapsed_seconds=time.monotonic()-started)
    if passed:
        # Unresolved pseudo-host rows never share the resolved native namespace.
        keys=np.rec.fromarrays([rows['RES'],rows['FILE_NUM'],np.where(resolved,native_rows,rows['HALO_INDEX'])],
                               names='res,file_num,native_or_pseudo_row')
        _,groups=np.unique(keys,return_inverse=True)
        output=destination/'host_groups.npz'
        with output.open('xb') as stream:
            np.savez_compressed(stream,parent_node_id=ids,targetid=rows['TARGETID'],host_group=groups,
                resolved_native_row=native_rows,source_row=rows['SOURCE_ROW'])
        report['output']=dict(path=str(output),sha256=sha256(output),bytes=output.stat().st_size)
    save_json(destination/'HOST_MAPPING_AUDIT.json',report)
    print('HOST_MAPPING_DIAGNOSTIC_COMPLETE',phase,passed,flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--phases',nargs='+',required=True)
    parser.add_argument('--generator-distance-certificate')
    args=parser.parse_args()
    for phase in args.phases:
        report=run(phase,args.generator_distance_certificate)
        if not report['cross_branch_host_mapping_pass']:
            raise SystemExit('Native host mapping failed; diagnostic preserved, no automatic retry: '+phase)
