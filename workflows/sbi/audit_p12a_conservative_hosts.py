"""Separate diagnostic of all-compatible native hosts and unchanged fold gates.

This is not an exact-host reconstruction, a label annotator, or a P4 release.
Any unmatched resolved low group fails closed. Ambiguous candidates are all
retained and every compatible observed high host is unioned with the low group.
"""
import argparse
import gc
import os
from pathlib import Path
import time
import numpy as np

from workflows.sbi import prepare_p12a_lowz_inputs as p12
from workflows.sbi.prepare_cfm_lowz_inputs import require_compute, save_json, sha256
from workflows.sbi.audit_p12a_lowz_native_hosts import NATIVE, GENERATOR_COMMIT, check_mapping
from workflows.sbi.p12a_conservative_hosts import compatible_candidates, brute_candidates
from workflows.sbi.p12a_host_fold_diagnostic import audit as fold_audit
from workflows.abacus_tweb.galaxy_position_mapping import reconstruct_native
from workflows.abacus_tweb.p4_build_spatial_manifest import DisjointSet


def run(phase):
    require_compute()
    if phase not in ('ph001','ph006'):raise PermissionError('Canary phases only')
    plan,_,_=p12.load_plan();root=Path(plan['output_root'])/phase
    destination=root/'conservative_host_diagnostic_v1'
    if destination.exists():raise FileExistsError('No automatic retry: '+str(destination))
    destination.mkdir();started=time.monotonic()
    import fitsio
    from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog
    sources=[Path(__file__),Path(__file__).with_name('p12a_conservative_hosts.py'),
             Path(__file__).with_name('p12a_host_fold_diagnostic.py'),
             Path(__file__).with_name('audit_p12a_lowz_native_hosts.py'),
             p12.REPO/'workflows/abacus_tweb/p4_build_spatial_manifest.py']
    source_hashes={str(p):sha256(p) for p in sources}
    p1path=root/'p1_canonical/manifest.json';p1=p12.read(p1path)
    identity_path=root/'galaxy_identity_v1/IDENTITY_COMPLETE.json';identity=p12.read(identity_path)
    if not identity['identity_pass'] or identity['p1_sha256']!=sha256(p1path):
        raise ValueError('Unqualified galaxy identity')
    inputs={str(p1path):sha256(p1path),str(identity_path):sha256(identity_path),
            identity['output']:identity['output_sha256'],p1['parent']:p1['parent_sha256'],
            p1['index']:p1['index_sha256']}
    for path,digest in inputs.items():
        if sha256(path)!=digest:raise ValueError('Source changed: '+path)
    table=fitsio.read(p1['parent'],columns=['TARGETID','SOURCE_ROW','RA','DEC','Z_COSMO',
                        'FILE_NUM','BOX_INDEX','HALO_INDEX','RES','CEN','HALO_MASS'])
    with np.load(p1['index'],allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['targetid'],table['TARGETID'])
        active=saved['active'].astype(bool)&saved['valid_target'].astype(bool)
        cap=saved['cap'].astype(int);shell=saved['shell'].astype(int);context=saved['context'].astype(bool)
    with np.load(identity['output'],allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['parent_node_id'],np.arange(len(table)))
        np.testing.assert_array_equal(saved['source_row'],table['SOURCE_ROW'])
        eligible=saved['supervised_eligible'].astype(bool)
        local_group=saved['branch_local_host_group']
    if np.any(eligible&~active) or int(eligible.sum())!=identity['eligible_rows']:
        raise ValueError('Eligibility mismatch')
    ids=np.flatnonzero(active);rows=table[ids];groups=local_group[ids]
    xyz=reconstruct_native(rows['RA'],rows['DEC'],rows['Z_COSMO'])
    low=rows['BOX_INDEX']<0;resolved=rows['RES']==1
    namespaces=dict(high_unresolved=int((~low&~resolved).sum()),
        low_resolved_below_cut=int((low&resolved&(rows['HALO_MASS']<10.)).sum()),
        low_unresolved_above_cut=int((low&~resolved&(rows['HALO_MASS']>10.)).sum()))
    dsu=DisjointSet(int(local_group.max())+1)
    records=[];native_sources=[];edge_low=[];edge_high=[];unmatched=[]
    total_candidates=0;ambiguous=0;low_groups=0;brute_checks=0
    for slab in np.unique(rows['FILE_NUM'][resolved]):
        path=NATIVE/f'AbacusSummit_base_c000_{phase}'/'halos/z0.200/halo_info'/f'halo_info_{int(slab):03d}.asdf'
        before=path.stat()
        cat=CompaSOHaloCatalog(path,fields=['N','x_L2com'],subsamples=False,cleaned=True,convert_units=True,verbose=False)
        if cat.header['SimName']!=f'AbacusSummit_base_c000_{phase}':raise ValueError('Wrong native simulation')
        native_xyz=np.asarray(cat.halos['x_L2com'],float)
        mass=np.asarray(cat.halos['N'],float)*float(cat.header['ParticleMassHMsun'])/1e10
        radii=(3*mass*1e10/(4*np.pi*float(cat.header['SODensityL1'])*2.77536627e11*float(cat.header['Omega_M'])))**(1/3)
        high_ids=np.flatnonzero(resolved&~low&(rows['FILE_NUM']==slab))
        _,high_qa=check_mapping(rows[high_ids],xyz[high_ids],mass,native_xyz,np.arange(len(mass)),radii)
        high_native,first=np.unique(rows['HALO_INDEX'][high_ids],return_index=True)
        high_groups=groups[high_ids[first]]
        low_ids=np.flatnonzero(resolved&low&(rows['FILE_NUM']==slab))
        native_keep=np.flatnonzero(mass>=10.)
        # Sorting once gives bounded group member slices for independent checks.
        order=np.argsort(groups[low_ids],kind='stable');ordered=low_ids[order]
        starts=np.r_[0,np.flatnonzero(np.diff(groups[ordered]))+1] if len(ordered) else np.array([],int)
        stops=np.r_[starts[1:],len(ordered)]
        slab_unmatched=0;slab_ambiguous=0;slab_links=0;slab_groups=0
        iterator=compatible_candidates(xyz[low_ids],rows['HALO_MASS'][low_ids],rows['CEN'][low_ids]==1,
            groups[low_ids],native_xyz[native_keep],mass[native_keep],radii[native_keep])
        for number,(group,candidates) in enumerate(iterator):
            low_groups+=1;slab_groups+=1;total_candidates+=len(candidates)
            if len(candidates)>1:ambiguous+=1;slab_ambiguous+=1
            # Predetermined samples plus the first zero/ambiguous case per slab.
            if number in (0,len(starts)//2,len(starts)-1) or (not len(candidates) and slab_unmatched==0) or (len(candidates)>1 and slab_ambiguous==1):
                members=ordered[starts[number]:stops[number]]
                expected=brute_candidates(xyz[members],rows['HALO_MASS'][members],rows['CEN'][members]==1,
                    native_xyz[native_keep],mass[native_keep],radii[native_keep])
                np.testing.assert_array_equal(candidates,expected);brute_checks+=1
            if not len(candidates):
                unmatched.append(group);slab_unmatched+=1
            candidate_native=native_keep[candidates]
            where=np.searchsorted(high_native,candidate_native)
            valid=where<len(high_native)
            valid[valid] &= high_native[where[valid]]==candidate_native[valid]
            for high_group in high_groups[where[valid]]:
                dsu.union(group,int(high_group));edge_low.append(group);edge_high.append(int(high_group));slab_links+=1
        after=path.stat()
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('Native source changed')
        native_sources.append(dict(path=str(path),bytes=before.st_size,mtime_ns=before.st_mtime_ns,
            cleaned=True,rows=len(mass),eligible_mass_rows=len(native_keep)))
        record=dict(slab=int(slab),high_mapping=high_qa,low_groups=slab_groups,
            unmatched=slab_unmatched,ambiguous=slab_ambiguous,cross_branch_links=slab_links)
        records.append(record);print('CONSERVATIVE_HOST_SLAB',phase,record,flush=True)
        del cat,native_xyz,mass,radii;gc.collect()
    coverage=not any(namespaces.values()) and not unmatched and all(r['high_mapping']['passed'] for r in records)
    folded=None;output=None
    if coverage:
        roots=dsu.roots();host_group=roots[local_group]
        if edge_low and np.any(roots[np.asarray(edge_low)]!=roots[np.asarray(edge_high)]):
            raise ValueError('Candidate union incomplete')
        p3path=root/'p3_fields/field_manifest.json';p3=p12.read(p3path)
        inputs[str(p3path)]=sha256(p3path);inputs[p1['points']]=sha256(p1['points'])
        origins={i:np.asarray(p3['components'][name]['grid']['origin_mpc'],float)
                 for i,name in ((0,'SGC'),(1,'NGC'))}
        points=np.load(p1['points'],mmap_mode='r')
        folded=fold_audit(points,cap,shell,context,eligible,host_group,origins)
        output=destination/'candidate_groups.npz'
        with output.open('xb') as stream:
            np.savez_compressed(stream,parent_node_id=ids,targetid=rows['TARGETID'],
                conservative_host_group=host_group[ids],edge_low=np.asarray(edge_low,dtype=np.int64),
                edge_high=np.asarray(edge_high,dtype=np.int64))
        output=dict(path=str(output),sha256=sha256(output))
    for path,digest in {**inputs,**source_hashes}.items():
        if sha256(path)!=digest:raise ValueError('Authority changed before diagnostic commit: '+path)
    report=dict(schema='p12a-conservative-host-diagnostic-v1',phase=phase,job=os.environ['SLURM_JOB_ID'],
        inputs=inputs,source_hashes=source_hashes,generator_commit=GENERATOR_COMMIT,native_sources=native_sources,
        native_raw_payloads_rehashed=False,namespace_checks=namespaces,slabs=records,
        low_resolved_groups=low_groups,compatible_native_candidates=total_candidates,
        ambiguous_low_groups=ambiguous,unmatched_low_groups=len(unmatched),unmatched_group_examples=unmatched[:30],
        cross_branch_edges=len(edge_low),independent_full_scan_checks=brute_checks,
        candidate_coverage_pass=coverage,completeness_conditional_on='catalogue FILE_NUM and pinned generator mass/NFW SO-radius contract',
        exact_host_identity_recovered=False,fold_diagnostic=folded,output=output,
        passed=coverage and folded is not None and folded['passed'],
        phase_roles_changed=False,galaxy_labels_changed=False,model_inputs_ready=False,
        production_host_sidecar_qualified=False,elapsed_seconds=time.monotonic()-started)
    save_json(destination/'CONSERVATIVE_HOST_AUDIT.json',report)
    print('CONSERVATIVE_HOST_DIAGNOSTIC_COMPLETE',phase,report['passed'],flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phases',nargs='+',default=['ph001','ph006'])
    for phase in parser.parse_args().phases:
        if not run(phase)['passed']:raise SystemExit('Conservative diagnostic failed; no automatic retry')
