"""Assemble field-only low-z P12 full and leave-one-phase-out loader products.

Reuse corrected parent arrays, not old eligibility, folds, selection, or scalers.
Only the counts normalization is consumed by the registered three-channel U-Net;
derived channels retain their physical values and are recomputed from each fit.
"""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import h5py

from workflows.sbi import prepare_p12a_lowz_inputs as p12
from workflows.sbi.prepare_cfm_lowz_inputs import require_compute, sha256
from workflows.abacus_tweb.p10_training_contract import atomic_json as save_json
from workflows.sbi.build_p12a_matched_response import selection_contract
from workflows.sbi.p12_prepare_crossfit_contracts import hardlink_or_copy, link_directory
from workflows.abacus_tweb.p6_field_patch_utils import fractional_cell_index
from workflows.abacus_tweb.p8_deterministic_common import linear_increments


def phase_products(phase):
    require_compute();plan,source,policy=p12.load_plan()
    if phase not in policy['training']+policy['development_and_calibration']:raise PermissionError(phase)
    root=Path(plan['output_root']);directory=root/phase
    contract=root/'training_contract_v1';out=contract/'phases'/phase
    marker=out/'LOWZ_PHASE_READY.json'
    if marker.exists():
        record=p12.read(marker)
        for path,digest in record['bindings'].items():
            if sha256(path)!=digest:raise ValueError('Phase product changed: '+path)
        return record
    out.mkdir(parents=True,exist_ok=False)
    p1path=directory/'p1_canonical/manifest.json';p1=p12.read(p1path)
    from workflows.sbi.p12a_phase_training_cores import build as build_cores
    p4=build_cores(phase);p4path=directory/'phase_training_cores_v1/CORE_COMPLETE.json'
    if not p4['pass'] or p4.get('training_split')!='whole_simulation_phase':
        raise ValueError('Whole-phase encoder ownership required')
    bindings={str(p1path):sha256(p1path),str(p4path):sha256(p4path)}
    for name,path in p4['artifacts'].items():
        if sha256(path)!=p4['artifact_sha256'][name]:raise ValueError('P4 payload changed')
        bindings[path]=p4['artifact_sha256'][name]
    prior=Path(source['full_contract'])/'phases'/phase
    prior_record=p12.read(prior/'phase_contract.json')
    old_p1=p12.read(prior_record['inputs']['p1_manifest'])
    if (sha256(prior_record['inputs']['p1_manifest'])!=prior_record['inputs']['p1_manifest_sha256']
            or old_p1['parent_sha256']!=p1['parent_sha256']
            or old_p1['truth_lineage']!=p1['truth_lineage']
            or sha256(prior_record['target']['path'])!=prior_record['target']['sha256']):
        raise ValueError('Corrected parent-array reuse lineage failed')
    for name in ('parent_targetid.npy','parent_redshift.npy','parent_eigenvalues.npy'):
        src=prior/name;hardlink_or_copy(src,out/name);bindings[str(out/name)]=sha256(src)
    with np.load(p1['index']) as index:
        np.testing.assert_array_equal(index['targetid'],np.load(out/'parent_targetid.npy',mmap_mode='r'))
        n_parent=len(index['targetid'])
    active_path=directory/'phase_training_cores_v1/active_assignment.npz';core_path=directory/'phase_training_cores_v1/cores.npz'
    with np.load(active_path) as saved:assignment={k:saved[k] for k in saved.files}
    with np.load(core_path) as saved:cores={k:saved[k] for k in saved.files}
    eligible=assignment['supervised_eligible'].astype(bool);parent=assignment['parent_node_id']
    shell=assignment['shell'];counts=np.bincount(shell[eligible],minlength=4)
    if np.any(counts==0) or len(np.unique(parent[eligible]))!=int(eligible.sum()):raise ValueError('Bad authoritative rows')
    weights=np.zeros(len(parent),np.float32);weights[eligible]=1/np.sqrt(counts[shell[eligible]])
    np.save(out/'active_row_weight.npy',weights)
    core_weights=np.bincount(assignment['core_id'],weights=weights,minlength=len(cores['core_id']))
    core_ids=np.flatnonzero(core_weights>0).astype(np.int32)
    training=phase in policy['training']
    np.save(out/('training_core_id.npy' if training else 'validation_core_id.npy'),core_ids)
    if training:np.save(out/'training_core_weight.npy',core_weights[core_ids])
    target=np.load(out/'parent_eigenvalues.npy',mmap_mode='r')
    selected=np.asarray(target[parent[eligible]],float)
    if not np.isfinite(selected).all() or np.any(np.diff(selected,axis=1)<0):raise ValueError('Invalid ordered truth')
    increments=linear_increments(selected)
    target_moments=dict(mean=increments.mean(axis=0).tolist(),second_moment=np.square(increments).mean(axis=0).tolist())
    phase_record=dict(schema_version='p10-phase-loader-contract-v1',phase=phase,
        role='training' if training else 'validation_and_selection',truth_present=True,
        truth_lineage=p1['truth_lineage'],parent_rows=n_parent,authoritative_rows=int(eligible.sum()),
        eligible_cores=len(core_ids),phase_weight_denominator=float(weights.sum(dtype=float)),
        shell_counts={str(i):int(v) for i,v in enumerate(counts)},
        target=dict(path=str(out/'parent_eigenvalues.npy'),sha256=sha256(out/'parent_eigenvalues.npy'),
                    authoritative_finite=True,authoritative_ordered=True),
        inputs=dict(p1_manifest=str(p1path),p1_manifest_sha256=sha256(p1path),
            assignment=str(active_path),assignment_sha256=sha256(active_path),
            cores=str(core_path),cores_sha256=sha256(core_path)),pass_=True)
    phase_record['pass']=phase_record.pop('pass_');save_json(out/'phase_contract.json',phase_record)
    adapter=contract/'adapters'/phase/'field';adapter.mkdir(parents=True,exist_ok=False)
    p3=p12.read(directory/'p3_fields/field_manifest.json')
    ordered=np.lexsort((parent[eligible],assignment['core_id'][eligible]))
    parents=parent[eligible][ordered];core=assignment['core_id'][eligible][ordered];caps=assignment['cap'][eligible][ordered]
    offsets=np.r_[0,np.cumsum(np.bincount(core,minlength=len(cores['core_id'])))]
    points=np.load(p1['points'],mmap_mode='r');frac=np.empty((len(parents),3),np.float32)
    cap_records={};response_records={};accum=np.zeros(3)
    for cap,name in ((0,'SGC'),(1,'NGC')):
        receipt_path=directory/'matched_cic_response_v1'/(name+'_COMPLETE.json');receipt=p12.read(receipt_path)
        if not receipt['geometry_basis_ready'] or receipt['binding']['p1_sha256']!=sha256(p1path):
            raise ValueError('Matched response is not bound to P1')
        for item in (receipt['output'],receipt['external_geometry']):
            if sha256(item['path'])!=item['sha256']:raise ValueError('Response payload changed')
            bindings[item['path']]=item['sha256']
        bindings[str(receipt_path)]=sha256(receipt_path)
        grid=receipt['grid'];which=caps==cap
        frac[which]=fractional_cell_index(points[parents[which],:3],grid['origin_mpc'],grid['cell_mpc'])
        cap_records[name]=dict(cap_id=cap,field_path=receipt['output']['path'],
            field_sha256=receipt['output']['sha256'],shape=grid['shape'],origin_mpc=grid['origin_mpc'],cell_mpc=grid['cell_mpc'])
        response_records[name]=receipt['output']
        with h5py.File(receipt['output']['path'],'r') as field:
            for sl in field['counts'].iter_chunks():
                values=np.log1p(np.asarray(field['counts'][sl],np.float32)[field['exposure_apodized'][sl]>1e-4]).astype(float)
                accum+=np.array([len(values),values.sum(),np.square(values).sum()])
    arrays={'core_voxel_start.npy':cores['voxel_start'],'core_voxel_stop.npy':cores['voxel_stop'],
        'core_fold.npy':cores['fold'],'core_cap.npy':cores['cap'],'core_active_offsets.npy':offsets,
        'core_active_parent.npy':parents,'core_active_frac_index.npy':frac}
    if offsets[-1]!=int(eligible.sum()) or not np.array_equal(caps,cores['cap'][core]) or not np.isfinite(frac).all():
        raise ValueError('Field adapter row/core identity failed')
    for name,array in arrays.items():np.save(adapter/name,array)
    adapter_manifest=dict(schema_version=1,phase=phase,stage='P12 matched-CIC field-only adapter',
        p3_manifest=str(directory/'p3_fields/field_manifest.json'),p4_cores=str(core_path),
        p4_active_assignment=str(active_path),points=p1['points'],channel_order=p3['channel_order'],
        caps=cap_records,n_cores=len(cores['core_id']),n_authoritative=len(parents),pass_=True)
    adapter_manifest['pass']=adapter_manifest.pop('pass_');save_json(adapter/'adapter_manifest.json',adapter_manifest)
    if accum[0]<=0:raise ValueError('No supported field voxels')
    field_moments=dict(count=int(accum[0]),mean=float(accum[1]/accum[0]),second_moment=float(accum[2]/accum[0]))
    for path in list(out.glob('*.npy'))+list(adapter.glob('*.npy'))+[out/'phase_contract.json',adapter/'adapter_manifest.json']:
        bindings[str(path)]=sha256(path)
    record=dict(phase=phase,bindings=bindings,field_moments=field_moments,target_moments=target_moments,
        response=response_records,field_only=True,pass_=True)
    record['pass']=record.pop('pass_');save_json(marker,record)
    print('P12_LOWZ_PHASE_READY',phase,flush=True)
    return record


def finish():
    require_compute();plan,source,policy=p12.load_plan();root=Path(plan['output_root'])
    full=root/'training_contract_v1';phases=policy['training']+policy['development_and_calibration']
    records={phase:phase_products(phase) for phase in phases}
    _,fits,_,_=selection_contract(root,policy)
    responses={p:r['response'] for p,r in records.items()}
    results={}
    for omitted in [None,*policy['training']]:
        train=[p for p in policy['training'] if p!=omitted]
        destination=full if omitted is None else root/'p12_crossfit_contracts'/('omit_'+omitted)
        validation=policy['development_and_calibration'] if omitted is None else [omitted]
        if omitted:
            for phase in train+validation:
                old=full/'phases'/phase;new=destination/'phases'/phase;new.mkdir(parents=True,exist_ok=True)
                value=p12.read(old/'phase_contract.json');value['role']='training' if phase in train else 'validation_and_selection'
                save_json(new/'phase_contract.json',value)
                for name in ('parent_targetid.npy','parent_redshift.npy','parent_eigenvalues.npy','active_row_weight.npy'):
                    hardlink_or_copy(old/name,new/name)
                if phase in train:
                    for name in ('training_core_id.npy','training_core_weight.npy'):hardlink_or_copy(old/name,new/name)
                else:hardlink_or_copy(old/'training_core_id.npy',new/'validation_core_id.npy')
                link_directory(full/'adapters'/phase/'field',destination/'adapters'/phase/'field')
        fit=fits['full' if omitted is None else 'oof_'+omitted]
        selection=p12.read(fit['path'])
        selection_manifest=dict(schema='p12a-matched-cic-selection-v1',pass_=True,cosmology='Planck18',
            density_unit='Mpc^-3',fit_phases=train,omitted_phase=omitted,selection=selection,
            contrast_epsilon=.001,phase_response=responses,source=fit)
        selection_manifest['pass']=selection_manifest.pop('pass_')
        selection_path=destination/'transforms/field/selection_manifest.json';save_json(selection_path,selection_manifest)
        mean=float(np.mean([records[p]['field_moments']['mean'] for p in train]))
        second=float(np.mean([records[p]['field_moments']['second_moment'] for p in train]))
        std=float(np.sqrt(max(second-mean*mean,0.)))
        if not np.isfinite(std) or std<=0:raise ValueError('Invalid counts normalization')
        channels={name:dict(policy='identity') for name in ('counts','expected_counts','log_count_ratio','ntilde_mpc3',
                    'exposure_binary','exposure_apodized','los_x','los_y','los_z')}
        channels['counts']=dict(policy='zscore',mean=mean,std=std)
        transform=dict(schema_version='p12a-lowz-unet-field-transform-v1',fit_phases=train,
            selection_manifest_sha256=sha256(selection_path),normalization=dict(channels=channels),
            fit_policy='equal phase supported-voxel log1p count moments',
            unused_derived_normalizations='identity; registered U-Net consumes raw derived contrast',pass_=True)
        transform['pass']=transform.pop('pass_');save_json(destination/'transforms/field/field_transform.json',transform)
        means=np.array([records[p]['target_moments']['mean'] for p in train]);seconds=np.array([records[p]['target_moments']['second_moment'] for p in train])
        tm=means.mean(axis=0);ts=np.sqrt(np.maximum(seconds.mean(axis=0)-tm*tm,0.))
        if not np.isfinite(ts).all() or np.any(ts<=0):raise ValueError('Invalid target scaler')
        scaler=dict(schema_version='p10-target-scaler-v1',representation='linear increments',
            definition=['lambda1','lambda2-lambda1','lambda3-lambda2'],fit_phases=train,
            mean=tm.tolist(),std=ts.tolist(),phase_not_feature=True,pass_=True)
        scaler['pass']=scaler.pop('pass_');save_json(destination/'transforms/target_scaler.json',scaler)
        ready=dict(schema_version='p12a-lowz-unet-loader-v1',phase_policy=source['phase_policy'],
            roles=dict(training=train,validation_and_selection=validation,sealed_blind_test=policy['final_test']),
            out_of_fold_phase=omitted,model='unet',field_adapter='matched_cic',
            readiness_scope='encoder_training_whole_phase_split',spatial_calibration_evaluation_qualified=False,
            phase_is_model_input=False,sealed_phase_opened=False,pass_=True)
        ready['pass']=ready.pop('pass_');save_json(destination/'TRAINING_LOADER_CANDIDATE.json',ready)
        # The actual training loader must read both fit types before publication.
        from workflows.abacus_tweb.p10_training_contract import P10PhaseBalancedLoader
        from workflows.abacus_tweb.p8_train_unet_patch import model_inputs
        import torch
        checks=[]
        with P10PhaseBalancedLoader(destination,allow_preparation=True) as loader:
            for phase in (train[0],validation[0]):
                adapter=loader.field_adapter(phase)
                ids=np.load(destination/'phases'/phase/('training_core_id.npy' if phase in train else 'validation_core_id.npy'))
                for core_id in (int(ids[0]),int(ids[-1])):
                    patch=adapter.extract(int(core_id),context_halo_voxels=48)
                    values,coords=model_inputs(patch,loader.field_normalization,'cpu')
                    if not torch.isfinite(values).all() or not torch.isfinite(coords).all():raise ValueError('Actual loader nonfinite')
                    checks.append(dict(phase=phase,core_id=core_id,input_shape=list(values.shape),parents=len(patch.authoritative_parent_id)))
        ready['actual_loader_checks']=checks;save_json(destination/'TRAINING_LOADER_READY.json',ready)
        results['full' if omitted is None else 'omit_'+omitted]=dict(path=str(destination),sha256=sha256(destination/'TRAINING_LOADER_READY.json'))
        print('P12_LOWZ_CONTRACT_READY',omitted or 'full',flush=True)
    completion=dict(contracts=results,phase_count=len(phases),full_and_twenty_oof=True,
        phase_roles_changed=False,learned_model_training_started=False,
        readiness_scope='encoder_training_whole_phase_split',spatial_calibration_evaluation_qualified=False)
    completion['pass']=True
    save_json(root/'LOWZ_TRAINING_PRODUCTS_COMPLETE.json',completion)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--phase');parser.add_argument('--finish',action='store_true')
    args=parser.parse_args()
    if args.finish:finish()
    elif args.phase:phase_products(args.phase)
    else:parser.error('Specify --phase or --finish')
