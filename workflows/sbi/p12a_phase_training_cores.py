"""Geometric core ownership for whole-phase-held-out P12 encoder training.

There are no within-phase training/validation folds in this contract. Fold=-1
explicitly means unavailable, not a passed spatial calibration/evaluation split.
"""
from pathlib import Path
import numpy as np
from workflows.sbi import prepare_p12a_lowz_inputs as p12
from workflows.sbi.prepare_cfm_lowz_inputs import require_compute, sha256
from workflows.abacus_tweb.p10_training_contract import atomic_json


def core_arrays(points, cap, shell, context, active, eligible, targetid, grids):
    if np.any(eligible & ~active) or np.any(active & ~context):raise ValueError('Invalid row populations')
    size=64/.6766
    parent_core=np.full(len(points),-1,np.int32)
    parts=[];offset=0
    for cid,name in ((0,'SGC'),(1,'NGC')):
        ids=np.flatnonzero(context & (cap==cid));grid=grids[name]
        origin=np.asarray(grid['origin_mpc'],float)
        index=np.floor((points[ids,:3]-origin)/size).astype(np.int32)
        unique,inverse,counts=np.unique(index,axis=0,return_inverse=True,return_counts=True)
        parent_core[ids]=offset+inverse
        lower=origin+unique*size;upper=lower+size
        start=np.maximum(0,np.floor((lower-origin)/grid['cell_mpc'])).astype(np.int32)
        stop=np.minimum(grid['shape'],np.ceil((upper-origin)/grid['cell_mpc'])).astype(np.int32)
        if np.any(stop<=start):raise ValueError('Empty geometric training core')
        parts.append(dict(cap=np.full(len(unique),cid,np.int8),core_index=unique,
            lower_mpc=lower,upper_mpc=upper,centroid_mpc=(lower+upper)/2,
            context_count=counts,voxel_start=start,voxel_stop=stop))
        offset+=len(unique)
    cores={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
    cores['core_id']=np.arange(offset,dtype=np.int32);cores['fold']=np.full(offset,-1,np.int8)
    cores['active_count']=np.bincount(parent_core[eligible],minlength=offset)
    ids=np.flatnonzero(active)
    assignment=dict(parent_node_id=ids,targetid=targetid[ids],core_id=parent_core[ids],
        cap=cap[ids],shell=shell[ids],fold=np.full(len(ids),-1,np.int8),
        supervised_eligible=eligible[ids],periodic_duplicate_image=~eligible[ids])
    if np.any(parent_core[context]<0) or int(cores['active_count'].sum())!=int(eligible.sum()):
        raise ValueError('Core ownership/count conservation failed')
    return cores,assignment


def build(phase):
    require_compute();plan,_,policy=p12.load_plan()
    if phase not in policy['training']+policy['development_and_calibration']:raise PermissionError(phase)
    root=Path(plan['output_root'])/phase;destination=root/'phase_training_cores_v1'
    marker=destination/'CORE_COMPLETE.json'
    if marker.exists():
        result=p12.read(marker)
        for path,digest in result['bindings'].items():
            if sha256(path)!=digest:raise ValueError('Training core input changed')
        return result
    identity_path=root/'galaxy_identity_v1/IDENTITY_COMPLETE.json'
    if not identity_path.exists():
        from workflows.sbi.prepare_p12a_galaxy_identity import run
        run(phase)
    p1path=root/'p1_canonical/manifest.json';p1=p12.read(p1path)
    identity=p12.read(identity_path);p3path=root/'p3_fields/field_manifest.json';p3=p12.read(p3path)
    if (not identity['identity_pass'] or identity['p1_sha256']!=sha256(p1path)
            or sha256(identity['output'])!=identity['output_sha256']
            or sha256(p1['index'])!=p1['index_sha256']):raise ValueError('Invalid corrected identity')
    with np.load(p1['index']) as f:
        index={k:f[k] for k in ('cap','shell','context','active','valid_target','targetid')}
    with np.load(identity['output']) as f:
        np.testing.assert_array_equal(f['parent_node_id'],np.arange(len(index['targetid'])))
        eligible=f['supervised_eligible'].astype(bool)
    points=np.load(p1['points'],mmap_mode='r')
    cores,assignment=core_arrays(points,index['cap'],index['shell'],index['context'].astype(bool),
        index['active'].astype(bool)&index['valid_target'].astype(bool),eligible,index['targetid'],
        {name:p3['components'][name]['grid'] for name in ('SGC','NGC')})
    destination.mkdir(exist_ok=False)
    artifacts={}
    for name,arrays in (('cores',cores),('active_assignment',assignment)):
        path=destination/(name+'.npz');np.savez_compressed(path,**arrays);artifacts[name]=str(path)
    bindings={str(p):sha256(p) for p in (p1path,p3path,identity_path,Path(identity['output']),Path(p1['index']),Path(p1['points']))}
    bindings.update({path:sha256(path) for path in artifacts.values()})
    result=dict(schema='p12a-phase-training-cores-v1',phase=phase,core_mpc_h=64.,
        training_split='whole_simulation_phase',within_phase_fold_value=-1,
        spatial_calibration_evaluation_qualified=False,phase_roles_changed=False,
        corrected_identity_host_binding=False,galaxy_identity=str(identity_path),
        artifacts=artifacts,artifact_sha256={k:bindings[p] for k,p in artifacts.items()},bindings=bindings)
    result['pass']=True;atomic_json(marker,result)
    print('P12_PHASE_CORES_COMPLETE',phase,len(cores['core_id']),flush=True)
    return result
