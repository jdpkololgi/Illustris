"""Independent saved-artifact checks for the complete-footprint boundary test."""
import argparse
import itertools
import json
from pathlib import Path
import h5py
import numpy as np
from numba import njit, prange
from workflows.sbi import e2e_boundary_mock as experiment
from workflows.sbi import e2e_boundary_models as models
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_coordinates as coord
from workflows.sbi import e2e_coupled_conditions as conditions
from workflows.sbi import e2e_coupled_target_products as targets
from workflows.sbi import e2e_footprint_tidal_operator as op


@njit(parallel=True)
def direct_sum(delta, target):
    # Independent direct accumulation checks FFT placement, origins and cropping.
    # Cuboid antiderivatives themselves have separate quadrature regression tests.
    partial=np.zeros((delta.shape[0],6))
    for i in prange(delta.shape[0]):
        for j in range(delta.shape[1]):
            for k in range(delta.shape[2]):
                weight=delta[i,j,k]
                if weight==0: continue
                kernel=op.cube_tensor(target[0]-i,target[1]-j,target[2]-k)
                for component in range(6):partial[i,component]+=weight*kernel[component]
    return partial.sum(axis=0)


def run(root,scores,refined):
    c.require_compute(); checks=[]; recomputed=[]
    scored=json.loads((scores/'SCORES.json').read_text())
    if not scored['passed']: raise ValueError('Scoring incomplete')
    factor=scored.get('source_factor',2);suffix=scored.get('source_suffix','')
    if factor not in (1,2) or (factor==1 and suffix!='_refined_v2') or (factor==2 and suffix):
        raise ValueError('Unknown source-resolution panel')
    identities=[(r['phase'],r['pair_id'],r['side']) for r in scored['cores']]
    if len(identities)!=64 or len(set(identities))!=64:
        raise ValueError('Incomplete or duplicated evaluation panel')
    for phase in experiment.TRAIN+experiment.EVAL:
        experiment.guard(phase)
        binding=json.loads((root/(phase+suffix)/'BINDING.json').read_text())
        native=c.verify_receipt(binding['native']['path'],payload=False)
        if native['subsample_fraction']!=.1 or native['processed_files']!=136:
            raise ValueError('Native particle provenance mismatch')
        for cap in ('NGC','SGC'):
            receipt=json.loads((root/(phase+suffix)/(cap+'_COMPLETE.json')).read_text())
            if c.sha256(receipt['output']['path'])!=receipt['output']['sha256']:
                raise ValueError('Stored core payload changed')
            for item in receipt['targets']:
                target=coord.verify_receipt(item['path'],payload=False)
                if (target['binding']['native_receipt_sha256']!=binding['native']['sha256']
                        or target['binding']['geometry_receipt_sha256']!=binding['geometry']['sha256']):
                    raise ValueError('Full-box reference and masked source do not share native inputs')
        checks.append(dict(phase=phase,fullbox_and_source_native_binding_identical=True))
    for phase in experiment.EVAL:
        for cap in ('NGC','SGC'):
            receipt=json.loads((root/(phase+suffix)/(cap+'_COMPLETE.json')).read_text())
            with np.load(receipt['output']['path']) as f:
                arrays={k:f[k] for k in f.files}
            correction=np.zeros(6);correction[[0,3,5]]=receipt['periodic_source_mean']/3
            np.testing.assert_allclose(arrays['periodic_dc']-arrays['periodic'],
                np.broadcast_to(correction,arrays['periodic'].shape),atol=2e-15,rtol=0)
            for i,identity in enumerate(receipt['identities']):
                if i%2==0:
                    support=targets.owned_core_arrays(conditions.load_pair(phase,identity['pair_id'])['support'])
                np.testing.assert_array_equal(arrays['support'][i],support[identity['side']])
                valid=arrays['support'][i]==1
                true=arrays['truth'][i]; t=arrays['isolated'][i]
                matrix_error=np.zeros((*t.shape[:-1],3,3))
                for component,(a,b) in enumerate(op.COMPONENTS):
                    matrix_error[...,a,b]=matrix_error[...,b,a]=t[...,component]-true[...,component]
                independent_mse=float(np.mean(np.sum(matrix_error[valid]**2,axis=(-1,-2))))
                row=next(r for r in scored['cores'] if (r['phase'],r['pair_id'],r['side'])==
                    (phase,identity['pair_id'],identity['side']))
                np.testing.assert_allclose(independent_mse,row['arms']['isolated']['primary']['tensor_mse'],rtol=1e-12)
                recomputed.append(independent_mse)
            checks.append(dict(phase=phase,cap=cap,original_condition_support_replayed=True,
                independently_recomputed_core_scores=len(receipt['identities'])))
            for ordinal in (0,len(receipt['identities'])-1):
                point=np.array([8,8,8])
                xyz=np.array(receipt['identities'][ordinal]['center_mpc_h'])+(-16+1+2*point)*3.383
                total=np.zeros(6)
                for source in ('NGC','SGC'):
                    with h5py.File(root/(phase+suffix)/(source+'_SOURCE.h5')) as f:
                        delta=f['delta'][:]*f['support'][:]
                        origin=np.array(f.attrs['origin_mpc_h']);cell=f.attrs['cell_mpc_h']
                    offsets=[(0,0,0)] if factor==2 else list(itertools.product((-.5,.5),repeat=3))
                    for shift in offsets:
                        offset=(xyz+np.array(shift)*3.383-origin)/cell
                        total+=direct_sum(delta,offset)/len(offsets)
                stored=arrays['isolated'][ordinal,8,8,8]
                error=float(np.max(abs(total-stored)))
                if error>2e-9: raise ValueError('Direct-sum/FFT mismatch: '+str(error))
                checks.append(dict(phase=phase,cap=cap,core=ordinal,direct_sum_max_abs=error))
                experiment.log(stage='direct_sum_verified',phase=phase,cap=cap,core=ordinal,error=error)
    np.testing.assert_allclose(np.sqrt(np.mean(recomputed)),scored['groups']['all']['isolated']['tensor_rmse'],rtol=1e-12)
    refinement=[]
    if refined is not None:
        if not json.loads((refined/'COMPLETE.json').read_text())['passed']:
            raise ValueError('Incomplete refinement')
        for cap in ('NGC','SGC'):
            coarse_record=json.loads((root/'ph016'/(cap+'_COMPLETE.json')).read_text())
            fine_record=json.loads((refined/(cap+'_COMPLETE.json')).read_text())
            if coarse_record['identities']!=fine_record['identities']:
                raise ValueError('Refinement changes scoring geometry')
            with np.load(coarse_record['output']['path']) as f, np.load(fine_record['output']['path']) as g:
                np.testing.assert_array_equal(f['support'],g['support'])
                np.testing.assert_array_equal(f['truth'],g['truth'])
                valid=f['support']==1
                base_error=np.sum((f['isolated']-f['truth'])**2*experiment.WEIGHTS,axis=-1)
                fine_error=np.sum((g['isolated']-g['truth'])**2*experiment.WEIGHTS,axis=-1)
                change=np.sum((g['isolated']-f['isolated'])**2*experiment.WEIGHTS,axis=-1)
                # Equal-core weighting, matching the primary score.
                mse=lambda a:float(np.mean([v[m].mean() for v,m in zip(a,valid)]))
                refinement.append(dict(cap=cap,base_rmse=np.sqrt(mse(base_error)),
                    refined_rmse=np.sqrt(mse(fine_error)),change_rmse=np.sqrt(mse(change)),
                    change_over_base_error=np.sqrt(mse(change)/mse(base_error))))
    c.atomic_json(scores/'VERIFIED.json',dict(passed=True,checks=checks,refinement=refinement,
        score_sha256=c.sha256(scores/'SCORES.json'),verifier_sha256=c.sha256(__file__),
        independent_core_score_count=len(recomputed),sealed_phase_access=False,source_factor=factor))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--scores',type=Path,required=True);p.add_argument('--refined',type=Path)
    a=p.parse_args();run(a.root,a.scores,a.refined)
