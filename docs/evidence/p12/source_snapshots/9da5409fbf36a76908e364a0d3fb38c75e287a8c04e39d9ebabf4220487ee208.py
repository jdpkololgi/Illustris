"""Bounded ph002 frozen encoder/response replay; never authorizes real inference."""
import argparse, hashlib, json, os, time
from pathlib import Path
import numpy as np
import torch
from workflows.abacus_tweb.p10_training_contract import P10PhaseBalancedLoader
from workflows.abacus_tweb import p8_train_unet_patch as u
from workflows.abacus_tweb.p8_deterministic_common import increments_to_eigenvalues, unscale_increments
from workflows.abacus_tweb.p6_p7_validate_model_convergence import compare, P6_GATES
from workflows.sbi.p12_export_unet_summaries import ntilde_at_rows
from workflows.sbi.p12_prepare_base_response_dataset import sample_random_support_distance
from workflows.abacus_tweb.p12a_coordinate_sample_audit import record
BASE=Path('/pscratch/sd/d/dkololgi/abacus/p10_multiphase')
ROOT=Path(__file__).resolve().parents[2]

def read(p):return json.loads(Path(p).read_text())

def pick(adapter,selection,ids):
    choices={}
    for core in ids:
        core=int(core); cap=int(adapter.core_cap[core]);name='NGC' if cap else 'SGC'
        grid=adapter.manifest['caps'][name]
        center=np.asarray(grid['origin_mpc'])+(adapter.core_start[core]+adapter.core_stop[core])*grid['cell_mpc']/2
        z=np.interp(np.linalg.norm(center),selection['cosmology']['radius_grid_mpc'],selection['cosmology']['redshift_grid'])
        shell=int(np.searchsorted([.15,.25,.35,.45,.55],z,side='right')-1)
        count=int(adapter.core_offsets[core+1]-adapter.core_offsets[core])
        if shell not in range(4) or count<16:continue
        key=(cap,shell)
        if key not in choices or (count,-core)>(choices[key][0],-choices[key][1]):choices[key]=(count,core)
    if len(choices)!=8:raise ValueError('missing cap/shell core')
    return [dict(cap=k[0],shell=k[1],rows=v[0],core=v[1]) for k,v in sorted(choices.items())]

def predict(model,patch,ck):
    x,p=u.model_inputs(patch,ck['normalization'],'cuda')
    with torch.inference_mode():
        latent=model.sample_latent(x,p)
        scaled=model.head(latent).cpu().numpy()
    return increments_to_eigenvalues(unscale_increments(scaled,ck['scaler'])).astype('f4')

def replay(output):
    if not torch.cuda.is_available() or not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('GPU allocation required')
    source=record(__file__,small=True); start=time.monotonic()
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    sr=read(BASE/'p12_oof_summaries/ph002/OOF_SUMMARY_COMPLETE.json')
    parent=np.load(sr['arrays']['parent_node_id'],mmap_mode='r');order=np.argsort(parent);sorted_parent=parent[order]
    stored=np.load(sr['arrays']['base_prediction'],mmap_mode='r')
    response=np.load(sr['arrays']['response'])
    cache=BASE/'p12a_random_support_parent_cache_v2/ph002'
    cached_distance=np.load(cache/'distance_to_support_boundary_mpc.npy',mmap_mode='r')
    cached_support=np.load(cache/'support_random.npy',mmap_mode='r')
    rm=read(BASE/'training_contract_r1_random/adapters/ph002/field/adapter_manifest.json')
    points=np.load(rm['points'],mmap_mode='r')
    result=dict(schema='p12a-frozen-mock-replay-v1',phase='ph002',source=source,job=os.environ['SLURM_JOB_ID'],
                tolerance=dict(stored_prediction_atol=1e-5,stored_prediction_rtol=1e-5,context_growth=P6_GATES),
                reference_is_existing_mock_path=True,loa_adapter_qualified=False,ready_for_desi_canary=False,runs=[])
    for mode,contract,checkpoint in [('oof',Path(sr['contract_root']),Path(sr['checkpoint'])),('fullfit',BASE/'training_contract',BASE/'arm_a_training/arm_a_r0_v1/unet/seed_42/best_checkpoint.pt')]:
        ck=torch.load(checkpoint,map_location='cuda',weights_only=False)
        model=u.UPatch().cuda().eval();model.load_state_dict(ck['state_dict'])
        loader=P10PhaseBalancedLoader(contract,include_blind=False);adapter=loader.field_adapter('ph002')
        selection=read(contract/'transforms/field/selection_manifest.json')
        ids=np.load(contract/'phases/ph002'/('validation_core_id.npy' if mode=='oof' else 'training_core_id.npy'))
        chosen=pick(adapter,selection,ids);run=dict(mode=mode,checkpoint=record(checkpoint,small=True),selected=chosen,cores=[])
        for meta in chosen:
            core=meta['core'];patch=adapter.extract(core,24,u.CHANNELS,alignment_voxels=8)
            got=predict(model,patch,ck);pid=patch.authoritative_parent_id
            row=order[np.searchsorted(sorted_parent,pid)]
            if not np.array_equal(parent[row],pid):raise ValueError('summary parent mismatch')
            checks={};diagnostics={}
            if mode=='oof':
                ref=np.asarray(stored[row]);checks['stored_prediction']=bool(np.allclose(got,ref,atol=1e-5,rtol=1e-5))
                diagnostics['stored_max_abs']=float(np.max(np.abs(got-ref)))
                z=response['redshift'][row];cap=response['cap'][row]
                nt=ntilde_at_rows(selection,cap,z)
                checks['ntilde_replay']=bool(np.array_equal(nt,response['ntilde_mpc3'][row]))
                sample=np.linspace(0,len(pid)-1,min(16,len(pid)),dtype=int)
                distance,support=sample_random_support_distance(rm,points,pid[sample])
                checks['boundary_cache_replay']=bool(np.array_equal(distance,cached_distance[row[sample]]))
                checks['support_cache_replay']=bool(np.array_equal(support,cached_support[row[sample]]))
            else:
                enlarged=adapter.extract(core,48,u.CHANNELS,alignment_voxels=8)
                ref=predict(model,enlarged,ck); metric=compare(got,ref)
                diagnostics['halo24_vs48']=metric
                checks['context_growth']=metric['nrmse']<=P6_GATES['worst_core_nrmse'] and metric['p95_abs_over_std']<=P6_GATES['prediction_p95']
                child_values=[];child_ref=[]
                axis=int(np.argmax(patch.core_stop-patch.core_start));middle=int((patch.core_stop[axis]+patch.core_start[axis])//2)
                for high in [False,True]:
                    lo=patch.core_start.copy();hi=patch.core_stop.copy()
                    use=patch.authoritative_frac_index_global[:,axis]>=middle
                    if high:lo[axis]=middle
                    else:hi[axis]=middle;use=~use
                    if not np.any(use):continue
                    child=adapter.extract_bounds(cap=patch.cap,core_start=lo,core_stop=hi,context_halo_voxels=24,channel_names=u.CHANNELS,alignment_voxels=8,core_id=core,fold=patch.fold,authoritative_parent_id=pid[use],authoritative_frac_index_global=patch.authoritative_frac_index_global[use])
                    child_values.append(predict(model,child,ck));child_ref.append(got[use])
                metric=compare(np.concatenate(child_values),np.concatenate(child_ref));diagnostics['subdivision']=metric
                checks['subdivision']=metric['nrmse']<=P6_GATES['subdivision_nrmse'] and metric['p95_abs_over_std']<=P6_GATES['subdivision_p95']
            checks['finite_ordered']=bool(np.isfinite(got).all() and (np.diff(got,axis=1)>=0).all())
            run['cores'].append(dict(**meta,checks=checks,diagnostics=diagnostics))
            print(mode,core,checks,flush=True)
        loader.close();result['runs'].append(run);del model
    result['pass_checks']=all(all(c['checks'].values()) for r in result['runs'] for c in r['cores'])
    result['elapsed_seconds']=time.monotonic()-start
    if source!=record(__file__,small=True):raise ValueError('source changed')
    with output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print('pass_checks',result['pass_checks'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    replay(a.output)
