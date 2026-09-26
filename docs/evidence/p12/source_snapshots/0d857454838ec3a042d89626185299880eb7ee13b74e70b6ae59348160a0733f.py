"""Golden mock replay of the observer patch adapter; no DESI posterior sampling."""
import sys,os,json,hashlib,dataclasses,argparse,importlib.util
from pathlib import Path
import numpy as np
import torch
import fitsio
ILL=Path('/global/u2/d/dkololgi/TNG/Illustris')
sys.path.insert(0,str(ILL))
spec=importlib.util.spec_from_file_location('observer',Path(__file__).with_name('p12a_observation_patch.py'));obs=importlib.util.module_from_spec(spec);spec.loader.exec_module(obs)
from workflows.abacus_tweb.p10_training_contract import P10PhaseBalancedLoader
from workflows.abacus_tweb import p8_train_unet_patch as u
from workflows.sbi.p12a_frozen_mock_replay import pick,predict
from workflows.sbi.p12a_blind_inference import reconstruct_fmpe
from workflows.sbi.p12_prepare_base_response_dataset import sample_random_support_distance
from workflows.sbi.p12_export_unet_summaries import ntilde_at_rows
from workflows.sbi.p12_train_base_response_fmpe import sample_posterior
B=Path('/pscratch/sd/d/dkololgi/abacus/p10_multiphase');R=B/'p12a_halo48_candidate_20260924_v1'
def read(p):return json.loads(Path(p).read_text())
def main(output):
    if output.exists():raise FileExistsError(output)
    torch.set_num_threads(8);torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False
    sr=read(R/'summaries/ph006/OOF_SUMMARY_COMPLETE.json');loader=P10PhaseBalancedLoader(Path(sr['contract_root']),include_blind=False);adapter=loader.field_adapter('ph006')
    p3=read(adapter.manifest['p3_manifest']);p1=read(p3['p1_manifest']);selection=read(Path(sr['contract_root'])/'transforms/field/selection_manifest.json')
    idx=np.load(p3['canonical_index']);print('index',idx.files,flush=True)
    points=np.load(p3['points'],mmap_mode='r');parent=np.load(sr['arrays']['parent_node_id']);order=np.argsort(parent);old=np.load(sr['arrays']['base_prediction'],mmap_mode='r')
    random_manifest=read(B/'training_contract_r1_random/adapters/ph006/field/adapter_manifest.json')
    response=np.load(sr['arrays']['response']);distance=np.load(B/'p12a_random_support_parent_cache_v2/ph006/distance_to_support_boundary_mpc.npy',mmap_mode='r')
    schema=read(p3['frozen_schema']);spline=read(p3['ntilde_spline']);angular=np.load(p3['angular_support']['path'])['support']
    ck=torch.load(sr['checkpoint'],map_location='cuda',weights_only=False);model=u.UPatch().cuda().eval();model.load_state_dict(ck['state_dict'])
    chosen=pick(adapter,selection,[r.core_id for r in loader.validation_refs()]);runs=[];contexts=[];references=[]
    with fitsio.FITS(p1['parent']) as f:
        print('columns',f[1].get_colnames()[:35],flush=True)
        for meta in chosen:
            patch=adapter.extract(meta['core'],48,u.CHANNELS,alignment_voxels=8);cap=meta['cap'];name='NGC' if cap else 'SGC';grid=adapter.manifest['caps'][name]
            frac=(points[:,:3]-grid['origin_mpc'])/grid['cell_mpc']-.5
            choose=(idx['cap']==cap)&idx['context']&np.all((frac>=patch.context_start-1)&(frac<patch.context_stop),axis=1)
            rows=np.flatnonzero(choose);d=f[1].read(rows=rows,columns=['RA','DEC','Z','TARGETID'])
            xyz=obs.observer_xyz(d['RA'],d['DEC'],d['Z']);coordinate_error=float(abs(xyz-points[rows,:3]).max())
            fields=obs.rebuild_fields(xyz,idx['shell'][rows],grid,patch.context_start,patch.context_stop,angular,p3['angular_support']['nside'],schema,spline,selection,0,name)
            values=np.stack([fields[n] for n in patch.channel_names]);errors={n:float(abs(values[i]-patch.values[i]).max()) for i,n in enumerate(patch.channel_names)}
            new=dataclasses.replace(patch,values=values);got=predict(model,new,ck);ref=predict(model,patch,ck);at=order[np.searchsorted(parent[order],patch.authoritative_parent_id)]
            checks={'coordinates':coordinate_error<=1e-8,'fields':all(np.allclose(values[i],patch.values[i],atol=1e-5,rtol=1e-5) for i in range(len(values))),'encoder':bool(np.allclose(got,ref,atol=1e-5,rtol=1e-5)),'stored_export':bool(np.allclose(ref,old[at],atol=1e-5,rtol=1e-5))}
            sample=np.linspace(0,len(at)-1,min(8,len(at)),dtype=int);at=at[sample]
            direct=f[1].read(rows=parent[at],columns=['TARGETID','Z','ZWARN'])
            dist,support=sample_random_support_distance(random_manifest,points,parent[at])
            nt=ntilde_at_rows(selection,response['cap'][at],direct['Z'].astype('f4'))
            checks['targetid']=bool(np.array_equal(direct['TARGETID'],idx['targetid'][parent[at]]))
            checks['redshift_success']=bool(np.all(direct['ZWARN']==0))
            checks['redshift']=bool(np.array_equal(direct['Z'].astype('f4'),response['redshift'][at]))
            checks['ntilde']=bool(np.array_equal(nt,response['ntilde_mpc3'][at]))
            checks['random_support_distance']=bool(np.array_equal(dist,distance[at]))
            checks['random_support']=bool(np.all(support))
            response_x=np.column_stack([direct['Z'].astype('f4'),np.log(nt),response['cap'][at],np.log1p(dist)])
            contexts.append(np.column_stack([got[sample],response_x]));references.append(np.column_stack([ref[sample],response_x]))
            runs.append(dict(**meta,rows_redeposited=len(rows),coordinate_max_abs=coordinate_error,field_max_abs=errors,checks=checks));print(runs[-1],flush=True)
    posterior,pck=reconstruct_fmpe(R/'posterior/fmpe_estimator.pt','cuda');draws=[]
    for x in [np.concatenate(contexts),np.concatenate(references)]:
        x=((x-pck['context_mean'])/pck['context_std']).astype('f4');torch.manual_seed(20260925);torch.cuda.manual_seed_all(20260925)
        draws.append(sample_posterior(posterior,x,32,8,'cuda'))
    result=dict(schema='p12a-observer-golden-replay-v1',phase='ph006',runs=runs,posterior_scaled_max_abs=float(abs(draws[0]-draws[1]).max()),posterior_parity=bool(np.allclose(*draws,atol=1e-5,rtol=1e-5)),observer_fields_pass=all(all(r['checks'].values()) for r in runs),random_response='independent P3b field sampling agrees with cache on 64 output galaxies; Loa field generation remains a separate input step',loa_source_refreeze_complete=False,ready_for_desi_canary=False,job=os.environ.get('SLURM_JOB_ID'))
    result['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('p12a_observation_patch.py')]}
    result['checkpoint_sha256']=hashlib.sha256((R/'posterior/fmpe_estimator.pt').read_bytes()).hexdigest()
    output.write_text(json.dumps(result,indent=2)+'\n');loader.close();print('DONE',result['observer_fields_pass'],result['posterior_parity'],flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.output)
