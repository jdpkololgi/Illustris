"""Frozen-weight halo48 candidate benchmark and isolated dataset preparation."""
import argparse,hashlib,json,sys,time,subprocess
from pathlib import Path
import numpy as np
import torch
from workflows.abacus_tweb import p8_train_unet_patch as u
from workflows.abacus_tweb.p10_training_contract import P10PhaseBalancedLoader
from workflows.abacus_tweb.p8_deterministic_common import increments_to_eigenvalues,unscale_increments,sha256
BASE=Path('/pscratch/sd/d/dkololgi/abacus/p10_multiphase')
PHASES=['ph000','ph002','ph003','ph004','ph005','ph006']

def read(p):return json.loads(Path(p).read_text())

def legacy(phase):return read(BASE/f'p12_oof_summaries/{phase}/OOF_SUMMARY_COMPLETE.json')

def benchmark(root):
    if not torch.cuda.is_available():raise RuntimeError('GPU required')
    torch.set_num_threads(8);torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False
    root.mkdir(parents=True,exist_ok=True)
    for phase in PHASES:
        out=root/f'{phase}.json'
        if out.exists():raise FileExistsError(out)
        sr=legacy(phase);contract=Path(sr['contract_root']);cp=Path(sr['checkpoint'])
        ck=torch.load(cp,map_location='cuda',weights_only=False);model=u.UPatch().cuda().eval();model.load_state_dict(ck['state_dict'])
        loader=P10PhaseBalancedLoader(contract,include_blind=False);adapter=loader.field_adapter(phase)
        selection=read(contract/'transforms/field/selection_manifest.json');groups={}
        refs=loader.validation_refs()
        for ref in refs:
            core=ref.core_id;cap=int(adapter.core_cap[core]);grid=adapter.manifest['caps']['NGC' if cap else 'SGC']
            center=np.asarray(grid['origin_mpc'])+(adapter.core_start[core]+adapter.core_stop[core])*grid['cell_mpc']/2
            z=np.interp(np.linalg.norm(center),selection['cosmology']['radius_grid_mpc'],selection['cosmology']['redshift_grid']);shell=int(np.searchsorted([.15,.25,.35,.45,.55],z,side='right')-1)
            count=int(adapter.core_offsets[core+1]-adapter.core_offsets[core])
            if shell in range(4) and count>0:groups.setdefault((cap,shell),[]).append((count,core))
        if len(groups)!=8:raise ValueError('missing benchmark strata')
        selected=[]
        for key,group in sorted(groups.items()):
            group.sort()
            for i in np.unique(np.linspace(0,len(group)-1,4,dtype=int)):selected.append(dict(cap=key[0],shell=key[1],core=group[i][1],rows=group[i][0]))
        oldparent=np.load(sr['arrays']['parent_node_id'],mmap_mode='r');order=np.argsort(oldparent);oldsorted=oldparent[order];oldbase=np.load(sr['arrays']['base_prediction'],mmap_mode='r')
        times=[];checks=[];torch.cuda.reset_peak_memory_stats()
        for j,meta in enumerate(selected):
            if j==0:
                patch=adapter.extract(meta['core'],24,u.CHANNELS,alignment_voxels=8);x,p=u.model_inputs(patch,ck['normalization'],'cuda')
                with torch.inference_mode():scaled=model(x,p).cpu().numpy()
                got=increments_to_eigenvalues(unscale_increments(scaled,ck['scaler'])).astype('f4');at=order[np.searchsorted(oldsorted,patch.authoritative_parent_id)]
                if not np.array_equal(oldparent[at],patch.authoritative_parent_id):raise ValueError('legacy parent mismatch')
                parity=float(np.max(np.abs(got-oldbase[at])))
                if not np.allclose(got,oldbase[at],atol=1e-5,rtol=1e-5):raise ValueError(f'{phase} legacy parity {parity}')
            torch.cuda.synchronize();start=time.monotonic()
            patch=adapter.extract(meta['core'],48,u.CHANNELS,alignment_voxels=8);x,p=u.model_inputs(patch,ck['normalization'],'cuda')
            with torch.inference_mode():scaled=model(x,p).cpu().numpy()
            got=increments_to_eigenvalues(unscale_increments(scaled,ck['scaler'])).astype('f4');torch.cuda.synchronize();times.append(time.monotonic()-start)
            checks.append(bool(np.isfinite(got).all() and (np.diff(got,axis=1)>=0).all()))
        report=dict(phase=phase,checkpoint_sha256=sha256(cp),source_sha256=sha256(Path(__file__)),context_halo_voxels=48,cores=len(refs),selected=selected,seconds=times,median_seconds=float(np.median(times)),p95_seconds=float(np.quantile(times,.95)),max_gpu_bytes=torch.cuda.max_memory_allocated(),legacy_max_abs=parity,pass_checks=all(checks),ready_for_desi_canary=False)
        with out.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
        print(phase,report['median_seconds'],report['p95_seconds'],report['cores'],report['max_gpu_bytes'],flush=True)
        loader.close();del model,ck;torch.cuda.empty_cache()

def prepare(root):
    from workflows.sbi import p12_prepare_base_response_dataset as d
    for phase in PHASES:
        r=read(root/f'summaries/{phase}/OOF_SUMMARY_COMPLETE.json');old=legacy(phase)
        if not r['pass'] or r.get('context_halo_voxels')!=48 or r.get('alignment_voxels')!=8:raise ValueError('candidate marker mismatch')
        if r['checkpoint_sha256']!=old['checkpoint_sha256']:raise ValueError('encoder changed')
        for key,path in r['arrays'].items():
            if sha256(Path(path))!=r['array_sha256'][key]:raise ValueError(f'{phase} candidate bytes changed: {key}')
        cache_root=BASE/f'p12a_random_support_parent_cache_v2/{phase}'
        cache=read(cache_root/'P12A_RANDOM_SUPPORT_CACHE_READY.json')
        if not cache['pass'] or cache['parent_sha256']!=r['array_sha256']['parent_node_id']:raise ValueError('response cache parent changed')
        if sha256(Path(cache['audit']['field_manifest']))!=cache['field_manifest_sha256']:raise ValueError('response field manifest changed')
        if sha256(cache_root/'distance_to_support_boundary_mpc.npy')!=cache['distance_sha256'] or sha256(cache_root/'support_random.npy')!=cache['support_sha256']:raise ValueError('response cache payload changed')
        for key in ['parent_node_id','truth','response']:
            if r['array_sha256'][key]!=old['array_sha256'][key]:
                # ZIP archive headers can differ; response contents must be exact.
                if key!='response':raise ValueError(f'{phase} {key} changed')
                with np.load(r['arrays'][key]) as a,np.load(old['arrays'][key]) as b:
                    if a.files!=b.files or not all(np.array_equal(a[k],b[k]) for k in a.files):raise ValueError('response changed')
    d.SUMMARY_ROOT=root/'summaries'
    sys.argv=['prepare','--output-root',str(root/'dataset'),'--workers','1']
    d.main()

def stage(mode,root,phase=None):
    manifest=read(root/'RUN_MANIFEST.json')
    if manifest['context_halo_voxels']!=48:raise ValueError('candidate contract changed')
    if mode=='export':
        if phase not in PHASES:raise ValueError('invalid phase')
        sr=manifest['legacy_summaries'][phase]
        if sha256(Path(sr['checkpoint']))!=sr['checkpoint_sha256']:raise ValueError('checkpoint changed')
        subprocess.run([sys.executable,'-m','workflows.sbi.p12_export_unet_summaries',
                        '--contract-root',sr['contract_root'],'--checkpoint',sr['checkpoint'],
                        '--phase',phase,'--context-halo','48','--output-root',str(root/'summaries')],check=True)
    elif mode=='fit':
        prepare(root)
        subprocess.run([sys.executable,'-m','workflows.sbi.p12_train_base_response_fmpe',
                        '--dataset-root',str(root/'dataset'),'--output-root',str(root/'posterior')],check=True)
    elif mode=='audit':
        subprocess.run([sys.executable,'-m','workflows.sbi.p12_calibration_diagnostics',
                        '--dataset-root',str(root/'dataset'),'--checkpoint',str(root/'posterior/fmpe_estimator.pt'),
                        '--output-root',str(root/'posterior/calibration_audit')],check=True)
        old=read(BASE/'p12a_base_response_v1/fmpe_seed42/P12A_COMPLETE.json')
        new=read(root/'posterior/P12A_COMPLETE.json')
        report=dict(schema='p12a-halo48-comparison-pending-review-v1',legacy_untempered=old['untempered'],
                    candidate_untempered=new['untempered'],candidate_calibration_audit=str(root/'posterior/calibration_audit/P12A_CALIBRATION_AUDIT.json'),
                    ready_for_desi_canary=False,science_release_ready=False,
                    outstanding=['scientific_comparison_review','independent_confirmation_exposure_ledger','Loa_adapter_golden_replay'])
        with (root/'COMPARISON_PENDING_REVIEW.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['benchmark','prepare','export','fit','audit']);p.add_argument('--root',type=Path,required=True);p.add_argument('--phase',choices=PHASES);a=p.parse_args()
    if a.mode=='benchmark':benchmark(a.root)
    elif a.mode=='prepare':prepare(a.root)
    else:stage(a.mode,a.root,a.phase)
