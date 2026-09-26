"""Frozen-weight halo48 candidate benchmark and isolated dataset preparation."""
import argparse,hashlib,json,sys,time
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
        for key in ['parent_node_id','truth','response']:
            if r['array_sha256'][key]!=old['array_sha256'][key]:
                # ZIP archive headers can differ; response contents must be exact.
                if key!='response':raise ValueError(f'{phase} {key} changed')
                with np.load(r['arrays'][key]) as a,np.load(old['arrays'][key]) as b:
                    if a.files!=b.files or not all(np.array_equal(a[k],b[k]) for k in a.files):raise ValueError('response changed')
    d.SUMMARY_ROOT=root/'summaries'
    sys.argv=['prepare','--output-root',str(root/'dataset'),'--workers','1']
    d.main()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['benchmark','prepare']);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    benchmark(a.root) if a.mode=='benchmark' else prepare(a.root)
