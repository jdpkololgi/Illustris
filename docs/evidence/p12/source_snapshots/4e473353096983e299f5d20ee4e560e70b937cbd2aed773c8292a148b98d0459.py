"""Diagnostic frozen-weight halo48 control; not a deployed model replacement."""
import argparse,json,os
from pathlib import Path
import numpy as np
import torch
from workflows.sbi.p12a_frozen_mock_replay import BASE,ROOT,read,predict,compare,P6_GATES,u,P10PhaseBalancedLoader,record

def main(output):
    if not torch.cuda.is_available() or not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('GPU allocation required')
    source=record(__file__,small=True);helper=record(ROOT/'workflows/sbi/p12a_frozen_mock_replay.py',small=True)
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    previous=ROOT/'docs/evidence/p12/P12A_FROZEN_MOCK_REPLAY_20260924_v2.json'
    selected=read(previous)['runs'][1]['selected']
    checkpoint=BASE/'arm_a_training/arm_a_r0_v1/unet/seed_42/best_checkpoint.pt'
    ck=torch.load(checkpoint,map_location='cuda',weights_only=False)
    model=u.UPatch().cuda().eval();model.load_state_dict(ck['state_dict'])
    loader=P10PhaseBalancedLoader(BASE/'training_contract',include_blind=False);adapter=loader.field_adapter('ph002')
    cores=[];all48=[];all64=[];allchild=[];allparent=[]
    for meta in selected:
        core=meta['core'];patch=adapter.extract(core,48,u.CHANNELS,alignment_voxels=8)
        got=predict(model,patch,ck);ref=predict(model,adapter.extract(core,64,u.CHANNELS,alignment_voxels=8),ck)
        growth=compare(got,ref);all48.append(got);all64.append(ref)
        axis=int(np.argmax(patch.core_stop-patch.core_start));middle=int((patch.core_stop[axis]+patch.core_start[axis])//2)
        children=[];parents=[]
        for high in [False,True]:
            lo=patch.core_start.copy();hi=patch.core_stop.copy();use=patch.authoritative_frac_index_global[:,axis]>=middle
            if high:lo[axis]=middle
            else:hi[axis]=middle;use=~use
            if not np.any(use):continue
            child=adapter.extract_bounds(cap=patch.cap,core_start=lo,core_stop=hi,context_halo_voxels=48,channel_names=u.CHANNELS,alignment_voxels=8,core_id=core,fold=patch.fold,authoritative_parent_id=patch.authoritative_parent_id[use],authoritative_frac_index_global=patch.authoritative_frac_index_global[use])
            children.append(predict(model,child,ck));parents.append(got[use])
        child=np.concatenate(children);parent=np.concatenate(parents);sub=compare(child,parent);allchild.append(child);allparent.append(parent)
        cores.append(dict(**meta,growth=growth,subdivision=sub));print(core,growth['nrmse'],sub['nrmse'],flush=True)
    aggregate_growth=compare(np.concatenate(all48),np.concatenate(all64));aggregate_sub=compare(np.concatenate(allchild),np.concatenate(allparent))
    checks=dict(growth=aggregate_growth['nrmse']<=P6_GATES['prediction_nrmse'] and aggregate_growth['p95_abs_over_std']<=P6_GATES['prediction_p95'],worst_core=max(c['growth']['nrmse'] for c in cores)<=P6_GATES['worst_core_nrmse'],subdivision=aggregate_sub['nrmse']<=P6_GATES['subdivision_nrmse'] and aggregate_sub['p95_abs_over_std']<=P6_GATES['subdivision_p95'])
    r=dict(schema='p12a-halo48-control-v1',source=source,helper=helper,selection_receipt=record(previous,small=True),checkpoint=record(checkpoint,small=True),cores=cores,aggregate_growth=aggregate_growth,aggregate_subdivision=aggregate_sub,checks=checks,pass_checks=all(checks.values()),deployed_halo_changed=False,ready_for_desi_canary=False,scope='eight previously selected dense training cores; no new summaries, posterior fitting, edge/sparse qualification or real-data inference')
    loader.close()
    if source!=record(__file__,small=True) or helper!=record(ROOT/'workflows/sbi/p12a_frozen_mock_replay.py',small=True):raise ValueError('source changed')
    with output.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
    print(checks,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    main(a.output)
