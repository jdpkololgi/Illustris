"""Finite disposable crossover on the bound encoder inputs; no production state."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from workflows.abacus_tweb import p10_train_arm_a as a
from workflows.sbi import e2e_wholefield_pilot_train as t


def main():
    t.setup(42)
    assert 'A100' in torch.cuda.get_device_name() and torch.cuda.get_device_properties(0).total_memory>75*2**30
    out=Path(os.environ['PROFILE_OUT'])
    b=json.loads(Path('configs/lowz_retraining_binding_20261118.json').read_text())['p12a_encoder']
    def checked(path,digest):
        raw=Path(path).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==digest
        return json.loads(raw)
    m=checked(b['manifest'],b['sha256'])
    counts={}
    for key,v in m['contracts'].items():
        root=Path(v['path'])
        roles=checked(root/'TRAINING_LOADER_READY.json',v['sha256'])['roles']
        counts[key]=sum(len(np.load(root/'phases'/p/'training_core_id.npy',mmap_mode='r')) for p in roles['training'])
    assert len(counts)==21
    loader=a.P10PhaseBalancedLoader(m['contracts']['full']['path'])
    phases=list(loader.training_phases[::7])
    assert len(phases)==3
    states={p:a.prepare_phase_runtime(loader,p,loader.target_scaler,training=True) for p in phases}
    adapters={p:loader.field_adapter(p) for p in phases}
    norm=loader.field_normalization
    rng=np.random.default_rng(20261118)
    cores={p:rng.permutation(list(states[p].expected_core_weight))[:36] for p in phases}
    u=a.unet_impl
    def prepare(ref):
        p,k=ref
        patch=adapters[p].extract(k,u.HALO_VOXELS,u.CHANNELS,alignment_voxels=u.ALIGNMENT_VOXELS)
        x,pts=u.model_inputs(patch,norm,'cpu')
        ids=patch.authoritative_parent_id
        return x,pts,torch.from_numpy(states[p].target_scaled[ids]),torch.from_numpy(states[p].parent_weight[ids])
    def trial(refs,prefetch):
        torch.manual_seed(42)
        model=u.UPatch(base=24,latent_channels=32).cuda()
        opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0001)
        sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=counts['full'])
        def update(ref,item):
            x,pts,y,w=[v.cuda() for v in item]
            pred=model(x,pts)
            loss=a.phase_equal_patch_objective((w*(pred-y).square().mean(1)).sum(),
                phase_weight_denominator=states[ref[0]].weight_denominator,
                phase_objective_scale=counts['full']/len(loader.training_phases))
            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True)
            opt.step();sched.step()
            return float(loss.detach())
        losses=[update(ref,prepare(ref)) for ref in refs[:3]]
        timed=refs[3:]
        torch.cuda.synchronize()
        start=time.perf_counter()
        with ThreadPoolExecutor(max_workers=1) as pool:
            fut=pool.submit(prepare,timed[0]) if prefetch else None
            for j,ref in enumerate(timed):
                item=fut.result() if prefetch else prepare(ref)
                if prefetch and j+1<len(timed):
                    fut=pool.submit(prepare,timed[j+1])
                losses.append(update(ref,item))
        torch.cuda.synchronize()
        elapsed=time.perf_counter()-start
        assert np.isfinite(losses).all()
        digest=hashlib.sha256()
        for v in model.state_dict().values():
            digest.update(v.detach().cpu().contiguous().numpy().tobytes())
        return dict(seconds_per_core=elapsed/len(timed),losses=losses,weights_sha256=digest.hexdigest())
    blocks=[]
    for block in range(4):
        refs=[(p,int(cores[p][i])) for i in range(block*9,block*9+9) for p in phases]
        order=['baseline','prefetch'] if block%2==0 else ['prefetch','baseline']
        rows={k:trial(refs,k=='prefetch') for k in order}
        x,y=rows['baseline'],rows['prefetch']
        assert x['losses']==y['losses'] and x['weights_sha256']==y['weights_sha256'], 'Numerical mismatch'
        blocks.append(dict(order=order,refs=refs,**rows))
        print(block, {k:r['seconds_per_core'] for k,r in rows.items()},flush=True)
    means={k:float(np.mean([z[k]['seconds_per_core'] for z in blocks])) for k in order}
    data=dict(means=means,blocks=blocks,full_epoch=counts['full'],all_epochs=sum(counts.values()),
              phases=phases,timed_per_arm=96,bitwise_equal=True)
    (out/'RESULT.json').write_text(json.dumps(data,indent=2))
    loader.close()


if __name__=='__main__':main()
