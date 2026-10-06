"""Matched coarse-checkpoint replay with fixed fine checkpoint and saved draws.

Historical local+wide sampler/operator: NOT the current full-cap sampler.
No new training or phase exposure. GPU spectral operator verified against CPU.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
import e2e_true_density_closure as b
from e2e_trace_shear_diagnostic import features,assess
import numpy as np
import torch

PARENT=b.ROOT.parent
OUT=b.ROOT/'verification/checkpoint_shear_20261005_v1'


class Spectral:
    def __init__(self,shape):
        self.shape=tuple(shape)
        axes=[torch.fft.fftfreq(n,device='cuda',dtype=torch.float64) for n in shape[:2]]
        axes.append(torch.fft.rfftfreq(shape[2],device='cuda',dtype=torch.float64))
        self.k=(axes[0][:,None,None],axes[1][None,:,None],axes[2][None,None,:])
        self.k2=sum(x*x for x in self.k);self.k2[0,0,0]=1

    def components(self,delta):
        spec=torch.fft.rfftn(delta)
        for a,d in b.op.COMPONENTS:
            work=spec*(self.k[a]*self.k[d]/self.k2)
            work[0,0,0]=spec[0,0,0]/3 if a==d else 0
            if a!=d:
                for axis in (a,d):
                    if self.shape[axis]%2==0:
                        idx=[slice(None)]*3;idx[axis]=self.shape[axis]//2;work[tuple(idx)]=0
            yield torch.fft.irfftn(work,s=self.shape)


class LocalOperator:
    def __init__(self):
        self.wide=Spectral((192,)*3);self.fine=Spectral((64,48,48))
        self.crop=tuple(slice(4*a,4*z) for a,z in b.op.layout()['joint_coarse_crop_in_wide'])
        self.cores=[tuple(slice(*s) for s in crop) for crop in b.op.layout()['owned_core_crops_in_joint']]

    def tensor(self,rho,wide):
        w=torch.as_tensor(wide-1,device='cuda',dtype=torch.float64)
        for axis in range(3):w=w.repeat_interleave(4,dim=axis)
        d=torch.as_tensor(rho-1,device='cuda',dtype=torch.float64)-w[self.crop]
        result=np.empty((2,16,16,16,6))
        for k,(cw,cf) in enumerate(zip(self.wide.components(w),self.fine.components(d))):
            total=cw[self.crop]+cf
            result[...,k]=torch.stack([total[s] for s in self.cores]).cpu().numpy()
        return result


def run(seed,limit):
    b.c.require_compute();torch.set_num_threads(4)
    early=PARENT/'cfm_mixed_20260926_v1/development'/f'seed{seed}_step13312_fine26624'
    late=PARENT/'cfm_pilot_final_eval_20260925_v1/results'/f'seed{seed}_step26624'
    bindings=[json.loads((p/'BINDING.json').read_text()) for p in (early,late)]
    assert bindings[0]['checkpoints']['fine']==bindings[1]['checkpoints']['fine']
    assert bindings[0]['normalizer']==bindings[1]['normalizer']
    assert all(v['draws']==32 and v['nfe']==128 for v in bindings)
    directory=OUT/f'seed{seed}';directory.mkdir(parents=True,exist_ok=True)
    operator=LocalOperator();rows=[];parity=None
    for phase in ('ph012','ph013'):
        geometry=b.auth.geometry(phase,b.FREEZE)
        selected=geometry if limit==0 else geometry[:limit]
        for row in selected:
            dest=directory/(row['pair_id']+'.json')
            if dest.exists():
                rows.append(json.loads(dest.read_text()));continue
            tick=time.monotonic();truth,prov=b.auth.truth(phase,row['pair_id'],b.FREEZE)
            full=b.tidal.eigensystem(truth['fullbox_tensor_cores']).mean((1,2,3))
            wide_truth=truth['coarse_rho_extended'][4:52,4:52,4:52]
            # Registered central wide crop, verified against the layout below.
            assert b.op.layout()['wide_crop_base_start']==[4,4,4]
            matched=b.tidal.eigensystem(operator.tensor(truth['rho_joint'],wide_truth)).mean((1,2,3))
            if parity is None:
                cpu=b.op.consistent_tensor(truth['rho_joint']-1,wide_truth-1,b.op.layout()['joint_coarse_crop_in_wide'],workers=4)
                cpu=np.stack([cpu[s] for s in operator.cores])
                gpu=operator.tensor(truth['rho_joint'],wide_truth)
                parity=float(abs(cpu-gpu).max());assert parity<1e-10
                print(json.dumps(dict(seed=seed,cpu_gpu_maxabs=parity)),flush=True)
            result=dict(phase=phase,pair_id=row['pair_id'],seed=seed,fullbox=full.tolist(),matched=matched.tolist(),arms={})
            for label,source,binding in zip(('coarse13312','coarse26624'),(early,late),bindings):
                folder=source/phase/row['pair_id']/'nfe128'
                completed=json.loads((folder/'COMPLETE.json').read_text())
                assert completed['binding']==b.c.digest(binding)
                eig=[];mass=[]
                for first in range(0,32,2):
                    path=folder/f'draw{first:03d}.npz'
                    assert b.c.sha256(path)==completed['chunks'][path.name]
                    with np.load(path) as f:
                        for rho,wide in zip(f['rho'],f['wide']):
                            tensor=operator.tensor(rho,wide)
                            vals=b.tidal.eigensystem(tensor).mean((1,2,3));eig.append(vals)
                            core=b.op.layout()['owned_core_crops_in_joint']
                            mass.append([rho[tuple(slice(*s) for s in cr)].mean()-1 for cr in core])
                eig=np.asarray(eig);np.testing.assert_allclose(eig.sum(-1),mass,atol=1e-10,rtol=0)
                fs=features(eig)
                result['arms'][label]=dict(fullbox={k:assess(v,features(full)[k]) for k,v in fs.items()},
                   matched={k:assess(v,features(matched)[k]) for k,v in fs.items()},
                   source=b.c.file_record(folder/'COMPLETE.json',content_hash=True))
                np.savez(directory/(row['pair_id']+'_'+label+'.npz'),eigenvalues=eig,mass=np.asarray(mass))
            result['seconds']=time.monotonic()-tick
            b.c.atomic_json(dest,result);rows.append(result)
            print(json.dumps(dict(seed=seed,pair=row['pair_id'],seconds=result['seconds'])),flush=True)
    b.c.atomic_json(directory/('COMPLETE.json' if limit==0 else f'PILOT_{limit}.json'),dict(seed=seed,cases=rows,
        source_sha256=b.c.sha256(__file__),bindings=[b.c.file_record(p/'BINDING.json',content_hash=True) for p in (early,late)],
        job=os.environ['SLURM_JOB_ID'],cpu_gpu_parity=parity,passed=True,
        scope='Local+wide historical sampler; not full-cap. Same fine26624,normalizer,NFE128,draw seeds.' ))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=[17,29],required=True)
    p.add_argument('--limit',type=int,default=0);a=p.parse_args();run(a.seed,a.limit)
