"""Fresh processes, real full low-z contract, disposable weights only."""
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
from workflows.abacus_tweb.p10_train_arm_a import sha256


def equal(x,y):
    if type(x) is not type(y): return False
    if isinstance(x,dict): return x.keys()==y.keys() and all(equal(v,y[k]) for k,v in x.items())
    if isinstance(x,(tuple,list)): return len(x)==len(y) and all(equal(v,w) for v,w in zip(x,y))
    if isinstance(x,torch.Tensor): return torch.equal(x,y)
    if isinstance(x,np.ndarray): return np.array_equal(x,y)
    return x==y


def main():
    out=Path(os.environ['RESTART_OUT'])
    binding=json.loads(Path('configs/lowz_retraining_binding_20261118.json').read_text())['p12a_encoder']
    assert sha256(Path(binding['manifest']))==binding['sha256']
    full=json.loads(Path(binding['manifest']).read_text())['contracts']['full']
    assert sha256(Path(full['path'])/'TRAINING_LOADER_READY.json')==full['sha256']
    def train(name,n,resume=False):
        print('trainer',name,n,resume,flush=True)
        subprocess.run([sys.executable,'-u','-m','workflows.abacus_tweb.p10_train_arm_a',
            '--contract-root',full['path'],'--output-root',str(out),'--run-name',name,
            '--model','unet','--seed','42','--ordered-prefetch','--disable-early-stopping',
            '--epochs','2','--min-epochs','2','--loss-log-every','1','--stop-after-updates',str(n),
            *(['--auto-resume'] if resume else [])],check=True)
    train('continuous',24)
    train('split',11)
    train('split',24,True)
    def d(name):return out/name/'unet/seed_42'
    x,y=[torch.load(d(k)/'arm_a_checkpoint.pt',map_location='cpu',weights_only=False) for k in ('continuous','split')]
    skip={'created_utc','maximum_memory','frozen_arguments'}
    args=[dict(v['frozen_arguments']) for v in (x,y)]
    for v in args:v.pop('run_name')
    assert args[0]==args[1] and x.keys()==y.keys() and x['global_step']==y['global_step']==24
    for k in x.keys()-skip: assert equal(x[k],y[k]),k
    assert (d('split')/'loss_trace.jsonl').read_bytes()==(d('continuous')/'loss_trace.jsonl').read_bytes()
    (out/'PASS.json').write_text(json.dumps(dict(steps=24,interrupted_at=11,
        equal_keys=sorted(x.keys()-skip),loss_trace_equal=True,production=False),indent=2))


if __name__=='__main__':main()
