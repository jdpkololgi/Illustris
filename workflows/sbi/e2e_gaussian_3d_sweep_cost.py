"""Direct whole-draw and entire-block-sweep timing, no posterior claim."""
import argparse
import gc
import os
from pathlib import Path
import socket
import time
import numpy as np
import torch
import e2e_gaussian_3d_cfm as g
import e2e_gaussian_3d_run as run
import e2e_true_coarse_fraction_saved as io


@torch.no_grad()
def measure(model,n,arm):
    data=g.JointGenerator(n);generator,_=run.generators()
    x,observation=data.batch(1,generator)
    width=n if arm=='whole' else 8
    noise=torch.randn((1,1,width,width,width),device='cuda',generator=generator)
    g.draw(model,observation,x,(0,)*3,width,noise,32)
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();times=[]
    for _ in range(3):
        started=time.monotonic()
        if arm=='whole':result=g.draw(model,observation,x,(0,)*3,n,noise,32)
        else:result=g.sweep(model,observation,x,generator,32)
        torch.cuda.synchronize();times.append(time.monotonic()-started)
        del result
    return dict(n=n,arm=arm,batch=1,nfe_per_solve=32,repetitions=3,
        blocks=1 if arm=='whole' else (n//8)**3,seconds=times,
        mean_seconds=float(np.mean(times)),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        measurement='one whole draw' if arm=='whole' else 'one entire sequential sweep',
        trained=False,accuracy_qualified=False)


def main(output):
    io.compute_guard();torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    rows=[];sources=[io.record(p) for p in (__file__,g.__file__,run.__file__)]
    for n in (8,16,24,32,48):
        for arm in ('whole','block'):
            model=run.initialize().eval();row=measure(model,n,arm);rows.append(row)
            print('ACTUAL SWEEP COST',row,flush=True)
            del model;gc.collect();torch.cuda.empty_cache()
    for source in sources:
        if io.sha(source['path'])!=source['sha256']:raise AssertionError('Source changed')
    io.write(output,dict(engineering_complete=True,rows=rows,sources=sources,
        job=os.environ['SLURM_JOB_ID'],node=socket.gethostname(),gpu=torch.cuda.get_device_name(),
        purpose='Direct cost confirmation, not additional training or accuracy selection'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);main(Path(p.parse_args().output))
