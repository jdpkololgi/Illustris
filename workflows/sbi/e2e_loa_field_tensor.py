"""Double-precision GPU implementation of the existing finite-context FFT tensor."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from scipy.ndimage import map_coordinates
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_operators as op
from workflows.sbi.e2e_coupled_physical_gate import eigenvalues


@torch.no_grad()
def tensor(delta,cell=6.766,crop=None):
    x=torch.as_tensor(delta,dtype=torch.float64,device='cuda')
    spectrum=torch.fft.rfftn(x)
    axes=[2*np.pi*torch.fft.fftfreq(n,d=cell,dtype=torch.float64,device='cuda') for n in x.shape[:2]]
    axes.append(2*np.pi*torch.fft.rfftfreq(x.shape[2],d=cell,dtype=torch.float64,device='cuda'))
    ks=(axes[0][:,None,None],axes[1][None,:,None],axes[2][None,None,:])
    k2=sum(k*k for k in ks);k2[0,0,0]=1;result=[]
    for a,b in op.COMPONENTS:
        work=spectrum*(ks[a]*ks[b]/k2)
        work[0,0,0]=spectrum[0,0,0]/3 if a==b else 0
        if a!=b:
            for axis in (a,b):
                if x.shape[axis]%2==0:
                    index=[slice(None)]*3;index[axis]=x.shape[axis]//2;work[tuple(index)]=0
        field=torch.fft.irfftn(work,s=x.shape)
        result.append(field if crop is None else field[crop].clone())
    return torch.stack(result,dim=-1).cpu().numpy()


def consistent(rho,wide):
    crop=tuple(slice(*s) for s in op.layout()['joint_coarse_crop_in_wide'])
    wide_crop=tuple(slice(4*s.start,4*s.stop) for s in crop)
    delta=np.asarray(rho,dtype=np.float64)-1;w=np.asarray(wide,dtype=np.float64)-1
    result=tensor(op.lift(w),crop=wide_crop)+tensor(delta-op.lift(w[crop]))
    trace=np.max(abs(result[...,[0,3,5]].sum(-1)-delta))
    if not np.isfinite(result).all() or trace>2e-10*max(1.,np.max(abs(delta))):
        raise ValueError('GPU physical trace identity failed')
    return result


def at_galaxies(field,xyz,grid,midpoint):
    origin=np.asarray(grid['origin_mpc_h'])+(np.asarray(midpoint)-[64,48,48])*3.383
    position=((xyz-origin)/6.766-.5).T
    if np.any(position<0) or np.any(position>np.asarray(field.shape[:3])[:,None]-1):
        raise ValueError('galaxy interpolation outside field; no extrapolation allowed')
    values=np.stack([map_coordinates(field[...,i],position,order=1,mode='nearest',prefilter=False)
                     for i in range(6)],axis=-1)
    return eigenvalues(values)


def check(root):
    c.require_compute();rows=[];rng=np.random.default_rng(3041)
    for shape in ((12,16,8),(9,11,13)):
        delta=rng.normal(size=shape)+.4
        a=op.tensor_from_delta(delta,6.766,workers=4);b=tensor(delta)
        rows.append(dict(kind='rectangular',shape=list(shape),max_abs=float(np.max(abs(a-b))),
                         passed=bool(np.allclose(a,b,rtol=1e-10,atol=1e-10))))
    with np.load(root/'benchmark/fp32.npz') as f:rho,wide=f['rho'][0],f['wide'][0]
    start=time.monotonic();a=op.consistent_tensor(rho-1,wide-1,op.layout()['joint_coarse_crop_in_wide'],workers=4)
    cpu=time.monotonic()-start;start=time.monotonic();b=consistent(rho,wide);gpu=time.monotonic()-start
    rows.append(dict(kind='full_physical',max_abs=float(np.max(abs(a-b))),
                     passed=bool(np.allclose(a,b,rtol=1e-10,atol=1e-10)),cpu_seconds=cpu,gpu_seconds=gpu))
    record=dict(checks=rows,passed=all(r['passed'] for r in rows),source=c.file_record(__file__,content_hash=True),
                reference_source=c.file_record(op.__file__,content_hash=True),precision='float64 FFT, unchanged operator')
    c.atomic_json(root/'TENSOR_CHECKS.json',record);print(json.dumps(record),flush=True)
    if not record['passed']:raise ValueError('GPU tensor parity failed')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();check(a.root)
