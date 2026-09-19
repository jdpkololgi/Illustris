"""Rectangular coupled-field physical charts and exact local interpolation."""
import gc
import json
import numpy as np
from scipy import fft

from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_coordinates as coord
from workflows.sbi.e2e_vdm_context_products import averaged_cubic_operator

LAYOUT=c.REPO/'configs/e2e_coupled_product_layout_v1.json'
COMPONENTS=((0,0),(0,1),(0,2),(1,1),(1,2),(2,2))


def layout():
    v=json.loads(LAYOUT.read_text())
    if v['schema']!='e2e-coupled-product-layout-v1' or v['training_authorized']:
        raise ValueError('unregistered coupled layout')
    return v


def blocks(x,factor=4):
    x=np.asarray(x)
    if x.ndim!=3 or any(n%factor for n in x.shape):
        raise ValueError('rectangular 3D block-aligned field required')
    d,h,w=x.shape
    return x.reshape(d//factor,factor,h//factor,factor,w//factor,factor)


def mean_pool(x,factor=4):
    return blocks(x,factor).mean((1,3,5))


def lift(x,factor=4):
    for axis in range(3):
        x=np.repeat(x,factor,axis=axis)
    return x


def project(x,factor=4):
    return np.asarray(x)-lift(mean_pool(x,factor),factor)


def encode(rho,factor=4):
    rho=np.asarray(rho,dtype=np.float64)
    if not np.isfinite(rho).all() or np.any(rho<=0):
        raise ValueError('positive finite physical density required; no clipping')
    return mean_pool(rho,factor),project(np.log(rho),factor)


def decode(coarse,residual,factor=4):
    coarse=np.asarray(coarse,dtype=np.float64)
    residual=np.asarray(residual,dtype=np.float64)
    if residual.shape!=tuple(factor*n for n in coarse.shape) or not np.isfinite(residual).all():
        raise ValueError('coarse/residual shape or finiteness failure')
    if not np.isfinite(coarse).all() or np.any(coarse<=0):
        raise ValueError('invalid positive coarse density')
    maximum=blocks(residual,factor).max((1,3,5))
    weights=np.exp(residual-lift(maximum,factor))
    return weights*lift(coarse/mean_pool(weights,factor),factor)


def sample_averaged_local(field,raw_starts,average,grid):
    """Same cubic tensor product as the previous builder, restricted to support.

    Avoids rebuilding an entire native2048² plane for every output sample.
    It is an exact sparse restriction, not a different interpolation/filter.
    """
    coord.validate_grid(grid)
    # Historical kernel names; all values are Mpc/h, multiplier exactly ONE.
    cfg=dict(raw_cell_mpc=grid['cell_mpc_h'],coordinate_h=1.,box_offset_mpc_h=-1000.,box_mpc_h=2000.)
    operators=[averaged_cubic_operator(starts,average,grid['origin_mpc_h'][a],cfg,field.shape[a])
               for a,starts in enumerate(raw_starts)]
    x,y,z=operators
    yi,zi=np.unique(y.indices),np.unique(z.indices)
    y,z=y[:,yi],z[:,zi]
    out=np.empty(tuple(op.shape[0] for op in operators),dtype=np.float64)
    for i in range(x.shape[0]):
        lo,hi=x.indptr[i:i+2]
        plane=np.zeros((len(yi),len(zi)),dtype=np.float64)
        for index,weight in zip(x.indices[lo:hi],x.data[lo:hi]):
            plane+=weight*np.asarray(field[index])[np.ix_(yi,zi)]
        out[i]=z.dot(y.dot(plane).T).T
    return out


def tensor_from_delta(delta,cell,workers=1,dc=True):
    """Rectangular spectral tensor with the inherited isotropic patch-DC closure.

    A finite patch mean is physical, unlike the zero global periodic-box mean.
    Adding mean(delta)/3 to each diagonal preserves its trace, not exterior tides.
    """
    delta=np.asarray(delta,dtype=np.float64)
    if delta.ndim!=3 or not np.isfinite(delta).all():
        raise ValueError('finite 3D density required')
    spectrum=fft.rfftn(delta,workers=workers)
    axes=[2*np.pi*fft.fftfreq(n,d=cell) for n in delta.shape[:2]]
    axes.append(2*np.pi*fft.rfftfreq(delta.shape[2],d=cell))
    ks=(axes[0][:,None,None],axes[1][None,:,None],axes[2][None,None,:])
    k2=sum(k*k for k in ks); k2[0,0,0]=1
    result=[]
    for a,b in COMPONENTS:
        work=spectrum*(ks[a]*ks[b]/k2)
        work[0,0,0]=spectrum[0,0,0]/3 if dc and a==b else 0
        if a!=b:
            for axis in (a,b):
                if delta.shape[axis]%2==0:
                    index=[slice(None)]*3; index[axis]=delta.shape[axis]//2
                    work[tuple(index)]=0
        result.append(fft.irfftn(work,s=delta.shape,workers=workers))
    return np.stack(result,axis=-1)


def consistent_tensor(delta,wide_delta,crop,cell=6.766,factor=4,workers=1):
    slices=tuple(slice(*s) for s in crop)
    background=lift(wide_delta[slices],factor)
    if background.shape!=delta.shape:
        raise ValueError('wide/joint crop does not align')
    wide_tensor=tensor_from_delta(lift(wide_delta,factor),cell,workers)
    tensor_slices=tuple(slice(factor*s.start,factor*s.stop) for s in slices)
    return wide_tensor[tensor_slices]+tensor_from_delta(delta-background,cell,workers)


def smoothed_spectrum(count_path,workers=32):
    c.require_compute()
    counts=np.load(c.guarded(count_path),mmap_mode='r',allow_pickle=False)
    n=counts.shape[0]
    if counts.shape!=(2048,)*3 or counts.dtype!=np.float32:
        raise ValueError('native2048 float32 counts required')
    mean=float(counts.mean(dtype=np.float64))
    if not mean>0:
        raise ValueError('nonpositive native mean')
    delta=np.empty(counts.shape,dtype=np.float64)
    for i in range(0,n,16):
        x=counts[i:i+16]
        if not np.isfinite(x).all() or np.min(x)<0:
            raise ValueError('invalid count payload')
        delta[i:i+16]=x/mean-1
    spectrum=fft.rfftn(delta,workers=workers,overwrite_x=True)
    del delta,counts; gc.collect()
    k=2*np.pi*fft.fftfreq(n,d=2000./n)
    kz=2*np.pi*fft.rfftfreq(n,d=2000./n)
    for i in range(n):
        spectrum[i]*=np.exp(-.5*7.**2*(k[i]**2+k[:,None]**2+kz[None,:]**2))
    spectrum[0,0,0]=0
    return spectrum,mean
