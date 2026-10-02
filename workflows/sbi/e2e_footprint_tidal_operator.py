"""Free-space tidal control for piecewise-constant cubic density cells.

Conventions: Laplacian(Phi)=delta; T=Hessian(Phi), components xx,xy,xz,yy,yz,zz.
The analytic cuboid integrals include the isotropic self term. Linear FFT
convolution is padded for every source-to-target displacement: no images,
periodic-box mean subtraction, fitted exterior, or additional smoothing.
See https://www.fatiando.org/choclo/latest/_modules/choclo/prism/_gravity.html
for the cuboid antiderivatives (gravity-gradient sign is opposite to ours).
"""
import math
import numpy as np
from numba import njit, prange
from scipy import fft

COMPONENTS=((0,0),(0,1),(0,2),(1,1),(1,2),(2,2))


@njit
def cube_tensor(x,y,z):
    """Integral of Hessian(-1/(4pi r)) over a unit source cube.

    x,y,z are target minus source-centre coordinates in cell units. No physical
    cell factor remains: the Hessian scales as h^-3 and volume as h^3.
    Targets on source faces/edges are not accepted by callers.
    """
    out=np.zeros(6,np.float64)
    for i in (-1,1):
        a=x+i*.5
        for j in (-1,1):
            b=y+j*.5
            for k in (-1,1):
                c=z+k*.5;r=math.sqrt(a*a+b*b+c*c);sign=i*j*k/(4*math.pi)
                # Principal atan, not atan2: its branch carries the interior
                # contact term and gives trace=1 inside the source cube.
                out[0]+=sign*math.atan(b*c/(a*r))
                out[3]+=sign*math.atan(a*c/(b*r))
                out[5]+=sign*math.atan(a*b/(c*r))
                out[1]-=sign*math.log(r+c if c>=0 else (a*a+b*b)/(r-c))
                out[2]-=sign*math.log(r+b if b>=0 else (a*a+c*c)/(r-b))
                out[4]-=sign*math.log(r+a if a>=0 else (b*b+c*c)/(r-a))
    return out


@njit(parallel=True)
def _kernel(dx,dy,dz,validx,validy,validz,offset):
    result=np.zeros((6,len(dx),len(dy),len(dz)),np.float64)
    for i in prange(len(dx)):
        if not validx[i]:continue
        for j in range(len(dy)):
            if not validy[j]:continue
            for k in range(len(dz)):
                if not validz[k]:continue
                v=cube_tensor(dx[i]+offset[0],dy[j]+offset[1],dz[k]+offset[2])
                for c in range(6):result[c,i,j,k]=v[c]
    return result


def kernel(source_shape,target_shape=None,offset=(0.,0.,0.),extra_padding=0):
    """Kernel for all required target-source lags, embedded at FFT indices."""
    source_shape=tuple(map(int,source_shape));target_shape=tuple(map(int,target_shape or source_shape))
    if len(source_shape)!=3 or len(target_shape)!=3 or min(*source_shape,*target_shape)<1:
        raise ValueError('Positive three-dimensional shapes required')
    offset=np.asarray(offset,dtype=float)
    if offset.shape!=(3,) or not np.isfinite(offset).all() or extra_padding<0:
        raise ValueError('Invalid origin offset or padding')
    if np.any(np.isclose(np.mod(offset,1),.5,atol=1e-12,rtol=0)):
        raise ValueError('Target centres on source voxel faces need a separate convention')
    shape=tuple(fft.next_fast_len(a+b-1+extra_padding) for a,b in zip(source_shape,target_shape))
    axes=[];valid=[]
    for n,ns,nt in zip(shape,source_shape,target_shape):
        index=np.arange(n);d=np.where(index<nt,index,index-n)
        axes.append(d);valid.append((index<nt)|(index>=n-ns+1))
    return _kernel(*axes,*valid,offset)


def source_delta(rho,support,coverage):
    """Fractional binary-footprint occupancy, never angular completeness weights."""
    rho=np.asarray(rho);support=np.asarray(support)
    if rho.shape!=support.shape or rho.shape!=np.shape(coverage):raise ValueError('Mask shape mismatch')
    if not np.isfinite(support).all() or np.any((support<0)|(support>1)):raise ValueError('Invalid mask')
    occupied=(support>0)&(np.asarray(coverage)>0)
    if np.any((support>0)&~occupied):raise ValueError('Observed support without density coverage')
    if not np.isfinite(rho[occupied]).all() or np.any(rho[occupied]<=0):raise ValueError('Invalid observed density')
    delta=np.zeros(rho.shape,np.float64)
    delta[occupied]=(rho[occupied]-1)*support[occupied]
    return delta


def convolve(delta,kernels,target_shape=None,workers=1):
    """Small-test/reference convenience; production streams the six components."""
    target_shape=tuple(target_shape or delta.shape);shape=kernels.shape[1:]
    d=fft.rfftn(delta,s=shape,workers=workers)
    crop=tuple(slice(0,n) for n in target_shape)
    out=[]
    for k in kernels:
        out.append(fft.irfftn(d*fft.rfftn(k,workers=workers),s=shape,workers=workers)[crop].copy())
    return np.stack(out,axis=-1)


def eigensystem(tensor):
    matrix=np.empty((*tensor.shape[:-1],3,3),np.float64)
    for c,(i,j) in enumerate(COMPONENTS):matrix[...,i,j]=matrix[...,j,i]=tensor[...,c]
    return np.linalg.eigvalsh(matrix)


def probabilities(eigenvalues,threshold=.2):
    if not np.isfinite(threshold) or not np.isfinite(eigenvalues).all():raise ValueError('Nonfinite classification')
    labels=np.sum(eigenvalues>threshold,axis=-1).astype('u1')
    return labels,np.stack([(labels==c).mean(axis=0) for c in range(4)],axis=-1)
