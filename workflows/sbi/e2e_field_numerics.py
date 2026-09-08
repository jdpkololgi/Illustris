"""Dependency-light reference transforms and globally addressed E2E noise.

These are numerical fixtures, not trained samplers. Haar uses local pairs and
does not wrap the parent boundary. FFT projections are coordinates on the
declared array; they do not assert physical periodicity of the survey.
"""
from __future__ import annotations

import hashlib
import itertools

import numpy as np
from scipy import fft

# Preserve the inherited dimensionless split, explicitly correcting the physical
# label for the actual 5-Mpc observer grid. This is a numerical audit candidate,
# not a frozen science-training cutoff.
AUDIT_LOW_CYCLES_PER_VOXEL = 0.1813799364234218 * 5.0 / (2*np.pi)


def low_projection(x, cutoff=AUDIT_LOW_CYCLES_PER_VOXEL):
    n = x.shape[0]
    if x.shape != (n,n,n):
        raise ValueError("expected one cubic parent")
    axes = np.meshgrid(fft.fftfreq(n),fft.fftfreq(n),fft.rfftfreq(n),indexing="ij",sparse=True)
    mask = sum(a*a for a in axes) <= cutoff**2
    spectrum = fft.rfftn(x,norm="ortho")
    return fft.irfftn(spectrum*mask,s=x.shape,norm="ortho")


def haar(x,depth=1,inverse=False):
    n = x.shape[0]
    if x.shape != (n,n,n) or depth not in (1,2) or n%(2**depth):
        raise ValueError("Haar audit permits cubic arrays and depths 1/2 only")
    result = np.asarray(x,dtype=np.float64).copy()
    levels = reversed(range(depth)) if inverse else range(depth)
    for level in levels:
        side = n//(2**level)
        region = result[:side,:side,:side]
        for axis in (reversed(range(3)) if inverse else range(3)):
            moved = np.moveaxis(region,axis,0)
            temp = moved.copy()
            if inverse:
                moved[::2] = (temp[:side//2]+temp[side//2:])/np.sqrt(2.)
                moved[1::2] = (temp[:side//2]-temp[side//2:])/np.sqrt(2.)
            else:
                moved[:side//2] = (temp[::2]+temp[1::2])/np.sqrt(2.)
                moved[side//2:] = (temp[::2]-temp[1::2])/np.sqrt(2.)
    return result


def _splitmix64(x):
    with np.errstate(over="ignore"):
        x = x + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def coordinate_noise(start,shape,sample_id):
    seed = np.uint64(int(hashlib.sha256(str(sample_id).encode()).hexdigest()[:16],16))
    axes = [(np.arange(n,dtype=np.int64)+int(s)).view(np.uint64) for s,n in zip(start,shape)]
    with np.errstate(over="ignore"):
        counter = (axes[0][:,None,None]*np.uint64(0xD6E8FEB86659FD93) ^
                   axes[1][None,:,None]*np.uint64(0xA5A3564E27F8862F) ^
                   axes[2][None,None,:]*np.uint64(0x9E3779B185EBCA87) ^ seed)
    a, b = _splitmix64(counter),_splitmix64(counter ^ np.uint64(0xDB4F0B9175AE2165))
    u = ((a >> np.uint64(11)).astype(np.float64)+.5)/2**53
    v = ((b >> np.uint64(11)).astype(np.float64)+.5)/2**53
    return np.sqrt(-2*np.log(u))*np.cos(2*np.pi*v)


def local_velocity(x):
    padded = np.pad(x,1,mode="constant")
    result = -.2*x.copy()
    for axis in range(3):
        lo, hi = [slice(1,-1)]*3,[slice(1,-1)]*3
        lo[axis],hi[axis] = slice(None,-2),slice(2,None)
        result += .025*(padded[tuple(lo)]+padded[tuple(hi)]-2*x)
    return result


def tiled_velocity(x,core=32,offset=0):
    n = x.shape[0]
    edges = sorted(set([0,n]+list(range(offset,n,core))))
    out = np.empty_like(x)
    for index in itertools.product(range(len(edges)-1),repeat=3):
        start = [edges[i] for i in index]
        stop = [edges[i+1] for i in index]
        halo_start = [max(0,x-1) for x in start]
        halo_stop = [min(n,x+1) for x in stop]
        region = tuple(slice(a,b) for a,b in zip(halo_start,halo_stop))
        trim = tuple(slice(a-h,b-h) for a,b,h in zip(start,stop,halo_start))
        out[tuple(slice(a,b) for a,b in zip(start,stop))] = local_velocity(x[region])[trim]
    return out


def evolve_high(initial,*,tiled=False,offset=0,steps=4):
    state = initial-low_projection(initial)
    dt = 1/steps
    def evaluate(x):
        velocity = tiled_velocity(x,offset=offset) if tiled else local_velocity(x)
        return velocity-low_projection(velocity)
    for _ in range(steps):
        v = evaluate(state)
        predictor = state+dt*v
        state = state+.5*dt*(v+evaluate(predictor))
    return state
