"""Exact three-residue representation of a piecewise-radial CIC expectation.

The quadrature is the registered factor-four midpoint rule. Its geometry-only
weights can be reused for full/OOF selection curves without refitting on a held
out phase. The three-residue compression is exact only when the CIC support
radius is smaller than every radial-bin width; that condition is enforced.
Coordinates and density units are P12 observer-frame Planck18 Mpc, NOT CFM Mpc/h.
"""
import numpy as np

from workflows.sbi.cfm_cic_quadrature import contract_lattice


SCHEMA='p12a-cic-radial-basis-v1'


def validate_geometry(radius_edges, cell):
    edges=np.asarray(radius_edges,dtype=np.float64)
    if (edges.ndim!=1 or len(edges)<3 or edges[0]!=0 or not np.isfinite(edges).all()
            or np.any(np.diff(edges)<=0) or not np.isfinite(cell) or cell<=0):
        raise ValueError('Invalid radial-bin/cell geometry')
    if np.sqrt(3.)*cell>=np.min(np.diff(edges)):
        raise ValueError('Three-residue compression cannot represent this CIC/bin geometry')
    return edges


def compress_lattice(angular_intensity, radial_bin, factor, cell, radius_edges):
    """Integrate an intensity lattice with density removed, by bin modulo 3."""
    edges=validate_geometry(radius_edges,cell)
    angular=np.asarray(angular_intensity,dtype=np.float64)
    bins=np.asarray(radial_bin)
    if (bins.shape!=angular.shape or not np.issubdtype(bins.dtype,np.integer)
            or np.any((angular>0)&((bins<0)|(bins>=len(edges)-1)))):
        raise ValueError('Nonzero angular intensity outside the radial bins')
    return np.stack([contract_lattice(np.where(bins%3==residue,angular,0.),factor,cell)
                     for residue in range(3)])


def expected_counts(weights, center_bin, density, valid):
    weights=np.asarray(weights,dtype=np.float64)
    bins=np.asarray(center_bin)
    density=np.asarray(density,dtype=np.float64);valid=np.asarray(valid,dtype=bool)
    if (weights.shape!=(3,*bins.shape) or not np.issubdtype(bins.dtype,np.integer)
            or density.ndim!=1 or valid.shape!=density.shape or len(density)<2
            or not np.isfinite(weights).all() or np.any(weights<0)
            or not np.isfinite(density).all() or np.any(density<0)
            or np.any((bins<0)|(bins>=len(density)))):
        raise ValueError('Invalid response weights/selection')
    result=np.zeros(bins.shape,dtype=np.float64)
    for delta in (-1,0,1):
        index=bins+delta;inside=(index>=0)&(index<len(density))
        safe=np.clip(index,0,len(density)-1)
        weight=np.take_along_axis(weights,(index%3)[None,...],axis=0)[0]
        result+=np.where(inside&valid[safe],weight*density[safe],0.)
    return result


def project_basis(origin,shape,cell,angular,cap,radius_edges,valid_bins,factor=4):
    import healpy as hp
    edges=validate_geometry(radius_edges,cell)
    valid=np.asarray(valid_bins,bool)
    origin=np.asarray(origin,float);shape=np.asarray(shape,int)
    if (origin.shape!=(3,) or not np.isfinite(origin).all() or shape.shape!=(3,)
            or np.any(shape<=0) or valid.shape!=(len(edges)-1,) or cap not in (0,1)):
        raise ValueError('Invalid P12 CIC projection inputs')
    axes=[origin[i]-cell+(np.arange((shape[i]+2)*factor)+.5)*cell/factor for i in range(3)]
    x,y,z=np.meshgrid(*axes,indexing='ij',sparse=True)
    radius=np.sqrt(x*x+y*y+z*z)
    bins=np.searchsorted(edges,radius,side='right')-1
    safe=np.clip(bins,0,len(valid)-1)
    included=(radius>0)&(bins>=0)&(bins<len(valid))&valid[safe]
    intensity=np.zeros(radius.shape,dtype=np.float64)
    if included.any():
        vectors=[np.broadcast_to(v,radius.shape)[included] for v in (x,y,z)]
        pixel=hp.vec2pix(256,*vectors)
        supported=np.asarray(angular['support'],bool)&(np.asarray(angular['domain'])//2==cap)
        response=np.asarray(angular['angular_response'],float)
        if not np.isfinite(response).all() or np.any(response<0):
            raise ValueError('Invalid angular response')
        intensity[included]=response[pixel]*supported[pixel]
    weights=compress_lattice(intensity,bins,factor,cell,edges)
    centers=[origin[i]+(np.arange(shape[i])+.5)*cell for i in range(3)]
    center_radius=np.sqrt(centers[0][:,None,None]**2+centers[1][None,:,None]**2+centers[2][None,None,:]**2)
    center_bin=np.clip(np.searchsorted(edges,center_radius,side='right')-1,0,len(valid)-1)
    return weights,center_bin.astype(np.int16)
