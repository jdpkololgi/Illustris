"""Observer-only reconstruction on a frozen P12 lattice (no truth inputs)."""
import numpy as np
from astropy.cosmology import Planck18

def observer_xyz(ra,dec,z):
    ra,dec,z=(np.asarray(v,dtype=np.float64) for v in (ra,dec,z))
    if not (ra.shape==dec.shape==z.shape) or not np.isfinite([ra,dec,z]).all() or np.any(z<=0):raise ValueError('invalid observer coordinates')
    # Exact P10 lookup and arithmetic; direct integration differs at sub-micro-Mpc scale.
    if np.any(z>.85):raise ValueError('outside frozen distance lookup')
    zg=np.linspace(0.,.85,17001,dtype=np.float64)
    radius=np.interp(z,zg,Planck18.comoving_distance(zg).value.astype(np.float64))
    ra=np.deg2rad(ra);dec=np.deg2rad(dec);c=np.cos(dec)
    return radius[:,None]*np.column_stack((c*np.cos(ra),c*np.sin(ra),np.sin(dec)))

def count_patch(xyz,shell,origin,cell,start,stop):
    """Deposit using global fractions and canonical shell/row summation order."""
    shape=np.asarray(stop)-start; out=np.zeros(tuple(shape),dtype='f4')
    u=(np.asarray(xyz)-origin)/cell-.5;i0=np.floor(u).astype('i8');frac=u-i0
    for s in [-1,0,1,2,3]:
        take=np.asarray(shell)==s
        for dx in [0,1]:
            for dy in [0,1]:
                for dz in [0,1]:
                    delta=np.array([dx,dy,dz]); idx=i0[take]+delta-start
                    w=np.prod(np.where(delta,frac[take],1-frac[take]),axis=1).astype('f4')
                    ok=np.all((idx>=0)&(idx<shape),axis=1)
                    np.add.at(out,tuple(idx[ok].T),w[ok])
    return out

def rebuild_fields(xyz,shell,grid,start,stop,angular_support,nside,schema,spline,selection,rotation,cap_name):
    from workflows.abacus_tweb.p3a_build_canonical_fields import GridSpec,field_block,cosmology_lookup
    from workflows.abacus_tweb.p6_field_patch_utils import derive_selection_channels,patch_redshift
    origin=np.asarray(grid['origin_mpc']);cell=grid['cell_mpc'];shape=np.asarray(stop)-start
    counts=count_patch(xyz,shell,origin,cell,start,stop)
    spec=GridSpec(tuple(origin+np.asarray(start)*cell),tuple(shape),cell,0.)
    z,r=cosmology_lookup();fields=field_block(spec,tuple(slice(0,int(n)) for n in shape),counts,angular_support,nside,z,r,spline,schema)
    curve=selection['rotations'][str(rotation)]['caps'][cap_name]
    redshift=patch_redshift(origin_mpc=origin,cell_mpc=cell,context_start=start,shape=tuple(shape),radius_grid_mpc=np.asarray(selection['cosmology']['radius_grid_mpc']),redshift_grid=np.asarray(selection['cosmology']['redshift_grid']))
    fields.update(derive_selection_channels(counts,fields['exposure_apodized'],redshift,cell_mpc=cell,grid_z=np.asarray(curve['grid_z']),ntilde=np.asarray(curve['ntilde']),**selection['contrast']))
    return fields
