"""Versioned real-space inverse for SecondGenMocks BGS v0.1, c000, z=0.2.

Distances approximate the original nbodykit/CLASS background with a fixed Astropy
massive-neutrino cosmology. Qualification measures errors against native objects;
a coordinate round trip is not independent evidence. Never use observed Z here.
"""
from functools import lru_cache
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from scipy.interpolate import CubicSpline

MAPPING_VERSION = 'bgs-v01-c000-galaxy-realspace-v1'
BOX_SIZE = 2000.
OBSERVER = np.array([-1000., -1000., -1000.])
MAX_REDSHIFT = 1.

@lru_cache(maxsize=1)
def cosmology():
    h = .6736
    return FlatLambdaCDM(H0=67.36, Om0=(.02237+.12)/h**2, Ob0=.02237/h**2,
                        Tcmb0=2.7255, Neff=3.0328, m_nu=[0., 0., .06])

@lru_cache(maxsize=1)
def _distance_spline():
    z = np.linspace(0., MAX_REDSHIFT, 8193)
    c = cosmology()
    return CubicSpline(z, c.comoving_distance(z).value*c.h, extrapolate=False)

def comoving_distance(z, *, direct=False):
    z = np.asarray(z, dtype=np.float64)
    if not np.all(np.isfinite(z)) or np.any((z < 0.) | (z > MAX_REDSHIFT)):
        raise ValueError('Z_COSMO must be finite and in [0, 1]; no observed-Z fallback')
    if direct:
        c = cosmology()
        return c.comoving_distance(z).value*c.h
    return _distance_spline()(z)

def reconstruct_native(ra, dec, z_cosmo, *, direct=False):
    ra, dec, z = np.broadcast_arrays(np.asarray(ra, dtype=np.float64),
                                     np.asarray(dec, dtype=np.float64),
                                     np.asarray(z_cosmo, dtype=np.float64))
    if ra.ndim != 1:
        raise ValueError('Expected one-dimensional galaxy arrays')
    if not np.all(np.isfinite(ra)) or not np.all(np.isfinite(dec)) or np.any(np.abs(dec)>90):
        raise ValueError('RA/DEC must be finite, DEC in [-90,90]')
    alpha, delta = np.deg2rad(ra), np.deg2rad(dec)
    unit = np.column_stack((np.cos(delta)*np.cos(alpha), np.cos(delta)*np.sin(alpha), np.sin(delta)))
    # Native halo/particle coordinates use a box centred at zero. The field grid
    # uses mod(native, L), not mod(native+L/2, L).
    native = comoving_distance(z, direct=direct)[:,None]*unit + OBSERVER
    return (native+BOX_SIZE/2) % BOX_SIZE - BOX_SIZE/2

def periodic_delta(a,b):
    return (np.asarray(a)-np.asarray(b)+BOX_SIZE/2)%BOX_SIZE-BOX_SIZE/2

def field_indices(native, ngrid=2048, boxsize=BOX_SIZE):
    # Match the native particle/halo annotation precision and nearest-cell rule.
    p = np.asarray(native, dtype=np.float32)
    if p.ndim!=2 or p.shape[1]!=3 or not np.all(np.isfinite(p)):
        raise ValueError('Expected finite N by 3 coordinates')
    return np.clip(np.floor(np.mod(p,boxsize)/(boxsize/ngrid)).astype(np.int64),0,ngrid-1)

def mapping_definition():
    return dict(version=MAPPING_VERSION, target='tidal_field_at_galaxy_real_space_position',
                input_columns=['RA','DEC','Z_COSMO'], cosmology=str(cosmology()),
                distance_implementation='Astropy massive-neutrino c000 approximation; 8193-knot cubic spline',
                observer_mpc_h=OBSERVER.tolist(),box_size_mpc_h=BOX_SIZE,
                simulation='AbacusSummit_base_c000',snapshot_z=.2,
                native_frame='[-L/2,L/2); grid mod(native,L)',
                sampling='nearest_cell_float32', no_observed_z_fallback=True)
