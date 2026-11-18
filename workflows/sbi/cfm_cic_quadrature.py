"""Separable evaluation of exactly the registered midpoint CIC quadrature.

The lattice contraction replaces scatter-add, not the numerical operator. A
one-native-cell input halo supplies the full support of each retained CIC basis.
"""
import numpy as np


def contract_lattice(intensity, factor, cell):
    from numpy.lib.stride_tricks import sliding_window_view
    if factor not in (2, 4) or not np.isfinite(cell) or cell <= 0:
        raise ValueError('Registered positive-cell quadrature required')
    value = np.asarray(intensity, dtype=np.float64)
    if (value.ndim != 3 or any(n % factor or n < 3*factor for n in value.shape)
            or not np.isfinite(value).all() or np.any(value < 0)):
        raise ValueError('Invalid padded intensity lattice')
    shape = tuple(n//factor-2 for n in value.shape)
    position = (np.arange(2*factor)+.5)/factor-1
    weights = 1-np.abs(position)
    for axis, n in enumerate(shape):
        windows = sliding_window_view(value, 2*factor, axis=axis)
        sl = [slice(None)]*windows.ndim
        sl[axis] = slice(factor//2, factor//2+n*factor, factor)
        value = np.einsum('...k,k->...', windows[tuple(sl)], weights, optimize=False)
    return value*(cell/factor)**3


def project(origin, shape, cell, angular, cap, plan, selection, factor=4):
    import healpy as hp
    from workflows.sbi import cfm_lowz_selection_response as lowz
    from workflows.sbi import e2e_coupled_coordinates as coord
    from workflows.sbi.prepare_cfm_lowz_local_condition_canary import density_at
    origin, shape = np.asarray(origin, float), np.asarray(shape, int)
    if origin.shape != (3,) or shape.shape != (3,) or (shape <= 0).any():
        raise ValueError('Invalid output box')
    axes = [origin[i]-cell+(np.arange((shape[i]+2)*factor)+.5)*cell/factor for i in range(3)]
    x,y,z = np.meshgrid(*axes, indexing='ij', sparse=True)
    radius = np.sqrt(x*x+y*y+z*z)
    redshift = coord.redshift(radius)
    density, valid = density_at(redshift, selection, cap)
    valid &= (radius > 0) & lowz.radial_mask(redshift, plan)
    intensity = np.zeros(radius.shape, dtype=np.float64)
    if valid.any():
        points = [np.broadcast_to(v, radius.shape)[valid] for v in (x,y,z)]
        pixel = hp.vec2pix(256, *points)
        support = angular['support'].astype(bool) & (angular['domain']//2 == lowz.CAPS.index(cap))
        intensity[valid] = density[valid]*angular['angular_response'][pixel]*support[pixel]
    return contract_lattice(intensity, factor, cell)
