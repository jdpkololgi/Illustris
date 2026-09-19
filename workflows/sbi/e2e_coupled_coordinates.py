"""Pinned DESI/Abacus Cartesian coordinates; explicit units, no fitted warp."""
from functools import lru_cache
import json
import numpy as np
from scipy.interpolate import CubicSpline
from workflows.sbi import e2e_coupled_contract as c

CONFIG = c.REPO/'configs/e2e_coupled_coordinates_v2.json'
ROOT = c.ROOT/'cartesian_v2'


def config():
    value = json.loads(CONFIG.read_text())
    if (value['schema'] != 'e2e-coupled-cartesian-v2'
            or value['distance_unit'] != 'Mpc/h'
            or value['raw_cell_mpc_h'] != 3.383
            or value['scientific_training_authorized']):
        raise ValueError('unregistered Cartesian coordinate authority')
    return value


@lru_cache(maxsize=1)
def table():
    cfg = config()
    path = c.REPO/cfg['table_path']
    if c.sha256(path) != cfg['table_sha256']:
        raise ValueError('DESI distance table checksum mismatch')
    values = np.loadtxt(path)
    z, radius = values[:, 0], values[:, 2]
    if not np.isfinite(values).all() or not np.all(np.diff(z)>0) or not np.all(np.diff(radius)>0):
        raise ValueError('invalid DESI distance table')
    return z, radius, CubicSpline(z, radius, extrapolate=False), CubicSpline(radius, z, extrapolate=False)


def radius_mpc_h(z):
    out = table()[2](z)
    if not np.isfinite(out).all():
        raise ValueError('redshift outside pinned distance table')
    return out


def redshift(radius):
    out = table()[3](radius)
    if not np.isfinite(out).all():
        raise ValueError('radius outside pinned distance table')
    return out


def sky_mpc_h(ra_deg, dec_deg, z):
    ra, dec = np.deg2rad(ra_deg), np.deg2rad(dec_deg)
    direction = np.stack((np.cos(ra)*np.cos(dec), np.sin(ra)*np.cos(dec), np.sin(dec)), axis=-1)
    return direction*radius_mpc_h(z)[..., None]


def selection_volume_jacobian(z, selection):
    """dV_old[Mpc^3]/dV_new[(Mpc/h)^3], preserving dN/dz/dOmega.

    The inherited radius curve is piecewise linear; differentiate that same
    interpolant, not a newly fitted cosmology or a refitted selection function.
    The h^-3 unit factor is INCLUDED here, not applied a second time downstream.
    """
    zz = np.asarray(selection['cosmology']['redshift_grid'])
    rr = np.asarray(selection['cosmology']['radius_grid_mpc'])
    z = np.asarray(z)
    if np.any((z<zz[0]) | (z>zz[-1])):
        raise ValueError('selection Jacobian outside inherited redshift grid')
    i = np.clip(np.searchsorted(zz, z, side='right')-1, 0, len(zz)-2)
    old_derivative = (rr[i+1]-rr[i])/(zz[i+1]-zz[i])
    old_radius = np.interp(z, zz, rr)
    new_radius = radius_mpc_h(z)
    new_derivative = table()[2](z, 1)
    ratio = np.divide(old_radius, new_radius, out=np.asarray(old_derivative/new_derivative).copy(), where=new_radius!=0)
    result = ratio**2*old_derivative/new_derivative
    if not np.isfinite(result).all() or np.any(result<=0):
        raise ValueError('invalid selection volume Jacobian')
    return result


def grid_record(spec):
    # GridSpec is only a numerical kernel adapter. Never serialize its legacy
    # '*_mpc' names for these Mpc/h values.
    return dict(origin_mpc_h=list(spec.origin), shape=list(spec.shape),
                cell_mpc_h=spec.cell_mpc, padding_mpc_h=spec.padding_mpc,
                distance_unit='Mpc/h', coordinate_sha256=c.sha256(CONFIG))


def validate_grid(grid):
    if (grid.get('distance_unit') != 'Mpc/h'
            or grid.get('coordinate_sha256') != c.sha256(CONFIG)
            or grid.get('cell_mpc_h') != config()['raw_cell_mpc_h']
            or 'origin_mpc' in grid):
        raise ValueError('wrong or ambiguous Cartesian grid authority/units')
    return grid


def provenance():
    return dict(**c.provenance(), coordinate_sha256=c.sha256(CONFIG))


def bind():
    base = c.bind_run()
    table()  # Verify pinned bytes before any Cartesian output.
    binding = dict(schema='e2e-coupled-cartesian-authority-v2',
                   base_config_sha256=base['config_sha256'],
                   coordinates=config(), coordinate_sha256=c.sha256(CONFIG))
    marker = ROOT/'CARTESIAN_AUTHORITY.json'
    if marker.exists():
        if json.loads(marker.read_text()) != binding:
            raise ValueError('Cartesian authority changed under existing products')
    else:
        c.atomic_json(marker, binding)
    return binding


def verify_receipt(path, payload=True):
    record = c.verify_receipt(path, payload=payload)
    if record.get('coordinate_sha256') != c.sha256(CONFIG):
        raise ValueError('Cartesian receipt has wrong coordinate authority')
    return record


def require_host_checks():
    for phase in config()['host_check_phases']:
        verify_receipt(ROOT/'coordinate_audit'/phase/'CORRECTED_COMPLETE.json')
