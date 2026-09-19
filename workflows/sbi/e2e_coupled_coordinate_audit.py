"""Independent small host-linkage check of the inherited sky-to-box mapping.

Real-space Z_COSMO, not observed Z, is compared with central host positions.
This is a pairing prerequisite, not a learned model score or a mapping fit.
"""
import argparse
import itertools
import json
from pathlib import Path

import fitsio
import numpy as np
from astropy.cosmology import Planck18
from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog

from workflows.sbi import e2e_coupled_contract as c


def periodic_difference(x,y,box=2000.):
    return (np.asarray(x)-np.asarray(y)+box/2)%box-box/2


def run(phase, samples=16384):
    c.require_compute()
    c.phase_guard(phase)
    if c.ROLES[phase] != 'train':
        raise PermissionError('convention diagnosis is training-only')
    sources = c.paths(phase)
    names = ['RA','DEC','Z_COSMO','Z','FILE_NUM','HALO_INDEX','BOX_INDEX','CEN','RES']
    with fitsio.FITS(sources['cutsky']) as f:
        rows = np.unique(np.linspace(0,f[1].get_nrows()-1,samples,dtype=np.int64))
        data = f[1].read(rows=rows,columns=names)
    selected = (data['CEN'] != 0) & (data['BOX_INDEX'] >= 0) & (data['Z_COSMO'] >= .15) & (data['Z_COSMO'] < .55)
    data = data[selected]
    # Three native slabs suffice to detect wrong units/offset/axis conventions.
    counts = {int(i): int((data['FILE_NUM']==i).sum()) for i in np.unique(data['FILE_NUM'])}
    slabs = sorted(counts, key=lambda i: (-counts[i], i))[:3]
    data = data[np.isin(data['FILE_NUM'],slabs)]
    host = np.empty((len(data),3),dtype=np.float64)
    host_l2 = np.empty_like(host)
    headers = []
    for slab in slabs:
        path = sources['snapshot_root']/'halo_info'/f'halo_info_{slab:03d}.asdf'
        cat = CompaSOHaloCatalog(path, fields=['x_com','x_L2com'], subsamples=False,
                               cleaned=False, convert_units=True, verbose=False)
        if cat.header['SimName'] != f'AbacusSummit_base_c000_{phase}':
            raise ValueError('host source phase mismatch')
        mask = data['FILE_NUM']==slab
        ids = data['HALO_INDEX'][mask]
        if np.any(ids < 0) or np.any(ids >= len(cat.halos)):
            raise ValueError('host index out of bounds')
        host[mask] = cat.halos['x_com'][ids]
        host_l2[mask] = cat.halos['x_L2com'][ids]
        headers.append(dict(source=c.file_record(path),
                            header={k:str(v) for k,v in cat.header.items()
                                    if k in ('SimName','Redshift','BoxSizeHMpc','H0','Omega_M','Omega_DE','hMpc')
                                    or k.startswith('omega')}))
        del cat
    ra,dec = np.deg2rad(data['RA']),np.deg2rad(data['DEC'])
    direction = np.column_stack((np.cos(ra)*np.cos(dec), np.sin(ra)*np.cos(dec), np.sin(dec)))
    radius = Planck18.comoving_distance(data['Z_COSMO']).value*.6766
    sky = direction*radius[:,None]
    variants = {}
    for offset in (-1000.,0.):
        for name,truth in (('x_com',host),('x_L2com',host_l2)):
            diff = periodic_difference(sky+offset,truth)
            norm = np.linalg.norm(diff,axis=1)
            variants[f'offset{offset:g}_{name}'] = dict(
                median_distance_mpc_h=float(np.median(norm)),
                p95_distance_mpc_h=float(np.quantile(norm,.95)),
                component_median_mpc_h=np.median(diff,axis=0).tolist(),
                rms_mpc_h=float(np.sqrt(np.mean(norm**2))))
    # Report simple discrete axis/sign diagnostics; never adopt them silently.
    axis_candidates = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((-1,1),repeat=3):
            for offset in (-1000.,0.):
                diff = periodic_difference(sky[:,perm]*signs+offset,host_l2)
                axis_candidates.append(dict(permutation=list(perm), signs=list(signs),offset=offset,
                    median_distance_mpc_h=float(np.median(np.linalg.norm(diff,axis=1)))))
    axis_candidates.sort(key=lambda row: row['median_distance_mpc_h'])
    result = dict(**c.provenance(), phase=phase, schema='e2e-coupled-coordinate-audit-v1',
                  source_code_sha256=c.sha256(__file__), samples=len(data), slabs=slabs,
                  box_index_counts={str(i):int((data['BOX_INDEX']==i).sum()) for i in np.unique(data['BOX_INDEX'])},
                  res_values={str(i):int((data['RES']==i).sum()) for i in np.unique(data['RES'])},
                  variants=variants, best_discrete_diagnostics=axis_candidates[:6], host_sources=headers,
                  coordinates_adopted=False,
                  inherited_mapping_within_R7=variants['offset-1000_x_L2com']['p95_distance_mpc_h'] < 7.)
    output = c.ROOT/'coordinate_audit'/phase
    output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/'linked_centrals.npz', **{name:data[name] for name in names},
                        sky_planck18_mpc_h=sky, host_x_com=host,host_x_L2com=host_l2)
    result['outputs'] = [c.file_record(output/'linked_centrals.npz',content_hash=True)]
    c.atomic_json(output/'AUDIT.json', result)
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',default='ph007')
    a=p.parse_args()
    print(json.dumps(run(a.phase),indent=2))
