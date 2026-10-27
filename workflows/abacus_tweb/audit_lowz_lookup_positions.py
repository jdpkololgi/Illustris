#!/usr/bin/env python3
"""Bounded file-0 index audit; execute only in an approved Slurm allocation.

Diagnostic candidates are hypotheses, not recovered production provenance.
No catalogue, labels or source data are changed. Outputs use exclusive creation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import fitsio
import numpy as np
from astropy.cosmology import FlatLambdaCDM

ROOT = Path('/global/cfs/cdirs/desi/cosmosim/SecondGenMocks/AbacusSummit')
CUT = ROOT / 'CutSky/BGS/v0.1/z0.200/cutsky_BGS_z0.200_AbacusSummit_base_c000_ph000.fits'
CUBE = ROOT / 'CubicBox/BGS/v0.1/z0.200/AbacusSummit_base_c000_ph000/BGS_box_ph000.fits'
HALO = Path('/global/cfs/cdirs/desi/public/cosmosim/AbacusSummit/AbacusSummit_base_c000_ph000/halos/z0.200/halo_info/halo_info_000.asdf')
COLS = ['FILE_NUM', 'HALO_INDEX', 'BOX_INDEX', 'RA', 'DEC', 'Z', 'Z_COSMO', 'RES', 'CEN', 'HALO_MASS']


def periodic_delta(a, b, box=2000.):
    return (a - b + box / 2) % box - box / 2


def reconstruct(rows, cosmology):
    ra, dec = np.deg2rad(rows['RA'].astype(float)), np.deg2rad(rows['DEC'].astype(float))
    distance = cosmology.comoving_distance(rows['Z_COSMO'].astype(float)).value * cosmology.h
    return distance[:, None] * np.column_stack((np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)))


def stats(x):
    x = np.asarray(x)
    good = np.isfinite(x)
    return {'n': len(x), 'n_finite': int(good.sum()),
            'median': float(np.median(x[good])) if good.any() else None,
            'p95': float(np.percentile(x[good], 95)) if good.any() else None,
            'max': float(np.max(x[good])) if good.any() else None}


def sample():
    with fitsio.FITS(CUT) as f:
        ids = np.linspace(0, f[1].get_nrows()-1, 8000, dtype='i8')
        rows = f[1].read(rows=ids, columns=COLS)
    keep = rows['FILE_NUM'] == 0
    return ids[keep], rows[keep]


def groups(rows):
    for low in (True, False):
        for res in (0, 1):
            for cen in (0, 1):
                yield f'{"low" if low else "high"}_res{res}_cen{cen}', ((rows['Z'] < .15) == low) & (rows['RES'] == res) & (rows['CEN'] == cen)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_STEP_ID'):
        ap.error('Requires an explicitly approved allocation and srun job step.')
    args.output.mkdir(parents=True, exist_ok=False)
    ids, rows = sample()
    report = {'job_id': os.environ['SLURM_JOB_ID'], 'sources': {}, 'candidate_results': {},
              'sampling': '8000 equally spaced original rows, then FILE_NUM=0',
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'limitations': ['Exact v0.1 export revision not recovered.',
                             'Astropy c000 distance models approximate the original nbodykit/CLASS implementation; validate against cubic galaxies.',
                             'Resolved filtered-index variants are hypotheses; they do not recover unresolved particle hosts.',
                             'Satellite galaxy coordinates need not equal host centres.']}
    for p in (CUT, CUBE, HALO):
        st = p.stat()
        report['sources'][str(p)] = {'bytes': st.st_size, 'mtime_ns': st.st_mtime_ns}
    report['groups'] = {name: int(mask.sum()) for name, mask in groups(rows)}
    # Two independently specified background approximations, neither called exact.
    h = .6736
    models = {
        'c000_massive_nu_approx': FlatLambdaCDM(H0=67.36, Om0=(.02237+.12)/h**2,
            Ob0=.02237/h**2, Tcmb0=2.7255, Neff=3.0328, m_nu=[0., 0., .06]),
        'header_flat_matter_approx': FlatLambdaCDM(H0=67.36, Om0=.315192, Tcmb0=0.),
    }
    sky = {name: reconstruct(rows, c) for name, c in models.items()}
    report['cosmology'] = {name: str(c) for name, c in models.items()}
    arrays = {'original_row': ids, **{key: rows[key] for key in COLS},
              **{f'sky_{name}': xyz for name, xyz in sky.items()}}
    high = rows['BOX_INDEX'] >= 0
    with fitsio.FITS(CUBE) as f:
        cubic = f[1].read(rows=rows['BOX_INDEX'][high].astype('i8'),
                          columns=['x', 'y', 'z', 'FILE_NUM', 'HALO_ID', 'HALO_MASS', 'cen'])
    if not np.all(cubic['FILE_NUM'] == 0):
        raise ValueError('File-0 cubic correspondence failed.')
    xyz_cube = np.column_stack([cubic[k] for k in ('x', 'y', 'z')])
    report['cubic_mass_exact_matches'] = int(np.sum(cubic['HALO_MASS'] == rows['HALO_MASS'][high]))
    report['cubic_coordinate_controls'] = {}
    for name, xyz in sky.items():
        for shift in (0., 1000.):
            residual = np.linalg.norm(periodic_delta(xyz[high] + shift, xyz_cube), axis=1)
            report['cubic_coordinate_controls'][f'{name}_shift{shift:g}'] = stats(residual)
    np.savez(args.output / 'sample_and_reconstruction.npz', **arrays)

    def persist():
        (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')

    persist()
    from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog
    for cleaned in (False, True):
        print(f'Loading one halo slab: cleaned={cleaned}', flush=True)
        cat = CompaSOHaloCatalog(str(HALO), cleaned=cleaned, convert_units=True,
                                 fields=['id', 'N', 'x_com', 'x_L2com'])
        table = cat.halos
        mass = np.asarray(table['N'], dtype=float) * float(cat.header['ParticleMassHMsun']) / 1e10
        report[f'halo_rows_clean{cleaned}'] = len(table)
        # Keep identity and filtering independent. Filtered variants deliberately
        # retain the array order of the public generator's halo cuts.
        mass_keep = mass >= 10.  # 1e11 Msun/h; catalogue masses are 1e10 Msun/h
        mappings = {'direct': None, 'mass_ge_1e11': np.flatnonzero(mass_keep)}
        l2 = np.asarray(table['x_L2com'])
        for origin in (0., -1000.):
            radius = np.linalg.norm(periodic_delta(l2, origin), axis=1)
            for model, c in models.items():
                rmax = c.comoving_distance(.16).value * c.h
                mappings[f'mass_radius_z016_origin{origin:g}_{model}'] = np.flatnonzero(mass_keep & (radius < rmax))
        for mapping_name, mapping in mappings.items():
            count = len(table) if mapping is None else len(mapping)
            valid = (rows['HALO_INDEX'] >= 0) & (rows['HALO_INDEX'] < count)
            ix = rows['HALO_INDEX'][valid].astype('i8')
            if mapping is not None:
                ix = mapping[ix]
            candidate = f'clean{cleaned}_{mapping_name}'
            mass_rel = np.full(len(rows), np.nan)
            mass_rel[valid] = np.abs(mass[ix] / rows['HALO_MASS'][valid] - 1.)
            entry = {'lookup_size': count, 'mass_relative_error': {g: stats(mass_rel[m]) for g, m in groups(rows)}, 'positions': {}}
            id_match = np.zeros(len(rows), dtype=bool)
            target_id = np.zeros(len(rows), dtype='i8')
            target_id[high] = cubic['HALO_ID']
            id_match[valid] = np.asarray(table['id'])[ix] == target_id[valid]
            entry['highz_cubic_halo_id_matches'] = int(id_match[high].sum())
            for field in ('x_com', 'x_L2com'):
                pos = np.full((len(rows), 3), np.nan)
                pos[valid] = np.asarray(table[field])[ix]
                for model, xyz in sky.items():
                    for origin in (0., -1000.):
                        residual = np.linalg.norm(periodic_delta(pos, xyz + origin), axis=1)
                        key = f'{field}_{model}_observer{origin:g}'
                        entry['positions'][key] = {g: stats(residual[m]) for g, m in groups(rows)}
                np.savez(args.output / f'{candidate}_{field}.npz',
                         original_row=ids, halo_row=ix, valid=valid, lookup_position=pos,
                         mass_relative_error=mass_rel)
            report['candidate_results'][candidate] = entry
            persist()
        del cat, table, mass, mass_keep, l2, radius, mappings
    report['completed'] = True
    persist()
    print(args.output / 'report.json', flush=True)


if __name__ == '__main__':
    main()
