#!/usr/bin/env python3
"""Recover/test a low-z selection-radius hypothesis using independent centrals.

One slab only. Requires a separately approved Slurm step. A fitted selection
interval is diagnostic evidence, not production-generator provenance. No labels
or input catalogues are changed, and no unresolved-host mapping is inferred.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from audit_lowz_lookup_positions import HALO, periodic_delta, stats


def feasible_intervals(base_ranks, event_radii, event_rows, parent_rows,
                       expected_indices, lower, upper):
    """Intervals where strict-radius selection reproduces every supplied rank.

base_ranks counts eligible rows before each parent at radius < lower.
Events must contain EVERY eligible halo in [lower, upper), including repeated
radii. The result lists open intervals; endpoints need separate float handling.
"""
    base = np.asarray(base_ranks, dtype=np.int64)
    event_radii, event_rows = np.asarray(event_radii), np.asarray(event_rows)
    parents, expected = np.asarray(parent_rows), np.asarray(expected_indices)
    if len(parents) == 0:
        raise ValueError('No independently matched training centrals.')
    if not lower < upper or np.any((event_radii < lower) | (event_radii >= upper)):
        raise ValueError('Invalid radius bracket/events.')
    boundaries = np.unique(np.r_[lower, event_radii, upper])
    accepted = []
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        ranks = base + np.sum(event_rows[event_radii <= lo, None] < parents[None, :], axis=0)
        if np.array_equal(ranks, expected):
            accepted.append([float(lo), float(hi)])
    return accepted


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--previous-output', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--radius-bracket-mpc-h', type=float, default=.05)
    args = ap.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_STEP_ID'):
        ap.error('Requires a separately approved allocation and srun step.')
    if not 0 < args.radius_bracket_mpc_h <= 1.:
        ap.error('Radius bracket must lie in (0,1] Mpc/h.')
    args.output.mkdir(parents=True, exist_ok=False)
    prior_path = args.previous_output / 'report.json'
    prior = json.loads(prior_path.read_text())
    if prior.get('completed') is not True:
        raise ValueError('Prior audit did not complete.')
    st = HALO.stat()
    if prior['sources'][str(HALO)] != {'bytes': st.st_size, 'mtime_ns': st.st_mtime_ns}:
        raise ValueError('Halo source changed since the original audit.')
    data_path = args.previous_output / 'sample_and_reconstruction.npz'
    with np.load(data_path) as f:
        s = {k: f[k] for k in f.files}
    if not np.all(s['FILE_NUM'] == 0):
        raise ValueError('This diagnostic is explicitly file-0 only.')
    report = {'job_id': os.environ['SLURM_JOB_ID'], 'eligible_for_production': False,
              'prior_report_sha256': hashlib.sha256(prior_path.read_bytes()).hexdigest(),
              'sample_sha256': hashlib.sha256(data_path.read_bytes()).hexdigest(),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'method': 'Geometry+mass uniquely identify centrals; alternating original-row order is training/heldout. Only training ranks choose the radius interval.',
              'limitations': ['File 0 only; exact export provenance remains unverified.',
                             'No inferred host mapping for RES=0.',
                             'A failed fixed radius hypothesis is not permission to tune mass cuts or reorder rows.']}

    def save():
        (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')

    from astropy.cosmology import FlatLambdaCDM
    from scipy.spatial import cKDTree
    from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog

    h = .6736
    cosmo = FlatLambdaCDM(H0=67.36, Om0=.14237/h**2, Ob0=.02237/h**2,
                          Tcmb0=2.7255, Neff=3.0328, m_nu=[0, 0, .06])
    radius0 = float(cosmo.comoving_distance(.16).value*h)
    lower, upper = radius0-args.radius_bracket_mpc_h, radius0+args.radius_bracket_mpc_h
    print('Loading cleaned file-0 halo positions and masses', flush=True)
    cat = CompaSOHaloCatalog(str(HALO), cleaned=True, convert_units=True,
                             fields=['id', 'N', 'x_L2com'])
    masses = np.asarray(cat.halos['N'], dtype=float)*float(cat.header['ParticleMassHMsun'])/1e10
    # Preserve generator's float32 position arithmetic for selection.
    shifted = np.array(cat.halos['x_L2com'], copy=True)
    shifted += 1000.
    shifted[shifted > 1000.] -= 2000.
    shifted[shifted < -1000.] += 2000.
    radius = np.sqrt(np.sum(shifted**2, axis=1)).astype(float)
    eligible = masses >= 10.
    tree_rows = np.flatnonzero(eligible & (radius < radius0+2.))
    tree = cKDTree(np.asarray(shifted[tree_rows], dtype=float))
    central = (s['Z'] < .15) & (s['RES'] == 1) & (s['CEN'] == 1)
    sample_rows = np.flatnonzero(central)
    parent = np.full(len(s['Z']), -1, dtype=np.int64)
    multiplicity = np.zeros(len(s['Z']), dtype=np.int32)
    for row in sample_rows:
        near = tree_rows[tree.query_ball_point(s['sky_c000_massive_nu_approx'][row], .01)]
        near = near[np.abs(masses[near]/s['HALO_MASS'][row]-1.) < 1e-5]
        multiplicity[row] = len(near)
        if len(near) == 1:
            parent[row] = near[0]
    del tree
    matched = sample_rows[parent[sample_rows] >= 0]
    matched = matched[np.argsort(s['original_row'][matched])]
    training, heldout = matched[::2], matched[1::2]
    report.update(radius0_mpc_h=radius0, radius_bracket_mpc_h=[lower, upper],
                  n_centrals=len(sample_rows), n_unique_matches=len(matched),
                  n_missing=int(np.sum(multiplicity[sample_rows] == 0)),
                  n_ambiguous=int(np.sum(multiplicity[sample_rows] > 1)),
                  n_training=len(training), n_heldout=len(heldout),
                  native_position_dtype=str(shifted.dtype))
    np.savez(args.output/'independent_central_matches.npz', original_row=s['original_row'],
             parent_raw_row=parent, match_multiplicity=multiplicity,
             training_sample_rows=training, heldout_sample_rows=heldout)
    save()
    if len(training) < 10 or len(heldout) < 10:
        report.update(completed=True, conclusion='Insufficient unique central matches; no radius inferred.')
        save()
        return
    base_rows = np.flatnonzero(eligible & (radius < lower))
    events = np.flatnonzero(eligible & (radius >= lower) & (radius < upper))
    report['n_radius_boundary_events'] = len(events)
    intervals = feasible_intervals(np.searchsorted(base_rows, parent[training]),
        radius[events], events, parent[training], s['HALO_INDEX'][training], lower, upper)
    report['training_feasible_intervals_mpc_h'] = intervals
    if not intervals:
        report.update(completed=True, conclusion='No single radius in the declared bracket reproduces the training indices.')
        save()
        return
    # Choose the first feasible interval by a fixed rule, without consulting heldout.
    rtest = sum(intervals[0])/2.
    mapping = np.flatnonzero(eligible & (radius.astype(float) < rtest))
    report['test_radius_mpc_h'] = rtest
    report['heldout_rank_exact_matches'] = int(np.sum(np.searchsorted(mapping, parent[heldout]) == s['HALO_INDEX'][heldout]))
    report['heldout_parent_in_mapping'] = int(np.sum(radius[parent[heldout]].astype(float) < rtest))
    lowres = (s['Z'] < .15) & (s['RES'] == 1)
    inbounds = lowres & (s['HALO_INDEX'] >= 0) & (s['HALO_INDEX'] < len(mapping))
    raw = np.full(len(s['Z']), -1, dtype=np.int64)
    raw[inbounds] = mapping[s['HALO_INDEX'][inbounds].astype('i8')]
    residual = np.full(len(s['Z']), np.nan)
    mass_error = np.full(len(s['Z']), np.nan)
    residual[inbounds] = np.linalg.norm(periodic_delta(shifted[raw[inbounds]], s['sky_c000_massive_nu_approx'][inbounds]), axis=1)
    mass_error[inbounds] = np.abs(masses[raw[inbounds]]/s['HALO_MASS'][inbounds]-1.)
    report['resolved_results'] = {}
    for name, mask in [('central', lowres & (s['CEN'] == 1)), ('satellite', lowres & (s['CEN'] == 0))]:
        report['resolved_results'][name] = {'n': int(mask.sum()), 'inbounds': int((mask & inbounds).sum()),
            'position_error_mpc_h': stats(residual[mask]), 'mass_relative_error': stats(mass_error[mask]),
            'mass_matches_rtol1e5': int(np.sum(mask & (mass_error < 1e-5))),
            'positions_within_001': int(np.sum(mask & (residual < .01)))}
    np.save(args.output/'candidate_filtered_to_raw.npy', mapping)
    np.savez(args.output/'candidate_sample_mapping.npz', original_row=s['original_row'],
             raw_halo_row=raw, position_error_mpc_h=residual, mass_relative_error=mass_error)
    report.update(completed=True, conclusion='Radius candidate tested on heldout centrals; inspect all gates before any expansion. No production authorization.')
    save()
    print(args.output/'report.json', flush=True)


if __name__ == '__main__':
    main()
