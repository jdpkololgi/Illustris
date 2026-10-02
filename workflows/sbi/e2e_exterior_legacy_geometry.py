"""Equivalent support-only legacy geometry with one coordinate validation.

Vectorized box sums avoid repeatedly reading the immutable coordinate authority
inside a 131072-candidate loop. Candidate RNG, order and selection are unchanged.
"""
import argparse
import json
import gc
from pathlib import Path
import h5py
import numpy as np
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_coordinates as coord
from workflows.sbi import e2e_coupled_geometry as geometry
from workflows.sbi import e2e_exterior_legacy as legacy


def box_fractions(support, centers, radius=16):
    prefix = np.pad(np.asarray(support, np.int64), ((1, 0),)*3)
    for axis in range(3):
        np.cumsum(prefix, axis=axis, out=prefix)
    low, high = np.asarray(centers)-radius, np.asarray(centers)+radius
    if np.any(low < 0) or np.any(high > np.asarray(support.shape)):
        raise ValueError('Candidate support cube leaves raw grid')
    total = np.zeros(len(low), np.int64)
    for i in (0, 1):
        for j in (0, 1):
            for k in (0, 1):
                index = (high[:, 0] if i else low[:, 0], high[:, 1] if j else low[:, 1],
                         high[:, 2] if k else low[:, 2])
                total += (1 if (i+j+k) % 2 else -1)*prefix[index]
    return total/(2*radius)**3


def build(phase):
    c.require_compute(); legacy.guard(phase)
    marker = legacy.ROOT/phase/'GEOMETRY.json'
    if marker.exists():
        return legacy.receipt(marker, phase, payload=False)
    buckets, sources = {}, []
    for cap in ('NGC', 'SGC'):
        path = legacy.ROOT/phase/(cap+'_RESPONSE.json')
        raw = legacy.receipt(path, phase); grid = raw['grid']; coord.validate_grid(grid)
        sources.append(c.file_record(path, content_hash=True))
        rng = np.random.default_rng(geometry.address(60918, phase, cap, 'pair-geometry'))
        centers = np.column_stack([rng.integers(2, (side-margin)//8+1, 65536)*8
                                   for side, margin in zip(grid['shape'], (96, 64, 64))])
        _, unique = np.unique(centers, axis=0, return_index=True)
        centers = centers[np.sort(unique)]
        with h5py.File(raw['outputs'][0]['path']) as f:
            support = f['support_random'][:].astype(bool)
        fraction = np.stack([box_fractions(support, centers+[32*side, 0, 0]) for side in (0, 1)], axis=1)
        del support; gc.collect()
        origin = np.asarray(grid['origin_mpc_h'])
        xyz = origin+(centers+[16, 0, 0])*3.383
        redshift = coord.redshift(np.linalg.norm(xyz, axis=1))
        shell = np.searchsorted((.15, .25, .35, .45, .55), redshift, side='right')-1
        eligible = (shell >= 0) & (shell < 4) & (fraction.min(1) >= .25)
        for center, z, sh, fractions in zip(centers[eligible], redshift[eligible], shell[eligible], fraction[eligible]):
            midpoint = (origin+(center+[16, 0, 0])*3.383-1000.) % 2000.
            pair = np.stack([(origin+(center+[32*side, 0, 0])*3.383-1000.) % 2000. for side in (0, 1)])
            kind = 'interior' if fractions.min() >= .95 else 'boundary'
            row = dict(phase=phase, role='legacy_nonblind_replication', cap=cap,
                       center=center.tolist(), grid=grid, shell=int(sh), redshift=float(z),
                       support_stratum=kind, science_core_support_fractions=fractions.tolist(),
                       source_midpoint_mpc_h=midpoint.tolist(), source_owned_core_centers_mpc_h=pair.tolist())
            buckets.setdefault((cap, int(sh), kind), []).append(row)
        print(json.dumps(dict(phase=phase, cap=cap, support_candidates=len(centers), eligible=int(eligible.sum()))), flush=True)
    selected, mids, cores = [], [], []
    for cap in ('NGC', 'SGC'):
        for shell in range(4):
            for kind in ('interior', 'boundary'):
                accepted = None
                for row in buckets.get((cap, shell, kind), []):
                    mid, pair = np.asarray(row['source_midpoint_mpc_h']), np.asarray(row['source_owned_core_centers_mpc_h'])
                    if geometry.admissible(mid, pair, mids, cores):
                        accepted = dict(row, pair_id=f'{phase}_{cap}_s{shell}_{kind}_00')
                        mids.append(mid); cores.extend(pair); break
                if accepted is None:
                    raise ValueError('No admissible legacy support-only core')
                # Independently verify selected coordinates with the original
                # helper; only 16 calls, not repeated for every candidate.
                expected_mid, expected_pair = geometry.domain_positions(accepted['center'], accepted['grid'])
                np.testing.assert_array_equal(expected_mid, mid)
                np.testing.assert_array_equal(expected_pair, pair)
                selected.append(accepted)
    record = dict(legacy.record_base(phase), pairs=selected, sources=sources, outputs=[], passed=True,
                  counts_or_targets_read=False, source_nonoverlap=True,
                  geometry_builder_sha256=c.sha256(__file__), selection_order_unchanged=True)
    c.atomic_json(marker, record)
    return record


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--phase', required=True); a = p.parse_args(); build(a.phase)
