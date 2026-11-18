"""Random-support-only low-z domain candidates; not a final training panel.

Keep existing owned-core sizes, .25/.95 support criteria and periodic source
non-overlap. Measure feasible counts instead of requiring old equal quotas.
No galaxy counts, matter targets or learned predictions are used for selection.
Joint non-overlap with the old-range panel and development group allocation
remain required before this diagnostic inventory can become training input.
"""
from __future__ import annotations

import argparse
from collections import Counter
import os
from pathlib import Path

import numpy as np

from workflows.sbi import cfm_lowz_selection_response as lowz
from workflows.sbi import e2e_coupled_coordinates as coord
from workflows.sbi import e2e_coupled_geometry as geometry
from workflows.sbi.prepare_cfm_lowz_inputs import require_compute, save_json, sha256

EDGES = np.array([.01, .03, .05, .075, .10, .15])


def prefix_sum(support):
    support = np.asarray(support)
    if support.ndim != 3 or support.size >= 2**32 or not np.isin(support, [0, 1]).all():
        raise ValueError('Expected bounded binary support cube')
    result = np.zeros(tuple(n+1 for n in support.shape), dtype=np.uint32)
    result[1:, 1:, 1:] = support
    for axis in range(3):
        np.cumsum(result, axis=axis, dtype=np.uint32, out=result)
    return result


def box_counts(prefix, start, stop):
    """Vectorized exact counts in boxes, with explicit zero padding."""
    start, stop = np.asarray(start, dtype=int), np.asarray(stop, dtype=int)
    if start.shape != stop.shape or start.ndim != 2 or start.shape[1] != 3 or np.any(stop <= start):
        raise ValueError('Invalid boxes')
    shape = np.asarray(prefix.shape)-1
    lo, hi = np.clip(start, 0, shape), np.clip(stop, 0, shape)
    result = np.zeros(len(start), dtype=np.int64)
    for x in (0, 1):
        for y in (0, 1):
            for z in (0, 1):
                ids = np.column_stack([(hi if upper else lo)[:, axis]
                                       for axis, upper in enumerate((x, y, z))])
                result += (-1)**(3-x-y-z)*prefix[tuple(ids.T)].astype(np.int64)
    return result


def candidates(phase, cap, record, angular, per_bin=8192):
    import h5py
    import healpy as hp
    support_map = angular['support'].astype(bool) & (angular['domain']//2 == lowz.CAPS.index(cap))
    pixels = np.flatnonzero(support_map)
    if not len(pixels):
        raise ValueError('Empty angular footprint')
    rng = np.random.default_rng(geometry.address(60918, phase, cap, 'lowz-domain-feasibility-v1'))
    with h5py.File(record['path'], 'r') as handle:
        if handle.attrs['phase'] != phase or handle.attrs['cap'] != cap:
            raise ValueError('Wrong response identity')
        # Only this label-free geometry dataset is read.
        dataset = handle['support_random']
        origin = np.asarray(handle.attrs['origin_mpc_h'])
        cell = float(handle.attrs['cell_mpc_h'])
        if cell != coord.config()['raw_cell_mpc_h']:
            raise ValueError('Unexpected raw-cell convention')
        bound = float(coord.radius_mpc_h(EDGES[-1]))+40*cell
        start = np.maximum(0, np.floor((-bound-origin)/cell).astype(int))
        stop = np.minimum(dataset.shape, np.ceil((bound-origin)/cell).astype(int))
        prefix = prefix_sum(dataset[tuple(slice(a, b) for a, b in zip(start, stop))])
        grid = dict(origin_mpc_h=origin.tolist(), shape=list(dataset.shape), cell_mpc_h=cell,
                    padding_mpc_h=0., distance_unit='Mpc/h', coordinate_sha256=sha256(coord.CONFIG))
    rows, diagnostics = [], []
    seen = set()
    for shell, (lo, hi) in enumerate(zip(EDGES[:-1], EDGES[1:])):
        selected_pixels = rng.choice(pixels, per_bin)
        direction = np.asarray(hp.pix2vec(256, selected_pixels)).T
        rlo, rhi = coord.radius_mpc_h([lo, hi])
        radius = rng.uniform(rlo**3, rhi**3, per_bin)**(1/3)
        midpoint = direction*radius[:, None]
        centers = np.rint(((midpoint-origin)/cell-[16, 0, 0])/8).astype(int)*8
        centers = np.unique(centers, axis=0)
        centers = np.asarray([x for x in centers if tuple(x) not in seen], dtype=int).reshape(-1, 3)
        seen.update(map(tuple, centers))
        xyz = origin+(centers+[16, 0, 0])*cell
        actual_z = coord.redshift(np.linalg.norm(xyz, axis=1))
        in_bin = (actual_z >= lo) & (actual_z < hi)
        centers, actual_z, xyz = centers[in_bin], actual_z[in_bin], xyz[in_bin]
        fractions = np.column_stack([box_counts(prefix, centers+offset-16-start,
                                                 centers+offset+16-start)/32**3
                                      for offset in (np.zeros(3, int), np.array([32, 0, 0]))])
        minimum = fractions.min(axis=1)
        accepted = minimum >= .25
        diagnostics.append(dict(z=[float(lo), float(hi)], draws=per_bin,
            unique_candidates_in_bin=len(centers), supported_candidates=int(accepted.sum()),
            interior_candidates=int((minimum >= .95).sum()),
            max_minimum_core_support=None if not len(minimum) else float(minimum.max())))
        for center, redshift, fraction in zip(centers[accepted], actual_z[accepted], fractions[accepted]):
            source_mid, source_cores = geometry.domain_positions(center, grid)
            joint_mid = origin+(center+[16, 0, 0])*cell
            rows.append(dict(phase=phase, cap=cap, lowz_bin=shell, center=center.tolist(), grid=grid,
                redshift=float(redshift), science_core_support_fractions=fraction.tolist(),
                support_stratum='interior' if min(fraction) >= .95 else 'boundary',
                joint_box_contains_observer=bool(np.all(np.abs(joint_mid) <= np.array([64, 48, 48])*cell)),
                source_midpoint_mpc_h=source_mid.tolist(), source_owned_core_centers_mpc_h=source_cores.tolist()))
    return rows, diagnostics


def run(phase):
    require_compute()
    if phase not in ('ph001', 'ph006'):
        raise ValueError('Two-canary feasibility scope only')
    plan, policy, source = lowz.load_plan()
    lowz.phase_guard(phase, policy)
    root = Path(plan['output_root'])
    output = root/('GEOMETRY_CANDIDATES_' + phase + '.json')
    if output.exists():
        raise FileExistsError(output)
    receipt_path = root/'responses_full'/phase/'RESPONSE_PREPARED.json'
    receipt = lowz.read_json(receipt_path)
    if receipt['phase'] != phase or receipt['code_bindings'] != lowz.code_bindings():
        raise ValueError('Response provenance mismatch')
    bindings = {str(path): sha256(path) for path in (Path(__file__), Path(geometry.__file__), Path(coord.__file__))}
    angular_path, angular_binding = lowz.source_payload(Path(source['root']), phase,
        'angular/ANGULAR_COMPLETE.json', plan['source_campaign']['sha256'])
    with np.load(angular_path, allow_pickle=False) as data:
        angular = {k: data[k] for k in ('support', 'domain')}
    rows, diagnostics = [], {}
    for cap, record in zip(lowz.CAPS, receipt['outputs']):
        if Path(record['path']).stat().st_size != record['bytes']:
            raise ValueError('Response size changed')
        new, diagnostics[cap] = candidates(phase, cap, record, angular)
        rows.extend(new)
        print(phase, cap, 'eligible candidates', len(new), flush=True)
    # Round-robin the fine redshift bins/caps. This is a deterministic feasible
    # subset, NOT a maximum-packing proof or a new equal-quota requirement.
    buckets = {(cap, shell): [r for r in rows if r['cap'] == cap and r['lowz_bin'] == shell]
               for cap in lowz.CAPS for shell in range(len(EDGES)-1)}
    cursors = Counter()
    chosen, mids, cores = [], [], []
    while True:
        added = False
        for key, bucket in buckets.items():
            while cursors[key] < len(bucket):
                row = bucket[cursors[key]]; cursors[key] += 1
                mid, pair = np.asarray(row['source_midpoint_mpc_h']), np.asarray(row['source_owned_core_centers_mpc_h'])
                if geometry.admissible(mid, pair, mids, cores):
                    chosen.append(row); mids.append(mid); cores.extend(pair); added = True
                    break
        if not added:
            break
    if any(sha256(path) != value for path, value in bindings.items()):
        raise ValueError('Geometry implementation changed during preparation')
    save_json(output, dict(phase=phase, job=os.environ['SLURM_JOB_ID'], diagnostics=diagnostics,
        candidate_count=len(rows), selected_nonoverlapping_candidates=chosen,
        selected_counts={str(key): sum(r['cap'] == key[0] and r['lowz_bin'] == key[1] for r in chosen)
                         for key in buckets},
        response_receipt_sha256=sha256(receipt_path), angular_source=angular_binding,
        code_bindings=bindings,
        counts_or_targets_read=False, original_support_and_nonoverlap_rules_preserved=True,
        full_response_hash_reverified=False, joint_old_range_nonoverlap_verified=False,
        development_partition_frozen=False, final_training_panel=False, model_inputs_ready=False))
    print('SAVED', output, 'selected', len(chosen), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('ph001', 'ph006'), required=True)
    run(parser.parse_args().phase)
