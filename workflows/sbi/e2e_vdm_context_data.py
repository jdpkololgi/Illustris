"""Separate role-aware products for the approved VDM context/diversity matrix.

Geometry selection reads random support, never counts or matter. No relaxation
of old train-only guards. No output qualifies model training by its presence.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import time

import h5py
import numpy as np
from astropy.cosmology import Planck18

from workflows.sbi.e2e_field_build_products import require_compute, sha256
from workflows.sbi.e2e_durable import publish_json, single_writer

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / 'configs/e2e_vdm_context_diversity_v1.json'
ROLES = {'ph000': 'train', 'ph002': 'train', 'ph003': 'train',
         'ph004': 'internal_selection', 'ph005': 'internal_confirmation'}
SCRATCH = Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v2/wide_pipeline_v1')


def guarded(path, phase=None):
    path = Path(path)
    for value in (str(path), str(path.resolve())):
        phases = set(re.findall(r'ph\d{3}', value))
        if not phases <= ROLES.keys() or (phase and phases and phases != {phase}):
            raise PermissionError('forbidden or mismatched field phase')
    return path.resolve()


def read_json(path):
    return json.loads(guarded(path).read_text())


def spec():
    c = read_json(CONFIG)
    if (c['schema'] != 'e2e-vdm-context-diversity-v1' or c['phase_roles'] != ROLES
            or c['forbidden_phases'] != ['ph001', 'ph006'] or c['production_ready']
            or c['candidates_per_cap'] > 65536 or c['source_separation_mpc_h'] < 162.384):
        raise ValueError('unregistered context experiment contract')
    return c


def output_root(root):
    root = Path(root).resolve()
    if root.parent != SCRATCH or not root.name.startswith('vdm_context_'):
        raise ValueError('separate vdm_context_ Scratch child required')
    return root


def address(*parts):
    return int(hashlib.sha256(':'.join(map(str, parts)).encode()).hexdigest()[:16], 16) % (2**63-1)


def source_center(center, grid, c):
    # center is a voxel boundary: the symmetric core's physical centre has no +.5.
    return np.mod((np.asarray(grid['origin_mpc']) + np.asarray(center)*c['raw_cell_mpc'])
                  *c['coordinate_h'] + c['box_offset_mpc_h'], c['box_mpc_h'])


def periodic_delta(a, b, box):
    return (np.asarray(a)-np.asarray(b)+box/2) % box-box/2


def admissible(position, prior, c):
    if not prior:
        return True
    delta = np.abs(periodic_delta(position, prior, c['box_mpc_h']))
    # Separate exact axis-aligned core non-overlap and Euclidean distance tests.
    core = c['science_grid']*c['fine_cell_mpc_h']
    return bool(np.all(np.linalg.norm(delta, axis=1) >= c['source_separation_mpc_h']-1e-9)
                and np.all(np.any(delta >= core-1e-9, axis=1)))


def candidates(source, c):
    phase, cap, grid = source['phase'], source['cap'], source['grid']
    if phase not in ROLES or source['role'] != ROLES[phase]:
        raise PermissionError('phase/role mismatch before reading observations')
    response = guarded(source['response_file']['path'], phase)
    expected = source['response_file']['recorded_sha256']
    if sha256(response) != expected:
        raise ValueError('response source hash drift')
    z = np.linspace(0, .8, 4001)
    radius = Planck18.comoving_distance(z).value
    rng = np.random.default_rng(address(c['seed'], phase, cap, 'geometry'))
    # Parent48 fine=96 raw. No observed periodic wrapping; 16 raw-cell halo.
    margin, align = 64, c['centre_alignment_raw']
    centres = np.column_stack([rng.integers(margin//align, (n-margin)//align+1,
                               size=c['candidates_per_cap'])*align for n in grid['shape']])
    buckets = {(s, t): [] for s in range(4) for t in c['support_strata']}
    counts = Counter()
    with h5py.File(response, 'r') as f:
        support = f['support_random'][:].astype(bool)
        if list(support.shape) != grid['shape']:
            raise ValueError('support/grid mismatch')
        seen = set()
        for centre in centres:
            key = tuple(map(int, centre))
            if key in seen:
                counts['duplicate'] += 1
                continue
            seen.add(key)
            xyz = np.asarray(grid['origin_mpc'])+centre*c['raw_cell_mpc']
            redshift = float(np.interp(np.linalg.norm(xyz), radius, z))
            shell = int(np.searchsorted(c['shell_edges'], redshift, side='right')-1)
            if not 0 <= shell < 4:
                counts['outside_shell'] += 1
                continue
            fraction = float(support[tuple(slice(v-16, v+16) for v in centre)].mean())
            if fraction < c['support_min']:
                counts['low_support'] += 1
                continue
            support_bin = 'interior' if fraction >= c['interior_min'] else 'boundary'
            buckets[shell, support_bin].append(dict(phase=phase, cap=cap, role=ROLES[phase],
                center=list(key), grid=grid, shell=shell, redshift=redshift,
                support_stratum=support_bin, science_core_support_fraction=fraction,
                source_center_mpc_h=source_center(centre, grid, c).tolist()))
    return buckets, dict(rejected=dict(counts), candidates=len(centres),
                         eligible={f'{s}:{t}': len(v) for (s,t),v in buckets.items()},
                         response_sha256=expected, phase=phase, cap=cap)


def select_phase(pools, phase, c):
    """Round-robin strata/caps; all candidate priority depends only on geometry."""
    slots = [(cap, s, t) for s in range(4) for t in c['support_strata'] for cap in c['caps']]
    quota = c['train_per_stratum'] if ROLES[phase] == 'train' else c['evaluation_per_stratum']
    rows, positions, rejected, cursors = [], [], Counter(), Counter()
    missing = []
    for ordinal in range(quota):
        for cap, shell, kind in slots:
            slot = (cap, shell, kind)
            pool = pools[cap][shell, kind]
            accepted = False
            while cursors[slot] < len(pool):
                row = pool[cursors[slot]]
                cursors[slot] += 1
                position = row['source_center_mpc_h']
                if not admissible(position, positions, c):
                    rejected['separation_or_core_overlap'] += 1
                    continue
                rows.append(dict(row, anchor_id=f'{phase}_{cap}_s{shell}_{kind}_{ordinal:02d}',
                                 small_train=phase in c['small_phases'] and ordinal == 0))
                positions.append(position)
                accepted = True
                break
            if not accepted:
                missing.append(dict(cap=cap, shell=shell, support=kind, ordinal=ordinal))
    return rows, dict(missing=missing, rejected=dict(rejected), quota=quota)


def verify_geometry(rows, c):
    for i, row in enumerate(rows):
        if row['phase'] not in ROLES or row['role'] != ROLES[row['phase']]:
            raise PermissionError('row role leakage')
        if any(v % c['centre_alignment_raw'] for v in row['center']):
            raise ValueError('unaligned centre')
        if not np.allclose(source_center(row['center'], row['grid'], c), row['source_center_mpc_h']):
            raise ValueError('source coordinates drift')
        prior = [r['source_center_mpc_h'] for r in rows[:i] if r['phase'] == row['phase']]
        if not admissible(row['source_center_mpc_h'], prior, c):
            raise ValueError('source alias or separation failure')


def screen(root):
    require_compute()
    root = output_root(root)
    c = spec()
    started = time.monotonic()
    with single_writer(root/'data'):
        destination = root/'data/GEOMETRY.json'
        if destination.exists():
            r = read_json(destination)
            if r['config_sha256'] != sha256(CONFIG) or r['source_sha256'] != sha256(__file__):
                raise ValueError('geometry resume code/config drift')
            verify_geometry(r['rows'], c)
            if not r['quotas_pass']:
                raise RuntimeError('previous geometry quota failure requires explicit review')
            return r
        source_path = guarded(c['screened_source'])
        source = read_json(source_path)
        rows, screens, phase_checks = [], [], {}
        for phase in ROLES:
            pools = {}
            for cap in c['caps']:
                item = next(s for s in source['sources'] if s['phase'] == phase and s['cap'] == cap)
                pools[cap], receipt = candidates(item, c)
                screens.append(receipt)
                print('CANDIDATES', phase, cap, receipt['eligible'], flush=True)
            selected, check = select_phase(pools, phase, c)
            rows.extend(selected)
            phase_checks[phase] = check
            print('SELECTED', phase, len(selected), 'missing', check['missing'], flush=True)
        verify_geometry(rows, c)
        passed = not any(v['missing'] for v in phase_checks.values())
        result = dict(schema='e2e-vdm-context-geometry-v1', rows=rows, quotas_pass=passed,
            screening=screens, phase_checks=phase_checks, config_sha256=sha256(CONFIG),
            source_sha256=sha256(__file__), input_sha256=sha256(source_path),
            target_values_used=False, counts_used=False, phase_roles=ROLES,
            elapsed_seconds=time.monotonic()-started, job=os.environ['SLURM_JOB_ID'], node=socket.gethostname())
        publish_json(destination, result)
        if not passed:
            raise RuntimeError('fixed geometry quotas unmet; no distance relaxation or training')
        print('GEOMETRY_COMPLETE', len(rows), flush=True)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['screen'])
    parser.add_argument('--root', required=True, type=Path)
    args = parser.parse_args()
    screen(args.root)


if __name__ == '__main__':
    main()
