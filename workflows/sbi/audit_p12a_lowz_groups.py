"""Audit low-z legacy duplicate keys on the two approved preparation phases."""
import json
import os
from pathlib import Path
import socket

import fitsio
import numpy as np

from workflows.abacus_tweb.p10_training_contract import atomic_json, sha256

ROOT = Path('/pscratch/sd/d/dkololgi/abacus/p12a_lowz_20261029_v1')


def audit(phase):
    if phase not in ('ph001', 'ph006'):
        raise ValueError('Two-phase audit only')
    path = ROOT / phase / 'p1_canonical/manifest.json'
    p1 = json.loads(path.read_text())
    table = fitsio.read(p1['parent'], columns=['TARGETID', 'SOURCE_ROW', 'Z', 'Z_COSMO',
                                               'FILE_NUM', 'HALO_INDEX', 'BOX_INDEX', 'RES', 'CEN'])
    points = np.load(ROOT / phase / 'p1_canonical/points.npy', mmap_mode='r')
    selected = (table['Z'] >= .01) & (table['Z'] < .15)
    ids = np.flatnonzero(selected)
    t = table[ids]
    keys = np.rec.fromarrays([t[k] for k in ('FILE_NUM', 'BOX_INDEX', 'HALO_INDEX')])
    _, inverse, count = np.unique(keys, return_inverse=True, return_counts=True)
    order = np.argsort(inverse, kind='stable')
    same = inverse[order[1:]] == inverse[order[:-1]]
    left, right = order[:-1][same], order[1:][same]
    probe = np.linspace(0, len(left)-1, min(8192, len(left)), dtype=int)
    separation = np.linalg.norm(points[ids[left[probe]], :3] - points[ids[right[probe]], :3], axis=1)
    record = dict(phase=phase, p1_sha256=sha256(path), source_sha256=sha256(Path(__file__)),
                  job=os.environ['SLURM_JOB_ID'], rows=len(ids),
                  unique_source_rows=len(np.unique(t['SOURCE_ROW'])),
                  legacy_key_groups=len(count), legacy_key_repeated_groups=int((count > 1).sum()),
                  rows_old_rule_would_discard=int(len(ids)-len(count)),
                  satellite_rows=int((t['CEN'] == 0).sum()),
                  repeated_pair_sample=len(probe),
                  sampled_pair_separation_mpc_quantiles=np.quantile(separation, [0, .5, .95, 1]).tolist(),
                  sampled_pairs_below_old_1000mpc_duplicate_gate=int((separation < 1000).sum()),
                  repeated_pairs_different_resolution=int((t['RES'][left] != t['RES'][right]).sum()),
                  low_branch_zcosmo_max=float(table['Z_COSMO'][table['BOX_INDEX'] < 0].max()),
                  high_branch_zcosmo_min=float(table['Z_COSMO'][table['BOX_INDEX'] >= 0].min()),
                  model_inputs_ready=False, grouping_qualified=False)
    destination = ROOT / phase / 'LOWZ_GROUP_AUDIT.json'
    if destination.exists():
        raise FileExistsError(destination)
    atomic_json(destination, record)
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    if not os.environ.get('SLURM_JOB_ID') or not socket.gethostname().startswith('nid'):
        raise RuntimeError('Approved compute-node step required')
    for phase in ('ph001', 'ph006'):
        audit(phase)
