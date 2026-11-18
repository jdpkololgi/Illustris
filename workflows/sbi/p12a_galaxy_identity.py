"""Separate catalogue/galaxy identity from branch-local host grouping.

No host key is ever used to remove a satellite. Low-branch rows retain their
SOURCE_ROW identities; a geometric bound must exclude their periodic copies.
High-branch copies require matching cubic galaxy IDs AND positional evidence.
Cross-branch host closure is deliberately a separate, explicit requirement.
"""
from __future__ import annotations

import numpy as np

from workflows.abacus_tweb.p4_build_spatial_manifest import splitmix64


def identity_arrays(table, active, unwrapped_xyz, box_mpc_h=2000., tolerance_mpc_h=.005):
    active = np.asarray(active, bool)
    xyz = np.asarray(unwrapped_xyz, float)
    n = len(table)
    if active.shape != (n,) or xyz.shape != (n, 3) or not np.isfinite(xyz).all():
        raise ValueError('Identity geometry shape/finiteness failure')
    if len(np.unique(table['SOURCE_ROW'])) != n or len(np.unique(table['TARGETID'])) != n:
        raise ValueError('Catalogue SOURCE_ROW and TARGETID must each be unique')
    low = table['BOX_INDEX'] < 0
    radii = np.linalg.norm(xyz, axis=1)
    low_radius = float(radii[active & low].max(initial=0.))
    maximum = float(radii[active].max(initial=0.))
    # Triangle inequality: two images of a single periodic object are at least
    # L apart in unwrapped real space. This bound rules out every low-low AND
    # low-high image pair; a failed bound is not permission to deduplicate them.
    clearance = box_mpc_h-low_radius-maximum
    if low[active].any() and clearance <= 2*tolerance_mpc_h:
        raise ValueError('Low-branch periodic-copy exclusion needs explicit mapping')
    ids = np.flatnonzero(active & ~low)
    order = ids[np.lexsort((table['BOX_INDEX'][ids], table['FILE_NUM'][ids]))]
    same = ((table['FILE_NUM'][order[1:]] == table['FILE_NUM'][order[:-1]])
            & (table['BOX_INDEX'][order[1:]] == table['BOX_INDEX'][order[:-1]]))
    left, right = order[:-1][same], order[1:][same]
    for key in ('RES', 'CEN', 'HALO_INDEX', 'HALO_MASS'):
        if not np.array_equal(table[key][left], table[key][right]):
            raise ValueError('Repeated cubic galaxy has inconsistent '+key)
    delta = xyz[left]-xyz[right]
    shifts = np.rint(delta/box_mpc_h)
    residual = np.linalg.norm(delta-shifts*box_mpc_h, axis=1)
    if np.any(np.all(shifts == 0, axis=1)) or np.any(residual > tolerance_mpc_h):
        raise ValueError('Repeated cubic ID is not a verified periodic galaxy copy')
    eligible = active.copy()
    starts = np.r_[0, np.flatnonzero(~same)+1] if len(order) else np.array([], int)
    stops = np.r_[starts[1:], len(order)]
    for start, stop in zip(starts, stops):
        if stop-start > 1:
            members = order[start:stop]
            keeper = members[np.argmin(splitmix64(table['TARGETID'][members]))]
            eligible[members] = False
            eligible[keeper] = True
    host_keys = np.rec.fromarrays([low, table['RES'], table['FILE_NUM'], table['HALO_INDEX']],
                                 names='low_branch,res,file_num,halo_index')
    _, host_group = np.unique(host_keys, return_inverse=True)
    return dict(supervised_eligible=eligible, periodic_duplicate_image=active & ~eligible,
                branch_local_host_group=host_group.astype(np.int64)), dict(
        identity_pass=True, active_rows=int(active.sum()), eligible_rows=int(eligible.sum()),
        low_branch_active_rows=int((low & active).sum()), low_branch_rows_removed=0,
        verified_high_branch_copy_rows_removed=int((active & ~eligible).sum()),
        low_branch_max_realspace_radius_mpc_h=low_radius,
        active_max_realspace_radius_mpc_h=maximum,
        low_branch_periodic_exclusion_margin_mpc_h=clearance,
        high_copy_max_periodic_position_residual_mpc_h=float(residual.max(initial=0.)),
        galaxy_key='low branch: SOURCE_ROW; high branch: FILE_NUM, BOX_INDEX with periodic position proof',
        host_key='branch, RES, FILE_NUM, HALO_INDEX; never a galaxy deduplication key',
        cross_branch_host_closure_qualified=False, fold_policy_qualified=False)
