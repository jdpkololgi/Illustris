"""All-compatible host candidates; no nearest-neighbour host assignment.

Completeness is conditional on the pinned generator's mass and NFW/SO-radius
contract. A spatial ball only generates candidates; every group member then
checks every candidate. No finite candidate cap or ranking is permitted.
"""
import numpy as np
from scipy.spatial import cKDTree

POSITION_TOLERANCE = .005
MASS_TOLERANCE = 1e-5
BOX = 2000.


def delta(a, b):
    d = np.asarray(a) - np.asarray(b)
    return d - BOX * np.rint(d / BOX)


def compatible_candidates(positions, masses, central, groups, native_positions,
                          native_mass, native_radius):
    """Return every compatible native row for each supplied local group.

    Input groups belong to one resolved FILE_NUM namespace. Native inputs are
    the entire cleaned slab, without any attempted low-redshift index filter.
    """
    positions, native_positions = map(lambda x: np.asarray(x, float),
                                     (positions, native_positions))
    masses, native_mass, native_radius = map(np.asarray, (masses, native_mass, native_radius))
    central = np.asarray(central, bool)
    groups = np.asarray(groups)
    if (positions.shape != (len(groups), 3) or masses.shape != groups.shape
            or central.shape != groups.shape or native_positions.shape != (len(native_mass), 3)
            or native_radius.shape != native_mass.shape or not np.isfinite(positions).all()
            or not np.isfinite(native_positions).all() or not np.isfinite(masses).all()
            or not np.isfinite(native_mass).all() or not np.isfinite(native_radius).all()
            or np.any(masses <= 0) or np.any(native_mass <= 0) or np.any(native_radius < 0)):
        raise ValueError('Invalid candidate inputs')
    order = np.argsort(groups, kind='stable')
    starts = np.r_[0, np.flatnonzero(np.diff(groups[order])) + 1] if len(order) else np.array([], int)
    stops = np.r_[starts[1:], len(order)]
    tree = cKDTree(np.mod(native_positions, BOX), boxsize=BOX)
    radius = float(native_radius.max(initial=0)) + POSITION_TOLERANCE
    for offset in range(0, len(starts), 8192):
        block_starts = starts[offset:offset+8192]
        anchors = order[block_starts]
        balls = tree.query_ball_point(np.mod(positions[anchors], BOX),
                                      np.nextafter(radius, np.inf), workers=8)
        for local, (start, candidates) in enumerate(zip(block_starts, balls)):
            stop = stops[offset+local]
            members = order[start:stop]
            candidates = np.asarray(sorted(candidates), dtype=np.int64)
            for member in members:
                good_mass = np.abs(native_mass[candidates]/masses[member]-1) < MASS_TOLERANCE
                candidates = candidates[good_mass]
                distance = np.linalg.norm(delta(native_positions[candidates], positions[member]), axis=1)
                limit = (POSITION_TOLERANCE if central[member]
                         else native_radius[candidates]+POSITION_TOLERANCE)
                candidates = candidates[distance <= limit]
                if not len(candidates):
                    break
            yield int(groups[order[start]]), candidates


def brute_candidates(positions, masses, central, native_positions, native_mass, native_radius):
    """Independent full-table enumeration used only on bounded audit samples."""
    good = np.ones(len(native_mass), bool)
    for xyz, mass, cen in zip(positions, masses, central):
        displacement = np.asarray(native_positions, float)-xyz
        displacement = (displacement+BOX/2) % BOX-BOX/2
        distance = np.sqrt(np.sum(displacement**2, axis=1))
        limit = POSITION_TOLERANCE if cen else native_radius+POSITION_TOLERANCE
        good &= (np.abs(native_mass/mass-1) < MASS_TOLERANCE) & (distance <= limit)
    return np.flatnonzero(good)
