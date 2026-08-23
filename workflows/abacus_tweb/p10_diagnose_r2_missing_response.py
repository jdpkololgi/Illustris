#!/usr/bin/env python3
"""Diagnose undefined target-derived R2 assignment pixels.

The diagnostic distinguishes fixed survey-boundary support from phase-dependent
target-empty pixels.  It consumes only the ph000/ph006 angular canaries and never
opens catalogue or truth files.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import healpy as hp
import numpy as np

from p10_audit_r2_response_ladder import atomic_json, sha256


def support_distance_rings(support: np.ndarray, nside: int, maximum: int) -> np.ndarray:
    support = np.asarray(support, dtype=bool)
    pixel = np.arange(support.size, dtype=np.int64)
    neighbours = hp.get_all_neighbours(nside, pixel, nest=False)
    valid = neighbours >= 0
    neighbour_support = np.zeros(neighbours.shape, dtype=bool)
    neighbour_support[valid] = support[neighbours[valid]]
    boundary = support & np.any(~valid | ~neighbour_support, axis=0)

    distance = np.full(support.size, maximum + 1, dtype=np.int16)
    distance[~support] = -1
    distance[boundary] = 0
    frontier = boundary.copy()
    for ring in range(1, maximum + 1):
        frontier_index = np.flatnonzero(frontier)
        if not frontier_index.size:
            break
        candidate = neighbours[:, frontier_index].ravel()
        candidate = candidate[candidate >= 0]
        candidate = np.unique(candidate)
        new_frontier = np.zeros_like(frontier)
        new_frontier[candidate] = support[candidate] & (distance[candidate] > maximum)
        distance[new_frontier] = ring
        frontier = new_frontier
    return distance


def binned_counts(distance: np.ndarray, select: np.ndarray) -> dict[str, int]:
    return {
        "ring_0": int(np.count_nonzero(select & (distance == 0))),
        "ring_1": int(np.count_nonzero(select & (distance == 1))),
        "rings_2_4": int(np.count_nonzero(select & (distance >= 2) & (distance <= 4))),
        "rings_5_8": int(np.count_nonzero(select & (distance >= 5) & (distance <= 8))),
        "rings_9_16": int(np.count_nonzero(select & (distance >= 9) & (distance <= 16))),
        "rings_17_32": int(np.count_nonzero(select & (distance >= 17) & (distance <= 32))),
        "beyond_32": int(np.count_nonzero(select & (distance > 32))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="/pscratch/sd/d/dkololgi/abacus/p10_multiphase/r2_assignment_canary_v1",
    )
    parser.add_argument("--nside", type=int, default=256)
    parser.add_argument("--maximum-rings", type=int, default=32)
    args = parser.parse_args()

    root = Path(args.root)
    maps = {}
    sources = {}
    for phase in ("ph000", "ph006"):
        path = root / phase / "assignment_angular_canary.npz"
        with np.load(path) as archive:
            support = np.asarray(archive["support"], dtype=bool)
            has_target = np.asarray(archive["has_target"], dtype=bool)
        maps[phase] = {
            "support": support,
            "has_target": has_target,
            "uncovered": support & ~has_target,
        }
        sources[phase] = {"path": str(path), "sha256": sha256(path)}

    if not np.array_equal(maps["ph000"]["support"], maps["ph006"]["support"]):
        raise ValueError("ph000/ph006 random support differs")
    support = maps["ph000"]["support"]
    distance = support_distance_rings(support, args.nside, args.maximum_rings)
    u0 = maps["ph000"]["uncovered"]
    u6 = maps["ph006"]["uncovered"]
    union = u0 | u6
    intersection = u0 & u6
    jaccard = float(intersection.sum() / max(union.sum(), 1))

    phase_rows = {}
    for phase, uncovered in (("ph000", u0), ("ph006", u6)):
        counts = binned_counts(distance, uncovered)
        n = int(uncovered.sum())
        phase_rows[phase] = {
            "uncovered_pixels": n,
            "boundary_ring_counts": counts,
            "fraction_within_4_rings": float(
                np.count_nonzero(uncovered & (distance <= 4)) / max(n, 1)
            ),
            "fraction_within_8_rings": float(
                np.count_nonzero(uncovered & (distance <= 8)) / max(n, 1)
            ),
            "fraction_beyond_32_rings": float(
                np.count_nonzero(uncovered & (distance > 32)) / max(n, 1)
            ),
        }

    report = {
        "schema_version": "p10-r2-missing-response-diagnostic-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "blind_phase_opened": False,
        "nside": args.nside,
        "ordering": "RING",
        "maximum_boundary_rings": args.maximum_rings,
        "sources": sources,
        "support_pixels": int(support.sum()),
        "phases": phase_rows,
        "cross_phase": {
            "uncovered_intersection": int(intersection.sum()),
            "uncovered_union": int(union.sum()),
            "uncovered_jaccard": jaccard,
            "ph000_only": int(np.count_nonzero(u0 & ~u6)),
            "ph006_only": int(np.count_nonzero(u6 & ~u0)),
        },
        "decision_diagnostics": {
            "fixed_geometry_if_jaccard_ge_0p99": jaccard >= 0.99,
            "boundary_localized_if_both_within_8_ge_0p99": bool(
                all(row["fraction_within_8_rings"] >= 0.99 for row in phase_rows.values())
            ),
            "neutral_fill_not_automatically_authorized": True,
        },
    }
    out = root / "R2_MISSING_RESPONSE_DIAGNOSTIC.json"
    atomic_json(out, report)
    print(out)
    print(report["cross_phase"])
    print(report["decision_diagnostics"])
    for phase, row in phase_rows.items():
        print(phase, row)


if __name__ == "__main__":
    main()
