#!/usr/bin/env python3
"""Export comoving Cartesian points from a staged mock wedge truth NPZ.

Writes ``<out-prefix>_points_xyz.npy`` (N,3) for ``build_abacus_graph.py --points-path``.
Node order matches the truth NPZ row order; graph + SBI cache must use the same ordering.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DEFAULT_TRUTH_NPZ = (
    "/pscratch/sd/d/dkololgi/abacus/SecondGen_Mocks/ph000/wedge/"
    "staged_mock_wedge_stage3_postcollision_rs7.npz"
)
DEFAULT_OUT_DIR = "/pscratch/sd/d/dkololgi/abacus/graph_constructions"
DEFAULT_PREFIX = "staged_mock_wedge_stage3_postcollision_rs7"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--truth-npz", type=Path, default=Path(DEFAULT_TRUTH_NPZ))
    p.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT_DIR))
    p.add_argument("--out-prefix", type=str, default=DEFAULT_PREFIX)
    p.add_argument(
        "--coord-keys",
        nargs=3,
        default=("x", "y", "z_comoving_cart"),
        metavar=("X", "Y", "Z"),
        help="NPZ keys for comoving Mpc/h Cartesian coordinates (default: x y z_comoving_cart).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    truth = args.truth_npz.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with np.load(truth) as data:
        keys = set(data.files)
        cols = []
        for key in args.coord_keys:
            resolved = key if key in keys else None
            if resolved is None:
                alt = {"z_comoving_cart": "z"}.get(key)
                if alt and alt in keys:
                    resolved = alt
            if resolved is None:
                raise KeyError(f"Coordinate key {key!r} not in {truth}. Have: {sorted(keys)}")
            cols.append(np.asarray(data[resolved], dtype=np.float64))
        xyz = np.stack(cols, axis=-1)

    prefix = args.out_prefix
    points_path = out_dir / f"{prefix}_points_xyz.npy"
    meta_path = out_dir / f"{prefix}_points_export.json"
    np.save(points_path, xyz)
    meta = {
        "truth_npz": str(truth),
        "n_points": int(xyz.shape[0]),
        "coord_keys": list(args.coord_keys),
        "points_xyz_path": str(points_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {xyz.shape[0]:,} points -> {points_path}")
    print(f"Metadata -> {meta_path}")


if __name__ == "__main__":
    main()
