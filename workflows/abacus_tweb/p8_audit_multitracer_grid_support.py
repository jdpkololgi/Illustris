#!/usr/bin/env python3
"""Measure Faint CIC support loss against the frozen Bright P3 grids.

This is diagnostic only: it never changes the catalogue, field grid, or gate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from workflows.abacus_tweb.p8_deterministic_common import atomic_json


ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p8_multitracer_v1")
P3 = Path("/pscratch/sd/d/dkololgi/abacus/p3_full_footprint/field_manifest.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--p3-manifest", type=Path, default=P3)
    parser.add_argument(
        "--products",
        nargs="+",
        default=["bf_oracle_assigned_v1", "bf_proxy_response_v1"],
    )
    return parser.parse_args()


def cic_support_audit(xyz: np.ndarray, grid: dict) -> dict:
    origin = np.asarray(grid["origin_mpc"], dtype=np.float64)
    shape = np.asarray(grid["shape"], dtype=np.int64)
    cell = float(grid["cell_mpc"])
    u = (np.asarray(xyz, dtype=np.float64) - origin) / cell - 0.5
    i0 = np.floor(u).astype(np.int64)
    frac = u - i0
    deposited = 0.0
    lost = 0.0
    complete = np.ones(len(xyz), dtype=bool)
    any_support = np.zeros(len(xyz), dtype=bool)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                offset = np.asarray([dx, dy, dz], dtype=np.int64)
                index = i0 + offset
                weight = (
                    np.where(dx, frac[:, 0], 1.0 - frac[:, 0])
                    * np.where(dy, frac[:, 1], 1.0 - frac[:, 1])
                    * np.where(dz, frac[:, 2], 1.0 - frac[:, 2])
                )
                valid = np.all((index >= 0) & (index < shape), axis=1)
                complete &= valid
                any_support |= valid & (weight > 0)
                deposited += float(np.sum(weight[valid], dtype=np.float64))
                lost += float(np.sum(weight[~valid], dtype=np.float64))
    total = deposited + lost
    return {
        "points": int(len(xyz)),
        "fully_supported_points": int(np.count_nonzero(complete)),
        "partially_supported_points": int(np.count_nonzero(any_support & ~complete)),
        "fully_outside_points": int(np.count_nonzero(~any_support)),
        "deposited_weight": deposited,
        "lost_weight": lost,
        "lost_weight_fraction": lost / total if total else 0.0,
        "fractional_index_min": np.min(u, axis=0).tolist() if len(u) else [None] * 3,
        "fractional_index_max": np.max(u, axis=0).tolist() if len(u) else [None] * 3,
        "grid_index_min": [0, 0, 0],
        "grid_index_max": (shape - 1).tolist(),
    }


def main() -> None:
    args = parse_args()
    p3 = json.loads(args.p3_manifest.read_text())
    result = {"schema_version": "p8-multitracer-grid-support-audit-v1", "products": {}}
    for product in args.products:
        manifest = json.loads(
            (args.root / "catalogues" / product / "manifest.json").read_text()
        )
        points = np.load(manifest["points"], mmap_mode="r")
        index = np.load(manifest["index"])
        bright = int(manifest["bright_prefix_rows"])
        tracer = np.asarray(index["tracer_type"], dtype=np.uint8)
        context = np.asarray(index["context"], dtype=bool)
        cap = np.asarray(index["cap"], dtype=np.uint8)
        result["products"][product] = {}
        for cap_id, cap_name in ((0, "SGC"), (1, "NGC")):
            selected = (np.arange(len(points)) >= bright) & (tracer == 1) & context & (cap == cap_id)
            result["products"][product][cap_name] = cic_support_audit(
                np.asarray(points[selected, :3]), p3["components"][cap_name]["grid"]
            )
    output = args.root / "fields" / "grid_support_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
